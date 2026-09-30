"""Vehicle detector fine-tuning and evaluation (offline; never imported by the pipeline or the API).

    python -m traffic_intel.ai.training train detection --config configs/training/detection_ua_detrac.yaml

Data: the dataset adapter (UA-DETRAC: official train / test splits, project classes) ->
    build_dataset()   YOLO folder: validation = whole TRAIN sequences (never frames of a sequence
                      in two splits); frames subsampled (adjacent frames are near-duplicates);
                      ignored regions greyed out and their labels dropped
    train_model()     Ultralytics fine-tuning from COCO-pretrained weights (transfer learning)
    evaluate()        through the PRODUCTION inference path (YOLODetector.detect + ClassMapper), so
                      COCO and fine-tuned models are scored identically: P / R at the pipeline's
                      confidence threshold, mAP50 and mAP50-95 (Ultralytics ap_per_class, 10 IoU
                      thresholds), per class, plus inference latency
    register()        best weights -> models/detection/<name>/<version>/ (ModelStore, metadata)
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field

from traffic_intel.ai.model_store import FileModelStore, ModelMetadata, TrainableModel
from traffic_intel.core.config import PROJECT_ROOT, get_settings, load_yaml
from traffic_intel.core.enums import ModelStatus, ModelTask, ObjectClass, class_id_of

log = logging.getLogger(__name__)
IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)
_TEST_LIMIT: int | None = None                       # smoke runs: first N test images only
GREY = 114                                            # Ultralytics letterbox fill value


# --------------------------------------------------------------------------- config
class DatasetCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = "ua_detrac"
    val_sequences: int = 10                  # whole train sequences held out for validation
    frame_step_train: int = 10               # keep every k-th annotated frame
    frame_step_eval: int = 10
    mask_ignored: bool = True
    output: str = "data/processed/ua_detrac_yolo"
    # Repeat-factor sampling (LVIS, Gupta et al. 2019) of TRAINING images for rare classes: an image is
    # repeated r = max over its classes of max(1, sqrt(t / f_c)) times (f_c = share of training images
    # containing class c; stochastic rounding, seeded). None = off. Validation / test are never touched.
    repeat_factor_threshold: float | None = None


class TrainCfg(BaseModel):
    model_config = ConfigDict(extra="allow")   # passed to Ultralytics train() as-is (recorded)
    epochs: int = 40
    imgsz: int = 960
    batch: int = 16
    patience: int = 10


class BaseModelCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str                                 # registry name of the fine-tuned model
    weights: str                              # COCO-pretrained starting point
    batch: int | None = None                  # override TrainCfg.batch (bigger models need less)


class DetectionExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    description: str = ""
    seed: int = 0
    dataset: DatasetCfg = Field(default_factory=DatasetCfg)
    classes: list[ObjectClass] = [ObjectClass.CAR, ObjectClass.BUS, ObjectClass.TRUCK]
    models: list[BaseModelCfg]
    train: TrainCfg = Field(default_factory=TrainCfg)
    eval_imgsz: list[int] = [960]
    eval_conf: float = 0.001                  # mAP needs the full precision / recall curve
    operating_conf: float = 0.25              # the pipeline's detector.conf_threshold (P / R reported here)
    baselines: bool = True                    # also evaluate the COCO-pretrained weights zero-shot


def load_detection_config(path: Path) -> DetectionExperimentConfig:
    return DetectionExperimentConfig.model_validate(load_yaml(path))


# --------------------------------------------------------------------------- dataset
def _split_sequences(train_seqs: list[tuple[str, str]], n_val: int) -> tuple[list[str], list[str]]:
    """Deterministic, weather-stratified choice of whole validation sequences."""
    ordered = sorted(train_seqs, key=lambda s: (s[1] or "", s[0]))
    step = max(1, len(ordered) // max(n_val, 1))
    val = [s for i, (s, _) in enumerate(ordered) if i % step == step // 2][:n_val]
    return [s for s, _ in train_seqs if s not in val], val


def build_dataset(cfg: DetectionExperimentConfig) -> Path:
    from traffic_intel.datasets.adapters.base import DetectionTrackingAdapter, adapter_for

    out = PROJECT_ROOT / cfg.dataset.output
    marker = out / "split.json"
    if marker.exists():
        log.info("YOLO dataset exists: %s", out)
        return out
    train_a = adapter_for(cfg.dataset.id, split="train")
    test_a = adapter_for(cfg.dataset.id, split="test")
    assert isinstance(train_a, DetectionTrackingAdapter) and isinstance(test_a, DetectionTrackingAdapter)
    seqs = train_a.sequences()
    train_ids, val_ids = _split_sequences([(s.sequence_id, s.metadata.get("weather") or "") for s in seqs],
                                          cfg.dataset.val_sequences)
    test_ids = [s.sequence_id for s in test_a.sequences()]
    yolo_idx = {class_id_of(c): i for i, c in enumerate(cfg.classes)}
    jobs = [(cfg.dataset.id, "train", sid, "train", cfg.dataset.frame_step_train) for sid in train_ids] + \
           [(cfg.dataset.id, "train", sid, "val", cfg.dataset.frame_step_eval) for sid in val_ids] + \
           [(cfg.dataset.id, "test", sid, "test", cfg.dataset.frame_step_eval) for sid in test_ids]
    for sub in ("train", "val", "test"):
        (out / "images" / sub).mkdir(parents=True, exist_ok=True)
        (out / "labels" / sub).mkdir(parents=True, exist_ok=True)
    stats: dict[str, dict[str, int]] = {s: {"sequences": 0, "images": 0, "boxes": 0, "dropped_in_ignored": 0}
                                        for s in ("train", "val", "test")}
    ignore: dict[str, list[list[float]]] = {}
    with ProcessPoolExecutor(max_workers=6) as ex:
        for res in ex.map(_export_sequence, jobs, [str(out)] * len(jobs), [yolo_idx] * len(jobs),
                          [cfg.dataset.mask_ignored] * len(jobs)):
            st = stats[res["split"]]
            st["sequences"] += 1
            for k in ("images", "boxes", "dropped_in_ignored"):
                st[k] += res[k]
            ignore[res["sequence"]] = res["ignore"]
    (out / "data.yaml").write_text(yaml.safe_dump({
        "path": str(out), "train": "images/train", "val": "images/val", "test": "images/test",
        "names": {i: c.value for i, c in enumerate(cfg.classes)}}, sort_keys=False), encoding="utf-8")
    (out / "ignore_regions.json").write_text(json.dumps(ignore), encoding="utf-8")
    marker.write_text(json.dumps({"dataset": cfg.dataset.id, "train_sequences": train_ids, "val_sequences": val_ids,
                                  "test_sequences": test_ids, "stats": stats,
                                  "frame_step": {"train": cfg.dataset.frame_step_train,
                                                 "eval": cfg.dataset.frame_step_eval},
                                  "classes": [c.value for c in cfg.classes], "mask_ignored": cfg.dataset.mask_ignored,
                                  "created": datetime.now(UTC).isoformat()}, indent=2), encoding="utf-8")
    log.info("YOLO dataset: %s", json.dumps(stats))
    return out


def _export_sequence(job: tuple[str, str, str, str, int], out: str, yolo_idx: dict[int, int],
                     mask: bool) -> dict[str, Any]:
    """One sequence -> images + YOLO labels (runs in a worker process)."""
    import cv2

    from traffic_intel.datasets.adapters.base import DetectionTrackingAdapter, adapter_for

    dsid, split, sid, sub, step = job
    a = adapter_for(dsid, split=split)
    assert isinstance(a, DetectionTrackingAdapter)
    info = next(s for s in a.sequences() if s.sequence_id == sid)
    n_img = n_box = dropped = 0
    ign = np.zeros((0, 4), np.float32)
    for fr in a.frames(sid):
        if fr.frame_index % step:
            continue
        src = Path(info.media or "") / f"img{fr.frame_index + 1:05d}.jpg"
        img = cv2.imread(str(src))
        if img is None:                                   # annotation without an image: skip, never guess
            continue
        h, w = img.shape[:2]
        ign = fr.ignore_regions
        b, c = fr.tracks.xyxy, fr.tracks.class_id
        keep = np.array([int(k) in yolo_idx for k in c], bool) if len(c) else np.zeros(0, bool)
        if mask and len(ign):
            for x1, y1, x2, y2 in ign.astype(int):
                img[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)] = GREY
            cx, cy = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2
            inside = ((cx[:, None] >= ign[None, :, 0]) & (cx[:, None] <= ign[None, :, 2])
                      & (cy[:, None] >= ign[None, :, 1]) & (cy[:, None] <= ign[None, :, 3])).any(1) \
                if len(b) else np.zeros(0, bool)
            dropped += int((inside & keep).sum())
            keep &= ~inside
        name = f"{sid}_{fr.frame_index:05d}"
        cv2.imwrite(str(Path(out) / "images" / sub / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        lines = []
        for (x1, y1, x2, y2), k in zip(b[keep], c[keep], strict=True):
            x1, x2 = max(0.0, float(x1)), min(float(w), float(x2))
            y1, y2 = max(0.0, float(y1)), min(float(h), float(y2))
            if x2 <= x1 or y2 <= y1:
                continue
            lines.append(f"{yolo_idx[int(k)]} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                         f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
        (Path(out) / "labels" / sub / f"{name}.txt").write_text("\n".join(lines), encoding="utf-8")
        n_img += 1
        n_box += len(lines)
    return {"split": sub, "sequence": sid, "images": n_img, "boxes": n_box, "dropped_in_ignored": dropped,
            "ignore": ign.tolist()}


# --------------------------------------------------------------------------- training
def repeat_factor_list(data_dir: Path, t: float, seed: int, n_classes: int) -> tuple[Path, dict[str, Any]]:
    """Training image list with rare-class images repeated (see DatasetCfg.repeat_factor_threshold).
    Returns a data yaml whose ``train`` is that list (val / test unchanged) and the sampling statistics."""
    images = sorted((data_dir / "images" / "train").glob("*.jpg"))
    present: list[set[int]] = []
    boxes: list[np.ndarray] = []
    for img in images:
        rows = [ln.split()[0] for ln in (data_dir / "labels" / "train" / f"{img.stem}.txt").read_text().splitlines()
                if ln.strip()]
        cls = np.array(rows, int)
        present.append(set(cls.tolist()))
        boxes.append(np.bincount(cls, minlength=n_classes))
    f = np.array([sum(c in p for p in present) for c in range(n_classes)], float) / max(len(images), 1)
    r_class = np.maximum(1.0, np.sqrt(t / np.maximum(f, 1e-12)))
    rng = np.random.default_rng(seed)
    out: list[str] = []
    before, after = np.zeros(n_classes, int), np.zeros(n_classes, int)
    for img, p, b in zip(images, present, boxes, strict=True):
        r = max((r_class[c] for c in p), default=1.0)
        k = int(np.floor(r)) + int(rng.random() < r - np.floor(r))
        out += [img.as_posix()] * k
        before += b
        after += k * b
    lst = data_dir / f"train_rfs_t{t:g}.txt"
    lst.write_text("\n".join(out) + "\n", encoding="utf-8")
    base = yaml.safe_load((data_dir / "data.yaml").read_text(encoding="utf-8"))
    y = data_dir / f"data_rfs_t{t:g}.yaml"
    y.write_text(yaml.safe_dump(base | {"train": lst.name}, sort_keys=False), encoding="utf-8")
    stats = {"threshold": t, "image_share_per_class": np.round(f, 4).tolist(),
             "repeat_factor_per_class": np.round(r_class, 3).tolist(), "images_before": len(images),
             "images_after": len(out), "boxes_per_class_before": before.tolist(),
             "boxes_per_class_after": after.tolist()}
    log.info("repeat-factor sampling: %s", stats)
    return y, stats


def train_model(base: BaseModelCfg, cfg: DetectionExperimentConfig, data_dir: Path, run_dir: Path) -> dict[str, Any]:
    from ultralytics import YOLO  # type: ignore[attr-defined]

    from traffic_intel.computer_vision.detectors.yolo import resolve_weights

    params = cfg.train.model_dump()
    if base.batch:
        params["batch"] = base.batch
    data_yaml, sampling = data_dir / "data.yaml", None
    if cfg.dataset.repeat_factor_threshold:
        data_yaml, sampling = repeat_factor_list(data_dir, cfg.dataset.repeat_factor_threshold, cfg.seed,
                                                 len(cfg.classes))
    model = YOLO(resolve_weights(base.weights, auto_download=True))
    t0 = time.perf_counter()
    model.train(data=str(data_yaml), project=str(run_dir), name=base.name, exist_ok=True,
                seed=cfg.seed, device=_device(), verbose=False, plots=True, **params)
    seconds = time.perf_counter() - t0
    wdir = run_dir / base.name / "weights"
    results = pd.read_csv(run_dir / base.name / "results.csv")
    results.columns = [c.strip() for c in results.columns]
    best_epoch = int(results["metrics/mAP50-95(B)"].idxmax()) + 1
    return {"best": str(wdir / "best.pt"), "train_seconds": round(seconds, 1), "epochs_run": len(results),
            "best_epoch": best_epoch, "hyperparameters": params,
            "val_mAP50-95": float(results["metrics/mAP50-95(B)"].max()), "sampling": sampling}


def _device() -> str:
    import torch

    return "0" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------- evaluation
def _test_set(data_dir: Path, classes: list[ObjectClass]) -> list[tuple[Path, np.ndarray, np.ndarray, np.ndarray]]:
    """(image, gt xyxy [N,4] px, gt project class ids [N], ignored regions [K,4]) per test image."""
    import cv2

    ignore = json.loads((data_dir / "ignore_regions.json").read_text(encoding="utf-8"))
    out = []
    ids = [class_id_of(c) for c in classes]
    size: tuple[int, int] | None = None
    for img in sorted((data_dir / "images" / "test").glob("*.jpg")):
        if size is None:
            first = cv2.imread(str(img))
            if first is None:
                raise OSError(f"unreadable test image {img}")
            size = (first.shape[1], first.shape[0])
        w, h = size
        rows = [ln.split() for ln in (data_dir / "labels" / "test" / f"{img.stem}.txt").read_text().splitlines()
                if ln.strip()]
        a = np.array(rows, float).reshape(-1, 5)
        xyxy = np.c_[(a[:, 1] - a[:, 3] / 2) * w, (a[:, 2] - a[:, 4] / 2) * h,
                     (a[:, 1] + a[:, 3] / 2) * w, (a[:, 2] + a[:, 4] / 2) * h]
        cls = np.array([ids[int(k)] for k in a[:, 0]], np.int64)
        seq = img.stem.rsplit("_", 1)[0]
        out.append((img, xyxy, cls, np.array(ignore.get(seq, []), float).reshape(-1, 4)))
    return out[::max(1, len(out) // _TEST_LIMIT)][:_TEST_LIMIT] if _TEST_LIMIT else out


def _box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return np.asarray(inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9))


def _match(pred_cls: np.ndarray, gt_cls: np.ndarray, iou: np.ndarray) -> np.ndarray:
    """Ultralytics' greedy matching (BaseValidator.match_predictions): [N_pred, 10] correct flags."""
    correct = np.zeros((len(pred_cls), len(IOU_THRESHOLDS)), bool)
    if not len(pred_cls) or not len(gt_cls):
        return correct
    iou = iou * (gt_cls[:, None] == pred_cls[None, :])                  # [gt, pred]
    for i, thr in enumerate(IOU_THRESHOLDS):
        m = np.array(np.nonzero(iou >= thr)).T
        if m.shape[0]:
            if m.shape[0] > 1:
                m = m[iou[m[:, 0], m[:, 1]].argsort()[::-1]]
                m = m[np.unique(m[:, 1], return_index=True)[1]]
                m = m[np.unique(m[:, 0], return_index=True)[1]]
            correct[m[:, 1].astype(int), i] = True
    return correct


def _in_regions(boxes: np.ndarray, regions: np.ndarray) -> np.ndarray:
    if not len(boxes) or not len(regions):
        return np.zeros(len(boxes), bool)
    cx, cy = (boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2
    r = regions
    return np.asarray(((cx[:, None] >= r[None, :, 0]) & (cx[:, None] <= r[None, :, 2])
                       & (cy[:, None] >= r[None, :, 1]) & (cy[:, None] <= r[None, :, 3])).any(1))


def evaluate(weights: str, data_dir: Path, cfg: DetectionExperimentConfig, imgsz: int, label: str,
             fig_dir: Path | None = None) -> dict[str, Any]:
    """Score one model on the held-out TEST split through the production detector adapter."""
    from ultralytics.utils.metrics import ap_per_class

    from traffic_intel.computer_vision.detectors.yolo import YOLODetector
    from traffic_intel.core.types import Frame

    class_map: dict[str | int, str] = {c.value: c.value for c in cfg.classes}  # model label -> ObjectClass
    det = YOLODetector(weights, conf_threshold=cfg.eval_conf, iou_threshold=0.7, imgsz=imgsz, max_det=300,
                       class_map=class_map, device="auto", half="auto")
    test = _test_set(data_dir, cfg.classes)
    tp_all, conf_all, pcls_all, tcls_all = [], [], [], []
    op_tp = op_fp = 0
    n_gt = 0
    confusion: dict[tuple[int, int], int] = {}
    t_infer = 0.0
    batch = 16
    now = datetime.now(UTC)
    det.warmup()
    for i in range(0, len(test), batch):
        chunk = test[i:i + batch]
        frames = [Frame("eval", k, now, _read(p)) for k, (p, *_rest) in enumerate(chunk)]
        t0 = time.perf_counter()
        results = det.detect(frames)
        t_infer += time.perf_counter() - t0
        for (_, gt, gcls, ign), d in zip(chunk, results, strict=True):
            keep = ~_in_regions(d.xyxy, ign)                              # detections in ignored regions do not count
            pb, pc, pcls = d.xyxy[keep], d.confidence[keep], d.class_id[keep]
            iou = _box_iou(gt, pb)
            correct = _match(pcls, gcls, iou)
            tp_all.append(correct)
            conf_all.append(pc)
            pcls_all.append(pcls)
            tcls_all.append(gcls)
            n_gt += len(gcls)
            op = pc >= cfg.operating_conf
            c50 = correct[:, 0]
            op_tp += int((c50 & op).sum())
            op_fp += int((~c50 & op).sum())
            _confusion(confusion, gt, gcls, pb[op], pcls[op])
    tp = np.concatenate(tp_all) if tp_all else np.zeros((0, 10), bool)
    conf = np.concatenate(conf_all) if conf_all else np.zeros(0)
    pcls = np.concatenate(pcls_all) if pcls_all else np.zeros(0)
    tcls = np.concatenate(tcls_all) if tcls_all else np.zeros(0)
    names = {class_id_of(c): c.value for c in cfg.classes}
    save = fig_dir / f"{label}_imgsz{imgsz}" if fig_dir else Path(".")
    if fig_dir:
        save.mkdir(parents=True, exist_ok=True)
    res = ap_per_class(tp, conf, pcls, tcls, plot=fig_dir is not None, save_dir=save, names=names)
    _, _, p, r, _f1, ap, uc = res[:7]
    per_class = {names[int(c)]: {"P_maxF1": round(float(p[j]), 4), "R_maxF1": round(float(r[j]), 4),
                                 "mAP50": round(float(ap[j, 0]), 4), "mAP50-95": round(float(ap[j].mean()), 4),
                                 "instances": int((tcls == c).sum())} for j, c in enumerate(uc)}
    out = {"model": label, "weights": str(weights), "imgsz": imgsz, "test_images": len(test), "instances": n_gt,
           "precision": round(op_tp / max(op_tp + op_fp, 1), 4), "recall": round(op_tp / max(n_gt, 1), 4),
           "operating_conf": cfg.operating_conf,
           "mAP50": round(float(ap[:, 0].mean()), 4) if len(ap) else 0.0,
           "mAP50-95": round(float(ap.mean()), 4) if len(ap) else 0.0,
           "ms_per_image": round(1000 * t_infer / max(len(test), 1), 2), "device": det.device_info.describe(),
           "per_class": per_class,
           "confusion_iou50": {f"{names.get(g, 'background')}->{names.get(pr, 'background')}": n
                               for (g, pr), n in sorted(confusion.items())}}
    log.info("%s @%d: P %.3f R %.3f mAP50 %.3f mAP50-95 %.3f (%.1f ms/img)", label, imgsz, out["precision"],
             out["recall"], out["mAP50"], out["mAP50-95"], out["ms_per_image"])
    return out


def _read(p: Path) -> np.ndarray:
    import cv2

    img = cv2.imread(str(p))
    if img is None:
        raise OSError(f"unreadable image {p}")
    return np.asarray(img, dtype=np.uint8)


def _confusion(acc: dict[tuple[int, int], int], gt: np.ndarray, gcls: np.ndarray, pb: np.ndarray,
               pcls: np.ndarray) -> None:
    """Class confusion at IoU 0.5 (operating threshold). -1 = background."""
    iou = _box_iou(gt, pb)
    used: set[int] = set()
    for g in range(len(gt)):
        j = int(np.argmax(iou[g])) if iou.shape[1] else -1
        if j >= 0 and iou[g, j] >= 0.5 and j not in used:
            used.add(j)
            key = (int(gcls[g]), int(pcls[j]))
        else:
            key = (int(gcls[g]), -1)
        acc[key] = acc.get(key, 0) + 1
    for j in range(len(pb)):
        if j not in used:
            acc[(-1, int(pcls[j]))] = acc.get((-1, int(pcls[j])), 0) + 1


# --------------------------------------------------------------------------- registry
class YoloWeights(TrainableModel):
    """A fine-tuned Ultralytics checkpoint as a registry artifact (model.pt)."""

    def __init__(self, path: str) -> None:
        self.path = path

    def save(self, directory: Path) -> None:
        shutil.copy2(self.path, directory / "model.pt")

    @classmethod
    def load(cls, directory: Path) -> YoloWeights:
        return cls(str(directory / "model.pt"))


def register(name: str, version: str, weights: str, cfg: DetectionExperimentConfig, split: dict[str, Any],
             train_info: dict[str, Any], metrics: dict[str, Any]) -> Path:
    import hashlib

    meta = ModelMetadata(
        task=ModelTask.DETECTION, name=name, version=version, framework="ultralytics",
        feature_schema_hash=hashlib.sha1(",".join(c.value for c in cfg.classes).encode()).hexdigest()[:12],
        trained_at=datetime.now(UTC),
        train_data={"dataset": cfg.dataset.id, "train_sequences": split["train_sequences"],
                    "val_sequences": split["val_sequences"], "test_sequences": len(split["test_sequences"]),
                    "images": split["stats"], "classes": split["classes"], "frame_step": split["frame_step"],
                    "sampling": train_info.get("sampling")},
        params={"type": "yolo", "base_weights": train_info["base_weights"], "seed": cfg.seed,
                **train_info["hyperparameters"], "best_epoch": train_info["best_epoch"],
                "epochs_run": train_info["epochs_run"], "train_seconds": train_info["train_seconds"]},
        metrics={"test": metrics}, status=ModelStatus.CANDIDATE)
    return FileModelStore(get_settings().models_dir).save(YoloWeights(weights), meta)


# --------------------------------------------------------------------------- experiment
def run_detection_experiment(cfg: DetectionExperimentConfig, only: list[str] | None = None,
                             smoke: bool = False, baselines_only: bool = False) -> dict[str, Any]:
    """``smoke``: 1 epoch on 5% of the training images, 160 test images, nothing registered -
    proves the whole pipeline runs before the full training.
    ``baselines_only``: evaluate the COCO-pretrained weights zero-shot, no training."""
    if smoke:
        cfg = cfg.model_copy(deep=True)
        cfg.name += "_smoke"
        cfg.train = TrainCfg.model_validate(cfg.train.model_dump() | {"epochs": 1, "fraction": 0.05})
        cfg.eval_imgsz = cfg.eval_imgsz[:1]
        global _TEST_LIMIT
        _TEST_LIMIT = 160
    exp_id = datetime.now().strftime("%Y%m%d-%H%M%S") + f"_{cfg.name}"
    out = PROJECT_ROOT / "experiments" / "detection" / exp_id
    # a smoke run is a pipeline check, not a result: its figures stay in its experiment folder
    fig = out / "figures" if smoke else PROJECT_ROOT / "reports" / "figures" / "detection" / exp_id
    out.mkdir(parents=True, exist_ok=True)
    data_dir = build_dataset(cfg)
    split = json.loads((data_dir / "split.json").read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    record: dict[str, Any] = {"experiment_id": exp_id, "config": cfg.model_dump(mode="json"), "dataset_split": split,
                              "started": datetime.now(UTC).isoformat(), "models": []}
    for base in cfg.models:
        if only and base.name not in only:
            continue
        if cfg.baselines or baselines_only:
            for s in cfg.eval_imgsz:
                m = evaluate(base.weights, data_dir, cfg, s, f"{Path(base.weights).stem}_coco_zero_shot", fig)
                rows.append(m | {"kind": "COCO-pretrained (zero-shot)", "train_seconds": 0})
            _write(out, record, rows)                   # recorded even if a later training is interrupted
        if baselines_only:
            continue
        info = train_model(base, cfg, data_dir, out / "runs")
        info["base_weights"] = base.weights
        best_metrics = []
        for s in cfg.eval_imgsz:
            m = evaluate(info["best"], data_dir, cfg, s, base.name, fig)
            rows.append(m | {"kind": "fine-tuned", "train_seconds": info["train_seconds"]})
            best_metrics.append(m)
        path = None if smoke else register(base.name, exp_id, info["best"], cfg, split, info,
                                           {"by_imgsz": best_metrics})
        record["models"].append({"name": base.name, "artifact": str(path), **info, "test": best_metrics})
        _write(out, record, rows)
    record["finished"] = datetime.now(UTC).isoformat()
    _write(out, record, rows)
    _plots(rows, fig)
    return record


def _write(out: Path, record: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    full = record | {"evaluations": rows}          # every evaluated model incl. baselines (per class, confusion)
    (out / "experiment.json").write_text(json.dumps(full, indent=2, default=str), encoding="utf-8")
    flat = [{k: v for k, v in r.items() if k not in ("per_class", "confusion_iou50")} |
            {f"{c}_mAP50-95": pc["mAP50-95"] for c, pc in r.get("per_class", {}).items()} for r in rows]
    pd.DataFrame(flat).to_csv(out / "metrics_test.csv", index=False)
    idx = out.parent / "index.csv"
    df = pd.DataFrame([{"experiment_id": record["experiment_id"], **r} for r in flat])
    old = pd.read_csv(idx) if idx.exists() else pd.DataFrame()
    if not old.empty:
        old = old[old["experiment_id"] != record["experiment_id"]]
    pd.concat([old, df], ignore_index=True).to_csv(idx, index=False)


def _plots(rows: list[dict[str, Any]], fig: Path) -> None:
    if not rows:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df["label"] = df["model"] + "@" + df["imgsz"].astype(str)
    ax = df.set_index("label")[["mAP50", "mAP50-95", "precision", "recall"]].plot.barh(figsize=(8, 0.6 * len(df) + 2))
    ax.set_xlim(0, 1)
    ax.set_title("UA-DETRAC official test split (held-out sequences)")
    plt.tight_layout()
    plt.savefig(fig / "model_comparison.png", dpi=130)
    plt.close("all")
