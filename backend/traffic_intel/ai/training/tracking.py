"""Multi-object tracker comparison on UA-DETRAC (offline; never imported by the pipeline or the API).

    python -m traffic_intel.ai.training train tracking --config configs/training/tracking_ua_detrac.yaml [--smoke]

Trackers are not learned: ByteTrack and BoT-SORT (the registered ``Tracker`` implementations) are
compared, and their association thresholds tuned with a small grid on the VALIDATION sequences (the
same whole sequences held out by the detection experiment). The chosen setting of each tracker is then
scored once on the official TEST sequences. Detections come from the configured detector through the
production adapter and are cached per sequence as Parquet (frame-level data stays in Parquet).

Metrics (CLEAR-MOT, IoU 0.5; ground truth and predictions centred in ignored regions are removed):
MOTA, MOTP (IoU), precision, recall, identity switches, tracker update time per frame.
"""

from __future__ import annotations

import itertools
import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict

from traffic_intel.core.config import PROJECT_ROOT, load_yaml

log = logging.getLogger(__name__)


class TrackerCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: str                                   # bytetrack | botsort (TRACKERS registry)
    grid: dict[str, list[Any]] = {}             # tuned on validation MOTA; other params = pipeline.yaml


class TrackingExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    description: str = ""
    dataset: str = "ua_detrac"
    detector_weights: str = "models/detection/yolo11m.pt"
    detector_imgsz: int = 960
    detector_conf: float = 0.1                  # low: the trackers' second (BYTE) round uses low-score boxes
    val_split_file: str = "data/processed/ua_detrac_yolo/split.json"   # validation sequences of the detector
    test_sequences: int | None = None          # None = all official test sequences
    cache_dir: str = "data/interim/ua_detrac_detections"
    trackers: list[TrackerCfg]


def load_tracking_config(path: Path) -> TrackingExperimentConfig:
    return TrackingExperimentConfig.model_validate(load_yaml(path))


def _cache_detections(cfg: TrackingExperimentConfig, split: str, seq: str, adapter: Any) -> pd.DataFrame:  # noqa: ANN401
    """Detections for every annotated frame of a sequence (production adapter), cached as Parquet."""
    import cv2

    from traffic_intel.computer_vision.detectors.yolo import YOLODetector
    from traffic_intel.core.types import Frame

    key = Path(cfg.detector_weights).stem if not cfg.detector_weights.startswith("registry:") \
        else cfg.detector_weights.split("/")[-1].replace("@", "_")
    out = PROJECT_ROOT / cfg.cache_dir / f"{key}_{cfg.detector_imgsz}" / f"{seq}.parquet"
    if out.exists():
        return pd.read_parquet(out)
    global _DETECTOR
    if _DETECTOR is None:
        _DETECTOR = YOLODetector(cfg.detector_weights, conf_threshold=cfg.detector_conf, imgsz=cfg.detector_imgsz,
                                 class_map={"car": "car", "bus": "bus", "truck": "truck"}, device="auto", half="auto")
    info = next(s for s in adapter.sequences() if s.sequence_id == seq)
    rows = []
    frames = [f.frame_index for f in adapter.frames(seq)]
    now = datetime.now(UTC)
    for i in range(0, len(frames), 16):
        chunk = frames[i:i + 16]
        imgs = [cv2.imread(str(Path(info.media) / f"img{k + 1:05d}.jpg")) for k in chunk]
        ok = [(k, im) for k, im in zip(chunk, imgs, strict=True) if im is not None]
        dets = _DETECTOR.detect([Frame(seq, k, now, np.asarray(im, dtype=np.uint8)) for k, im in ok])
        for (k, _), d in zip(ok, dets, strict=True):
            for b, c, s in zip(d.xyxy, d.class_id, d.confidence, strict=True):
                rows.append((k, float(s), int(c), *(float(v) for v in b)))
    df = pd.DataFrame(rows, columns=["frame", "conf", "class_id", "x1", "y1", "x2", "y2"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return df


_DETECTOR: Any = None


def _in_regions(boxes: np.ndarray, regions: np.ndarray) -> np.ndarray:
    if not len(boxes) or not len(regions):
        return np.zeros(len(boxes), bool)
    cx, cy = (boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2
    r = regions
    return np.asarray(((cx[:, None] >= r[None, :, 0]) & (cx[:, None] <= r[None, :, 2])
                       & (cy[:, None] >= r[None, :, 1]) & (cy[:, None] <= r[None, :, 3])).any(1))


def run_tracker(tracker_type: str, params: dict[str, Any], seqs: list[tuple[str, str]], cfg: TrackingExperimentConfig,
                adapters: dict[str, Any]) -> dict[str, Any]:
    import cv2

    from traffic_intel.ai.evaluation.detection import MOTEvaluator
    from traffic_intel.core.registry import ComponentSpec
    from traffic_intel.core.types import Detections, Frame, Tracks
    from traffic_intel.tracking.interfaces import TRACKERS

    total = MOTEvaluator()
    upd_s = 0.0
    n_frames = 0
    needs_image = tracker_type == "botsort"
    now = datetime.now(UTC)
    for split, seq in seqs:
        a = adapters[split]
        info = next(s for s in a.sequences() if s.sequence_id == seq)
        dets = _cache_detections(cfg, split, seq, a).groupby("frame")
        by_frame = {int(k): g for k, g in dets}
        tracker = TRACKERS.create(ComponentSpec(type=tracker_type, params=params), fps=info.fps)
        ev = MOTEvaluator()
        blank = np.zeros((info.frame_size[1], info.frame_size[0], 3), np.uint8) if info.frame_size else \
            np.zeros((540, 960, 3), np.uint8)
        for fr in a.frames(seq):
            g = by_frame.get(fr.frame_index)
            d = Detections(g[["x1", "y1", "x2", "y2"]].to_numpy(np.float32), g["conf"].to_numpy(np.float32),
                           g["class_id"].to_numpy(np.int64)) if g is not None else \
                Detections(np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int64))
            img = cv2.imread(str(Path(info.media) / f"img{fr.frame_index + 1:05d}.jpg")) if needs_image else blank
            t0 = time.perf_counter()
            tr = tracker.update(d, Frame(seq, fr.frame_index, now,
                                         np.asarray(img, dtype=np.uint8) if img is not None else blank))
            upd_s += time.perf_counter() - t0
            n_frames += 1
            ig = fr.ignore_regions
            keep_p = ~_in_regions(tr.xyxy, ig)
            keep_g = ~_in_regions(fr.tracks.xyxy, ig)
            ev.add(Tracks(tr.track_id[keep_p], tr.xyxy[keep_p], tr.confidence[keep_p], tr.class_id[keep_p]),
                   Tracks(fr.tracks.track_id[keep_g], fr.tracks.xyxy[keep_g], fr.tracks.confidence[keep_g],
                          fr.tracks.class_id[keep_g]))
        for k in ("gt_total", "fn", "fp", "idsw", "matches"):
            setattr(total, k, getattr(total, k) + getattr(ev, k))
        total.iou_sum += ev.iou_sum
    return total.results() | {"frames": n_frames, "ms_per_frame": round(1000 * upd_s / max(n_frames, 1), 3)}


def run_tracking_experiment(cfg: TrackingExperimentConfig, smoke: bool = False) -> dict[str, Any]:
    from traffic_intel.core.config import load_pipeline_config
    from traffic_intel.datasets.adapters.base import adapter_for

    exp_id = datetime.now().strftime("%Y%m%d-%H%M%S") + f"_{cfg.name}" + ("_smoke" if smoke else "")
    out = PROJECT_ROOT / "experiments" / "tracking" / exp_id
    out.mkdir(parents=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False), encoding="utf-8")
    adapters: dict[str, Any] = {"train": adapter_for(cfg.dataset, split="train"),
                                "test": adapter_for(cfg.dataset, split="test")}
    split = json.loads((PROJECT_ROOT / cfg.val_split_file).read_text(encoding="utf-8"))
    val = [("train", s) for s in split["val_sequences"]]
    test_ids = sorted(s.sequence_id for s in adapters["test"].sequences())
    if cfg.test_sequences:
        test_ids = test_ids[::max(1, len(test_ids) // cfg.test_sequences)][:cfg.test_sequences]
    test = [("test", s) for s in test_ids]
    if smoke:
        val, test = val[:2], test[:2]
    base = dict(load_pipeline_config().tracker.params)
    grid_rows, results = [], []
    for t in cfg.trackers:
        keys = list(t.grid)
        combos = list(itertools.product(*[t.grid[k] for k in keys])) or [()]
        if smoke:
            combos = combos[:1]
        best: tuple[float, dict[str, Any]] | None = None
        for values in combos:
            params = {**base, **dict(zip(keys, values, strict=True))}
            r = run_tracker(t.type, params, val, cfg, adapters)
            grid_rows.append({"tracker": t.type, "params": json.dumps({k: params[k] for k in keys}), **r})
            log.info("val %s %s: MOTA %.4f IDsw %d", t.type, {k: params[k] for k in keys}, r["MOTA"], r["id_switches"])
            if best is None or r["MOTA"] > best[0]:
                best = (r["MOTA"], params)
        assert best is not None
        r = run_tracker(t.type, best[1], test, cfg, adapters)
        results.append({"tracker": t.type, "chosen_on_validation": {k: best[1][k] for k in keys},
                        "val_MOTA": best[0], **r})
        log.info("TEST %s: %s", t.type, r)
    pd.DataFrame(grid_rows).to_csv(out / "grid_search_val.csv", index=False)
    pd.DataFrame(results).to_csv(out / "metrics_test.csv", index=False)
    record = {"experiment_id": exp_id, "detector": cfg.detector_weights, "detector_imgsz": cfg.detector_imgsz,
              "val_sequences": [s for _, s in val], "test_sequences": [s for _, s in test], "results": results,
              "created": datetime.now(UTC).isoformat()}
    (out / "experiment.json").write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    idx = out.parent / "index.csv"
    old = pd.read_csv(idx) if idx.exists() else pd.DataFrame()
    df = pd.DataFrame(results).assign(experiment_id=exp_id, detector=cfg.detector_weights)
    df["chosen_on_validation"] = df["chosen_on_validation"].map(json.dumps)
    pd.concat([old, df], ignore_index=True).to_csv(idx, index=False)
    return record


__all__ = ["TrackingExperimentConfig", "load_tracking_config", "run_tracking_experiment"]
