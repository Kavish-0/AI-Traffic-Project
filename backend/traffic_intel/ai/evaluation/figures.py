"""Evaluation figures under ``reports/figures/``, built from finished experiment records and the
registered models - nothing is retrained and no number is typed by hand.

    python -m traffic_intel.ai.training figures [--no-rescore] [--no-samples]

    forecasting/<exp>/  RMSE by horizon and prediction vs actual (copied from the run), plus error over
                        time and by hour of day for the validation-selected model vs persistence. These
                        re-score the REGISTERED models on the rebuilt test windows; the recomputed RMSE
                        must match the experiment record or the figure is not written.
    anomaly/<exp>/      score timeline (threshold + ground-truth incident periods), PR curves
    detection/<exp>/    PR/F1 curves (written by the experiment), per-class mAP, test confusion matrix,
                        class distribution per split, sample test predictions, Ultralytics training plots
    congestion/, incident/   written by their experiments
    INDEX.md            every figure, by task
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import yaml

from traffic_intel.ai.evaluation import plots
from traffic_intel.ai.evaluation.summary import (
    EXP,
    FORECAST_CONFIGS,
    _latest,
    detection_rows,
    detection_runs,
    selected_model,
)
from traffic_intel.core.config import PROJECT_ROOT

FIG = PROJECT_ROOT / "reports" / "figures"
log = logging.getLogger(__name__)


def _copy(src: Path, dst: Path) -> Path | None:
    if not src.exists():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


# --------------------------------------------------------------------------- forecasting
def forecasting_figures(exp_dir: Path, rescore: bool = True) -> list[Path]:
    from traffic_intel.ai.evaluation.metrics import rmse
    from traffic_intel.ai.model_store import FileModelStore
    from traffic_intel.ai.training.experiment import UNITS, ExperimentConfig, prepare_data
    from traffic_intel.core.config import get_settings
    from traffic_intel.core.enums import ModelTask
    from traffic_intel.forecasting.interfaces import Forecaster

    cfg = ExperimentConfig.model_validate(yaml.safe_load((exp_dir / "config.yaml").read_text(encoding="utf-8")))
    selected, tgt = selected_model(exp_dir), cfg.primary_target
    out = FIG / "forecasting" / exp_dir.name
    copied = [_copy(exp_dir / "plots" / f"rmse_by_horizon_{tgt}.png", out / f"rmse_by_horizon_{tgt}.png"),
              _copy(exp_dir / "plots" / f"pred_vs_actual_{selected}_{tgt}.png",
                    out / f"pred_vs_actual_{selected}_{tgt}.png")]
    files = [f for f in copied if f is not None]
    if not rescore:
        return files
    info = json.loads((exp_dir / "dataset.json").read_text(encoding="utf-8"))
    data = prepare_data(cfg)
    test = data.windows("test", int(info["window_lookback_steps"]))
    k = data.pre.target_columns.index(tgt)
    y = data.pre.inverse_targets(test.y())[..., k]
    recorded = pd.read_csv(exp_dir / "metrics_test.csv")
    store = FileModelStore(get_settings().models_dir)
    preds: dict[str, np.ndarray] = {}
    for name in dict.fromkeys([selected, "persistence"]):
        model = cast("Forecaster", store.load(ModelTask.FORECASTING, name, exp_dir.name))
        p = data.pre.inverse_targets(model.predict(test))[..., k]
        for j, h in enumerate(cfg.horizons_min):      # the rebuilt windows + registered model reproduce the record
            want = recorded[(recorded["model"] == name) & (recorded["target"] == tgt)
                            & (recorded["horizon_min"] == h)]["RMSE"].iloc[0]
            got = rmse(y[:, j], p[:, j])
            if not np.isclose(got, want, rtol=2e-3, atol=1e-3):
                raise RuntimeError(f"{exp_dir.name}/{name} +{h} min: re-scored RMSE {got:.4f} != recorded {want:.4f}")
        preds[name] = p
    times = test.origin_times()
    span = pd.Timestamp(info["periods"]["test"]["end"]) - pd.Timestamp(info["periods"]["test"]["start"])
    freq = "1D" if span > pd.Timedelta(days=10) else "6h" if span > pd.Timedelta(days=2) else "1h"
    unit = UNITS.get(tgt, "")
    n = len(info["series"])
    files.append(plots.error_profile(times, y, preds, cfg.horizons_min, unit, data.ds.tz,
                                     out / f"error_over_time_{tgt}.png",
                                     f"{cfg.name}: test MAE over time - {tgt}, {n} series pooled", "time", freq))
    files.append(plots.error_profile(times, y, preds, cfg.horizons_min, unit, data.ds.tz,
                                     out / f"error_by_hour_{tgt}.png",
                                     f"{cfg.name}: test MAE by hour of day - {tgt}, {n} series pooled", "hour"))
    log.info("forecasting figures %s: %s re-scored, RMSE matches the record", exp_dir.name, list(preds))
    return files


# --------------------------------------------------------------------------- anomaly
def anomaly_figures(exp_dir: Path, days: int = 3, fig_dir: Path | None = None) -> list[Path]:
    """Needs ``scores.parquet`` (written by anomaly runs since the figures were added)."""
    if not (exp_dir / "scores.parquet").exists():
        log.warning("%s has no scores.parquet - rerun the anomaly experiment for its figures", exp_dir.name)
        return []
    sc = pd.read_parquet(exp_dir / "scores.parquet")
    m = pd.read_csv(exp_dir / "metrics.csv")
    ev = pd.read_csv(exp_dir / "events.csv")
    ev["start"], ev["end"] = pd.to_datetime(ev["start"], utc=True), pd.to_datetime(ev["end"], utc=True)
    out = fig_dir or FIG / "anomaly" / exp_dir.name
    ranked = m.sort_values("pr_auc", ascending=False)
    learned = [d for d in ("isolation_forest", "autoencoder") if d in set(m["detector"])]
    stat = [d for d in ranked["detector"] if d not in learned and d != "fused_probability"][:2]
    show = [d for d in ["fused_probability", *stat, *learned] if d in set(m["detector"])]
    thr = dict(zip(m["detector"], m["threshold"], strict=True))
    # the series and the `days`-day window holding the most traffic-impacting incidents
    imp = ev[ev["impactful"].astype(bool)]
    sid = imp["series_id"].value_counts().index[0] if len(imp) else str(sc["series_id"].iloc[0])
    s = sc[sc["series_id"] == sid].sort_values("timestamp")
    t = pd.DatetimeIndex(pd.to_datetime(s["timestamp"], utc=True))
    starts = pd.date_range(t.min().floor("1D"), t.max() - pd.Timedelta(days=days), freq="1D") \
        if t.max() - t.min() > pd.Timedelta(days=days) else pd.DatetimeIndex([t.min()])
    ev_s = ev[ev["series_id"] == sid]

    def n_imp(a: pd.Timestamp) -> int:
        e = ev_s[ev_s["impactful"].astype(bool)]
        return int(((e["start"] >= a) & (e["start"] < a + pd.Timedelta(days=days))).sum())

    a = max(starts, key=n_imp)
    w = (t >= a) & (t < a + pd.Timedelta(days=days))
    events = [(r["start"], r["end"], bool(r["impactful"])) for _, r in ev_s.iterrows()
              if r["end"] >= a and r["start"] < a + pd.Timedelta(days=days)]
    files = [plots.score_timeline(t[w], {d: (s[d].to_numpy()[w], float(thr[d])) for d in show}, events, "UTC",
                                  out / "score_timeline.png",
                                  f"Anomaly scores on {sid} (test period, {days} days) - synthetic injected incidents")]
    valid = sc["avg_speed_kmh"].notna().to_numpy()
    y = sc["gt_incident"].to_numpy()[valid].astype(bool)
    files.append(plots.pr_curves(y, {d: np.nan_to_num(sc[d].to_numpy()[valid], nan=0.0) for d in show},
                                 out / "pr_curves.png", "Anomaly detectors: precision-recall on test windows"))
    return files


# --------------------------------------------------------------------------- detection
_CLASS_BGR = {"car": (214, 120, 42), "bus": (52, 104, 235), "truck": (122, 175, 27)}   # plots.SERIES 0-2


def detection_figures(samples: bool = True) -> list[Path]:
    """Per run: test confusion matrices + Ultralytics training plots. Across runs (newest evaluation of each
    model x input size): per-class mAP, class distribution, sample predictions."""
    from traffic_intel.ai.training.detection import DetectionExperimentConfig

    runs = detection_runs()
    if not runs:
        return []
    files: list[Path] = []
    finetuned: dict[str, str] = {}
    cfg: DetectionExperimentConfig | None = None
    for exp_dir in runs:
        rec = json.loads((exp_dir / "experiment.json").read_text(encoding="utf-8"))
        cfg = DetectionExperimentConfig.model_validate(rec["config"])
        out = FIG / "detection" / exp_dir.name
        names = [*(c.value for c in cfg.classes), "background"]
        # older records keep the confusion of the fine-tuned models only; newer ones of every evaluated model
        evals = rec.get("evaluations") or [ev for mod in rec["models"] for ev in mod.get("test", [])]
        for ev in evals:
            cm = np.zeros((len(names), len(names)))
            for key, n in ev["confusion_iou50"].items():
                g, p = key.split("->")
                cm[names.index(g), names.index(p)] = n
            files.append(plots.confusion(cm, names, out / f"confusion_{ev['model']}_imgsz{ev['imgsz']}.png",
                                         f"{ev['model']} @{ev['imgsz']}: test, IoU 0.5, "
                                         f"conf >= {ev['operating_conf']}", "predicted", "ground truth"))
        for mod in rec["models"]:
            run = Path(mod["best"]).parent.parent
            for f in ("results.png", "labels.jpg", "confusion_matrix_normalized.png", "val_batch0_pred.jpg",
                      "val_batch0_labels.jpg"):
                c = _copy(run / f, out / f"ultralytics_{mod['name']}" / f)
                if c:
                    files.append(c)
            if Path(mod["best"]).exists():
                finetuned[mod["name"]] = mod["best"]
    assert cfg is not None
    classes = [c.value for c in cfg.classes]
    top = FIG / "detection"
    m = detection_rows()
    rows = []
    for _, r in m.iterrows():
        label = f"{r['model']} @{r['imgsz']}"
        rows += [{"class": c, "model": label, "v": r[f"{c}_mAP50-95"]} for c in classes]
        rows.append({"class": "all (mean)", "model": label, "v": r["mAP50-95"]})
    files.append(plots.grouped_bars(pd.DataFrame(rows), "class", "model", "v", top / "per_class_map.png",
                                    "UA-DETRAC test: mAP50-95 per class (all evaluated models)", "mAP50-95",
                                    ylim=(0, 1)))
    data_dir = PROJECT_ROOT / "data" / "processed" / "ua_detrac_yolo"
    if (data_dir / "labels").exists():
        files.append(_class_distribution(data_dir, classes, top / "class_distribution.png"))
    if samples and (data_dir / "images" / "test").exists():
        base = next((b for b in cfg.models if Path(b.weights).stem == "yolo11m"), cfg.models[0])
        chosen = [(f"{Path(base.weights).stem}_coco_zero_shot", base.weights), *finetuned.items()]
        for label, weights in chosen:
            files.append(_sample_predictions(weights, label, data_dir, cfg, top / f"samples_{label}.jpg"))
    return files


def _class_distribution(data_dir: Path, classes: list[str], path: Path) -> Path:
    rows = []
    for split in ("train", "val", "test"):
        counts = np.zeros(len(classes), int)
        for f in (data_dir / "labels" / split).glob("*.txt"):
            for ln in f.read_text().splitlines():
                if ln.strip():
                    counts[int(ln.split()[0])] += 1
        total = int(counts.sum())
        rows += [{"class": c, "split": f"{split} ({total:,} boxes)", "share": 100 * n / max(total, 1)}
                 for c, n in zip(classes, counts, strict=True)]
    return plots.grouped_bars(pd.DataFrame(rows), "class", "split", "share", path,
                              "UA-DETRAC boxes per class and split (every 10th frame)",
                              "share of boxes (%)", fmt="{:.1f}%", ylim=(0, 100))


def _sample_predictions(weights: str, label: str, data_dir: Path, cfg: Any, path: Path, n: int = 4) -> Path:  # noqa: ANN401
    """Four test images from different sequences: ground truth (white) vs detections (class colour)."""
    import cv2

    from traffic_intel.ai.training.detection import _read, _test_set
    from traffic_intel.computer_vision.detectors.yolo import YOLODetector
    from traffic_intel.core.enums import class_id_of
    from traffic_intel.core.types import Frame

    test = _test_set(data_dir, cfg.classes)
    picks = [test[int(len(test) * q)] for q in np.linspace(0.08, 0.92, n)]
    det = YOLODetector(weights, conf_threshold=cfg.operating_conf, iou_threshold=0.7, imgsz=cfg.train.imgsz,
                       max_det=300, class_map={c.value: c.value for c in cfg.classes}, device="auto", half="auto")
    name_of = {class_id_of(c): c.value for c in cfg.classes}
    now = pd.Timestamp.now(tz="UTC").to_pydatetime()
    tiles = []
    for k, (img_path, gt, _gcls, _ign) in enumerate(picks):
        img = _read(img_path)
        d = det.detect([Frame("sample", k, now, img)])[0]
        for b in gt.astype(int):
            cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (255, 255, 255), 1)
        for b, c, s in zip(d.xyxy.astype(int), d.class_id, d.confidence, strict=True):
            col = _CLASS_BGR.get(name_of.get(int(c), ""), (180, 180, 180))
            cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), col, 2)
            cv2.putText(img, f"{name_of.get(int(c), c)} {s:.2f}", (b[0], max(12, b[1] - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)
        cv2.putText(img, img_path.stem, (8, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                    cv2.LINE_AA)
        tiles.append(img)
    h, w = tiles[0].shape[:2]
    tiles = [cv2.resize(t, (w, h)) for t in tiles]
    grid = np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)])
    head = np.full((44, grid.shape[1], 3), 252, np.uint8)
    cv2.putText(head, f"{label} @{cfg.train.imgsz} on UA-DETRAC test | white: ground truth | blue car, orange bus, "
                f"green truck (conf >= {cfg.operating_conf}) | grey areas: ignored regions",
                (10, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (11, 11, 11), 1, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.vstack([head, grid]), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return path


# --------------------------------------------------------------------------- all
def build_all(rescore: bool = True, samples: bool = True) -> list[Path]:
    files: list[Path] = []
    for suffix, _title in FORECAST_CONFIGS:
        d = _latest("forecasting", suffix)
        if d is not None:
            files += forecasting_figures(d, rescore)
    runs = sorted(p for p in (EXP / "anomaly").glob("*") if p.is_dir() and (p / "report.md").exists()
                  and not p.name.endswith("_smoke"))
    if runs:
        files += anomaly_figures(runs[-1])
    files += detection_figures(samples)
    write_index()
    return files


_NOTES = {"forecasting": "per experiment: RMSE by horizon, prediction vs actual (selected model), error over time and "
                         "by hour of day (selected model vs persistence, re-scored from the registry)",
          "congestion": "per experiment: test macro-F1 by model and horizon, confusion matrix of the model chosen "
                        "on validation",
          "anomaly": "score timeline with alarm threshold and ground-truth incident periods; precision-recall curves",
          "incident": "AI City 2021 Track 4 (real video): window scores, PR/ROC, reliability (calibration)",
          "detection": "UA-DETRAC test: PR/F1/P/R curves per model and input size, per-class mAP, confusion matrix, "
                       "class distribution, sample predictions; ultralytics_* = training curves and validation "
                       "samples from the Ultralytics run"}


def write_index() -> Path:
    lines = ["# Evaluation figures", "",
             "Generated by `python -m traffic_intel.ai.training figures` and by the experiments themselves. "
             "Every figure comes from a recorded experiment (the folder name is the experiment id).", ""]
    for task in sorted(p.name for p in FIG.iterdir() if p.is_dir()):
        lines += [f"## {task}", "", _NOTES.get(task, ""), ""]
        for f in sorted((FIG / task).rglob("*")):
            if f.suffix.lower() in (".png", ".jpg"):
                rel = f.relative_to(FIG).as_posix()
                lines.append(f"- [{f.relative_to(FIG / task).as_posix()}]({rel})")
        lines.append("")
    out = FIG / "INDEX.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out
