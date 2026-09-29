"""Incident / anomaly detection evaluated on REAL labelled video: AI City 2021 Track 4.

    python -m traffic_intel.ai.training train incident --config configs/training/incident_aic21.yaml [--smoke]

What is (and is not) measured
    The only real incident labels on disk are the 18 "anomaly" intervals of the AIC21 Track 4 training
    videos (stalled vehicles and crashes share one label). The cameras are uncalibrated, so the live
    system's METRIC traffic features and track detectors cannot be computed on them. This module therefore
    evaluates detection approaches on scale-normalised PIXEL motion (distances in bounding-box heights).
    The models it fits are evaluation artefacts kept in the experiment folder - they are NOT deployed
    (the live anomaly models consume the canonical metric features; see configs/training/anomaly.yaml).

Stages
    1 extract   production perception (Detector + Tracker interfaces from configs/pipeline.yaml) on every
                training video at ``target_fps`` -> frame-level tracks as Parquet (data/interim/...)
    2 windows   per ``window_s`` window (sliding by ``step_s``): tracks visible, normalised speed, stationary
                tracks, longest stationary duration, stationary area, speed / count change vs. the video's
                own recent past (causal)
    3 split     BY VIDEO, stratified: labelled videos are spread over train / val / test
    4 methods   stationary_rule   longest ID-free stationary persistence of a box (the pixel analogue of
                                  the live stopped-vehicle rule); threshold chosen on validation
                isolation_forest  sklearn, fitted on NORMAL windows of training videos
                autoencoder       sklearn MLP 16-4-16 reconstructing normal windows (same family as the
                                  live learned detectors); score = reconstruction error
    5 metrics   window level: ROC-AUC, PR-AUC, precision / recall / F1 at the validation threshold;
                event level (video): first alarm -> precision / recall / F1 and detection delay;
                AIC21 timing score (F1 within 10 s of the start, RMSE, S4) with the rule's back-dated onset
    6 calibration  Platt (logistic) and isotonic mapping of the best score, fitted on validation windows,
                Brier score and reliability on test windows
"""

from __future__ import annotations

import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field

from traffic_intel.core.config import PROJECT_ROOT, load_yaml

log = logging.getLogger(__name__)
FEATURES = ["n_tracks", "speed_norm", "frac_stationary", "n_stationary", "max_stationary_s",
            "stationary_area", "speed_change", "count_change", "max_persist_s", "n_persist", "persist_area"]


class ExtractCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    weights: str = "models/detection/yolo11m.pt"   # the production detector unless overridden
    imgsz: int = 1280                              # the pipeline's inference size (distant vehicles)
    conf_threshold: float = 0.1                    # low: ByteTrack's second round uses low-score boxes
    tracker_params: dict[str, Any] = {}            # overrides of pipeline.yaml tracker.params
    target_fps: float = 5.0
    batch: int = 8
    out_dir: str = "data/interim/aic21_track4"


class WindowCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    window_s: float = 10.0
    step_s: float = 5.0
    stationary_lag_s: float = 5.0                  # displacement measured over this lag
    stationary_rel: float = 0.25                   # < 0.25 box heights in lag -> stationary sample
    baseline_s: float = 120.0                      # causal baseline for speed / count change
    # ID-free stationary persistence: how long a box has kept overlapping the same place (robust to
    # detector flicker and track-id switches, which fragment tracks of distant stopped vehicles)
    persist_step_s: float = 2.0
    persist_iou: float = 0.5
    persist_conf: float = 0.25                     # = the pipeline's detector.conf_threshold
    persist_min_s: float = 30.0                    # n_persist / persist_area count boxes persisting this long


class SplitCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    val_fraction: float = 0.2
    test_fraction: float = 0.2
    seed: int = 0


class IncidentExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    description: str = ""
    dataset: str = "aicity21_anomaly"
    extract: ExtractCfg = Field(default_factory=ExtractCfg)
    windows: WindowCfg = Field(default_factory=WindowCfg)
    split: SplitCfg = Field(default_factory=SplitCfg)
    isolation_forest: dict[str, Any] = {"n_estimators": 300, "max_samples": 2048}
    autoencoder: dict[str, Any] = {"hidden": [16, 4, 16], "max_iter": 400}
    event_tolerance_s: float = 10.0
    videos: list[str] | None = None                # restrict (smoke runs)


def load_incident_config(path: Path) -> IncidentExperimentConfig:
    return IncidentExperimentConfig.model_validate(load_yaml(path))


# --------------------------------------------------------------------------- 1 extraction
def extract_tracks(video: Path, out: Path, cfg: ExtractCfg) -> dict[str, Any]:
    """Production detector + tracker over one video at ``target_fps`` -> Parquet (frame-level stays here):
    ``<video>.parquet`` tracks, ``<video>_det.parquet`` all detections above ``conf_threshold``."""
    import cv2

    from traffic_intel.computer_vision.detectors.yolo import YOLODetector
    from traffic_intel.core.config import load_pipeline_config
    from traffic_intel.core.registry import ComponentSpec
    from traffic_intel.core.types import Frame
    from traffic_intel.tracking.interfaces import TRACKERS

    pipeline = load_pipeline_config()
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, round(fps / cfg.target_fps))
    dparams = dict(pipeline.detector.params)
    vehicles = {k: v for k, v in (dparams.get("class_map") or {}).items() if v in ("car", "bus", "truck", "motorcycle")}
    det = YOLODetector(cfg.weights, conf_threshold=cfg.conf_threshold,
                       iou_threshold=float(dparams.get("iou_threshold", 0.5)), imgsz=cfg.imgsz,
                       class_map=vehicles, device="auto", half="auto")
    tracker = TRACKERS.create(ComponentSpec(type=pipeline.tracker.type,
                                            params={**pipeline.tracker.params, **cfg.tracker_params}), fps=fps / step)
    rows: list[tuple[float, int, int, float, float, float, float, float]] = []
    det_rows: list[tuple[float, int, float, float, float, float, float]] = []
    buf: list[tuple[int, np.ndarray]] = []
    t0 = time.perf_counter()
    now = datetime.now(UTC)

    def flush() -> None:
        frames = [Frame(video.stem, i, now, img) for i, img in buf]
        for fr, d in zip(frames, det.detect(frames), strict=True):
            tr = tracker.update(d, fr)
            i = fr.index
            for b, c, s in zip(d.xyxy, d.class_id, d.confidence, strict=True):
                det_rows.append((i / fps, int(c), float(s), float(b[0]), float(b[1]), float(b[2]), float(b[3])))
            for tid, b, c, s in zip(tr.track_id, tr.xyxy, tr.class_id, tr.confidence, strict=True):
                rows.append((i / fps, int(tid), int(c), float(s), float(b[0]), float(b[1]), float(b[2]), float(b[3])))
        buf.clear()

    i = 0
    while True:
        if i % step:
            if not cap.grab():
                break
        else:
            ok, img = cap.read()
            if not ok:
                break
            buf.append((i, img))
            if len(buf) == cfg.batch:
                flush()
        i += 1
    if buf:
        flush()
    cap.release()
    df = pd.DataFrame(rows, columns=["t_s", "track_id", "class_id", "conf", "x1", "y1", "x2", "y2"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    dets = pd.DataFrame(det_rows, columns=["t_s", "class_id", "conf", "x1", "y1", "x2", "y2"])
    dets.to_parquet(out.with_name(f"{out.stem}_det.parquet"), index=False)
    info = {"video": video.stem, "fps": fps, "frames": n, "processed_every": step, "duration_s": n / fps,
            "rows": len(df), "detections": len(dets), "tracks": int(df["track_id"].nunique()) if len(df) else 0,
            "seconds": round(time.perf_counter() - t0, 1), "imgsz": cfg.imgsz, "conf": cfg.conf_threshold,
            "tracker": pipeline.tracker.type, "tracker_overrides": cfg.tracker_params}
    out.with_suffix(".json").write_text(json.dumps(info), encoding="utf-8")
    return info


# --------------------------------------------------------------------------- 2 windows
def persistence_series(dets: pd.DataFrame, frame_area: float, cfg: WindowCfg) -> pd.DataFrame:
    """Every ``persist_step_s``: how long each box has kept overlapping (IoU >= persist_iou) a box at the
    previous sampled instant (one missed instant is bridged). Track ids are not used, so detector flicker
    and id switches do not reset it. Returns t, max_persist_s, n_persist, persist_area per instant."""
    from traffic_intel.ai.training.detection import _box_iou

    cols = ["t", "max_persist_s", "n_persist", "persist_area"]
    d = dets[dets["conf"] >= cfg.persist_conf]
    if d.empty:
        return pd.DataFrame(columns=cols)
    times = np.sort(d["t_s"].unique())
    _, first = np.unique(np.floor(times / cfg.persist_step_s), return_index=True)
    sample_t = times[first]
    boxes = {t: g[["x1", "y1", "x2", "y2"]].to_numpy(float) for t, g in d[d["t_s"].isin(sample_t)].groupby("t_s")}
    hist: list[tuple[float, np.ndarray, np.ndarray]] = []            # last two instants: (t, boxes, persistence)
    rows = []
    for t in sample_t:
        now_boxes = boxes.get(t, np.zeros((0, 4)))
        persist = np.zeros(len(now_boxes))
        unmatched = np.arange(len(now_boxes))
        for t_prev, b_prev, p_prev in reversed(hist):     # previous instant first, then the one before
            if not len(unmatched) or not len(b_prev):
                continue
            iou = _box_iou(now_boxes[unmatched], b_prev)
            j = iou.argmax(1)
            ok = iou[np.arange(len(unmatched)), j] >= cfg.persist_iou
            persist[unmatched[ok]] = p_prev[j[ok]] + (t - t_prev)
            unmatched = unmatched[~ok]
        long = persist >= cfg.persist_min_s
        lb = now_boxes[long]
        area = float(((lb[:, 2] - lb[:, 0]) * (lb[:, 3] - lb[:, 1])).sum() / frame_area) if long.any() else 0.0
        rows.append((float(t), float(persist.max()) if len(persist) else 0.0, int(long.sum()), float(area)))
        hist = (hist + [(float(t), now_boxes, persist)])[-2:]
    return pd.DataFrame(rows, columns=cols)


def window_features(tracks: pd.DataFrame, duration_s: float, frame_area: float, cfg: WindowCfg,
                    dets: pd.DataFrame | None = None) -> pd.DataFrame:
    """Causal per-window features from pixel tracks (scale-normalised by box height) and, when
    ``dets`` is given, ID-free stationary persistence."""
    starts = np.arange(0.0, max(duration_s - cfg.window_s, 0.0) + 1e-9, cfg.step_s)
    out = _track_windows(tracks, starts, frame_area, cfg)
    pers = persistence_series(dets, frame_area, cfg) if dets is not None else pd.DataFrame()
    for f in ("max_persist_s", "n_persist", "persist_area"):
        vals = np.zeros(len(starts))
        if len(pers):
            t = pers["t"].to_numpy()
            v = pers[f].to_numpy(float)
            for k, s in enumerate(starts):
                m = (t >= s) & (t < s + cfg.window_s)
                vals[k] = v[m].max() if m.any() else 0.0
        out[f] = vals
    return out


def _track_windows(tracks: pd.DataFrame, starts: np.ndarray, frame_area: float, cfg: WindowCfg) -> pd.DataFrame:
    track_features = [f for f in FEATURES if f not in ("max_persist_s", "n_persist", "persist_area")]
    if tracks.empty:
        return pd.DataFrame({"t_start": starts, "t_end": starts + cfg.window_s, **{f: 0.0 for f in track_features}})
    t = tracks.sort_values(["track_id", "t_s"]).copy()
    t["cx"], t["cy"] = (t["x1"] + t["x2"]) / 2, (t["y1"] + t["y2"]) / 2
    t["h"] = (t["y2"] - t["y1"]).clip(lower=4.0)
    t["area"] = (t["x2"] - t["x1"]) * (t["y2"] - t["y1"])
    # displacement over the trailing lag, same track (merge_asof on time within each track)
    lagged = t[["track_id", "t_s", "cx", "cy"]].rename(columns={"t_s": "t_prev", "cx": "cx_prev", "cy": "cy_prev"})
    t["t_key"] = t["t_s"] - cfg.stationary_lag_s
    t = pd.merge_asof(t.sort_values("t_key"), lagged.sort_values("t_prev"), left_on="t_key", right_on="t_prev",
                      by="track_id", direction="backward", tolerance=cfg.stationary_lag_s)
    t = t.sort_values(["track_id", "t_s"])
    dt = (t["t_s"] - t["t_prev"]).where(lambda s: s > 0)
    disp = np.hypot(t["cx"] - t["cx_prev"], t["cy"] - t["cy_prev"]) / t["h"]
    t["speed_norm"] = disp / dt                                      # box heights per second
    t["stationary"] = (disp / dt * cfg.stationary_lag_s) < cfg.stationary_rel
    # continuous stationary duration per track (resets when the track moves)
    run = (t["stationary"] != t.groupby("track_id")["stationary"].shift()).cumsum()
    first = t.groupby(run)["t_s"].transform("min")
    t["stationary_s"] = np.where(t["stationary"], t["t_s"] - first + cfg.stationary_lag_s, 0.0)
    rows = []
    ts = t["t_s"].to_numpy()
    for s in starts:
        m = (ts >= s) & (ts < s + cfg.window_s)
        w = t[m]
        if w.empty:
            rows.append({"t_start": s, "t_end": s + cfg.window_s, **{f: 0.0 for f in track_features}})
            continue
        per_frame = w.groupby("t_s")["track_id"].nunique()
        stat = w[w["stationary"]]
        rows.append({
            "t_start": s, "t_end": s + cfg.window_s,
            "n_tracks": float(per_frame.mean()),
            "speed_norm": float(w["speed_norm"].median()) if w["speed_norm"].notna().any() else 0.0,
            "frac_stationary": float(w["stationary"].mean()),
            "n_stationary": float(stat["track_id"].nunique()),
            "max_stationary_s": float(w["stationary_s"].max()),
            "stationary_area": float(stat.groupby("t_s")["area"].sum().mean() / frame_area) if len(stat) else 0.0,
        })
    out = pd.DataFrame(rows)
    k = max(1, int(cfg.baseline_s / cfg.step_s))
    base_speed = out["speed_norm"].shift(1).rolling(k, min_periods=3).median()
    base_count = out["n_tracks"].shift(1).rolling(k, min_periods=3).median()
    out["speed_change"] = (1.0 - out["speed_norm"] / base_speed.where(base_speed > 0)).fillna(0.0).clip(-5, 5)
    out["count_change"] = (out["n_tracks"] / base_count.where(base_count > 0) - 1.0).fillna(0.0).clip(-5, 5)
    return out


def _labelled(windows: pd.DataFrame, intervals: list[tuple[float, float]]) -> np.ndarray:
    c = (windows["t_start"] + windows["t_end"]).to_numpy() / 2
    y = np.zeros(len(c), bool)
    for s, e in intervals:
        y |= (c >= s) & (c <= e)
    return y


# --------------------------------------------------------------------------- 3 split
def split_videos(videos: list[str], labelled: set[str], cfg: SplitCfg) -> dict[str, list[str]]:
    rng = np.random.default_rng(cfg.seed)
    out: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for group in (sorted(v for v in videos if v in labelled), sorted(v for v in videos if v not in labelled)):
        g = list(rng.permutation(group))
        n_test = max(1, round(len(g) * cfg.test_fraction)) if g else 0
        n_val = max(1, round(len(g) * cfg.val_fraction)) if len(g) > 2 else 0
        out["test"] += g[:n_test]
        out["val"] += g[n_test:n_test + n_val]
        out["train"] += g[n_test + n_val:]
    return {k: sorted(v, key=int) for k, v in out.items()}


# --------------------------------------------------------------------------- 4/5 methods + metrics
def _ramp_scores(raw_train_normal: np.ndarray, raw: np.ndarray) -> np.ndarray:
    """Same normalisation as the live learned detectors: 0 = within normal 99%, 1 = far beyond."""
    from traffic_intel.anomaly.detectors.statistical import ramp

    qn = float(np.quantile(raw_train_normal, 0.99))
    qx = max(float(np.quantile(raw_train_normal, 0.999)) * 2.0, qn + 1e-9)
    return np.array([ramp(float(r), qn, qx) for r in raw])


def fit_methods(train: pd.DataFrame, cfg: IncidentExperimentConfig) -> dict[str, Any]:
    from sklearn.ensemble import IsolationForest
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler

    normal = train[~train["y"]]
    scaler = StandardScaler().fit(normal[FEATURES])
    Xn = scaler.transform(normal[FEATURES])
    iso = IsolationForest(random_state=cfg.split.seed, n_jobs=-1, **cfg.isolation_forest).fit(Xn)
    ae_p = dict(cfg.autoencoder)
    ae = MLPRegressor(hidden_layer_sizes=tuple(ae_p.pop("hidden", [16, 4, 16])), early_stopping=True,
                      random_state=cfg.split.seed, **ae_p).fit(Xn, Xn)
    raw_iso = -iso.score_samples(Xn)
    raw_ae = np.mean((ae.predict(Xn) - Xn) ** 2, axis=1)
    return {"scaler": scaler, "isolation_forest": iso, "autoencoder": ae, "raw_iso_normal": raw_iso,
            "raw_ae_normal": raw_ae, "n_normal_windows": len(normal)}


def score_methods(df: pd.DataFrame, fitted: dict[str, Any]) -> dict[str, np.ndarray]:
    X = fitted["scaler"].transform(df[FEATURES])
    return {
        # the stopped-vehicle rule on ID-free persistence (the track-based duration is a learned-model input)
        "stationary_rule": df["max_persist_s"].to_numpy(float),
        "isolation_forest": _ramp_scores(fitted["raw_iso_normal"], -fitted["isolation_forest"].score_samples(X)),
        "autoencoder": _ramp_scores(fitted["raw_ae_normal"],
                                    np.mean((fitted["autoencoder"].predict(X) - X) ** 2, axis=1)),
    }


def _best_threshold(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import precision_recall_curve

    p, r, thr = precision_recall_curve(y, s)
    f1 = 2 * p * r / np.maximum(p + r, 1e-12)
    return float(thr[int(np.nanargmax(f1[:-1]))]) if len(thr) else float("inf")


def window_metrics(y: np.ndarray, s: np.ndarray, thr: float) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score

    pred = s >= thr
    p, r, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    both = len(set(y.tolist())) == 2
    return {"roc_auc": float(roc_auc_score(y, s)) if both else float("nan"),
            "pr_auc": float(average_precision_score(y, s)) if both else float("nan"),
            "precision": float(p), "recall": float(r), "f1": float(f1), "threshold": float(thr),
            "positive_rate": float(y.mean())}


def event_metrics(df: pd.DataFrame, s: np.ndarray, thr: float, intervals: dict[str, list[tuple[float, float]]],
                  tol: float) -> dict[str, Any]:
    """Per video: the first alarm. TP = labelled video, alarm inside [start - tol, end]."""
    tp = fp = fn = 0
    delays: list[float] = []
    for vid, g in df.assign(score=s).groupby("video", sort=False):
        alarm = g[g["score"] >= thr]
        t_alarm = float(alarm["t_end"].iloc[0]) if len(alarm) else None      # decision at the window's end
        iv = intervals.get(str(vid), [])
        if not iv:
            fp += t_alarm is not None
            continue
        hit = next(((st, en) for st, en in iv if t_alarm is not None and st - tol <= t_alarm <= en), None)
        if hit is not None and t_alarm is not None:
            tp += 1
            delays.append(t_alarm - hit[0])
        else:
            fn += 1
            fp += t_alarm is not None
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "median_delay_s": float(np.median(delays)) if delays else float("nan")}


# --------------------------------------------------------------------------- experiment
def run_incident_experiment(cfg: IncidentExperimentConfig, smoke: bool = False) -> dict[str, Any]:
    from traffic_intel.ai.evaluation.incidents import TimedPrediction, anomaly_timing_score
    from traffic_intel.datasets.adapters.base import IncidentAdapter, adapter_for

    exp_id = datetime.now().strftime("%Y%m%d-%H%M%S") + f"_{cfg.name}" + ("_smoke" if smoke else "")
    out = PROJECT_ROOT / "experiments" / "incident" / exp_id
    # a smoke run is a pipeline check, not a result: its figures stay in its experiment folder
    fig = out / "figures" if smoke else PROJECT_ROOT / "reports" / "figures" / "incident" / exp_id
    out.mkdir(parents=True)
    fig.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False), encoding="utf-8")
    a = adapter_for(cfg.dataset)
    assert isinstance(a, IncidentAdapter)
    ann = a.annotations()
    videos = a.videos()
    intervals: dict[str, list[tuple[float, float]]] = {}
    for x in ann:
        intervals.setdefault(x.video_id, []).append((x.start_s, x.end_s if x.end_s is not None else 1e9))
    ids = sorted(videos, key=int)
    if cfg.videos:
        ids = [v for v in ids if v in cfg.videos]

    # 1 extraction (resumable: a video with a Parquet file is not processed again)
    ex_dir = PROJECT_ROOT / cfg.extract.out_dir
    meta: dict[str, dict[str, Any]] = {}
    for n, vid in enumerate(ids, 1):
        pq = ex_dir / f"{vid}.parquet"
        if not pq.exists():
            info = extract_tracks(videos[vid], pq, cfg.extract)
            log.info("[%d/%d] video %s: %d track rows in %.0f s", n, len(ids), vid, info["rows"], info["seconds"])
        meta[vid] = json.loads(pq.with_suffix(".json").read_text(encoding="utf-8"))

    # 2 windows
    import cv2

    cap = cv2.VideoCapture(str(videos[ids[0]]))
    frame_area = float(cap.get(3) * cap.get(4))
    cap.release()
    parts = []
    for vid in ids:
        w = window_features(pd.read_parquet(ex_dir / f"{vid}.parquet"), meta[vid]["duration_s"], frame_area,
                            cfg.windows, dets=pd.read_parquet(ex_dir / f"{vid}_det.parquet"))
        w["video"] = vid
        w["y"] = _labelled(w, intervals.get(vid, []))
        parts.append(w)
    data = pd.concat(parts, ignore_index=True)
    data.to_parquet(out / "windows.parquet", index=False)

    # 3 split by video
    split = split_videos(ids, set(intervals), cfg.split)
    (out / "split.json").write_text(json.dumps({k: {"videos": v, "labelled": [x for x in v if x in intervals]}
                                                for k, v in split.items()}, indent=2), encoding="utf-8")
    sets = {k: data[data["video"].isin(v)].reset_index(drop=True) for k, v in split.items()}
    log.info("windows: %s; positive windows: %s", {k: len(v) for k, v in sets.items()},
             {k: int(v["y"].sum()) for k, v in sets.items()})

    # 4 fit on normal TRAIN windows; thresholds on VALIDATION; report TEST
    t0 = time.perf_counter()
    fitted = fit_methods(sets["train"], cfg)
    fit_s = time.perf_counter() - t0
    scores = {k: score_methods(v, fitted) for k, v in sets.items()}
    rows, events = [], []
    for method in scores["test"]:
        thr = _best_threshold(sets["val"]["y"].to_numpy(), scores["val"][method])
        wm = window_metrics(sets["test"]["y"].to_numpy(), scores["test"][method], thr)
        em = event_metrics(sets["test"], scores["test"][method], thr, intervals, cfg.event_tolerance_s)
        rows.append({"method": method, **{f"window_{k}": v for k, v in wm.items()},
                     **{f"event_{k}": v for k, v in em.items()}})
        events.append((method, thr))
    # AIC21 timing score: rule alarms back-dated to the moment the vehicle became stationary
    preds = []
    thr_rule = dict(events)["stationary_rule"]
    for vid, g in sets["test"].assign(score=scores["test"]["stationary_rule"]).groupby("video"):
        al = g[g["score"] >= thr_rule]
        if len(al):
            r = al.iloc[0]
            preds.append(TimedPrediction(str(vid), max(0.0, float(r["t_end"]) - float(r["score"]))))
    test_labels = [x for x in ann if x.video_id in split["test"]]
    aic = anomaly_timing_score(test_labels, preds, window_s=cfg.event_tolerance_s)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / "metrics_test.csv", index=False)

    # 6 calibration of the best (validation PR-AUC) score
    from sklearn.metrics import average_precision_score

    best = max(scores["val"], key=lambda m: average_precision_score(sets["val"]["y"], scores["val"][m])
               if sets["val"]["y"].any() else 0.0)
    calib = _calibration(sets, scores, best, fig)
    _plots(sets, scores, dict(events), intervals, fig)
    record = {"experiment_id": exp_id, "created": datetime.now(UTC).isoformat(), "detector": cfg.extract.model_dump(),
              "videos": len(ids), "windows": {k: len(v) for k, v in sets.items()},
              "positive_windows": {k: int(v["y"].sum()) for k, v in sets.items()},
              "normal_train_windows": fitted["n_normal_windows"], "fit_seconds": round(fit_s, 2),
              "metrics": rows, "aic21_timing_rule": aic, "calibration": calib, "split": split}
    (out / "experiment.json").write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    import joblib

    joblib.dump({k: fitted[k] for k in ("scaler", "isolation_forest", "autoencoder")}, out / "fitted_models.joblib")
    (out / "report.md").write_text(_report(record, metrics, cfg), encoding="utf-8")
    idx = out.parent / "index.csv"
    old = pd.read_csv(idx) if idx.exists() else pd.DataFrame()
    pd.concat([old, metrics.assign(experiment_id=exp_id)], ignore_index=True).to_csv(idx, index=False)
    return record


def _calibration(sets: dict[str, pd.DataFrame], scores: dict[str, dict[str, np.ndarray]], method: str,
                 fig: Path) -> dict[str, Any]:
    """Platt / isotonic maps fitted on VALIDATION windows, judged on TEST windows (Brier, reliability)."""
    from sklearn.calibration import calibration_curve
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss

    yv, sv = sets["val"]["y"].to_numpy(), scores["val"][method]
    yt, st = sets["test"]["y"].to_numpy(), scores["test"][method]
    if len(set(yv.tolist())) < 2 or len(set(yt.tolist())) < 2:
        return {"method": method, "note": "validation or test windows contain one class only - not calibrated"}
    raw = np.clip(st / max(float(np.nanmax(sv)), 1e-9), 0, 1)             # uncalibrated score scaled to [0, 1]
    platt = LogisticRegression().fit(sv.reshape(-1, 1), yv)
    iso = IsotonicRegression(out_of_bounds="clip").fit(sv, yv)
    p_platt = platt.predict_proba(st.reshape(-1, 1))[:, 1]
    p_iso = iso.predict(st)
    res = {"method": method, "fitted_on": "validation windows", "evaluated_on": "test windows",
           "brier_uncalibrated_scaled": float(brier_score_loss(yt, raw)),
           "brier_platt": float(brier_score_loss(yt, p_platt)),
           "brier_isotonic": float(brier_score_loss(yt, p_iso)),
           "brier_constant_base_rate": float(brier_score_loss(yt, np.full(len(yt), yv.mean())))}
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(5, 5))
    for name, p in (("uncalibrated (scaled)", raw), ("Platt", p_platt), ("isotonic", p_iso)):
        fp, mp = calibration_curve(yt, p, n_bins=10, strategy="quantile")
        plt.plot(mp, fp, marker="o", label=name)
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("predicted probability")
    plt.ylabel("observed frequency (test windows)")
    plt.title(f"Reliability: {method}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig / "reliability.png", dpi=130)
    plt.close("all")
    return res


def _plots(sets: dict[str, pd.DataFrame], scores: dict[str, dict[str, np.ndarray]], thr: dict[str, float],
           intervals: dict[str, list[tuple[float, float]]], fig: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve

    y = sets["test"]["y"].to_numpy()
    if len(set(y.tolist())) == 2:
        fig_, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
        for m, s in scores["test"].items():
            fpr, tpr, _ = roc_curve(y, s)
            p, r, _ = precision_recall_curve(y, s)
            a1.plot(fpr, tpr, label=m)
            a2.plot(r, p, label=m)
        a1.plot([0, 1], [0, 1], "k--", lw=1)
        a1.set_xlabel("false positive rate")
        a1.set_ylabel("true positive rate")
        a1.set_title("ROC (test windows)")
        a2.set_xlabel("recall")
        a2.set_ylabel("precision")
        a2.set_title("Precision-recall (test windows)")
        a1.legend()
        plt.tight_layout()
        plt.savefig(fig / "roc_pr.png", dpi=130)
        plt.close("all")
    test = sets["test"]
    for vid in [v for v in test["video"].unique() if v in intervals][:6]:
        g = test[test["video"] == vid]
        idx = g.index.to_numpy()
        fig_, ax = plt.subplots(len(scores["test"]), 1, figsize=(10, 2.2 * len(scores["test"])), sharex=True)
        for k, (m, s) in enumerate(scores["test"].items()):
            ax[k].plot(g["t_end"], s[idx], lw=1)
            ax[k].axhline(thr[m], color="r", ls="--", lw=1)
            for st, en in intervals[vid]:
                ax[k].axvspan(st, min(en, g["t_end"].max()), color="orange", alpha=0.25)
            ax[k].set_ylabel(m, fontsize=8)
        ax[-1].set_xlabel("video time (s) - shaded: labelled anomaly; dashed: threshold chosen on validation")
        fig_.suptitle(f"AIC21 Track 4 test video {vid}")
        plt.tight_layout()
        plt.savefig(fig / f"scores_video_{vid}.png", dpi=110)
        plt.close("all")


def _report(rec: dict[str, Any], metrics: pd.DataFrame, cfg: IncidentExperimentConfig) -> str:
    cols = ["method", "window_roc_auc", "window_pr_auc", "window_precision", "window_recall", "window_f1",
            "event_precision", "event_recall", "event_f1", "event_median_delay_s"]
    lines = [f"# Incident detection on real video: {rec['experiment_id']}", "",
             "AI City 2021 Track 4 training videos, split by video. Pixel-space features (uncalibrated cameras);",
             "the fitted models are evaluation artefacts, not deployed.", "",
             f"- Detector: `{cfg.extract.weights}` @ {cfg.extract.imgsz}, {cfg.extract.target_fps} processed fps",
             f"- Videos: {rec['videos']}; windows {rec['windows']}; positive windows {rec['positive_windows']}",
             "- Split (labelled videos): " + ", ".join(f"{k} {len(v)}" for k, v in rec["split"].items()), "",
             "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in metrics.iterrows():
        lines.append("| " + " | ".join(str(r[c]) if c == "method" else f"{r[c]:.3f}" for c in cols) + " |")
    lines += ["", f"AIC21 timing score (stationary rule, onset back-dated): `{rec['aic21_timing_rule']}`", "",
              f"Calibration: `{rec['calibration']}`", ""]
    return "\n".join(lines)


__all__ = ["run_incident_experiment", "load_incident_config", "window_features", "split_videos",
           "event_metrics", "window_metrics"]
