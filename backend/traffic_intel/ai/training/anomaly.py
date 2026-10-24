"""Anomaly-detector training + evaluation (offline).

    python -m traffic_intel.ai.training train anomaly --config configs/training/anomaly.yaml [--promote]

Data: the synthetic junction series (``forecasting/data/synthetic_series.py``) generated WITH
ground-truth labels (injected capacity drops). Chronological split: the learned detectors
(Isolation Forest, autoencoder) are fitted on the first ``train_days`` using NORMAL windows only
(no injected incident within ``margin_min``); everything is evaluated on the remaining days.

Metrics per detector (and for the fused series-level incident probability):
    pr_auc, roc_auc       window-level ranking quality of the score vs "incident active"
    precision / recall / f1  window level at the configured alarm threshold (minutes with an alarm vs
                          minutes with an incident active; accuracy is not used - incidents are rare)
    event_recall          share of incidents with an alarm (score >= threshold) between start
                          and end + margin_min
    event_recall_impact   same, only for incidents that actually hurt traffic (peak demand /
                          reduced capacity >= impact_load_ratio) - a capacity drop at 3 a.m.
                          is invisible in traffic data and cannot be detected by any method
    false_alarms_per_h    alarm onsets outside [start - margin, end + margin] of any incident,
                          per hour of normal operation
    median_delay_min      alarm time - incident start, over detected incidents

Honest caveat: synthetic incidents are capacity drops in a queueing model; results validate
the machinery and relative behaviour, not real-world detection rates.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field

from traffic_intel.ai.model_store import FileModelStore, ModelMetadata
from traffic_intel.anomaly.detectors.learned import AutoencoderDetector, IsolationForestDetector
from traffic_intel.anomaly.features import FEATURE_NAMES
from traffic_intel.anomaly.interfaces import SERIES_ANOMALY_DETECTORS
from traffic_intel.core.config import PROJECT_ROOT, get_settings
from traffic_intel.core.enums import ModelTask
from traffic_intel.core.registry import ComponentSpec

log = logging.getLogger(__name__)

LEARNED = {"isolation_forest": IsolationForestDetector, "autoencoder": AutoencoderDetector}


class AnomalyExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = "synthetic_anomaly"
    description: str = ""
    dataset: dict[str, Any] = Field(default_factory=lambda: {"days": 21, "seed": 11})
    train_days: int = 14
    statistical: list[ComponentSpec] = Field(default_factory=list)
    learned: dict[str, dict[str, Any]] = Field(
        default_factory=lambda: {"isolation_forest": dict[str, Any](), "autoencoder": dict[str, Any]()})
    history_windows: int = 30
    alarm_threshold: float = 0.5
    # the fused probability is a different scale: series evidence alone is capped near 0.38 by design
    # (a physical cue is needed for "possible incident"), so it alarms at the risk engine's
    # "elevated incident probability" threshold
    fused_alarm_threshold: float = 0.3
    margin_min: int = 15
    impact_load_ratio: float = 0.9


class AnomalyExperimentResult(BaseModel):
    experiment_id: str
    directory: str
    metrics: list[dict[str, Any]]


def load_anomaly_config(path: Path) -> AnomalyExperimentConfig:
    return AnomalyExperimentConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def _dataset(cfg: AnomalyExperimentConfig) -> pd.DataFrame:
    from traffic_intel.forecasting.data.synthetic_series import SyntheticSeriesConfig, generate_series

    sc = SyntheticSeriesConfig(**cfg.dataset)
    df = generate_series(sc, weather_hourly=None, holiday_country="IN", with_labels=True)
    return df.sort_values(["series_id", "timestamp"]).reset_index(drop=True)


def _events(g: pd.DataFrame, impact_load_ratio: float) -> list[dict[str, Any]]:
    """Contiguous gt_incident intervals of one series."""
    on = g["gt_incident"].to_numpy() > 0
    t = g["timestamp"].to_numpy()
    load = g["gt_load_ratio"].to_numpy()
    out, i = [], 0
    while i < len(on):
        if on[i]:
            j = i
            while j + 1 < len(on) and on[j + 1]:
                j += 1
            out.append({"start": pd.Timestamp(t[i]), "end": pd.Timestamp(t[j]),
                        "peak_load": float(np.nanmax(load[i:j + 1])),
                        "impactful": bool(np.nanmax(load[i:j + 1]) >= impact_load_ratio)})
            i = j + 1
        else:
            i += 1
    return out


def _near_incident(times: pd.Series, events: list[dict[str, Any]], margin: pd.Timedelta) -> np.ndarray:
    mask = np.zeros(len(times), dtype=bool)
    for e in events:
        mask |= ((times >= e["start"] - margin) & (times <= e["end"] + margin)).to_numpy()
    return mask


def _series_scores(detector: Any, g: pd.DataFrame, history: int) -> np.ndarray:
    """Replay a (stateless) series detector window by window, as the engine does online."""
    out = np.zeros(len(g))
    cols = g.drop(columns=[c for c in g.columns if c.startswith("gt_")])
    for k in range(len(g)):
        if cols["avg_speed_kmh"].iloc[k] != cols["avg_speed_kmh"].iloc[k]:     # NaN row (sensor gap)
            continue
        h = cols.iloc[max(0, k - history + 1):k + 1]
        h = h[h["avg_speed_kmh"].notna()]
        sig = detector.score(str(g["series_id"].iloc[0]), h)
        out[k] = max((s.score for s in sig), default=0.0)
    return out


def _metrics(name: str, scores: np.ndarray, g_all: list[tuple[pd.DataFrame, list[dict[str, Any]]]],
             cfg: AnomalyExperimentConfig) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score

    margin = pd.Timedelta(minutes=cfg.margin_min)
    y, s, delays, detected, detected_imp, n_imp, n_ev, fa, normal_min = [], [], [], 0, 0, 0, 0, 0, 0
    thr = cfg.fused_alarm_threshold if name == "fused_probability" else cfg.alarm_threshold
    offset = 0
    for g, events in g_all:
        sc = scores[offset:offset + len(g)]
        offset += len(g)
        valid = g["avg_speed_kmh"].notna().to_numpy()
        y.extend(g["gt_incident"].to_numpy()[valid])
        s.extend(sc[valid])
        alarm = (sc >= thr) & valid
        t = g["timestamp"]
        for e in events:
            n_ev += 1
            n_imp += e["impactful"]
            w = ((t >= e["start"]) & (t <= e["end"] + margin)).to_numpy() & alarm
            if w.any():
                detected += 1
                detected_imp += e["impactful"]
                delays.append((t[w].iloc[0] - e["start"]).total_seconds() / 60.0)
        near = _near_incident(t, events, margin)
        onsets = alarm & ~np.concatenate([[False], alarm[:-1]])
        fa += int((onsets & ~near).sum())
        normal_min += int((valid & ~near).sum())
    y_arr = np.asarray(y, dtype=bool)
    s_arr = np.nan_to_num(np.asarray(s, dtype=float), nan=0.0)
    p, r, f1, _ = precision_recall_fscore_support(y_arr, s_arr >= thr, average="binary", zero_division=0)
    both = y_arr.any() and not y_arr.all()
    return {"detector": name,
            "pr_auc": round(float(average_precision_score(y_arr, s_arr)), 4) if both else None,
            "roc_auc": round(float(roc_auc_score(y_arr, s_arr)), 4) if both else None,
            "threshold": thr, "precision": round(float(p), 4), "recall": round(float(r), 4), "f1": round(float(f1), 4),
            "incident_share": round(float(y_arr.mean()), 4),
            "events": n_ev, "event_recall": round(detected / n_ev, 3) if n_ev else None,
            "impactful_events": n_imp, "event_recall_impact": round(detected_imp / n_imp, 3) if n_imp else None,
            "false_alarms_per_h": round(fa / max(normal_min / 60.0, 1e-9), 3),
            "median_delay_min": round(float(np.median(delays)), 1) if delays else None}


def run_anomaly_experiment(cfg: AnomalyExperimentConfig, *, experiments_root: Path | None = None,
                           store: FileModelStore | None = None) -> AnomalyExperimentResult:
    from traffic_intel.anomaly.fusion import WeightedLogisticFuser

    default_root = experiments_root is None          # figures go to reports/ only for the real experiments folder
    experiments_root = experiments_root or PROJECT_ROOT / "experiments" / "anomaly"
    store = store or FileModelStore(get_settings().models_dir)
    exp_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{cfg.name}"
    out = experiments_root / exp_id
    out.mkdir(parents=True, exist_ok=False)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False), "utf-8")

    df = _dataset(cfg)
    split = df["timestamp"].min() + pd.Timedelta(days=cfg.train_days)
    margin = pd.Timedelta(minutes=cfg.margin_min)
    train, test = df[df["timestamp"] < split], df[df["timestamp"] >= split]
    normal_parts = []
    for _, g in train.groupby("series_id", sort=False):
        ev = _events(g, cfg.impact_load_ratio)
        normal_parts.append(g[~_near_incident(g["timestamp"], ev, margin)])
    train_normal = pd.concat(normal_parts)
    train_normal = train_normal[train_normal["avg_speed_kmh"].notna()]
    log.info("anomaly experiment %s: train %s rows (%s normal), test %s rows", exp_id, len(train),
             len(train_normal), len(test))

    detectors: dict[str, Any] = {}
    fitted: dict[str, Any] = {}
    for kind, params in cfg.learned.items():
        det = LEARNED[kind](**params)
        det.fit(train_normal.drop(columns=["gt_incident", "gt_load_ratio"]))
        fitted[kind] = det
        detectors[kind] = det
    for spec in cfg.statistical:
        detectors[spec.type] = SERIES_ANOMALY_DETECTORS.create(spec)

    groups = [(g.reset_index(drop=True), _events(g, cfg.impact_load_ratio))
              for _, g in test.groupby("series_id", sort=False)]
    all_scores: dict[str, np.ndarray] = {}
    for name, det in detectors.items():
        parts = []
        for g, _ in groups:
            if name in fitted:
                sc = det.score_frame(g.drop(columns=["gt_incident", "gt_load_ratio"]))
                parts.append(np.nan_to_num(sc, nan=0.0))
            else:
                parts.append(_series_scores(det, g, cfg.history_windows))
        all_scores[name] = np.concatenate(parts)
        log.info("scored %s", name)

    # fused series-level incident probability (no camera track evidence in this dataset)
    fuser = WeightedLogisticFuser()
    fam_of = {"sudden_slowdown": "speed_drop", "queue_growth": "queue_growth", "density_zscore": "density_change",
              "flow_imbalance": "flow_imbalance", "ewma": "anomaly_score", "zscore": "anomaly_score",
              "isolation_forest": "anomaly_score", "autoencoder": "anomaly_score"}
    n = len(next(iter(all_scores.values())))
    fused = np.zeros(n)
    for k in range(n):
        evidence: dict[str, float] = {}
        for name, sc in all_scores.items():
            f = fam_of.get(name)
            if f is not None:
                evidence[f] = max(evidence.get(f, 0.0), float(sc[k]))
        fused[k] = fuser.probability(evidence)[0]
    all_scores["fused_probability"] = fused

    metrics = [_metrics(name, sc, groups, cfg) for name, sc in all_scores.items()]
    pd.DataFrame(metrics).to_csv(out / "metrics.csv", index=False)
    scored = pd.concat([g[["timestamp", "series_id", "avg_speed_kmh", "gt_incident"]] for g, _ in groups],
                       ignore_index=True)
    scored = scored.assign(**{name: sc for name, sc in all_scores.items()})
    scored.to_parquet(out / "scores.parquet", index=False)              # for the evaluation figures
    events = [{"series_id": g["series_id"].iloc[0], **{k: (str(v) if isinstance(v, pd.Timestamp) else v)
                                                       for k, v in e.items()}} for g, evs in groups for e in evs]
    pd.DataFrame(events).to_csv(out / "events.csv", index=False)
    (out / "dataset.json").write_text(json.dumps({
        "generator": cfg.dataset, "train_until": str(split), "train_rows": len(train),
        "train_normal_rows": len(train_normal), "test_rows": len(test), "test_events": len(events),
        "features": FEATURE_NAMES}, indent=2), encoding="utf-8")

    for kind, det in fitted.items():
        store.save(det, ModelMetadata(
            task=ModelTask.ANOMALY, name=kind, version=exp_id, framework="sklearn",
            feature_schema_hash=str(abs(hash(tuple(FEATURE_NAMES))) % 10**8), trained_at=datetime.now(UTC),
            train_data={"experiment_id": exp_id, "dataset": "synthetic", "train_until": str(split),
                        "normal_rows": len(train_normal)},
            params={"type": kind, **det.params},
            metrics=next(m for m in metrics if m["detector"] == kind)))
    (out / "report.md").write_text(_report(exp_id, cfg, metrics, len(events)), encoding="utf-8")
    from traffic_intel.ai.evaluation.figures import anomaly_figures

    anomaly_figures(out, fig_dir=None if default_root else out / "figures")
    return AnomalyExperimentResult(experiment_id=exp_id, directory=str(out), metrics=metrics)


def _report(exp_id: str, cfg: AnomalyExperimentConfig, metrics: list[dict[str, Any]], n_events: int) -> str:
    lines = [f"# Anomaly experiment `{exp_id}`", "",
             "Synthetic junction series with injected capacity drops (ground truth). Learned detectors fitted on "
             f"normal windows of the first {cfg.train_days} days; all detectors evaluated on the remaining days "
             f"({n_events} injected incidents). Alarm threshold {cfg.alarm_threshold} "
             f"(fused probability: {cfg.fused_alarm_threshold}); margin {cfg.margin_min} min.",
             "", "| detector | PR-AUC | ROC-AUC | precision | recall | F1 | event recall | recall (impactful) "
             "| false alarms / h | median delay (min) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for m in sorted(metrics, key=lambda m: -(m["pr_auc"] or 0)):
        lines.append(f"| {m['detector']} | {m['pr_auc']} | {m['roc_auc']} | {m['precision']} | {m['recall']} "
                     f"| {m['f1']} | {m['event_recall']} | {m['event_recall_impact']} "
                     f"| {m['false_alarms_per_h']} | {m['median_delay_min']} |")
    lines += ["", "Incident share of test windows (PR-AUC of a random score): "
              f"{metrics[0]['incident_share']}.", "",
              "Caveat: synthetic capacity drops in a queueing model - this validates the detectors' behaviour "
              "and the evaluation machinery, not real-world detection rates. Incidents during low demand "
              "do not change traffic and are undetectable from traffic data (see 'recall (impactful)')."]
    return "\n".join(lines) + "\n"
