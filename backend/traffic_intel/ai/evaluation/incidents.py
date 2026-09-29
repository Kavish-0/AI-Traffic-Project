"""Incident / anomaly timing evaluation against labelled intervals (``IncidentAnnotation``).

Re-implementation of the AI City Challenge traffic-anomaly score as published (NOT the official
evaluation server - numbers are for internal comparison):

    a prediction (video, time, score) is a TRUE POSITIVE if it lies within ``window_s`` (10 s) of a
    labelled anomaly's start and that anomaly is not yet matched; other predictions are FALSE POSITIVES
    F1        from TP / FP / FN
    RMSE      of (prediction time - true start) over true positives
    NRMSE     = min(RMSE, cap_s) / cap_s          (cap 300 s)
    S4        = F1 x (1 - NRMSE)

The project's own outputs are hedged incidents with a type and probability. ``predictions_from_incidents``
turns them into timed predictions using the manifest's ``positive_system_types`` - any non-normal type
(stopped vehicle, possible incident, abnormal queue, sudden slowdown, wrong way) counts as a detection of
a labelled "anomaly"; nothing is renamed "accident".
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TimedPrediction:
    video_id: str
    time_s: float
    score: float = 1.0
    incident_type: str = ""


def predictions_from_incidents(rows: Iterable[dict[str, Any]], positive_types: Sequence[str],
                               min_probability: float = 0.0) -> list[TimedPrediction]:
    """rows: dicts with video_id, time_s (first detection, seconds from video start), incident_type, probability."""
    pos = {t.lower() for t in positive_types}
    return [TimedPrediction(str(r["video_id"]), float(r["time_s"]), float(r.get("probability", 1.0)),
                            r.get("incident_type", "")) for r in rows
            if str(r.get("incident_type", "")).lower() in pos and float(r.get("probability", 1.0)) >= min_probability]


def anomaly_timing_score(labels: Iterable[Any], predictions: Iterable[TimedPrediction], *, window_s: float = 10.0,
                         cap_s: float = 300.0) -> dict[str, float]:
    """labels: objects with video_id and start_s (IncidentAnnotation)."""
    gt: dict[str, list[float]] = defaultdict(list)
    for a in labels:
        gt[str(a.video_id)].append(float(a.start_s))
    preds: dict[str, list[TimedPrediction]] = defaultdict(list)
    for p in predictions:
        preds[p.video_id].append(p)
    tp = fp = 0
    errors: list[float] = []
    for vid in set(gt) | set(preds):
        starts = sorted(gt.get(vid, []))
        used = [False] * len(starts)
        for p in sorted(preds.get(vid, []), key=lambda x: (-x.score, x.time_s)):
            best, err = -1, math.inf
            for i, s in enumerate(starts):
                e = abs(p.time_s - s)
                if not used[i] and e <= window_s and e < err:
                    best, err = i, e
            if best >= 0:
                used[best] = True
                tp += 1
                errors.append(p.time_s - starts[best])
            else:
                fp += 1
    n_gt = sum(len(v) for v in gt.values())
    fn = n_gt - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / n_gt if n_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    rmse = math.sqrt(sum(e * e for e in errors) / len(errors)) if errors else float("nan")
    nrmse = min(rmse, cap_s) / cap_s if errors else 1.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4), "rmse_s": round(rmse, 3) if errors else float("nan"), "nrmse": round(nrmse, 4),
            "s4": round(f1 * (1.0 - nrmse), 4)}
