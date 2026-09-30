"""Detection and tracking evaluation against canonical ground truth (``AnnotatedFrame``).

Detection  per-class Average Precision at IoU >= 0.5 (all-point interpolated PR curve, VOC style),
           plus precision / recall at a confidence threshold. Unmatched predictions whose centre lies
           in a dataset "ignored region" (UA-DETRAC) are discarded, as in the DETRAC protocol.
Tracking   CLEAR-MOT: per frame, previous ID correspondences are kept while IoU >= threshold, the rest
           is matched with the Hungarian algorithm; MOTA = 1 - (FN + FP + IDSW) / GT, MOTP = mean IoU
           of matched pairs. IDF1 needs a global identity assignment and is left to py-motmetrics
           if required later.

Both work on the project's own types (Detections / Tracks / ObjectClass ids), so the detector and
tracker are evaluated exactly as the pipeline runs them.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from traffic_intel.core.enums import OBJECT_CLASSES
from traffic_intel.core.types import Detections, Tracks


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return np.asarray(inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9))


def _in_regions(boxes: np.ndarray, regions: np.ndarray) -> np.ndarray:
    if len(boxes) == 0 or len(regions) == 0:
        return np.zeros(len(boxes), dtype=bool)
    cx, cy = (boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2
    r = regions
    return np.asarray(((cx[:, None] >= r[None, :, 0]) & (cx[:, None] <= r[None, :, 2])
                       & (cy[:, None] >= r[None, :, 1]) & (cy[:, None] <= r[None, :, 3])).any(axis=1))


@dataclass
class DetectionEvaluator:
    iou_threshold: float = 0.5
    conf_threshold: float = 0.25
    _scores: dict[int, list[tuple[float, bool]]] = field(default_factory=lambda: defaultdict(list))
    _n_gt: dict[int, int] = field(default_factory=lambda: defaultdict(int))

    def add(self, pred: Detections, gt: Detections, ignore_regions: np.ndarray | None = None) -> None:
        ignore = ignore_regions if ignore_regions is not None else np.zeros((0, 4))
        for c in set(pred.class_id.tolist()) | set(gt.class_id.tolist()):
            p = pred.class_id == c
            g = gt.class_id == c
            pb, ps, gb = pred.xyxy[p], pred.confidence[p], gt.xyxy[g]
            self._n_gt[c] += int(g.sum())
            order = np.argsort(-ps)
            iou = iou_matrix(pb[order], gb)
            used = np.zeros(len(gb), dtype=bool)
            ign = _in_regions(pb[order], ignore)
            for i, k in enumerate(order):
                j = int(np.argmax(iou[i])) if len(gb) else -1
                if j >= 0 and iou[i, j] >= self.iou_threshold and not used[j]:
                    used[j] = True
                    self._scores[c].append((float(ps[k]), True))
                elif not ign[i]:
                    self._scores[c].append((float(ps[k]), False))

    def results(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for c in sorted(set(self._n_gt) | set(self._scores)):
            s = sorted(self._scores[c], key=lambda x: -x[0])
            tp = np.cumsum([m for _, m in s]) if s else np.array([])
            fp = np.cumsum([not m for _, m in s]) if s else np.array([])
            n = self._n_gt[c]
            recall = tp / n if n else np.zeros_like(tp, dtype=float)
            precision = tp / np.maximum(tp + fp, 1)
            ap = _average_precision(recall, precision) if n else float("nan")
            at = [m for sc, m in s if sc >= self.conf_threshold]
            out[OBJECT_CLASSES[c].value] = {
                "ap50": round(ap, 4), "n_gt": n, "n_pred": len(s),
                "precision": round(sum(at) / len(at), 4) if at else 0.0,
                "recall": round(sum(at) / n, 4) if n else float("nan")}
        valid = [v["ap50"] for v in out.values() if v["n_gt"] > 0]
        out["mean"] = {"ap50": round(float(np.mean(valid)), 4) if valid else float("nan")}
        return out


def _average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    if len(recall) == 0:
        return 0.0
    r = np.concatenate([[0.0], recall, [1.0]])
    p = np.concatenate([[1.0], precision, [0.0]])
    for i in range(len(p) - 2, -1, -1):
        p[i] = max(p[i], p[i + 1])
    idx = np.where(r[1:] != r[:-1])[0]
    return float(np.sum((r[idx + 1] - r[idx]) * p[idx + 1]))


@dataclass
class MOTEvaluator:
    iou_threshold: float = 0.5
    gt_total: int = 0
    fn: int = 0
    fp: int = 0
    idsw: int = 0
    iou_sum: float = 0.0
    matches: int = 0
    _last: dict[int, int] = field(default_factory=dict)    # gt id -> predicted id of its last match

    def add(self, pred: Tracks, gt: Tracks) -> None:
        gi, pi = gt.track_id.tolist(), pred.track_id.tolist()
        iou = iou_matrix(gt.xyxy, pred.xyxy)
        self.gt_total += len(gi)
        matched_g, matched_p, pairs = set(), set(), []
        for g, gid in enumerate(gi):                           # 1) keep previous correspondences
            pid = self._last.get(gid)
            if pid in pi:
                p = pi.index(pid)
                if iou[g, p] >= self.iou_threshold and p not in matched_p:
                    matched_g.add(g)
                    matched_p.add(p)
                    pairs.append((g, p))
        rest_g = [g for g in range(len(gi)) if g not in matched_g]
        rest_p = [p for p in range(len(pi)) if p not in matched_p]
        if rest_g and rest_p:                                  # 2) Hungarian on the rest
            cost = 1.0 - iou[np.ix_(rest_g, rest_p)]
            for r, c in zip(*linear_sum_assignment(cost), strict=True):
                if iou[rest_g[r], rest_p[c]] >= self.iou_threshold:
                    g, p = rest_g[r], rest_p[c]
                    matched_g.add(g)
                    matched_p.add(p)
                    pairs.append((g, p))
                    if gi[g] in self._last and self._last[gi[g]] != pi[p]:
                        self.idsw += 1
        for g, p in pairs:
            self._last[gi[g]] = pi[p]
            self.iou_sum += float(iou[g, p])
        self.matches += len(pairs)
        self.fn += len(gi) - len(matched_g)
        self.fp += len(pi) - len(matched_p)

    def results(self) -> dict[str, float]:
        return {"MOTA": round(1.0 - (self.fn + self.fp + self.idsw) / max(self.gt_total, 1), 4),
                "MOTP_iou": round(self.iou_sum / max(self.matches, 1), 4),
                "precision": round(self.matches / max(self.matches + self.fp, 1), 4),
                "recall": round(self.matches / max(self.gt_total, 1), 4),
                "id_switches": self.idsw, "fn": self.fn, "fp": self.fp, "gt": self.gt_total}


def evaluate_detection(pairs: Iterable[tuple[Detections, Detections, np.ndarray | None]],
                       iou_threshold: float = 0.5, conf_threshold: float = 0.25) -> dict[str, dict[str, float]]:
    ev = DetectionEvaluator(iou_threshold=iou_threshold, conf_threshold=conf_threshold)
    for pred, gt, ign in pairs:
        ev.add(pred, gt, ign)
    return ev.results()
