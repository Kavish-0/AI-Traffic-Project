"""Series-level baseline detectors: rolling statistics, moving averages, z-scores, EWMA.

Each detector scores the NEWEST 60 s window of an approach against the windows
immediately before it (a local, adaptive baseline - no global thresholds that
break between sites or times of day). Each returns 0 or 1 ``AnomalySignal``
with a score in [0, 1] and the numbers that produced it in ``details``.

sudden_slowdown   relative drop of space-mean speed vs an EWMA baseline, gated by a
                  z-score so ordinary fluctuation does not fire:
                      drop  = (baseline - current) / baseline
                      z     = (baseline - current) / max(robust std (1.4826 x MAD), floor x baseline)
                      score = ramp(drop; drop_min -> drop_full) x min(1, z / z_full)
                  e.g. 45, 43, 42, 40, 18 -> baseline ~42, drop 57%, z >> 3 -> score ~0.9
                  When the 85th-percentile speed is available the same score is computed for it
                  and the smaller one is used: a real stream slowdown slows the FAST vehicles too,
                  while one stopped vehicle drags the mean down but leaves p85 unchanged (that case
                  is the stopped-vehicle detector's evidence and must not be counted twice).
queue_growth      queue rising: least-squares slope over the last windows (m/window) and
                  rise above the recent median queue.
density_zscore    robust z-score (median / MAD) of density_ratio vs recent windows; upward only.
                  The density that the window's STOPPED vehicles account for is removed first
                  (stopped_vehicles x density per vehicle in the region): a single broken-down car
                  is the stopped-vehicle detector's evidence, not a traffic-density change.
flow_imbalance    vehicles accumulating: sum(inflow - outflow) / sum(inflow) over the last
                  windows, plus outflow collapse vs its recent median.
ewma              EWMA control chart on several features (speed down, density /
                  occupancy / queue up): |x - ewma| / ewm-std beyond L.
zscore            moving-average + rolling-std z-score on the same features.

Why 60 s windows: at signalised junctions queues and speeds oscillate within
every signal cycle; one-minute aggregates average most of a cycle, so the
baseline is the normal cyclic state and only departures from it score.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from traffic_intel.anomaly.interfaces import SERIES_ANOMALY_DETECTORS, SeriesAnomalyDetector
from traffic_intel.core.enums import AnomalyScope, AnomalyType
from traffic_intel.core.schemas import AnomalySignal


def ramp(x: float, lo: float, hi: float) -> float:
    """0 below lo, 1 above hi, linear in between."""
    if hi <= lo:
        return float(x >= hi)
    return float(min(1.0, max(0.0, (x - lo) / (hi - lo))))


def _series(history: pd.DataFrame, col: str) -> np.ndarray:
    return pd.to_numeric(history[col], errors="coerce").to_numpy(dtype=float) if col in history else np.array([])


class _Base(SeriesAnomalyDetector):
    type: AnomalyType = AnomalyType.STATISTICAL_OUTLIER

    def __init__(self, baseline_windows: int = 10, min_baseline_windows: int = 4, min_quality: float = 0.5) -> None:
        self.baseline_windows = baseline_windows
        self.min_baseline_windows = min_baseline_windows
        self.min_quality = min_quality

    def _ready(self, history: pd.DataFrame) -> bool:
        if len(history) < self.min_baseline_windows + 1:
            return False
        q = history["quality"].iloc[-1] if "quality" in history else 1.0
        return bool(pd.isna(q) or q >= self.min_quality)

    def _signal(self, approach_id: str, history: pd.DataFrame, score: float, details: dict[str, Any],
                type_: AnomalyType | None = None) -> list[AnomalySignal]:
        if score <= 0:
            return []
        ts = pd.Timestamp(history["timestamp"].iloc[-1]).to_pydatetime()
        return [AnomalySignal(approach_id=approach_id, camera_id=None, timestamp=ts, type=type_ or self.type,
                              scope=AnomalyScope.SERIES, detector=self.name, score=round(min(1.0, score), 3),
                              details={k: (round(v, 3) if isinstance(v, float) else v) for k, v in details.items()})]


@SERIES_ANOMALY_DETECTORS.register("sudden_slowdown")
class SuddenSlowdownDetector(_Base):
    name = "sudden_slowdown"
    type = AnomalyType.SUDDEN_SLOWDOWN

    def __init__(self, ewma_alpha: float = 0.3, drop_min: float = 0.25, drop_full: float = 0.6, z_full: float = 3.0,
                 std_floor_ratio: float = 0.1, min_baseline_kmh: float = 15.0, **kw: Any) -> None:
        super().__init__(**kw)
        self.alpha, self.drop_min, self.drop_full, self.z_full = ewma_alpha, drop_min, drop_full, z_full
        self.std_floor_ratio, self.min_baseline_kmh = std_floor_ratio, min_baseline_kmh

    def _drop_score(self, v: np.ndarray) -> tuple[float, dict[str, Any]] | None:
        cur, prev = v[-1], v[:-1][-self.baseline_windows:]
        prev = prev[np.isfinite(prev)]
        if not np.isfinite(cur) or len(prev) < self.min_baseline_windows:
            return None
        base = float(pd.Series(prev).ewm(alpha=self.alpha).mean().iloc[-1])
        if base < self.min_baseline_kmh:
            return None
        drop = (base - cur) / base
        # robust spread: one abnormal window already in the baseline must not mask the next one
        mad = 1.4826 * float(np.median(np.abs(prev - np.median(prev))))
        std = max(mad, self.std_floor_ratio * base)
        z = (base - cur) / std
        score = ramp(drop, self.drop_min, self.drop_full) * min(1.0, max(0.0, z) / self.z_full)
        return score, {"current_kmh": float(cur), "baseline_kmh": base, "drop_pct": 100 * drop, "z": float(z),
                       "recent_kmh": [round(float(x), 1) for x in v[-6:]]}

    def score(self, approach_id: str, history: pd.DataFrame) -> list[AnomalySignal]:
        if not self._ready(history):
            return []
        mean = self._drop_score(_series(history, "avg_speed_kmh"))
        if mean is None:
            return []
        score, details = mean
        p85 = self._drop_score(_series(history, "speed_p85_kmh")) if "speed_p85_kmh" in history else None
        if p85 is not None:
            details["p85_drop_pct"] = p85[1]["drop_pct"]
            score = min(score, p85[0])
        return self._signal(approach_id, history, score, details)


@SERIES_ANOMALY_DETECTORS.register("queue_growth")
class QueueGrowthDetector(_Base):
    name = "queue_growth"
    type = AnomalyType.SUDDEN_QUEUE

    def __init__(self, trend_windows: int = 4, rate_full_m_per_window: float = 8.0, rise_min_m: float = 10.0,
                 rise_full_m: float = 30.0, **kw: Any) -> None:
        super().__init__(**kw)
        self.trend_windows, self.rate_full = trend_windows, rate_full_m_per_window
        self.rise_min, self.rise_full = rise_min_m, rise_full_m

    def score(self, approach_id: str, history: pd.DataFrame) -> list[AnomalySignal]:
        if not self._ready(history):
            return []
        q = _series(history, "queue_length_m")
        recent = q[-self.trend_windows:]
        base = q[:-self.trend_windows][-self.baseline_windows:]
        base = base[np.isfinite(base)]
        if len(recent) < 2 or not np.isfinite(recent).all() or len(base) < 2:
            return []
        slope = float(np.polyfit(np.arange(len(recent)), recent, 1)[0])
        rise = float(recent[-1] - np.median(base))
        score = ramp(slope, 0.0, self.rate_full) * ramp(rise, self.rise_min, self.rise_full)
        exceeds = history["queue_exceeds_view_fraction"].iloc[-1] if "queue_exceeds_view_fraction" in history else 0
        if pd.notna(exceeds) and exceeds > 0.5 and rise > 0:
            score = max(score, 0.6)                 # queue running out of the camera view
        return self._signal(approach_id, history, score, {
            "queue_m": float(recent[-1]), "baseline_median_m": float(np.median(base)), "rise_m": rise,
            "slope_m_per_window": slope, "recent_m": [round(float(x), 1) for x in q[-6:]]})


@SERIES_ANOMALY_DETECTORS.register("density_zscore")
class DensityZScoreDetector(_Base):
    name = "density_zscore"
    type = AnomalyType.UNUSUAL_DENSITY

    def __init__(self, z_min: float = 3.0, z_full: float = 6.0, min_delta: float = 0.08, **kw: Any) -> None:
        super().__init__(**kw)
        self.z_min, self.z_full, self.min_delta = z_min, z_full, min_delta

    def score(self, approach_id: str, history: pd.DataFrame) -> list[AnomalySignal]:
        if not self._ready(history):
            return []
        d = _series(history, "density_ratio")
        cur, prev = d[-1], d[:-1][-self.baseline_windows:]
        prev = prev[np.isfinite(prev)]
        if not np.isfinite(cur) or len(prev) < self.min_baseline_windows:
            return []
        explained = 0.0
        if "stopped_vehicles" in history and "mean_vehicles_in_region" in history:
            n_stop, n_reg = history["stopped_vehicles"].iloc[-1], history["mean_vehicles_in_region"].iloc[-1]
            if pd.notna(n_stop) and pd.notna(n_reg) and n_reg > 0:
                explained = float(min(n_stop, n_reg)) * float(cur) / float(n_reg)
        adj = cur - explained
        med = float(np.median(prev))
        mad = 1.4826 * float(np.median(np.abs(prev - med)))
        z = (adj - med) / max(mad, 0.02)
        score = ramp(z, self.z_min, self.z_full) * float(adj - med >= self.min_delta)
        return self._signal(approach_id, history, score, {"density_ratio": float(cur), "baseline_median": med,
                                                          "explained_by_stopped": explained, "robust_z": float(z)})


@SERIES_ANOMALY_DETECTORS.register("flow_imbalance")
class FlowImbalanceDetector(_Base):
    name = "flow_imbalance"
    type = AnomalyType.UNUSUAL_FLOW

    def __init__(self, windows: int = 3, min_inflow: int = 5, ratio_min: float = 0.25, ratio_full: float = 0.6,
                 collapse_ratio: float = 0.3, **kw: Any) -> None:
        super().__init__(**kw)
        self.windows, self.min_inflow = windows, min_inflow
        self.ratio_min, self.ratio_full, self.collapse_ratio = ratio_min, ratio_full, collapse_ratio

    def score(self, approach_id: str, history: pd.DataFrame) -> list[AnomalySignal]:
        if not self._ready(history):
            return []
        inflow, outflow = _series(history, "inflow"), _series(history, "outflow")
        i_r, o_r = inflow[-self.windows:], outflow[-self.windows:]
        if not (np.isfinite(i_r).all() and np.isfinite(o_r).all()) or i_r.sum() < self.min_inflow:
            return []
        ratio = float((i_r.sum() - o_r.sum()) / max(i_r.sum(), 1.0))
        base_out = outflow[:-self.windows][-self.baseline_windows:]
        base_out = base_out[np.isfinite(base_out)]
        collapse = bool(len(base_out) >= 2 and np.median(base_out) > 0
                        and o_r[-1] < self.collapse_ratio * np.median(base_out))
        score = ramp(ratio, self.ratio_min, self.ratio_full)
        if collapse and ratio > 0:
            score = max(score, 0.5)
        return self._signal(approach_id, history, score, {
            "accumulation_ratio": ratio, "inflow_recent": int(i_r.sum()), "outflow_recent": int(o_r.sum()),
            "outflow_collapse": collapse})


_BAD_DIRECTION = {"avg_speed_kmh": -1.0, "density_ratio": 1.0, "occupancy_pct": 1.0, "queue_length_m": 1.0}


class _MultiFeature(_Base):
    def __init__(self, columns: list[str] | None = None, L: float = 3.0, full: float = 6.0, **kw: Any) -> None:
        super().__init__(**kw)
        self.columns = columns or list(_BAD_DIRECTION)
        self.L, self.full = L, full

    def _deviation(self, x: np.ndarray) -> float | None:  # pragma: no cover - abstract
        raise NotImplementedError

    def score(self, approach_id: str, history: pd.DataFrame) -> list[AnomalySignal]:
        if not self._ready(history):
            return []
        stats: dict[str, float] = {}
        for c in self.columns:
            x = _series(history, c)
            if len(x) == 0 or not np.isfinite(x[-1]):
                continue
            dev = self._deviation(x[np.isfinite(x)])
            if dev is not None:
                stats[c] = _BAD_DIRECTION.get(c, 1.0) * dev    # >0 = deviation in the harmful direction
        if not stats:
            return []
        worst = max(stats, key=stats.get)  # type: ignore[arg-type]
        return self._signal(approach_id, history, ramp(stats[worst], self.L, self.full),
                            {"feature": worst, "statistic": stats[worst],
                             "all": {k: round(v, 2) for k, v in stats.items()}})


@SERIES_ANOMALY_DETECTORS.register("ewma")
class EWMAControlChart(_MultiFeature):
    """EWMA mean and variance over previous windows; statistic = (x - ewma) / ewm-std."""

    name = "ewma"

    def __init__(self, lam: float = 0.3, std_floor: float = 1e-3, **kw: Any) -> None:
        super().__init__(**kw)
        self.lam, self.std_floor = lam, std_floor

    def _deviation(self, x: np.ndarray) -> float | None:
        prev = x[:-1][-self.baseline_windows * 2:]
        if len(prev) < self.min_baseline_windows:
            return None
        s = pd.Series(prev)
        mean = float(s.ewm(alpha=self.lam).mean().iloc[-1])
        std = float(np.sqrt(s.ewm(alpha=self.lam).var().iloc[-1] or 0.0))
        return float((x[-1] - mean) / max(std, self.std_floor + 0.05 * abs(mean)))


@SERIES_ANOMALY_DETECTORS.register("zscore")
class RollingZScore(_MultiFeature):
    """Moving average + rolling standard deviation of the previous windows."""

    name = "zscore"

    def _deviation(self, x: np.ndarray) -> float | None:
        prev = x[:-1][-self.baseline_windows:]
        if len(prev) < self.min_baseline_windows:
            return None
        return float((x[-1] - prev.mean()) / max(prev.std(), 0.05 * abs(prev.mean()) + 1e-3))
