"""Pluggable window-feature calculators.

Each calculator reads a ``WindowData`` and returns a dict. Keys that are
``TrafficFeatures`` fields fill those fields; any other key must be a float
and lands in ``TrafficFeatures.extra``. Calculators run in order and may read
earlier results from ``window.results``.

Add a feature:
    @FEATURE_CALCULATORS.register("my_feature")
    class MyFeature(FeatureCalculator):
        def compute(self, w): return {"my_metric": ...}
then list it in ``feature_engine.params.calculators`` (or it is picked up by
``DEFAULT_CALCULATORS`` if you add it there).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from typing import Any

import numpy as np

from traffic_intel.core.enums import OBJECT_CLASSES, VEHICLE_CLASS_IDS, ObjectClass, ZoneType
from traffic_intel.core.registry import Registry
from traffic_intel.traffic.features.samples import FrameSample, WindowData, head_offset, tail_offset


class FeatureCalculator(ABC):
    name: str = ""

    @abstractmethod
    def compute(self, w: WindowData) -> dict[str, Any]: ...


FEATURE_CALCULATORS: Registry[FeatureCalculator] = Registry("feature_calculator")

DEFAULT_CALCULATORS = [
    "counts", "speed", "density", "occupancy", "flow", "stopped", "queue", "gaps", "trajectories",
]


def _longest_run(times: list[float], max_gap: float) -> float:
    """Longest time span covered by consecutive samples no more than ``max_gap`` apart."""
    best, start = 0.0, times[0]
    for prev, cur in zip(times, times[1:], strict=False):
        if cur - prev > max_gap:
            best = max(best, prev - start)
            start = cur
    return max(best, times[-1] - start)


def _vehicle_mask(s: FrameSample) -> np.ndarray:
    return s.is_vehicle


# --------------------------------------------------------------------------- #
@FEATURE_CALCULATORS.register("counts")
class CountCalculator(FeatureCalculator):
    """Unique tracks present in the ROI during the window: >= min_track_points
    samples and a continuous run of >= min_presence_s inside the ROI. A track's
    class is its latest majority-vote class. Tracks that
    span a window boundary are present in both windows by definition;
    throughput is measured by ``flow``, not by counts."""

    name = "counts"

    def compute(self, w: WindowData) -> dict[str, Any]:
        times: dict[int, list[float]] = defaultdict(list)
        cls_of: dict[int, int] = {}
        for s in w.samples:
            for tid, cid in zip(s.track_id.tolist(), s.class_id.tolist(), strict=True):
                times[tid].append(s.t)
                cls_of[tid] = cid
        # A "continuous" run tolerates a couple of missed frames.
        frame_dt = w.duration_s / max(w.expected_frames, 1.0)
        max_gap = 2.5 * frame_dt
        counted = [
            tid for tid, ts in times.items()
            if len(ts) >= w.cfg.min_track_points and _longest_run(ts, max_gap) >= w.cfg.min_presence_s - 1e-9
        ]
        by_class: Counter[int] = Counter(cls_of[t] for t in counted)
        count_by_class = {OBJECT_CLASSES[c]: int(by_class.get(c, 0)) for c in range(len(OBJECT_CLASSES))}
        vehicles = sum(n for c, n in by_class.items() if c in VEHICLE_CLASS_IDS)
        pcu = float(sum(w.pcu[c] * n for c, n in by_class.items() if c in VEHICLE_CLASS_IDS))
        return {
            "vehicle_count": int(vehicles),
            "count_by_class": count_by_class,
            "pedestrian_count": int(count_by_class[ObjectClass.PEDESTRIAN]),
            "pcu_count": round(pcu, 2),
        }


@FEATURE_CALCULATORS.register("speed")
class SpeedCalculator(FeatureCalculator):
    """Space-mean speed = mean of instantaneous speeds of all vehicles in the
    measurement region, over all sampled instants (this is the definition of
    space-mean speed for spatial snapshots; the harmonic mean applies to
    *point* observations such as loop detectors). Stopped vehicles contribute
    ~0 km/h and are deliberately included."""

    name = "speed"

    def compute(self, w: WindowData) -> dict[str, Any]:
        values: list[np.ndarray] = []
        per_track: dict[int, list[float]] = defaultdict(list)
        for s in w.samples:
            m = _vehicle_mask(s) & s.in_measure & np.isfinite(s.speed_kmh)
            values.append(s.speed_kmh[m])
            for tid, v in zip(s.track_id[m].tolist(), s.speed_kmh[m].tolist(), strict=True):
                per_track[tid].append(v)
        v = np.concatenate(values) if values else np.zeros(0)
        edges = list(w.cfg.speed_histogram_edges_kmh)
        labels = [f"{int(a)}-{int(b)}" for a, b in zip(edges[:-1], edges[1:], strict=True)] + [f"{int(edges[-1])}+"]
        hist = dict.fromkeys(labels, 0)
        for speeds in per_track.values():
            idx = int(np.searchsorted(edges, float(np.mean(speeds)), side="right")) - 1
            hist[labels[min(max(idx, 0), len(labels) - 1)]] += 1
        enough = len(v) >= w.cfg.min_speed_samples
        return {
            "avg_speed_kmh": round(float(v.mean()), 2) if enough else None,
            "median_speed_kmh": round(float(np.median(v)), 2) if enough else None,
            "speed_p85_kmh": round(float(np.percentile(v, 85)), 2) if enough else None,
            "speed_distribution": hist,
            "speed_std_kmh": round(float(v.std()), 2) if enough else float("nan"),
            "speed_sample_count": float(len(v)),
            "speed_vehicle_count": float(len(per_track)),
        }


@FEATURE_CALCULATORS.register("density")
class DensityCalculator(FeatureCalculator):
    """Time-averaged vehicles (PCU if enabled) in the measurement region per
    km per lane; ratio against jam density."""

    name = "density"

    def compute(self, w: WindowData) -> dict[str, Any]:
        per_sample = []
        for s in w.samples:
            m = _vehicle_mask(s) & s.in_measure
            per_sample.append(float(w.pcu[s.class_id[m]].sum()) if w.cfg.use_pcu else float(m.sum()))
        mean_in_region = float(np.mean(per_sample)) if per_sample else 0.0
        lane_km = w.geometry.roi_length_m / 1000.0 * w.geometry.lanes
        density = mean_in_region / lane_km if lane_km > 0 else 0.0
        return {
            "density_veh_km": round(density, 2),
            "density_ratio": round(density / w.cfg.jam_density_veh_km_lane, 4),
            "mean_vehicles_in_region": round(mean_in_region, 3),
        }


@FEATURE_CALCULATORS.register("occupancy")
class OccupancyCalculator(FeatureCalculator):
    """With a virtual loop: % of sampled instants a vehicle footprint covers the
    loop (like an inductive loop). Without one: space occupancy = % of lane
    length covered by typical vehicle lengths."""

    name = "occupancy"

    def compute(self, w: WindowData) -> dict[str, Any]:
        if not w.samples:
            return {"occupancy_pct": 0.0, "occupancy_method": 0.0 if w.geometry.has_loop else 1.0}
        if w.geometry.has_loop:
            occ = 100.0 * float(np.mean([s.loop_occupied for s in w.samples]))
            return {"occupancy_pct": round(occ, 2), "occupancy_method": 0.0}
        lane_m = w.geometry.roi_length_m * w.geometry.lanes
        vals = []
        for s in w.samples:
            m = _vehicle_mask(s) & s.in_measure
            vals.append(float(w.lengths[s.class_id[m]].sum()) / lane_m if lane_m > 0 else 0.0)
        return {"occupancy_pct": round(min(100.0, 100.0 * float(np.mean(vals))), 2), "occupancy_method": 1.0}


@FEATURE_CALCULATORS.register("flow")
class FlowCalculator(FeatureCalculator):
    """inflow/outflow = forward crossings of entry/exit lines (vehicles).
    flow = outflow (else inflow) scaled to veh/h; with no lines, the
    fundamental relation q = k * v * lanes is used and flagged."""

    name = "flow"

    def compute(self, w: WindowData) -> dict[str, Any]:
        inflow = outflow = reverse = 0
        for e in w.events:
            if e.class_id not in VEHICLE_CLASS_IDS:
                continue
            if not e.forward:
                reverse += 1
            elif e.kind is ZoneType.ENTRY_LINE:
                inflow += 1
            elif e.kind is ZoneType.EXIT_LINE:
                outflow += 1
        per_h = 3600.0 / w.duration_s
        if w.geometry.has_exit_line:
            flow, source = outflow * per_h, 0.0
        elif w.geometry.has_entry_line:
            flow, source = inflow * per_h, 1.0
        else:
            k = float(w.results.get("density_veh_km", 0.0))  # type: ignore[arg-type]
            v = w.results.get("avg_speed_kmh")
            speed = float(v) if v is not None else None  # type: ignore[arg-type]
            flow, source = (k * speed * w.geometry.lanes if speed is not None else 0.0), 2.0
        return {
            "inflow": inflow,
            "outflow": outflow,
            "flow_veh_h": round(flow, 1),
            "flow_source": source,
            "reverse_crossings": float(reverse),
        }


@FEATURE_CALCULATORS.register("stopped")
class StoppedCalculator(FeatureCalculator):
    """stopped_vehicles = time-averaged number of vehicles that have been below
    the stopped-speed threshold for >= stopped_min_s (rounded)."""

    name = "stopped"

    def compute(self, w: WindowData) -> dict[str, Any]:
        per_sample = []
        unique: set[int] = set()
        for s in w.samples:
            m = _vehicle_mask(s) & s.stopped
            per_sample.append(int(m.sum()))
            unique.update(s.track_id[m].tolist())
        return {
            "stopped_vehicles": int(round(float(np.mean(per_sample)))) if per_sample else 0,
            "stopped_vehicles_max": float(max(per_sample, default=0)),
            "stopped_vehicles_unique": float(len(unique)),
        }


def queue_length_at(s: FrameSample, w: WindowData) -> tuple[float, int]:
    """Queue length (m) and queued-vehicle count in one sample.

    A queue is a chain of queued (slow/stopped) vehicles in the queue zone
    that starts within ``queue_head_max_m`` of the stop line and continues
    while the bumper gap to the next queued vehicle is <= ``queue_max_gap_m``.
    Computed per lane; the approach queue is the longest lane queue.
    Length = distance from the stop line to the tail of the last vehicle; the
    tail is located using the anchor reference (front/rear/center of vehicle).
    """
    cfg = w.cfg
    m = _vehicle_mask(s) & s.queued & s.in_queue_zone & np.isfinite(s.along_m) & (s.along_m > -2.0)
    if not m.any():
        return 0.0, 0
    lanes = s.lane[m] if (s.lane[m] >= 0).all() else np.floor(s.lateral_m[m] / cfg.lane_width_m).astype(int)
    along = s.along_m[m]
    lengths = w.lengths[s.class_id[m]]
    best_len, best_n = 0.0, 0
    for lane in np.unique(lanes):
        sel = lanes == lane
        order = np.argsort(along[sel])
        a, L = along[sel][order], lengths[sel][order]
        if a[0] > cfg.queue_head_max_m:
            continue
        anchor = w.geometry.anchor_reference
        tail, head = a + tail_offset(anchor, L), a - head_offset(anchor, L)
        n = 1
        for i in range(1, len(a)):
            if head[i] - tail[i - 1] > cfg.queue_max_gap_m:     # bumper gap to the vehicle ahead
                break
            n += 1
        length = max(0.0, float(tail[n - 1]))
        if length > best_len:
            best_len, best_n = length, n
    return best_len, best_n


@FEATURE_CALCULATORS.register("queue")
class QueueCalculator(FeatureCalculator):
    name = "queue"

    def compute(self, w: WindowData) -> dict[str, Any]:
        if not w.samples:
            return {"queue_length_m": 0.0, "queue_max_m": 0.0, "queue_vehicles_mean": 0.0,
                    "queue_exceeds_view_fraction": 0.0}
        res = [queue_length_at(s, w) for s in w.samples]
        lengths = np.array([r[0] for r in res])
        limit = w.cfg.queue_view_limit_ratio * w.geometry.roi_length_m
        return {
            "queue_length_m": round(float(lengths.mean()), 1),
            "queue_max_m": round(float(lengths.max()), 1),
            "queue_vehicles_mean": round(float(np.mean([r[1] for r in res])), 2),
            # Queue tail at the edge of the view: the true queue may be longer.
            "queue_exceeds_view_fraction": round(float(np.mean(lengths >= limit)), 3),
        }


@FEATURE_CALCULATORS.register("gaps")
class GapCalculator(FeatureCalculator):
    """Mean bumper-to-bumper gap between consecutive vehicles in the same lane:
    distance from the tail of the vehicle ahead to the nose of the one behind,
    using typical class lengths and the anchor reference."""

    name = "gaps"

    def compute(self, w: WindowData) -> dict[str, Any]:
        gaps: list[float] = []
        spacings: list[float] = []
        for s in w.samples:
            m = _vehicle_mask(s) & s.in_measure & np.isfinite(s.along_m)
            if m.sum() < 2:
                continue
            lanes = s.lane[m] if (s.lane[m] >= 0).all() else np.floor(s.lateral_m[m] / w.cfg.lane_width_m)
            along, L = s.along_m[m], w.lengths[s.class_id[m]]
            for lane in np.unique(lanes):
                sel = lanes == lane
                if sel.sum() < 2:
                    continue
                order = np.argsort(along[sel])
                a, ll = along[sel][order], L[sel][order]
                anchor = w.geometry.anchor_reference
                sp = np.diff(a)
                spacings.extend(sp.tolist())
                bumper = (a[1:] - head_offset(anchor, ll[1:])) - (a[:-1] + tail_offset(anchor, ll[:-1]))
                gaps.extend(np.clip(bumper, 0.0, None).tolist())
        return {
            "avg_gap_m": round(float(np.mean(gaps)), 2) if gaps else None,
            "avg_spacing_m": round(float(np.mean(spacings)), 2) if spacings else float("nan"),
        }


@FEATURE_CALCULATORS.register("trajectories")
class TrajectoryStatsCalculator(FeatureCalculator):
    """Statistics over trajectories that *ended* in this window."""

    name = "trajectories"

    def compute(self, w: WindowData) -> dict[str, Any]:
        vehicles = [t for t in w.finished if t.object_class.is_vehicle]
        if not vehicles:
            return {"trajectory_stats": {"n": 0.0}}
        durations = [(t.last_seen - t.first_seen).total_seconds() for t in vehicles]
        straight = []
        for t in vehicles:
            p = np.asarray(t.path_ground_m)
            if len(p) >= 2:
                path = float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
                if path > 1.0:
                    straight.append(float(np.linalg.norm(p[-1] - p[0])) / path)
        speeds = [t.mean_speed_kmh for t in vehicles if t.mean_speed_kmh is not None]
        along = [t.displacement_along_travel_m for t in vehicles if t.displacement_along_travel_m is not None]
        stats = {
            "n": float(len(vehicles)),
            "mean_duration_s": round(float(np.mean(durations)), 2),
            "mean_distance_m": round(float(np.mean([t.distance_m for t in vehicles])), 2),
            "mean_straightness": round(float(np.mean(straight)), 3) if straight else float("nan"),
            "mean_track_speed_kmh": round(float(np.mean(speeds)), 2) if speeds else float("nan"),
            "completed_entry_to_exit": float(sum(t.crossed_entry and t.crossed_exit for t in vehicles)),
            "reverse_direction_tracks": float(sum(a < -2.0 for a in along)),
            "max_stopped_s": round(float(max(t.max_stopped_s for t in vehicles)), 1),
        }
        return {"trajectory_stats": stats}
