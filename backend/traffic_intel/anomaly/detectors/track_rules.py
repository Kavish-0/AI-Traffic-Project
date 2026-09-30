"""Track-level anomaly detectors (run inside the camera worker, every processed frame).

All three use kinematics the pipeline already computes (TrajectoryStore: ground
positions, smoothed speed, stopped duration) and the camera's configured zone
geometry. Nothing about road directions is hard-coded: the expected direction of
travel comes from each lane zone's ``direction`` (camera YAML), falling back to
the camera's ``travel_direction``.

stopped_vehicle   a vehicle nearly stationary for >= ``min_stopped_s``. Vehicles
                  waiting inside the queue zone are normal at a signal and only count
                  after ``max_queue_wait_s`` (a very long wait). Confidence rises with
                  duration, with trajectory reliability, and when other traffic keeps
                  moving past (an isolated stop is more suspicious than a jam).
wrong_way         moving against the lane's configured direction: net displacement
                  against it >= ``min_distance_m`` within ``window_s``, heading cosine
                  <= ``cos_threshold``, speed >= ``min_speed_kmh`` - and the movement must be
                  PHYSICAL: most steps go against the lane (``min_step_fraction``) and no step
                  is faster than ``max_step_speed_kmh``. A tracker identity switch makes a track
                  "teleport" backwards in one frame (dense queues); that is not wrong-way driving.
trajectory_rules  erratic motion: sharp heading changes at speed (swerving) and
                  sustained driving outside all lane zones. Windows containing an implausible
                  step (identity switch / detection glitch) are skipped, not reported.

Every signal carries a score in [0, 1] (strength of evidence) - never a verdict.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from traffic_intel.anomaly.interfaces import TRACK_ANOMALY_DETECTORS, TrackAnomalyDetector, TrackContext
from traffic_intel.core.enums import AnomalyScope, AnomalyType, ObjectClass
from traffic_intel.core.schemas import AnomalySignal
from traffic_intel.core.types import TrackedFrame
from traffic_intel.tracking.trajectory import Trajectory, TrajectoryPoint


def _location(ctx: TrackContext, g: tuple[float, float] | None) -> dict[str, Any]:
    if g is None:
        return {}
    lon, lat = ctx.geometry.projector.ground_to_lonlat(np.array([g], dtype=np.float64))[0]
    along = float(ctx.geometry.distance_upstream(np.array([g], dtype=np.float32))[0])
    return {"lon": round(float(lon), 7), "lat": round(float(lat), 7), "ground_x_m": round(g[0], 2),
            "ground_y_m": round(g[1], 2), "distance_upstream_of_stop_line_m": round(along, 1)}


def _signal(ctx: TrackContext, tracked: TrackedFrame, type_: AnomalyType, detector: str, score: float,
            tr: Trajectory, details: dict[str, Any]) -> AnomalySignal:
    g = tr.last.ground_m if tr.points else None
    return AnomalySignal(
        approach_id=ctx.approach_id, camera_id=ctx.camera_id, timestamp=tracked.timestamp, type=type_,
        scope=AnomalyScope.TRACK, detector=detector, score=round(float(min(1.0, max(0.0, score))), 3),
        track_ids=[tr.track_id], ground_xy_m=g,
        details={"road_id": ctx.road_id, "junction_id": ctx.junction_id, "object_class": tr.object_class,
                 "location": _location(ctx, g), **details},
    )


def _is_vehicle(tr: Trajectory) -> bool:
    return ObjectClass(tr.object_class).is_vehicle


def _steps(pts: list[TrajectoryPoint], max_step_speed_kmh: float) -> tuple[np.ndarray, bool]:
    """Ground displacement per step and whether every step is physically plausible."""
    g = np.array([p.ground_m for p in pts], dtype=np.float64)
    ts = np.array([p.t for p in pts], dtype=np.float64)
    seg = np.diff(g, axis=0)
    dt = np.maximum(np.diff(ts), 1e-3)
    step_kmh = np.linalg.norm(seg, axis=1) / dt * 3.6
    return seg, bool((step_kmh <= max_step_speed_kmh).all())


def _expected_direction(ctx: TrackContext, anchor: tuple[float, float]) -> np.ndarray:
    lane = int(ctx.geometry.lane_of(np.array([anchor], np.float32))[0])
    for z in ctx.geometry.camera.zones:
        if z.type.value == "lane" and z.lane_index == lane and z.direction is not None:
            d = np.asarray(z.direction, np.float64)
            return d / (np.linalg.norm(d) or 1.0)
    return np.asarray(ctx.geometry.travel_direction, np.float64)


class _Reemit:
    """Emit when a condition first holds, then at most every ``repeat_s`` while it holds."""

    def __init__(self, repeat_s: float) -> None:
        self.repeat_s = repeat_s
        self._last: dict[int, float] = {}

    def due(self, track_id: int, t: float) -> bool:
        last = self._last.get(track_id)
        if last is None or t - last >= self.repeat_s:
            self._last[track_id] = t
            return True
        return False

    def forget_missing(self, alive: set[int]) -> None:
        for tid in [k for k in self._last if k not in alive]:
            del self._last[tid]


@TRACK_ANOMALY_DETECTORS.register("stopped_vehicle")
class StoppedVehicleDetector(TrackAnomalyDetector):
    name = "stopped_vehicle"

    def __init__(self, min_stopped_s: float = 30.0, max_speed_kmh: float = 3.0, exclude_queue_zone: bool = True,
                 max_queue_wait_s: float = 180.0, moving_speed_kmh: float = 15.0, repeat_s: float = 30.0) -> None:
        self.min_stopped_s = min_stopped_s
        self.max_speed_kmh = max_speed_kmh
        self.exclude_queue_zone = exclude_queue_zone
        self.max_queue_wait_s = max_queue_wait_s
        self.moving_speed_kmh = moving_speed_kmh
        self._emit = _Reemit(repeat_s)

    def update(self, tracked: TrackedFrame, ctx: TrackContext) -> list[AnomalySignal]:
        t = tracked.timestamp.timestamp()
        ids = [int(i) for i in tracked.tracks.track_id]
        trajs = [tr for i in ids if (tr := ctx.store.get(i)) is not None and tr.points]
        moving = sum(1 for tr in trajs if tr.speed_kmh is not None and tr.speed_kmh >= self.moving_speed_kmh)
        out = []
        for tr in trajs:
            if not _is_vehicle(tr) or tr.speed_kmh is None or tr.speed_kmh > self.max_speed_kmh:
                continue
            dur = tr.stopped_duration_s
            anchor = tr.last.anchor_px
            in_queue = bool(ctx.geometry.in_queue_zone(np.array([anchor], np.float32))[0]) \
                if ctx.geometry.queue_zones else False
            needed = self.max_queue_wait_s if (self.exclude_queue_zone and in_queue) else self.min_stopped_s
            if dur < needed or not self._emit.due(tr.track_id, t):
                continue
            reliability = float(np.mean([p.reliable for p in tr.points])) if tr.points else 0.0
            duration_term = min(1.0, 0.5 + 0.5 * (dur - needed) / max(needed, 1.0))
            context_term = 1.0 if moving > 0 else 0.7        # everyone stopped -> more likely congestion
            conf = duration_term * (0.5 + 0.5 * reliability) * context_term
            out.append(_signal(ctx, tracked, AnomalyType.STOPPED_VEHICLE, self.name, conf, tr, {
                "stopped_duration_s": round(dur, 1), "confidence": round(conf, 3), "in_queue_zone": in_queue,
                "threshold_s": needed, "other_vehicles_moving": moving,
            }))
        self._emit.forget_missing(set(ids) | set(ctx.store.tracks))
        return out


@TRACK_ANOMALY_DETECTORS.register("wrong_way")
class WrongWayDetector(TrackAnomalyDetector):
    name = "wrong_way"

    def __init__(self, min_distance_m: float = 8.0, cos_threshold: float = -0.5, min_speed_kmh: float = 5.0,
                 window_s: float = 4.0, repeat_s: float = 10.0, min_step_fraction: float = 0.7,
                 max_step_speed_kmh: float = 150.0) -> None:
        self.min_distance_m = min_distance_m
        self.cos_threshold = cos_threshold
        self.min_speed_kmh = min_speed_kmh
        self.window_s = window_s
        self.min_step_fraction = min_step_fraction
        self.max_step_speed_kmh = max_step_speed_kmh
        self._emit = _Reemit(repeat_s)

    def update(self, tracked: TrackedFrame, ctx: TrackContext) -> list[AnomalySignal]:
        t = tracked.timestamp.timestamp()
        out = []
        for i in tracked.tracks.track_id:
            tr = ctx.store.get(int(i))
            if tr is None or not _is_vehicle(tr) or tr.speed_kmh is None or tr.speed_kmh < self.min_speed_kmh:
                continue
            d = tr.direction
            if d is None:
                continue
            pts = [p for p in tr.points if p.ground_m is not None and p.reliable and p.t >= t - self.window_s]
            if len(pts) < 3:
                continue
            expected = _expected_direction(ctx, tr.last.anchor_px)
            cos = float(np.dot(d, expected))
            g0, g1 = pts[0].ground_m, pts[-1].ground_m
            assert g0 is not None and g1 is not None  # filtered above
            along = float(np.dot(np.subtract(g1, g0), expected))
            if cos > self.cos_threshold or along > -self.min_distance_m:
                continue
            seg, plausible = _steps(pts, self.max_step_speed_kmh)
            moving = np.linalg.norm(seg, axis=1) > 0.05
            against = (seg @ np.asarray(expected, dtype=np.float64)) < 0
            if not plausible or not moving.any() or against[moving].mean() < self.min_step_fraction:
                continue                                      # identity switch / jitter, not a vehicle driving back
            if not self._emit.due(tr.track_id, t):
                continue
            conf = min(1.0, (-along) / (2 * self.min_distance_m)) * min(1.0, (-cos - 0.3) / 0.6)
            out.append(_signal(ctx, tracked, AnomalyType.WRONG_WAY, self.name, conf, tr, {
                "heading_cosine_vs_expected": round(cos, 3), "distance_against_m": round(-along, 1),
                "speed_kmh": round(tr.speed_kmh, 1), "expected_direction": [round(float(v), 3) for v in expected],
                "confidence": round(conf, 3),
            }))
        return out


@TRACK_ANOMALY_DETECTORS.register("trajectory_rules")
class TrajectoryRulesDetector(TrackAnomalyDetector):
    name = "trajectory_rules"

    def __init__(self, min_speed_kmh: float = 15.0, max_heading_rate_deg_s: float = 30.0, off_lane_s: float = 2.0,
                 window_s: float = 2.0, repeat_s: float = 10.0, max_step_speed_kmh: float = 150.0) -> None:
        self.max_step_speed_kmh = max_step_speed_kmh
        self.min_speed_kmh = min_speed_kmh
        self.max_heading_rate = max_heading_rate_deg_s
        self.off_lane_s = off_lane_s
        self.window_s = window_s
        self._emit = _Reemit(repeat_s)
        self._off_since: dict[int, float] = {}

    def update(self, tracked: TrackedFrame, ctx: TrackContext) -> list[AnomalySignal]:
        t = tracked.timestamp.timestamp()
        out = []
        has_lanes = bool(ctx.geometry.lanes)
        for i in tracked.tracks.track_id:
            tr = ctx.store.get(int(i))
            if tr is None or not _is_vehicle(tr) or tr.speed_kmh is None or tr.speed_kmh < self.min_speed_kmh:
                self._off_since.pop(int(i), None)
                continue
            reasons: dict[str, float] = {}
            pts = [p for p in tr.points if p.ground_m is not None and p.reliable and p.t >= t - self.window_s]
            if len(pts) >= 5 and _steps(pts, self.max_step_speed_kmh)[1]:
                g = np.array([p.ground_m for p in pts])
                ts = np.array([p.t for p in pts])
                seg = np.diff(g, axis=0)
                keep = np.linalg.norm(seg, axis=1) > 0.3            # ignore jitter-sized steps
                if keep.sum() >= 3:
                    ang = np.unwrap(np.arctan2(seg[keep, 0], seg[keep, 1]))
                    span = ts[1:][keep][-1] - ts[1:][keep][0]
                    if span > 0:
                        rate = math.degrees(abs(ang[-1] - ang[0])) / span
                        if rate > self.max_heading_rate:
                            reasons["heading_rate_deg_s"] = round(rate, 1)
            if has_lanes:
                anchor = np.array([tr.last.anchor_px], np.float32)
                lane = int(ctx.geometry.lane_of(anchor)[0])
                # off-lane = on the monitored road area (ROI) but in no lane zone, e.g. on the shoulder;
                # vehicles that have left the ROI (past the exit line) are not "off lane".
                if lane < 0 and bool(ctx.geometry.in_roi(anchor)[0]):
                    since = self._off_since.setdefault(tr.track_id, t)
                    if t - since >= self.off_lane_s:
                        reasons["off_lane_s"] = round(t - since, 1)
                else:
                    self._off_since.pop(tr.track_id, None)
            if not reasons or not self._emit.due(tr.track_id, t):
                continue
            score = min(1.0, 0.4 + 0.3 * len(reasons) + 0.3 * min(1.0, reasons.get("heading_rate_deg_s", 0.0)
                                                                       / (3 * self.max_heading_rate)))
            out.append(_signal(ctx, tracked, AnomalyType.ABNORMAL_TRAJECTORY, self.name, score, tr,
                               {**reasons, "speed_kmh": round(tr.speed_kmh, 1)}))
        return out
