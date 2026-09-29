"""Default traffic feature engine (``feature_engine.type: default``).

Per frame:   tracks -> ground projection -> trajectories -> line crossings
             -> FrameSample (objects in ROI with speed/stopped/queue/lane/position)
Per window:  samples + crossings + finished trajectories -> calculators -> TrafficFeatures

Windows are aligned to multiples of ``window_s`` in *source* time (UTC epoch),
so replayed and live footage produce identical window boundaries. Windows in
which no frames arrived are still emitted (quality 0) - gaps are explicit.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any

import numpy as np

from traffic_intel.computer_vision.zones import CameraGeometry
from traffic_intel.core.config import CameraConfig
from traffic_intel.core.enums import IS_VEHICLE, ZoneType
from traffic_intel.core.schemas import TrackSummary, TrafficFeatures
from traffic_intel.core.types import TrackedFrame
from traffic_intel.tracking.trajectory import Trajectory, TrajectoryPoint, TrajectoryStore
from traffic_intel.traffic.features.calculators import DEFAULT_CALCULATORS, FEATURE_CALCULATORS
from traffic_intel.traffic.features.samples import (
    CrossingEvent,
    FeatureEngineConfig,
    FrameSample,
    GeometryFacts,
    WindowData,
)
from traffic_intel.traffic.interfaces import FEATURE_ENGINES, TrafficFeatureEngine

_FIELDS = set(TrafficFeatures.model_fields)
_IS_VEHICLE = np.array(IS_VEHICLE, dtype=bool)


def _dt(t: float) -> datetime:
    return datetime.fromtimestamp(t, UTC)



def _is_stopped(tr: Trajectory, stopped_speed_kmh: float, min_duration_s: float) -> bool:
    return tr.speed_kmh is not None and tr.speed_kmh < stopped_speed_kmh and tr.stopped_duration_s >= min_duration_s

class DefaultFeatureEngine(TrafficFeatureEngine):
    def __init__(self, camera: CameraConfig, geometry: CameraGeometry, fps: float, **params: Any) -> None:
        self.camera = camera
        self.geometry = geometry
        self.fps = fps
        self.cfg = FeatureEngineConfig(**params)
        self.store = TrajectoryStore(self.cfg.trajectory)
        self.calculators = [FEATURE_CALCULATORS.get(n)() for n in (self.cfg.calculators or DEFAULT_CALCULATORS)]
        self.facts = GeometryFacts(
            roi_length_m=geometry.roi_length_m,
            lanes=geometry.lane_count,
            has_loop=geometry.has_loop,
            has_entry_line=geometry.has_line(ZoneType.ENTRY_LINE),
            has_exit_line=geometry.has_line(ZoneType.EXIT_LINE),
            anchor_reference=(
                geometry.anchor_reference if self.cfg.anchor_reference == "auto" else self.cfg.anchor_reference
            ),
        )
        self._pcu = self.cfg.pcu_table()
        self._lengths = self.cfg.length_table()
        self._entry_ids = {ln.id for ln in geometry.lines if ln.kind is ZoneType.ENTRY_LINE}
        self._exit_ids = {ln.id for ln in geometry.lines if ln.kind is ZoneType.EXIT_LINE}
        self._samples: list[FrameSample] = []
        self._events: list[CrossingEvent] = []
        self._finished: list[TrackSummary] = []        # retained for window stats
        self._finished_out: list[TrackSummary] = []    # not yet handed out
        self._window_start: float | None = None
        self._last_t: float | None = None

    # ------------------------------------------------------------------ #
    def update(self, tracked: TrackedFrame) -> None:
        t = tracked.timestamp.timestamp()
        if self._window_start is None:
            self._window_start = math.floor(t / self.cfg.window_s) * self.cfg.window_s
        self._last_t = t
        geo = self.geometry
        tracks = tracked.tracks
        n = len(tracks)
        anchors = tracks.anchor_points()
        ground = tracks.ground_xy if tracks.ground_xy is not None else geo.projector.image_to_ground(anchors)
        tracks.ground_xy = ground

        h, w = tracked.image_shape
        m = self.cfg.edge_margin_px
        box = tracks.xyxy
        truncated = (box[:, 0] <= m) | (box[:, 2] >= w - m) | (box[:, 3] >= h - m)
        finite = np.all(np.isfinite(ground), axis=1) if n else np.zeros(0, bool)
        reliable = ~truncated & finite & geo.projector.in_calibrated_region(anchors)

        prev = {tid: self.store.get(int(tid)) for tid in tracks.track_id.tolist()}
        prev_pts = {tid: tr.last for tid, tr in prev.items() if tr is not None and tr.points}
        trajs = self.store.update(t, tracks, anchors, ground, reliable)
        self._detect_crossings(t, tracks.track_id, anchors, ground, trajs, prev_pts)

        in_roi = geo.in_roi(anchors)
        in_measure = in_roi & geo.in_measurement_region(anchors) & finite
        idx = np.flatnonzero(in_roi)
        for i in idx:
            trajs[i].roi_points += 1

        speed = np.array([np.nan if trajs[i].speed_kmh is None else trajs[i].speed_kmh for i in idx],
                         dtype=np.float32)
        stop_kmh = self.cfg.trajectory.stopped_speed_kmh
        stopped = np.array([_is_stopped(trajs[i], stop_kmh, self.cfg.stopped_min_s) for i in idx], dtype=np.bool_)
        queued = np.isfinite(speed) & (speed < self.cfg.queue_speed_kmh)
        g = ground[idx] if n else np.zeros((0, 2), np.float32)
        class_ids = np.array([trajs[i].class_id for i in idx], dtype=np.int64)
        vehicle_boxes = box[idx][_IS_VEHICLE[class_ids]]

        self._samples.append(FrameSample(
            t=t,
            track_id=tracks.track_id[idx],
            class_id=class_ids,
            xyxy=box[idx],
            in_measure=in_measure[idx],
            speed_kmh=speed,
            stopped=stopped,
            queued=queued,
            in_queue_zone=geo.in_queue_zone(anchors[idx]),
            lane=geo.lane_of(anchors[idx]),
            along_m=geo.distance_upstream(g),
            lateral_m=geo.lateral_offset(g),
            loop_occupied=geo.loop_occupied(vehicle_boxes),
        ))

        for tr in self.store.expire(t):
            self._finish(tr)

    def _detect_crossings(self, t: float, track_ids: np.ndarray, anchors: np.ndarray, ground: np.ndarray,
                          trajs: list[Trajectory], prev_pts: dict[int, TrajectoryPoint]) -> None:
        if not prev_pts or not self.geometry.lines:
            return
        rows = [i for i, tid in enumerate(track_ids.tolist()) if tid in prev_pts]
        if not rows:
            return
        p_prev = np.array([prev_pts[int(track_ids[i])].anchor_px for i in rows], dtype=np.float32)
        p_cur = anchors[rows]
        for line, mask in self.geometry.line_crossings(p_prev, p_cur):
            for k in np.flatnonzero(mask):
                i = rows[k]
                tr: Trajectory = trajs[i]
                forward = self._is_forward(tr, prev_pts[int(track_ids[i])].ground_m, ground[i])
                key = line.id if forward else f"{line.id}:reverse"
                if key in tr.crossed:
                    continue            # each track counts once per line and direction
                tr.crossed[key] = True
                self._events.append(CrossingEvent(t, tr.track_id, tr.class_id, line.id, line.kind, forward))

    def _is_forward(self, tr: Trajectory, g_prev: tuple[float, float] | None, g_cur: np.ndarray) -> bool:
        u = self.geometry.travel_direction
        if g_prev is not None and np.all(np.isfinite(g_cur)):
            step = np.asarray(g_cur, dtype=np.float64) - np.asarray(g_prev)
            if float(np.linalg.norm(step)) > 1e-3:
                return float(step @ u) > 0
        d = tr.direction
        return True if d is None else float(d @ u) > 0

    def _finish(self, tr: Trajectory) -> None:
        if tr.roi_points == 0:
            return
        s = tr.to_summary(self.camera.id, self.camera.approach_id, self.geometry.travel_direction,
                          self._entry_ids, self._exit_ids)
        self._finished.append(s)
        self._finished_out.append(s)

    # ------------------------------------------------------------------ #
    def _compute(self, start: float, end: float, *, live: bool) -> TrafficFeatures:
        samples = [s for s in self._samples if start <= s.t < end]
        events = [e for e in self._events if start <= e.t < end]
        finished = [f for f in self._finished if start <= f.last_seen.timestamp() < end]
        wd = WindowData(
            start=start, end=end, samples=samples, events=events, finished=finished,
            expected_frames=(end - start) * self.fps, geometry=self.facts, cfg=self.cfg,
            pcu=self._pcu, lengths=self._lengths,
        )
        fields: dict[str, Any] = {}
        extra: dict[str, float] = {}
        for calc in self.calculators:
            out = calc.compute(wd)
            wd.results.update(out)
            for k, v in out.items():
                if k in _FIELDS:
                    fields[k] = v
                elif isinstance(v, int | float) and math.isfinite(float(v)):
                    extra[k] = float(v)
        if "trajectory_stats" in fields:
            fields["trajectory_stats"] = {k: v for k, v in fields["trajectory_stats"].items() if math.isfinite(v)}
        extra["roi_length_m"] = round(self.facts.roi_length_m, 2)
        extra["lanes"] = float(self.facts.lanes)
        quality = min(1.0, len(samples) / wd.expected_frames) if wd.expected_frames > 0 else 0.0
        defaults: dict[str, Any] = dict(
            vehicle_count=0, count_by_class={}, pcu_count=0.0, avg_speed_kmh=None, speed_p85_kmh=None,
            density_veh_km=0.0, density_ratio=0.0, occupancy_pct=0.0, flow_veh_h=0.0, inflow=0, outflow=0,
            queue_length_m=0.0, stopped_vehicles=0, avg_gap_m=None, pedestrian_count=0,
        )
        return TrafficFeatures(
            approach_id=self.camera.approach_id,
            camera_id=self.camera.id,
            window_start=_dt(start),
            window_s=round(end - start, 3),
            quality=round(quality, 3),
            **{**defaults, **fields},
            extra=extra,
        )

    def live_features(self, now: datetime) -> TrafficFeatures:
        t = now.timestamp()
        return self._compute(t - self.cfg.live_window_s, t + 1e-6, live=True)

    def close_windows(self, until: datetime) -> list[TrafficFeatures]:
        if self._window_start is None:
            return []
        t = until.timestamp()
        out = []
        ws = self.cfg.window_s
        while t >= self._window_start + ws:
            out.append(self._compute(self._window_start, self._window_start + ws, live=False))
            self._window_start += ws
        if out:
            self._prune(t)
        return out

    def flush(self) -> list[TrafficFeatures]:
        """End of stream: finish all trajectories and emit the last (partial) window."""
        if self._window_start is None or self._last_t is None:
            return []
        for tr in self.store.flush():
            self._finish(tr)
        end = self._last_t + 1.0 / self.fps
        out = [self._compute(self._window_start, end, live=False)] if end > self._window_start else []
        self._window_start = None
        return out

    def _prune(self, now: float) -> None:
        assert self._window_start is not None
        keep_from = min(self._window_start, now - self.cfg.live_window_s)
        self._samples = [s for s in self._samples if s.t >= keep_from]
        self._events = [e for e in self._events if e.t >= keep_from]
        self._finished = [f for f in self._finished if f.last_seen.timestamp() >= keep_from]

    def finished_tracks(self) -> list[TrackSummary]:
        out, self._finished_out = self._finished_out, []
        return out


@FEATURE_ENGINES.register("default")
def _create(**kwargs: Any) -> DefaultFeatureEngine:
    return DefaultFeatureEngine(**kwargs)
