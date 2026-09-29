"""Trajectory extraction: per-track history and kinematics in ground metres.

For every tracked object we keep a time-bounded history of anchor points
(bottom-centre of the box, in pixels and projected ground metres) and derive:

* velocity  - least-squares slope of ground position vs time over the last
              ``speed_window_s`` (a first-order Savitzky-Golay estimate;
              robust to per-frame box jitter), optionally EMA-smoothed.
* speed     - |velocity| in km/h; None while unreliable (too young, too few
              valid points, outside the calibrated area, truncated box, or an
              implausible value).
* direction - unit velocity vector (ground) and compass heading.
* distance  - integral of the smoothed speed over time (integrating raw
              point-to-point jumps would turn jitter into phantom metres).
* stopped   - continuous time with speed < ``stopped_speed_kmh``.

Class labels are a confidence-weighted vote over the history, so a vehicle
that flickers car/truck is still counted once, as its majority class.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from traffic_intel.core.enums import class_from_id
from traffic_intel.core.schemas import TrackSummary
from traffic_intel.core.types import BoolArray, FloatArray, Tracks


class TrajectoryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    history_s: float = Field(10.0, gt=0)          # kinematic history kept per track
    max_age_s: float = Field(3.5, gt=0)           # unseen this long -> trajectory finished
    speed_window_s: float = Field(1.0, gt=0)      # regression window for velocity
    # Near standstill, box jitter dominates and |v| is biased upwards; below
    # slow_speed_kmh the velocity is also estimated over the longer slow_window_s
    # and the lower of the two is used.
    slow_speed_kmh: float = Field(12.0, ge=0)
    slow_window_s: float = Field(3.0, gt=0)
    speed_min_span_s: float = Field(0.5, gt=0)    # min time covered by points in the window
    speed_min_points: int = Field(3, ge=2)
    speed_min_track_s: float = Field(0.8, ge=0)   # no speed from tracks younger than this
    speed_ema_alpha: float = Field(0.5, gt=0, le=1)  # 1 = no smoothing
    max_speed_kmh: float = Field(160.0, gt=0)     # above = measurement error -> None
    stopped_speed_kmh: float = Field(5.0, ge=0)
    summary_sample_hz: float = Field(2.0, gt=0)   # polyline sampling in TrackSummary
    summary_max_points: int = Field(600, ge=2)


@dataclass(slots=True)
class TrajectoryPoint:
    t: float                     # POSIX seconds (source time)
    anchor_px: tuple[float, float]
    ground_m: tuple[float, float] | None
    bbox: tuple[float, float, float, float]
    reliable: bool               # usable for metric kinematics


@dataclass
class Trajectory:
    track_id: int
    first_seen: float
    cfg: TrajectoryConfig
    points: deque[TrajectoryPoint] = field(default_factory=deque)
    class_votes: dict[int, float] = field(default_factory=dict)
    last_seen: float = 0.0
    confidence: float = 0.0
    velocity_ms: FloatArray | None = None     # ground m/s
    speed_kmh: float | None = None
    distance_m: float = 0.0
    stopped_since: float | None = None
    max_stopped_s: float = 0.0
    speed_samples: list[float] = field(default_factory=list)   # for mean speed
    crossed: dict[str, bool] = field(default_factory=dict)      # line id -> forward?
    roi_points: int = 0
    summary_path: list[tuple[float, tuple[float, float], tuple[float, float] | None]] = field(
        default_factory=list
    )
    _last_kin_t: float | None = None

    # ---------------------------------------------------------------- #
    @property
    def class_id(self) -> int:
        return max(self.class_votes.items(), key=lambda kv: kv[1])[0]

    @property
    def object_class(self) -> str:
        return class_from_id(self.class_id).value

    @property
    def age_s(self) -> float:
        return self.last_seen - self.first_seen

    @property
    def last(self) -> TrajectoryPoint:
        return self.points[-1]

    @property
    def stopped_duration_s(self) -> float:
        return 0.0 if self.stopped_since is None else self.last_seen - self.stopped_since

    @property
    def direction(self) -> FloatArray | None:
        if self.velocity_ms is None:
            return None
        n = float(np.linalg.norm(self.velocity_ms))
        return None if n < 1e-3 else (self.velocity_ms / n).astype(np.float32)

    def heading_deg(self, bearing_deg: float = 0.0) -> float | None:
        """Compass heading of motion; ``bearing_deg`` = compass direction of ground +y."""
        d = self.direction
        if d is None:
            return None
        return (bearing_deg + math.degrees(math.atan2(float(d[0]), float(d[1])))) % 360.0

    def mean_speed_kmh(self) -> float | None:
        return float(np.mean(self.speed_samples)) if self.speed_samples else None

    # ---------------------------------------------------------------- #
    def add(self, p: TrajectoryPoint, class_id: int, confidence: float) -> None:
        cfg = self.cfg
        self.points.append(p)
        self.last_seen = p.t
        self.confidence = confidence
        self.class_votes[class_id] = self.class_votes.get(class_id, 0.0) + confidence
        while self.points and self.points[0].t < p.t - cfg.history_s:
            self.points.popleft()
        if not self.summary_path or p.t - self.summary_path[-1][0] >= 1.0 / cfg.summary_sample_hz:
            self.summary_path.append((p.t, p.anchor_px, p.ground_m))
            if len(self.summary_path) > cfg.summary_max_points:
                self.summary_path = self.summary_path[::2]
        self._update_kinematics(p.t)

    def _regress(self, t: float, window_s: float) -> FloatArray | None:
        """Least-squares ground velocity (m/s) over reliable points in [t - window, t]."""
        cfg = self.cfg
        if self.age_s < cfg.speed_min_track_s:
            return None
        pts = [q for q in self.points if q.reliable and q.t >= t - window_s and q.ground_m is not None]
        if len(pts) < cfg.speed_min_points:
            return None
        ts = np.array([q.t for q in pts], dtype=np.float64)
        if ts[-1] - ts[0] < cfg.speed_min_span_s:
            return None
        xy = np.array([q.ground_m for q in pts], dtype=np.float64)
        tc = ts - ts.mean()
        slope = (tc @ (xy - xy.mean(axis=0))) / float(tc @ tc)
        return slope.astype(np.float32)

    def _update_kinematics(self, t: float) -> None:
        cfg = self.cfg
        v = self._regress(t, cfg.speed_window_s)
        if v is not None and float(np.linalg.norm(v)) * 3.6 < cfg.slow_speed_kmh:
            # Only accept the long window when it is *slower*: that removes jitter bias at
            # standstill without delaying the detection of a vehicle that has just stopped.
            v_slow = self._regress(t, cfg.slow_window_s)
            if v_slow is not None and float(np.linalg.norm(v_slow)) < float(np.linalg.norm(v)):
                v = v_slow
        if v is not None and float(np.linalg.norm(v)) * 3.6 > cfg.max_speed_kmh:
            v = None                                                  # implausible -> unreliable
        if v is None:
            # A speed older than one regression window is stale -> report unknown.
            if self._last_kin_t is None or t - self._last_kin_t > cfg.speed_window_s:
                self.speed_kmh = None
            return

        a = cfg.speed_ema_alpha
        stale = self._last_kin_t is None or t - self._last_kin_t > cfg.history_s
        self.velocity_ms = v if (self.velocity_ms is None or stale) else (
            a * v + (1 - a) * self.velocity_ms
        ).astype(np.float32)
        speed = float(np.linalg.norm(self.velocity_ms)) * 3.6
        if self._last_kin_t is not None:
            # Short unreliable gaps (occlusion, box truncation) are bridged with the new speed.
            self.distance_m += (speed / 3.6) * max(0.0, t - self._last_kin_t)
        self._last_kin_t = t
        self.speed_kmh = speed
        self.speed_samples.append(speed)

        if speed < cfg.stopped_speed_kmh:
            if self.stopped_since is None:
                self.stopped_since = t
            self.max_stopped_s = max(self.max_stopped_s, t - self.stopped_since)
        else:
            self.stopped_since = None

    def full_path(self) -> list[tuple[float, tuple[float, float], tuple[float, float] | None]]:
        """Sampled path plus the latest point (sampling may have skipped it)."""
        path = list(self.summary_path)
        if self.points and (not path or self.last.t > path[-1][0]):
            path.append((self.last.t, self.last.anchor_px, self.last.ground_m))
        return path

    def displacement_along(self, direction: FloatArray) -> float | None:
        g = [g for _, _, g in self.full_path() if g is not None]
        if len(g) < 2:
            return None
        return float((np.asarray(g[-1]) - np.asarray(g[0])) @ direction)

    def to_summary(
        self, camera_id: str, approach_id: str, travel_direction: FloatArray | None,
        entry_ids: set[str], exit_ids: set[str],
    ) -> TrackSummary:
        from datetime import UTC

        path = self.full_path()
        return TrackSummary(
            camera_id=camera_id,
            approach_id=approach_id,
            track_id=self.track_id,
            object_class=class_from_id(self.class_id),
            first_seen=datetime.fromtimestamp(self.first_seen, UTC),
            last_seen=datetime.fromtimestamp(self.last_seen, UTC),
            path_ground_m=[(round(g[0], 2), round(g[1], 2)) for _, _, g in path if g is not None],
            path_image_px=[(round(a[0], 1), round(a[1], 1)) for _, a, _ in path],
            mean_speed_kmh=self.mean_speed_kmh(),
            max_stopped_s=round(self.max_stopped_s, 2),
            crossed_entry=any(self.crossed.get(i) for i in entry_ids),
            crossed_exit=any(self.crossed.get(i) for i in exit_ids),
            distance_m=round(self.distance_m, 2),
            displacement_along_travel_m=(
                None if travel_direction is None else self.displacement_along(travel_direction)
            ),
        )


@dataclass(frozen=True, slots=True)
class TrackedObject:
    """Per-object view of the current frame, including its recent trajectory."""

    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]
    center: tuple[float, float]
    anchor: tuple[float, float]
    ground_xy: tuple[float, float] | None
    timestamp: datetime
    speed_kmh: float | None
    direction: tuple[float, float] | None
    distance_m: float
    stopped_s: float
    age_s: float
    trajectory: list[tuple[datetime, float, float]]    # (time, anchor x px, anchor y px)


class TrajectoryStore:
    def __init__(self, cfg: TrajectoryConfig | None = None) -> None:
        self.cfg = cfg or TrajectoryConfig()
        self.tracks: dict[int, Trajectory] = {}

    def update(
        self,
        t: float,
        tracks: Tracks,
        anchors: FloatArray,
        ground: FloatArray | None,
        reliable: BoolArray,
    ) -> list[Trajectory]:
        out = []
        for i in range(len(tracks)):
            tid = int(tracks.track_id[i])
            traj = self.tracks.get(tid)
            if traj is None:
                traj = self.tracks[tid] = Trajectory(tid, first_seen=t, cfg=self.cfg)
            g = None
            if ground is not None and np.all(np.isfinite(ground[i])):
                g = (float(ground[i, 0]), float(ground[i, 1]))
            traj.add(
                TrajectoryPoint(
                    t=t,
                    anchor_px=(float(anchors[i, 0]), float(anchors[i, 1])),
                    ground_m=g,
                    bbox=tuple(float(v) for v in tracks.xyxy[i]),  # type: ignore[arg-type]
                    reliable=bool(reliable[i]) and g is not None,
                ),
                int(tracks.class_id[i]),
                float(tracks.confidence[i]),
            )
            out.append(traj)
        return out

    def expire(self, t: float) -> list[Trajectory]:
        """Remove and return trajectories not seen for ``max_age_s``."""
        done = [tr for tr in self.tracks.values() if t - tr.last_seen > self.cfg.max_age_s]
        for tr in done:
            del self.tracks[tr.track_id]
        return done

    def flush(self) -> list[Trajectory]:
        done = list(self.tracks.values())
        self.tracks.clear()
        return done

    def get(self, track_id: int) -> Trajectory | None:
        return self.tracks.get(track_id)

    def tracked_objects(self, track_ids: list[int], timestamp: datetime, tail: int = 30) -> list[TrackedObject]:
        from datetime import UTC

        objs = []
        for tid in track_ids:
            tr = self.tracks.get(tid)
            if tr is None or not tr.points:
                continue
            p = tr.last
            x1, y1, x2, y2 = p.bbox
            d = tr.direction
            objs.append(TrackedObject(
                track_id=tid,
                class_id=tr.class_id,
                class_name=tr.object_class,
                confidence=tr.confidence,
                bbox=p.bbox,
                center=((x1 + x2) / 2, (y1 + y2) / 2),
                anchor=p.anchor_px,
                ground_xy=p.ground_m,
                timestamp=timestamp,
                speed_kmh=tr.speed_kmh,
                direction=None if d is None else (float(d[0]), float(d[1])),
                distance_m=tr.distance_m,
                stopped_s=tr.stopped_duration_s,
                age_s=tr.age_s,
                trajectory=[(datetime.fromtimestamp(q.t, UTC), q.anchor_px[0], q.anchor_px[1])
                            for q in list(tr.points)[-tail:]],
            ))
        return objs
