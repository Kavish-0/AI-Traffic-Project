"""Synthetic incident scenarios, run end-to-end through the REAL pipeline from tracking onwards.

    kinematic simulation (IDM car-following per lane, events injected)
        -> vehicle boxes projected through a known camera homography (perfect tracker)
        -> PerceptionPipeline.process_detections: ground projection, trajectories, feature engine,
           track anomaly detectors (stopped_vehicle, wrong_way, trajectory_rules), 60 s windows,
           congestion classifier -> TrafficState
        -> IntelligenceEngine: series detectors, incident fusion, risk

Only the detector and the tracker are replaced (by ground truth), so the scenarios test the
incident logic, not YOLO. The simulated road is a 2-lane mid-block approach (no signal, no
queue zone) seen by a camera covering 0-60 m upstream of the exit line.

    normal             steady traffic
    sudden_slowdown    at the event all traffic slows to ~10 km/h (e.g. hazard ahead / heavy rain)
    stopped_vehicle    one vehicle stops in lane 2 (breakdown); following lane-2 traffic changes lane
    wrong_way          one vehicle drives against the configured lane direction
    queue_formation    demand rises above a downstream bottleneck's capacity: a queue grows from the
                       exit line while traffic keeps discharging (no stopped vehicle)
    multiple_signals   a vehicle stops in lane 2 near the exit and blocks it (no lane change); lane 1
                       slows (rubbernecking) -> stopped vehicle + queue growth + slowdown together

``python -m traffic_intel.incidents scenarios`` prints the timeline of each.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np
from pydantic import BaseModel

from traffic_intel.core.config import (
    CalibrationConfig,
    CameraConfig,
    NetworkConfig,
    PipelineConfig,
    ZoneConfig,
    get_settings,
    load_network,
    load_pipeline_config,
)
from traffic_intel.core.enums import IncidentStatus, ObjectClass, ZoneType, class_id_of
from traffic_intel.core.schemas import AnomalySignal, TrafficFeatures, TrafficState
from traffic_intel.core.types import Detections, Frame, Tracks
from traffic_intel.incidents.engine import IntelligenceEngine, IntelligenceUpdate
from traffic_intel.synthetic.scene import DEFAULT_CLASSES, _Agent, _bbox, ground_to_image_matrix, project
from traffic_intel.tracking.interfaces import Tracker


class ScenarioSpec(BaseModel):
    name: str
    description: str
    expected_type: str                          # incident type the engine should report after the event
    seed: int = 11
    duration_s: float = 900.0
    event_at_s: float = 600.0                   # 10 normal windows first (series baselines)
    fps: float = 5.0
    lanes: int = 2
    demand_veh_h_lane: float = 450.0
    demand_after_veh_h_lane: float | None = None
    speed_cap_kmh: float | None = None          # all lanes after the event
    lane_speed_cap_kmh: dict[int, float] = {}   # lane -> cap after the event
    block_lane: int | None = None               # a vehicle stops in this lane ...
    block_along_m: float = 35.0                 # ... at this distance upstream of the exit line
    divert_blocked_lane: bool = True            # following traffic changes lane (else it queues)
    bottleneck_headway_s: float | None = None   # after the event: min headway per lane at the exit line
    wrong_way: bool = False


SCENARIOS: dict[str, ScenarioSpec] = {s.name: s for s in [
    ScenarioSpec(name="normal", description="steady free-flowing traffic", expected_type="normal"),
    ScenarioSpec(name="sudden_slowdown", description="all traffic slows from ~45 to ~10 km/h",
                 expected_type="sudden_slowdown", speed_cap_kmh=10.0),
    ScenarioSpec(name="stopped_vehicle", description="a vehicle breaks down in lane 2, others pass in lane 1",
                 expected_type="stopped_vehicle", block_lane=1, block_along_m=35.0, demand_veh_h_lane=200.0),
    ScenarioSpec(name="wrong_way", description="a vehicle drives against the lane direction",
                 expected_type="wrong_way", wrong_way=True, demand_veh_h_lane=250.0),
    ScenarioSpec(name="queue_formation", description="demand exceeds a downstream bottleneck; queue grows",
                 expected_type="abnormal_queue", demand_veh_h_lane=450.0, demand_after_veh_h_lane=950.0,
                 bottleneck_headway_s=5.0),
    ScenarioSpec(name="multiple_signals", description="lane-blocking stopped vehicle + queue + slowdown",
                 expected_type="possible_incident", block_lane=1, block_along_m=4.0, divert_blocked_lane=False,
                 demand_veh_h_lane=600.0, lane_speed_cap_kmh={0: 20.0}),
]}

LANE_WIDTH_M = 3.5
FRAME_SIZE = (1280, 720)
_CLASS_MIX = {"car": 0.7, "motorcycle": 0.15, "truck": 0.1, "bus": 0.05}


# --------------------------------------------------------------------------- #
def scenario_camera(approach_id: str = "junction_a_north", lanes: int = 2) -> CameraConfig:
    """Calibrated camera over a mid-block approach; lane directions configured as (0, -1)."""
    H = ground_to_image_matrix(FRAME_SIZE)
    W = lanes * LANE_WIDTH_M

    def poly(x0: float, x1: float, a0: float, a1: float) -> list[tuple[float, float]]:
        return [(round(float(x), 1), round(float(y), 1))
                for x, y in project(H, np.array([[x0, a1], [x1, a1], [x1, a0], [x0, a0]]))]

    def line(a: float) -> list[tuple[float, float]]:
        return [(round(float(x), 1), round(float(y), 1)) for x, y in project(H, np.array([[-0.5, a], [W + 0.5, a]]))]

    ground = np.array([[0, 0], [7, 0], [0, 30], [7, 30], [0, 60], [7, 60]], dtype=np.float64)
    img = project(H, ground)
    return CameraConfig(
        id="cam_scenario", name="Synthetic incident scenario", approach_id=approach_id,
        source="synthetic://scenario", source_type="synthetic", process_fps=5.0, location=(77.5946, 12.9719),
        calibration=CalibrationConfig(
            image_points_px=[(float(x), float(y)) for x, y in img],
            ground_points_m=[(float(x), float(y)) for x, y in ground],
            origin_lonlat=(77.5946, 12.9721), bearing_deg=0.0, image_size=FRAME_SIZE),
        zones=[
            ZoneConfig(id="roi", type=ZoneType.ROI, points_px=poly(-3.0, W + 3.0, -3.0, 62.0)),
            *[ZoneConfig(id=f"lane_{i + 1}", type=ZoneType.LANE, lane_index=i + 1, direction=(0.0, -1.0),
                         points_px=poly(i * LANE_WIDTH_M, (i + 1) * LANE_WIDTH_M, -3.0, 62.0)) for i in range(lanes)],
            ZoneConfig(id="entry", type=ZoneType.ENTRY_LINE, points_px=line(50.0)),
            ZoneConfig(id="exit", type=ZoneType.EXIT_LINE, points_px=line(0.0)),
            ZoneConfig(id="loop", type=ZoneType.VIRTUAL_LOOP, points_px=poly(0.0, W, 3.0, 8.0)),
        ],
    )


@dataclass
class _Veh(_Agent):
    target_x: float | None = None
    blocker: bool = False
    frozen: bool = False
    ghost: bool = False


class TrafficSim:
    """Minimal IDM car-following with the scenario's events (deterministic for a seed)."""

    SPAWN_M, DESPAWN_M = 110.0, -12.0

    def __init__(self, spec: ScenarioSpec) -> None:
        self.s = spec
        self.rng = np.random.default_rng(spec.seed)
        self.agents: list[_Veh] = []
        self.next_id = 1
        self.t = -60.0                                   # warm-up
        self.next_arrival = [self._headway(self.t) + self.t for _ in range(spec.lanes)]
        self.pending: list[list[str]] = [[] for _ in range(spec.lanes)]
        self.last_release = [-1e9] * spec.lanes
        self.blocker_id: int | None = None
        self.ghost_spawned = False

    def _demand(self, t: float) -> float:
        s = self.s
        return s.demand_after_veh_h_lane if (s.demand_after_veh_h_lane and t >= s.event_at_s) else s.demand_veh_h_lane

    def _headway(self, t: float) -> float:
        return float(self.rng.exponential(3600.0 / self._demand(t)))

    def _event(self) -> bool:
        return self.t >= self.s.event_at_s

    def _v0(self, a: _Veh) -> float:
        v0 = a.v0
        if self._event():
            if self.s.speed_cap_kmh is not None:
                v0 = min(v0, self.s.speed_cap_kmh / 3.6 * (0.9 + 0.2 * ((a.id * 7919) % 100) / 100))
            cap = self.s.lane_speed_cap_kmh.get(a.lane)
            if cap is not None:
                v0 = min(v0, cap / 3.6)
        return v0

    def _spawn(self, lane: int, name: str) -> bool:
        lane_agents = [a for a in self.agents if a.lane == lane]
        last = max(lane_agents, key=lambda a: a.along, default=None)
        spec = DEFAULT_CLASSES[name]
        if last is not None and self.SPAWN_M - (last.along + last.length) < spec.min_gap_m + 8.0:
            return False
        v0 = max(10.0, self.rng.normal(spec.desired_speed_kmh, spec.speed_std_kmh)) / 3.6
        v = v0 if last is None else min(v0, last.v)
        x = LANE_WIDTH_M * (lane + 0.5) + float(self.rng.normal(0, 0.2))
        self.agents.append(_Veh(self.next_id, ObjectClass(name), lane, x, self.SPAWN_M, v, v0, spec.length_m,
                                spec.width_m, spec.height_m, spec.min_gap_m, spec.headway_s))
        self.next_id += 1
        return True

    def step(self, h: float) -> None:
        s, t = self.s, self.t
        names, shares = list(_CLASS_MIX), np.array(list(_CLASS_MIX.values()))
        # the blocking vehicle is chosen ~20 s before the event so that it stops at about event_at_s;
        # following traffic reacts (changes lane / arrivals use the other lane) once it starts braking
        choose_blocker = s.block_lane is not None and t >= s.event_at_s - 20.0
        blocker = next((a for a in self.agents if a.id == self.blocker_id), None)
        braking = blocker is not None and (blocker.frozen or blocker.along < s.block_along_m + 40.0)
        blocked = s.block_lane if braking else None
        if blocked is not None and s.divert_blocked_lane and self.pending[blocked]:
            other = next(i for i in range(s.lanes) if i != blocked)
            self.pending[other].extend(self.pending[blocked])
            self.pending[blocked] = []
        for lane in range(s.lanes):
            while self.next_arrival[lane] <= t:
                target = lane
                if blocked == lane and s.divert_blocked_lane:
                    target = next(i for i in range(s.lanes) if i != lane)
                self.pending[target].append(names[int(self.rng.choice(len(names), p=shares))])
                self.next_arrival[lane] += self._headway(t)
            if self.pending[lane] and self._spawn(lane, self.pending[lane][0]):
                self.pending[lane].pop(0)

        if choose_blocker and self.blocker_id is None:              # designate the vehicle that will stop
            cands = [a for a in self.agents if a.lane == s.block_lane and a.along > s.block_along_m + 25
                     and not a.ghost and a.cls is ObjectClass.CAR]
            if cands:
                b = min(cands, key=lambda a: a.along)
                b.blocker, self.blocker_id = True, b.id
        if s.wrong_way and self._event() and not self.ghost_spawned:
            spec = DEFAULT_CLASSES["car"]
            self.agents.append(_Veh(self.next_id, ObjectClass.CAR, -2, LANE_WIDTH_M * (s.lanes - 0.5), -8.0,
                                    30 / 3.6, 30 / 3.6, spec.length_m, spec.width_m, spec.height_m,
                                    direction=-1.0, ghost=True))
            self.next_id += 1
            self.ghost_spawned = True

        # lane changes away from a blocked lane (gradual lateral move)
        if blocked is not None and s.divert_blocked_lane:
            other = next(i for i in range(s.lanes) if i != blocked)
            for a in self.agents:
                if a.lane == blocked and not a.blocker and s.block_along_m < a.along < s.block_along_m + 80:
                    tgt = [b for b in self.agents if b.lane == other]
                    if all(abs(b.along - a.along) > 8.0 for b in tgt):
                        a.lane, a.target_x = other, LANE_WIDTH_M * (other + 0.5)

        acc: dict[int, float] = {}
        for lane in range(s.lanes):
            lane_agents = sorted((a for a in self.agents if a.lane == lane), key=lambda a: a.along)
            for i, a in enumerate(lane_agents):
                if a.frozen:
                    acc[a.id] = 0.0
                    continue
                gaps: list[tuple[float, float]] = []
                if i > 0:
                    lead = lane_agents[i - 1]
                    gaps.append((a.along - (lead.along + lead.length), a.v - lead.v))
                if a.blocker and a.along > s.block_along_m:
                    gaps.append((a.along - s.block_along_m, a.v))
                if (s.bottleneck_headway_s and self._event() and a.along > 0.3 and i == 0
                        and t - self.last_release[lane] < s.bottleneck_headway_s):
                    gaps.append((a.along - 0.5, a.v))
                v0 = self._v0(a)
                accel = 1.5 * (1 - (a.v / v0) ** 4)
                for gap, dv in gaps:
                    s_star = a.s0 + max(0.0, a.v * a.T + a.v * dv / (2 * math.sqrt(1.5 * 2.0)))
                    accel = min(accel, 1.5 * (1 - (a.v / v0) ** 4 - (s_star / max(gap, 0.1)) ** 2))
                acc[a.id] = max(accel, -9.0)
        for a in self.agents:
            if a.ghost:
                a.along -= a.direction * a.v * h
                continue
            if a.frozen:
                continue
            prev = a.along
            a.v = max(0.0, a.v + acc.get(a.id, 0.0) * h)
            a.along -= a.v * h
            if prev > 0.0 >= a.along and 0 <= a.lane < s.lanes:
                self.last_release[a.lane] = t
            if a.blocker and a.v < 0.05 and a.along - s.block_along_m < a.s0 + 1.5:   # IDM stops s0 short
                a.frozen, a.v = True, 0.0
            if a.target_x is not None:
                d = a.target_x - a.x
                a.x += max(-1.0 * h, min(1.0 * h, d))
                if abs(d) < 0.01:
                    a.target_x = None
        self.agents = [a for a in self.agents if self.DESPAWN_M <= a.along <= self.SPAWN_M * 1.05]
        self.t += h

    def tracks(self, H: np.ndarray) -> Tracks:
        ids, boxes, classes = [], [], []
        for a in self.agents:
            bb = _bbox(H, a, FRAME_SIZE)
            if bb is not None:
                ids.append(a.id)
                boxes.append(bb)
                classes.append(class_id_of(a.cls))
        n = len(ids)
        return Tracks(track_id=np.array(ids, dtype=np.int64), xyxy=np.array(boxes, dtype=np.float32).reshape(-1, 4),
                      confidence=np.full(n, 0.9, dtype=np.float32), class_id=np.array(classes, dtype=np.int64))


class _ScriptedTracker(Tracker):
    """Returns the ground-truth tracks set by the scenario runner (perfect tracking)."""

    def __init__(self) -> None:
        self.next: Tracks = Tracks()

    def update(self, detections: Detections, frame: Frame) -> Tracks:
        return self.next

    def reset(self) -> None:
        self.next = Tracks()


class _NoDetector:
    def detect(self, frames: list[Frame]) -> list[Detections]:
        return [Detections() for _ in frames]

    def warmup(self) -> None:
        pass


# --------------------------------------------------------------------------- #
@dataclass
class ScenarioResult:
    spec: ScenarioSpec
    updates: list[IntelligenceUpdate] = field(default_factory=list)
    signals: list[AnomalySignal] = field(default_factory=list)       # track-level signals from the camera
    windows: list[tuple[TrafficFeatures, TrafficState]] = field(default_factory=list)   # what the camera published

    @property
    def track_signals(self) -> int:
        return len(self.signals)

    @property
    def event_time(self) -> datetime:
        return self.updates[0].timestamp - timedelta(seconds=60) + timedelta(seconds=self.spec.event_at_s) \
            if self.updates else datetime.min

    def before(self) -> list[IntelligenceUpdate]:
        return [u for u in self.updates if u.timestamp <= self.event_time]

    def after(self) -> list[IntelligenceUpdate]:
        return [u for u in self.updates if u.timestamp > self.event_time]

    def types_after(self) -> list[str]:
        return [u.incident.incident_type for u in self.after() if u.incident is not None]

    def max_probability(self, updates: list[IntelligenceUpdate] | None = None) -> float:
        ups = self.updates if updates is None else updates
        return max((u.incident.probability for u in ups if u.incident is not None), default=0.0)

    def final_type(self) -> str:
        types = self.types_after()
        return types[-1] if types else "normal"

    def peak_risk(self, updates: list[IntelligenceUpdate] | None = None) -> IntelligenceUpdate | None:
        ups = [(u, u.risk) for u in (self.updates if updates is None else updates) if u.risk is not None]
        order = ["low", "medium", "high", "critical"]
        best = max(ups, key=lambda ur: (order.index(ur[1].level.value), ur[1].risk_score), default=None)
        return best[0] if best else None


def run_scenario(spec: ScenarioSpec | str, *, network: NetworkConfig | None = None,
                 pipeline: PipelineConfig | None = None, engine: IntelligenceEngine | None = None,
                 store: Any | None = None, start: datetime | None = None) -> ScenarioResult:
    from traffic_intel.pipeline.perception import PerceptionPipeline

    spec = SCENARIOS[spec] if isinstance(spec, str) else spec
    pipeline = pipeline or load_pipeline_config()
    network = network or load_network(get_settings().config_dir / pipeline.network_file)
    camera = scenario_camera(lanes=spec.lanes)
    if store is not None:          # scenarios are independent: close what a previous run left open
        for inc in store.list_incidents(approach_id=camera.approach_id, active_only=True):
            store.set_status(inc["incident_id"], IncidentStatus.RESOLVED, note="closed: new synthetic scenario run")
    tracker = _ScriptedTracker()
    pipe = PerceptionPipeline.build(camera, pipeline, network, FRAME_SIZE, spec.fps,
                                    detector=_NoDetector(), tracker=tracker)  # type: ignore[arg-type]
    engine = engine or IntelligenceEngine.build(network, pipeline, store=store, with_forecast=False)
    sim = TrafficSim(spec)
    H = ground_to_image_matrix(FRAME_SIZE)
    dt, sub = 1.0 / spec.fps, 2
    while sim.t < 0:
        sim.step(dt / sub)
    start = start or datetime.fromisoformat("2026-09-01T08:00:00+05:30")
    image = np.zeros((FRAME_SIZE[1], FRAME_SIZE[0], 3), dtype=np.uint8)
    result = ScenarioResult(spec)
    for k in range(int(round(spec.duration_s * spec.fps))):
        tracker.next = sim.tracks(H)
        frame = Frame(camera_id=camera.id, index=k, timestamp=start + timedelta(seconds=k * dt), image=image)
        res = pipe.process_detections(frame, Detections())
        if res.signals:
            result.signals.extend(res.signals)
            engine.ingest_signals(res.signals)
        for f, st in zip(res.window_features, res.window_states, strict=True):
            result.windows.append((f, st))
            result.updates.append(engine.process_window(f, st))
        for _ in range(sub):
            sim.step(dt / sub)
    return result
