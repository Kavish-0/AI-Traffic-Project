"""Synthetic traffic scene: ground truth + noisy detections, no video or GPU.

A single signalised approach (default 2 lanes) simulated with the Intelligent
Driver Model (Treiber et al. 2000) per lane, a fixed-time signal at the stop
line, Poisson arrivals and a heterogeneous class mix. Vehicle footprints are
projected through a known camera homography into pixel bounding boxes
(including an approximate vehicle height), then degraded like a real
detector: box jitter, missed detections, car<->truck class flips and
occasional low-confidence frames.

Ground frame (matches configs/cameras/cam_01.yaml): x = across the road
(0 = left road edge), y = metres *upstream* of the stop line; traffic moves
towards -y. The anchor of a box (bottom-centre) corresponds to the vehicle's
front bumper, the part nearest to the camera.

Everything is deterministic for a given ``SceneConfig`` (seeded).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import cv2
import numpy as np
from pydantic import BaseModel, Field

from traffic_intel.core.enums import VEHICLE_CLASS_IDS, ObjectClass, class_id_of
from traffic_intel.core.types import Detections


class ClassSpec(BaseModel):
    share: float
    length_m: float
    width_m: float
    height_m: float
    desired_speed_kmh: float
    speed_std_kmh: float = 5.0
    min_gap_m: float = 2.0      # IDM s0
    headway_s: float = 1.2      # IDM T


DEFAULT_CLASSES: dict[str, ClassSpec] = {
    "car": ClassSpec(share=0.6, length_m=4.5, width_m=1.8, height_m=1.5, desired_speed_kmh=45),
    "motorcycle": ClassSpec(share=0.22, length_m=2.0, width_m=0.8, height_m=1.4, desired_speed_kmh=42,
                            min_gap_m=1.0, headway_s=0.8),
    "bus": ClassSpec(share=0.06, length_m=12.0, width_m=2.5, height_m=3.2, desired_speed_kmh=38,
                     min_gap_m=2.5, headway_s=1.5),
    "truck": ClassSpec(share=0.12, length_m=9.0, width_m=2.5, height_m=3.0, desired_speed_kmh=38,
                       min_gap_m=2.5, headway_s=1.5),
}


class SignalConfig(BaseModel):
    cycle_s: float = 60.0
    green_s: float = 27.0
    amber_s: float = 3.0
    offset_s: float = 0.0       # time within the cycle at t = 0


class SceneConfig(BaseModel):
    name: str = "default"
    seed: int = 7
    start: datetime = datetime.fromisoformat("2026-09-01T08:00:00+05:30")
    duration_s: float = Field(180.0, gt=0)
    warmup_s: float = Field(60.0, ge=0)
    fps: float = Field(10.0, gt=0)
    frame_size: tuple[int, int] = (1280, 720)
    lanes: int = Field(2, ge=1)
    lane_width_m: float = 3.5
    spawn_along_m: float = 110.0
    despawn_along_m: float = -12.0
    arrival_rate_veh_h_per_lane: float = 500.0
    pedestrian_rate_h: float = 60.0
    signal: SignalConfig | None = SignalConfig()
    classes: dict[str, ClassSpec] = DEFAULT_CLASSES
    # IDM
    max_accel: float = 1.5
    comfort_decel: float = 2.0
    # detector noise
    jitter_px: float = 1.5
    miss_prob: float = 0.04
    class_flip_prob: float = 0.03
    low_conf_prob: float = 0.08
    calibration_noise_px: float = 1.0
    substeps: int = 4


SCENARIOS: dict[str, SceneConfig] = {
    "default": SceneConfig(),
    "free_flow": SceneConfig(name="free_flow", arrival_rate_veh_h_per_lane=250.0, signal=None),
    "congested": SceneConfig(
        name="congested", arrival_rate_veh_h_per_lane=950.0,
        signal=SignalConfig(cycle_s=90.0, green_s=22.0, amber_s=3.0),
    ),
    "small": SceneConfig(name="small", duration_s=20.0, warmup_s=30.0),
}

# Camera: ground rectangle x 0..7 m, y 0..60 m seen as a trapezoid (same geometry as
# cam_01.yaml, scaled from 1920x1080 to 1280x720).
_GROUND_CORNERS = np.array([[0, 60], [7, 60], [7, 0], [0, 0]], dtype=np.float32)
_IMAGE_CORNERS_1080 = np.array([[820, 420], [1100, 420], [1500, 1000], [400, 1000]], dtype=np.float32)


def image_corners(frame_size: tuple[int, int]) -> np.ndarray:
    return np.asarray(_IMAGE_CORNERS_1080 * np.array([frame_size[0] / 1920, frame_size[1] / 1080]), dtype=np.float32)


def ground_to_image_matrix(frame_size: tuple[int, int]) -> np.ndarray:
    return np.asarray(cv2.getPerspectiveTransform(_GROUND_CORNERS, image_corners(frame_size)), dtype=np.float64)


def project(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    p = np.hstack([np.asarray(pts, np.float64).reshape(-1, 2), np.ones((len(pts), 1))]) @ H.T
    return np.asarray(p[:, :2] / p[:, 2:3])


@dataclass
class _Agent:
    id: int
    cls: ObjectClass
    lane: int                   # -1 = pedestrian on the sidewalk
    x: float                    # lateral centre
    along: float                # front position (metres upstream of stop line)
    v: float                    # m/s (towards -y); pedestrians may walk either way
    v0: float
    length: float
    width: float
    height: float
    s0: float = 2.0
    T: float = 1.2
    direction: float = 1.0      # +1 = travelling towards -y (downstream)


@dataclass
class ObjectTruth:
    id: int
    cls: ObjectClass
    x: float
    along: float
    speed_kmh: float
    length: float
    bbox: tuple[float, float, float, float] | None     # None = not visible in the frame


@dataclass
class SyntheticScene:
    cfg: SceneConfig
    timestamps: list[datetime]
    truth: list[list[ObjectTruth]]
    detections: list[Detections]
    H_ground_to_image: np.ndarray
    calibration_image_points: list[tuple[float, float]]
    calibration_ground_points: list[tuple[float, float]]
    signal_red: list[bool] = field(default_factory=list)

    @property
    def n_frames(self) -> int:
        return len(self.timestamps)


def _signal_state(sig: SignalConfig | None, t: float) -> str:
    if sig is None:
        return "green"
    phase = (t + sig.offset_s) % sig.cycle_s
    if phase < sig.green_s:
        return "green"
    if phase < sig.green_s + sig.amber_s:
        return "amber"
    return "red"


def _bbox(H: np.ndarray, a: _Agent, frame_size: tuple[int, int]) -> tuple[float, float, float, float] | None:
    """Footprint rectangle projected to the image, extended upwards by the
    object's height (scaled with the local lateral pixels-per-metre)."""
    x0, x1 = a.x - a.width / 2, a.x + a.width / 2
    front, rear = a.along, a.along + a.length
    if a.direction < 0:                       # pedestrian walking upstream: front is the far end
        front, rear = a.along - a.length, a.along
    fp = project(H, np.array([[x0, front], [x1, front], [x0, rear], [x1, rear]]))
    if not np.all(np.isfinite(fp)):
        return None
    ppm_near = float(np.linalg.norm(fp[1] - fp[0])) / a.width
    ppm_far = float(np.linalg.norm(fp[3] - fp[2])) / a.width
    top = min(fp[0, 1] - a.height * ppm_near, fp[2, 1] - a.height * ppm_far)
    bx1, bx2 = float(fp[:, 0].min()), float(fp[:, 0].max())
    by1, by2 = float(top), float(fp[:, 1].max())
    w, h = frame_size
    if bx2 < 0 or bx1 > w or by2 < 0 or by1 > h:
        return None
    cx1, cy1, cx2, cy2 = max(bx1, 0.0), max(by1, 0.0), min(bx2, w - 1.0), min(by2, h - 1.0)
    visible = (cx2 - cx1) * (cy2 - cy1) / max((bx2 - bx1) * (by2 - by1), 1e-6)
    if visible < 0.4 or (cx2 - cx1) < 3 or (cy2 - cy1) < 3:
        return None
    return (cx1, cy1, cx2, cy2)


def generate_scene(cfg: SceneConfig | str = "default") -> SyntheticScene:
    if isinstance(cfg, str):
        cfg = SCENARIOS[cfg]
    rng = np.random.default_rng(cfg.seed)
    H = ground_to_image_matrix(cfg.frame_size)
    dt = 1.0 / cfg.fps
    sub = dt / cfg.substeps
    names = list(cfg.classes)
    shares = np.array([cfg.classes[n].share for n in names])
    shares = shares / shares.sum()

    agents: list[_Agent] = []
    next_id = 1
    next_arrival = [float(rng.exponential(3600.0 / cfg.arrival_rate_veh_h_per_lane)) - cfg.warmup_s
                    for _ in range(cfg.lanes)]
    pending: list[list[str]] = [[] for _ in range(cfg.lanes)]   # classes waiting to enter
    next_ped = (float(rng.exponential(3600.0 / cfg.pedestrian_rate_h)) - cfg.warmup_s
                if cfg.pedestrian_rate_h > 0 else math.inf)

    def spawn_vehicle(lane: int, name: str) -> bool:
        nonlocal next_id
        lane_agents = [a for a in agents if a.lane == lane]
        last = max(lane_agents, key=lambda a: a.along, default=None)
        spec = cfg.classes[name]
        if last is not None and cfg.spawn_along_m - (last.along + last.length) < spec.min_gap_m + 8.0:
            return False
        v0 = max(10.0, rng.normal(spec.desired_speed_kmh, spec.speed_std_kmh)) / 3.6
        v = v0 if last is None else min(v0, last.v)
        lateral = cfg.lane_width_m * (lane + 0.5) + float(rng.normal(0, 0.25 if name != "motorcycle" else 0.6))
        agents.append(_Agent(next_id, ObjectClass(name), lane, lateral, cfg.spawn_along_m, v, v0,
                             spec.length_m, spec.width_m, spec.height_m, spec.min_gap_m, spec.headway_s))
        next_id += 1
        return True

    def step(t: float, h: float) -> None:
        nonlocal next_ped, next_id
        for lane in range(cfg.lanes):
            while next_arrival[lane] <= t:
                pending[lane].append(names[int(rng.choice(len(names), p=shares))])
                next_arrival[lane] += float(rng.exponential(3600.0 / cfg.arrival_rate_veh_h_per_lane))
            if pending[lane] and spawn_vehicle(lane, pending[lane][0]):
                pending[lane].pop(0)
        while next_ped <= t:
            d = 1.0 if rng.random() < 0.5 else -1.0
            start = cfg.spawn_along_m * 0.7 if d > 0 else cfg.despawn_along_m
            agents.append(_Agent(next_id, ObjectClass.PEDESTRIAN, -1, -1.5 + float(rng.normal(0, 0.2)), start,
                                 1.3, 1.3, 0.5, 0.5, 1.7, direction=d))
            next_id += 1
            next_ped += float(rng.exponential(3600.0 / cfg.pedestrian_rate_h))

        sig = _signal_state(cfg.signal, t)
        acc: dict[int, float] = {}
        for lane in range(cfg.lanes):
            lane_agents = sorted((a for a in agents if a.lane == lane), key=lambda a: a.along)
            for i, a in enumerate(lane_agents):
                gaps: list[tuple[float, float]] = []            # (gap, closing speed)
                if i > 0:
                    lead = lane_agents[i - 1]
                    gaps.append((a.along - (lead.along + lead.length), a.v - lead.v))
                if sig != "green" and a.along > 0.3:
                    can_stop = a.along - 1.0 >= a.v ** 2 / (2 * cfg.comfort_decel * 1.6)
                    if sig == "red" or can_stop:
                        gaps.append((a.along - 1.0, a.v))
                a_free = cfg.max_accel * (1 - (a.v / a.v0) ** 4)
                accel = a_free
                for s, dv in gaps:
                    s_star = a.s0 + max(0.0, a.v * a.T + a.v * dv / (2 * math.sqrt(cfg.max_accel * cfg.comfort_decel)))
                    accel = min(accel, cfg.max_accel * (1 - (a.v / a.v0) ** 4 - (s_star / max(s, 0.1)) ** 2))
                acc[a.id] = max(accel, -9.0)
        for a in agents:
            if a.lane >= 0:
                a.v = max(0.0, a.v + acc[a.id] * h)
            a.along -= a.direction * a.v * h
        agents[:] = [a for a in agents if cfg.despawn_along_m <= a.along <= cfg.spawn_along_m * 1.05]

    # warm-up (not recorded)
    t = -cfg.warmup_s
    while t < 0:
        for _ in range(cfg.substeps):
            step(t, sub)
            t += sub

    timestamps, truth, dets, red = [], [], [], []
    n = int(round(cfg.duration_s * cfg.fps))
    flip = {class_id_of(ObjectClass.CAR): class_id_of(ObjectClass.TRUCK),
            class_id_of(ObjectClass.TRUCK): class_id_of(ObjectClass.CAR)}
    for k in range(n):
        t = k * dt
        frame_truth = []
        boxes, confs, clss = [], [], []
        for a in agents:
            bb = _bbox(H, a, cfg.frame_size)
            frame_truth.append(ObjectTruth(a.id, a.cls, a.x, a.along, a.v * 3.6, a.length, bb))
            if bb is None or rng.random() < cfg.miss_prob:
                continue
            jit = rng.normal(0, cfg.jitter_px, 4)
            b = np.clip(np.array(bb) + jit, 0, [cfg.frame_size[0] - 1, cfg.frame_size[1] - 1] * 2)
            if b[2] - b[0] < 2 or b[3] - b[1] < 2:
                continue
            cid = class_id_of(a.cls)
            if cid in flip and rng.random() < cfg.class_flip_prob:
                cid = flip[cid]
            conf = rng.uniform(0.2, 0.5) if rng.random() < cfg.low_conf_prob else rng.uniform(0.6, 0.95)
            boxes.append(b)
            confs.append(conf)
            clss.append(cid)
        timestamps.append(cfg.start + timedelta(seconds=t))
        truth.append(frame_truth)
        red.append(_signal_state(cfg.signal, t) == "red")
        dets.append(Detections(
            np.array(boxes, dtype=np.float32).reshape(-1, 4),
            np.array(confs, dtype=np.float32),
            np.array(clss, dtype=np.int64),
        ))
        for _ in range(cfg.substeps):
            step(t, sub)
            t += sub

    # Six calibration points (lane edges at 0/30/60 m), with picking noise.
    ground_pts = np.array([[0, 0], [7, 0], [0, 30], [7, 30], [0, 60], [7, 60]], dtype=np.float64)
    img_pts = project(H, ground_pts) + rng.normal(0, cfg.calibration_noise_px, (6, 2))
    return SyntheticScene(
        cfg=cfg, timestamps=timestamps, truth=truth, detections=dets, H_ground_to_image=H,
        calibration_image_points=[(round(float(x), 1), round(float(y), 1)) for x, y in img_pts],
        calibration_ground_points=[(float(x), float(y)) for x, y in ground_pts],
        signal_red=red,
    )


# --------------------------------------------------------------------------- #
# Ground truth aggregates (same definitions as the feature engine)
# --------------------------------------------------------------------------- #
@dataclass
class WindowTruth:
    vehicles_present: int
    mean_speed_kmh: float | None
    outflow: int
    inflow: int
    mean_queue_m: float
    mean_stopped: float
    mean_vehicles_in_region: float


def truth_for_window(
    scene: SyntheticScene,
    start: datetime,
    end: datetime,
    *,
    region_along: tuple[float, float] = (0.0, 60.0),
    count_region_along: tuple[float, float] | None = None,
    entry_along: float = 50.0,
    min_frames: int = 3,
    queue_speed_kmh: float = 7.0,
    queue_visible_only: bool = True,
    queue_head_max_m: float = 15.0,
    queue_max_gap_m: float = 15.0,
    stopped_speed_kmh: float = 5.0,
) -> WindowTruth:
    """``region_along`` = metric measurement region (speeds, queue visibility);
    ``count_region_along`` = ROI used for presence counts (defaults to region_along)."""
    count_region = count_region_along or region_along
    frames = [i for i, ts in enumerate(scene.timestamps) if start <= ts < end]
    presence: dict[int, int] = {}
    speeds: list[float] = []
    queues: list[float] = []
    stopped: list[int] = []
    in_region: list[int] = []
    prev_along: dict[int, float] = {}
    outflow = inflow = 0
    if frames and frames[0] > 0:
        prev_along = {o.id: o.along for o in scene.truth[frames[0] - 1]}
    for i in frames:
        objs = [o for o in scene.truth[i] if class_id_of(o.cls) in VEHICLE_CLASS_IDS]
        region = [o for o in objs if region_along[0] <= o.along <= region_along[1] and o.bbox is not None]
        for o in objs:
            if o.bbox is not None and count_region[0] <= o.along <= count_region[1]:
                presence[o.id] = presence.get(o.id, 0) + 1
        speeds.extend(o.speed_kmh for o in region)
        in_region.append(len(region))
        stopped.append(sum(o.speed_kmh < stopped_speed_kmh for o in region))
        for o in objs:
            p = prev_along.get(o.id)
            if p is not None and p > 0 >= o.along:
                outflow += 1
            if p is not None and p > entry_along >= o.along:
                inflow += 1
        prev_along = {o.id: o.along for o in scene.truth[i]}
        # queue per lane
        best = 0.0
        lanes = {int(o.x // scene.cfg.lane_width_m) for o in objs}
        for lane in lanes:
            q = sorted((o for o in objs if int(o.x // scene.cfg.lane_width_m) == lane
                        and o.speed_kmh < queue_speed_kmh and o.along > -2.0
                        and (not queue_visible_only or o.along <= region_along[1])), key=lambda o: o.along)
            if not q or q[0].along > queue_head_max_m:
                continue
            tail = q[0]
            for a, b in zip(q, q[1:], strict=False):
                if b.along - (a.along + a.length) > queue_max_gap_m:
                    break
                tail = b
            tail_m = tail.along + tail.length
            best = max(best, min(tail_m, region_along[1]) if queue_visible_only else tail_m)
        queues.append(best)
    return WindowTruth(
        vehicles_present=sum(1 for n in presence.values() if n >= min_frames),
        mean_speed_kmh=float(np.mean(speeds)) if speeds else None,
        outflow=outflow,
        inflow=inflow,
        mean_queue_m=float(np.mean(queues)) if queues else 0.0,
        mean_stopped=float(np.mean(stopped)) if stopped else 0.0,
        mean_vehicles_in_region=float(np.mean(in_region)) if in_region else 0.0,
    )
