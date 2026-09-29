"""Anomaly detectors: series baselines (slowdown, queue, density, flow, EWMA, z-score) and track rules
(stopped vehicle, wrong-way) on hand-built, exactly-known inputs."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd
import pytest
from conftest import T0, simple_camera, tracked_frame

from traffic_intel.anomaly.detectors.statistical import (
    DensityZScoreDetector,
    EWMAControlChart,
    FlowImbalanceDetector,
    QueueGrowthDetector,
    RollingZScore,
    SuddenSlowdownDetector,
)
from traffic_intel.anomaly.detectors.track_rules import StoppedVehicleDetector, WrongWayDetector
from traffic_intel.anomaly.interfaces import TrackContext
from traffic_intel.computer_vision.homography import HomographyProjector
from traffic_intel.computer_vision.zones import CameraGeometry
from traffic_intel.core.config import CameraConfig
from traffic_intel.core.enums import AnomalyType, ObjectClass, ZoneType
from traffic_intel.traffic.features.engine import DefaultFeatureEngine

CAR = ObjectClass.CAR


def history(**cols: list[float]) -> pd.DataFrame:
    n = len(next(iter(cols.values())))
    df = pd.DataFrame({"timestamp": [T0 + timedelta(minutes=i + 1) for i in range(n)], "quality": 1.0, **cols})
    return df


def score_of(det: Any, h: pd.DataFrame) -> float:
    sig = det.score("junction_a_north", h)
    return max((s.score for s in sig), default=0.0)


# --------------------------------------------------------------------------- sudden slowdown
def test_sudden_slowdown_example_from_the_specification() -> None:
    """45, 43, 42, 40, 18, 7 km/h -> abnormal slowdown (score rises as the drop deepens)."""
    det = SuddenSlowdownDetector()
    speeds = [45.0, 43.0, 42.0, 40.0, 18.0, 7.0]
    assert score_of(det, history(avg_speed_kmh=speeds[:4])) == 0.0          # not enough baseline yet
    s18 = score_of(det, history(avg_speed_kmh=speeds[:5]))
    s7 = score_of(det, history(avg_speed_kmh=speeds))
    assert s18 >= 0.5 and s7 >= 0.9 and s7 >= s18
    (sig,) = det.score("a", history(avg_speed_kmh=speeds))
    assert sig.type is AnomalyType.SUDDEN_SLOWDOWN
    assert sig.details["drop_pct"] > 75 and sig.details["recent_kmh"][-1] == 7.0


def test_ordinary_fluctuation_is_not_a_slowdown() -> None:
    rng = np.random.default_rng(0)
    speeds = list(42 + rng.normal(0, 2.5, 30))
    det = SuddenSlowdownDetector()
    assert max(score_of(det, history(avg_speed_kmh=speeds[: k + 1])) for k in range(5, 30)) == 0.0


def test_slowdown_needs_the_fast_vehicles_to_slow_too() -> None:
    """One stopped vehicle drags the MEAN speed down but not p85 -> not a stream slowdown."""
    det = SuddenSlowdownDetector()
    mean = [42, 41, 43, 42, 40, 18]
    assert score_of(det, history(avg_speed_kmh=mean)) > 0.5
    assert score_of(det, history(avg_speed_kmh=mean, speed_p85_kmh=[48, 47, 49, 48, 47, 46])) == 0.0
    assert score_of(det, history(avg_speed_kmh=mean, speed_p85_kmh=[48, 47, 49, 48, 47, 20])) > 0.5


# --------------------------------------------------------------------------- queue / density / flow
def test_queue_growth() -> None:
    det = QueueGrowthDetector()
    assert score_of(det, history(queue_length_m=[0, 0, 0, 0, 0, 12, 28, 45])) > 0.8
    assert score_of(det, history(queue_length_m=[20, 22, 19, 21, 20, 22, 21, 20])) == 0.0   # standing queue
    assert score_of(det, history(queue_length_m=[0, 0, 0, 0, 0, 0, 0, 8])) == 0.0            # small


def test_density_change_explained_by_stopped_vehicles_is_not_counted_twice() -> None:
    det = DensityZScoreDetector()
    base = [0.05, 0.06, 0.05, 0.04, 0.05, 0.06]
    jump = history(density_ratio=base + [0.30], stopped_vehicles=[0] * 6 + [0],
                   mean_vehicles_in_region=[1.0] * 6 + [6.0])
    assert score_of(det, jump) > 0.5
    explained = history(density_ratio=base + [0.15], stopped_vehicles=[0] * 6 + [2],
                        mean_vehicles_in_region=[1.0] * 6 + [3.0])       # 2 of 3 vehicles are the stopped ones
    assert score_of(det, explained) == 0.0


def test_flow_imbalance_vehicles_accumulating() -> None:
    det = FlowImbalanceDetector()
    ok = history(inflow=[10] * 8, outflow=[10] * 8)
    assert score_of(det, ok) == 0.0
    acc = history(inflow=[10] * 5 + [12, 12, 12], outflow=[10] * 5 + [3, 2, 2])
    (sig,) = det.score("a", acc)
    assert sig.type is AnomalyType.UNUSUAL_FLOW and sig.score > 0.5


@pytest.mark.parametrize("det", [EWMAControlChart(), RollingZScore()])
def test_multi_feature_outlier_detectors(det: Any) -> None:
    rng = np.random.default_rng(1)
    n = 20
    h = history(avg_speed_kmh=list(40 + rng.normal(0, 1.0, n)), density_ratio=list(0.1 + rng.normal(0, 0.01, n)),
                occupancy_pct=list(10 + rng.normal(0, 0.5, n)), queue_length_m=[0.0] * n)
    assert score_of(det, h) == 0.0
    h.loc[n - 1, "avg_speed_kmh"] = 12.0
    (sig,) = det.score("a", h)
    assert sig.type is AnomalyType.STATISTICAL_OUTLIER and sig.details["feature"] == "avg_speed_kmh"


# --------------------------------------------------------------------------- track rules
def _ctx(camera: CameraConfig) -> tuple[DefaultFeatureEngine, TrackContext]:
    size = camera.calibration.image_size  # type: ignore[union-attr]
    geo = CameraGeometry(camera, size, HomographyProjector(camera.calibration, size))  # type: ignore[arg-type]
    eng = DefaultFeatureEngine(camera=camera, geometry=geo, fps=5.0, window_s=60.0)
    return eng, TrackContext(camera_id=camera.id, approach_id=camera.approach_id, junction_id="junction_a",
                             road_id="road_a_sb", geometry=geo, store=eng.store)


def _no_queue_zone(**overrides: Any) -> CameraConfig:
    cam = simple_camera(**overrides)
    return cam.model_copy(update={"zones": [z for z in cam.zones if z.type is not ZoneType.QUEUE_ZONE]})


def _run(eng: DefaultFeatureEngine, ctx: TrackContext, det: Any, frames: list[list[tuple]]) -> list:
    out = []
    for k, objs in enumerate(frames):
        tf = tracked_frame(k, objs, fps=5.0)
        eng.update(tf)
        out.extend(det.update(tf, ctx))
    return out


def test_stopped_vehicle_reports_track_location_duration_and_confidence() -> None:
    """Car 1 stands in lane 2 at 30 m for 45 s while car-after-car passes in lane 1."""
    eng, ctx = _ctx(_no_queue_zone())
    frames = []
    for k in range(int(45 * 5)):
        t = k / 5.0
        objs = [(1, CAR, 5.25, 30.0)]
        passing_y = 58.0 - (t % 6.0) * 12.5              # a car every 6 s at 45 km/h in lane 1
        objs.append((100 + int(t // 6.0), CAR, 1.75, passing_y))
        frames.append(objs)
    det = StoppedVehicleDetector(min_stopped_s=30.0)
    sigs = [s for s in _run(eng, ctx, det, frames) if s.track_ids == [1]]
    assert sigs, "stopped vehicle not reported"
    s = sigs[0]
    assert s.type is AnomalyType.STOPPED_VEHICLE
    assert 30.0 <= s.details["stopped_duration_s"] <= 33.0
    assert s.details["road_id"] == "road_a_sb" and s.details["junction_id"] == "junction_a"
    loc = s.details["location"]
    assert {"lon", "lat"} <= set(loc) and loc["distance_upstream_of_stop_line_m"] == pytest.approx(30.0, abs=1.5)
    assert 0.0 < s.details["confidence"] <= 1.0 and s.score == pytest.approx(s.details["confidence"], abs=1e-3)
    assert not any(s.track_ids != [1] for s in _run(*_ctx(_no_queue_zone()), det, frames))   # movers never flagged


def test_waiting_in_the_queue_zone_is_normal_until_a_very_long_wait() -> None:
    eng, ctx = _ctx(simple_camera())                     # queue zone covers the approach (signal)
    frames = [[(1, CAR, 1.75, 3.0)] for _ in range(int(60 * 5))]
    assert _run(eng, ctx, StoppedVehicleDetector(min_stopped_s=30.0, max_queue_wait_s=180.0), frames) == []
    eng, ctx = _ctx(simple_camera())
    assert _run(eng, ctx, StoppedVehicleDetector(min_stopped_s=30.0, max_queue_wait_s=50.0), frames)


def _two_cars(frames_n: int = 40) -> list[list[tuple]]:
    """Car 1 drives with the lane direction (towards -y), car 2 against it (towards +y), 36 km/h."""
    return [[(1, CAR, 1.75, 50.0 - 2.0 * k), (2, CAR, 5.25, 5.0 + 2.0 * k)] for k in range(frames_n)]


def test_wrong_way_uses_the_configured_lane_direction() -> None:
    eng, ctx = _ctx(simple_camera())
    sigs = _run(eng, ctx, WrongWayDetector(), _two_cars())
    assert sigs and {s.track_ids[0] for s in sigs} == {2}
    assert sigs[0].type is AnomalyType.WRONG_WAY and sigs[0].details["location"]
    # Reverse the configured direction of every lane: now car 1 is the wrong-way one.
    cam = simple_camera()
    flipped = cam.model_copy(update={"zones": [
        z.model_copy(update={"direction": (0.0, 1.0)}) if z.type is ZoneType.LANE else z for z in cam.zones]})
    eng, ctx = _ctx(flipped)
    assert {s.track_ids[0] for s in _run(eng, ctx, WrongWayDetector(), _two_cars())} == {1}


def test_identity_switch_is_not_wrong_way() -> None:
    """A tracker ID switch makes a track jump ~12 m backwards in ONE frame, then drive on normally."""
    eng, ctx = _ctx(simple_camera())
    frames = []
    for k in range(40):
        y = 50.0 - 2.0 * k                     # 36 km/h with the lane direction
        if k >= 10:
            y += 12.0                          # identity switch at frame 10: now following a car 12 m behind
        frames.append([(1, CAR, 1.75, y)])
    assert _run(eng, ctx, WrongWayDetector(), frames) == []

