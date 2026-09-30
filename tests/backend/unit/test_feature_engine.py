"""Traffic feature engine: hand-computable micro scenarios + synthetic dataset vs ground truth."""

from __future__ import annotations

import functools
from datetime import timedelta
from typing import Any

import numpy as np
import pytest
from conftest import T0, simple_camera, tracked_frame

from traffic_intel.computer_vision.homography import HomographyProjector
from traffic_intel.computer_vision.zones import CameraGeometry
from traffic_intel.core.enums import ObjectClass
from traffic_intel.traffic.features.calculators import FEATURE_CALCULATORS, FeatureCalculator
from traffic_intel.traffic.features.engine import DefaultFeatureEngine
from traffic_intel.traffic.features.samples import WindowData

CAR, BUS, PED = ObjectClass.CAR, ObjectClass.BUS, ObjectClass.PEDESTRIAN


def engine(**params: Any) -> DefaultFeatureEngine:
    cam = simple_camera()
    size = cam.calibration.image_size  # type: ignore[union-attr]
    geo = CameraGeometry(cam, size, HomographyProjector(cam.calibration, size))  # type: ignore[arg-type]
    params.setdefault("window_s", 10.0)
    return DefaultFeatureEngine(camera=cam, geometry=geo, fps=10.0, **params)


def run(eng: DefaultFeatureEngine, frames: list[list[tuple[int, ObjectClass, float, float]]]) -> list:
    out = []
    for k, objs in enumerate(frames):
        tf = tracked_frame(k, objs)
        eng.update(tf)
        out.extend(eng.close_windows(tf.timestamp))
    return out


def test_single_vehicle_speed_flow_and_counts() -> None:
    """One car at 36 km/h (10 m/s) drives from y=58 to beyond the stop line in 7 s.
    Its trajectory is finished 3.5 s after it disappears (lost-track timeout)."""
    eng = engine(window_s=20.0)
    frames = [[(1, CAR, 1.75, 58.0 - 1.0 * k)] if k < 70 else [] for k in range(201)]
    (w,) = run(eng, frames)
    assert w.vehicle_count == 1 and w.count_by_class[CAR] == 1
    assert w.inflow == 1 and w.outflow == 1
    assert w.flow_veh_h == pytest.approx(180.0)            # 1 vehicle / 20 s
    assert w.avg_speed_kmh == pytest.approx(36.0, abs=0.5)
    assert w.median_speed_kmh == pytest.approx(36.0, abs=0.5)
    assert w.speed_distribution["30-40"] == 1
    assert w.stopped_vehicles == 0 and w.queue_length_m == 0.0
    assert w.quality == pytest.approx(1.0)
    assert w.trajectory_stats["n"] == 1 and w.trajectory_stats["completed_entry_to_exit"] == 1


def test_standing_queue_length_stopped_count_and_density() -> None:
    """Three cars standing at the stop line in lane 1 (front bumpers at 1, 7, 13 m)."""
    eng = engine(window_s=10.0)
    queue = [(1, CAR, 1.75, 1.0), (2, CAR, 1.75, 7.0), (3, CAR, 1.75, 13.0)]
    (w,) = run(eng, [queue] * 101)
    assert w.vehicle_count == 3
    # time-averaged: each car counts as stopped only after the 0.8 s speed warm-up + stopped_min_s (2 s)
    assert w.stopped_vehicles == 2
    assert w.extra["stopped_vehicles_max"] == 3 and w.extra["stopped_vehicles_unique"] == 3
    # queue tail = last front bumper + car length (anchor = front bumper): 13 + 4.5
    assert w.extra["queue_max_m"] == pytest.approx(17.5, abs=0.3)
    assert w.avg_speed_kmh == pytest.approx(0.0, abs=0.5)
    lane_km = w.extra["roi_length_m"] / 1000 * 2
    assert w.density_veh_km == pytest.approx(3 / lane_km, rel=0.01)
    assert w.density_ratio == pytest.approx(w.density_veh_km / 150, rel=0.01)
    assert w.occupancy_pct == pytest.approx(100.0)       # car at y=1..5.5 overlaps loop 3..8
    assert w.avg_gap_m == pytest.approx(1.5, abs=0.1)    # 6 m spacing - 4.5 m length
    assert w.flow_veh_h == 0.0


def test_queue_breaks_at_large_gap_and_needs_head_near_stop_line() -> None:
    eng = engine(window_s=10.0, queue_max_gap_m=10.0, queue_head_max_m=15.0)
    cars = [(1, CAR, 1.75, 1.0), (2, CAR, 1.75, 7.0), (3, CAR, 1.75, 30.0),     # 18.5 m gap -> not queued
            (4, CAR, 5.25, 20.0), (5, CAR, 5.25, 26.0)]                           # lane 2 starts too far
    (w,) = run(eng, [cars] * 101)
    assert w.extra["queue_max_m"] == pytest.approx(11.5, abs=0.3)                # 7 + 4.5


def test_pcu_weighting_and_pedestrians_excluded() -> None:
    eng = engine(window_s=10.0)
    objs = [(1, BUS, 1.75, 20.0), (2, PED, -0.5, 30.0)]
    (w,) = run(eng, [objs] * 101)
    assert w.vehicle_count == 1 and w.pedestrian_count == 1 and w.pcu_count == pytest.approx(3.0)
    lane_km = w.extra["roi_length_m"] / 1000 * 2
    assert w.density_veh_km == pytest.approx(3.0 / lane_km, rel=0.01)         # bus = 3 PCU


def test_wrong_way_movement_counted_as_reverse_not_inflow() -> None:
    eng = engine(window_s=10.0)
    frames = [[(1, CAR, 5.25, -1.0 + 1.0 * k)] for k in range(60)] + [[] for _ in range(41)]
    (w,) = run(eng, frames)
    assert w.inflow == 0 and w.outflow == 0 and w.extra["reverse_crossings"] == 2
    assert w.trajectory_stats["reverse_direction_tracks"] == 1


def test_windows_align_to_clock_and_gaps_are_explicit() -> None:
    eng = engine(window_s=10.0)
    eng.update(tracked_frame(0, []))
    later = tracked_frame(0, [], t0=T0 + timedelta(seconds=35))
    eng.update(later)
    out = eng.close_windows(later.timestamp)
    assert [w.window_start for w in out] == [T0 + timedelta(seconds=s) for s in (0, 10, 20)]
    assert out[1].quality == 0.0 and out[1].vehicle_count == 0 and out[1].avg_speed_kmh is None


def test_live_features_and_flush() -> None:
    eng = engine(window_s=60.0, live_window_s=2.0)
    frames = [[(1, CAR, 1.75, 50.0 - 1.0 * k)] for k in range(30)]
    run(eng, frames)
    live = eng.live_features(T0 + timedelta(seconds=2.9))
    assert live.window_s == pytest.approx(2.0, abs=0.01) and live.vehicle_count == 1
    assert live.avg_speed_kmh == pytest.approx(36.0, abs=1.0)
    (partial,) = eng.flush()
    # the partial window covers only the 3 s that were observed, completely
    assert partial.window_s == pytest.approx(3.0) and partial.quality == pytest.approx(1.0)
    assert len(eng.finished_tracks()) == 1


def test_custom_calculator_plugs_in() -> None:
    @FEATURE_CALCULATORS.register("test_heavy_share")
    class HeavyShare(FeatureCalculator):
        def compute(self, w: WindowData) -> dict[str, Any]:
            counts = w.results["count_by_class"]
            total = max(1, int(w.results["vehicle_count"]))  # type: ignore[call-overload]
            return {"heavy_share": (counts[BUS] + counts[ObjectClass.TRUCK]) / total}  # type: ignore[index]

    eng = engine(window_s=10.0, calculators=["counts", "test_heavy_share"])
    (w,) = run(eng, [[(1, BUS, 1.75, 20.0), (2, CAR, 5.25, 20.0)]] * 101)
    assert w.extra["heavy_share"] == pytest.approx(0.5)


def test_unknown_config_key_rejected() -> None:
    with pytest.raises(Exception, match="speed_smoothing"):
        engine(speed_smoothing="savgol")


# --------------------------------------------------------------------------- #
# Synthetic dataset (IDM microsimulation + noisy detections) vs ground truth
# --------------------------------------------------------------------------- #
@functools.cache
def _run_scenario(name: str) -> tuple[list, Any, Any]:
    from traffic_intel.core.config import get_settings, load_network, load_pipeline_config
    from traffic_intel.pipeline.perception import PerceptionPipeline
    from traffic_intel.synthetic.sources import SyntheticDetector, SyntheticVideoSource, synthetic_camera

    st = get_settings()
    pc = load_pipeline_config(st)
    net = load_network(st.config_dir / pc.network_file)
    src = SyntheticVideoSource(synthetic_camera(name), render=False)
    pipe = PerceptionPipeline.build(synthetic_camera(name), pc, net, src.frame_size, src.fps,
                                    detector=SyntheticDetector(name))
    states = []
    for f in src.frames():
        states.extend(pipe.process(f).window_states)
    states.extend(pipe.flush()[1])           # the last window closes at end of stream
    return states, pipe, src.scene


def _extent(pipe: Any, poly: np.ndarray) -> tuple[float, float]:
    g = pipe.projector.image_to_ground(poly)
    a = pipe.geometry.distance_upstream(g[np.all(np.isfinite(g), axis=1)])
    return float(a.min()), float(a.max())


@pytest.mark.parametrize("scenario", ["default", "free_flow"])
def test_synthetic_scenarios_match_ground_truth(scenario: str) -> None:
    from traffic_intel.synthetic.scene import truth_for_window

    states, pipe, scene = _run_scenario(scenario)
    assert len(states) == 3
    region, roi = _extent(pipe, pipe.geometry.measurement_region), _extent(pipe, pipe.geometry.roi)
    for s in states:
        tr = truth_for_window(scene, s.window_start, s.timestamp, region_along=region, count_region_along=roi)
        assert s.vehicle_count == pytest.approx(tr.vehicles_present, abs=1)
        assert s.outflow == pytest.approx(tr.outflow, abs=1)
        assert s.inflow == pytest.approx(tr.inflow, abs=2)
        assert s.average_speed_kmh == pytest.approx(tr.mean_speed_kmh, abs=1.5)
        assert s.queue_length_m == pytest.approx(tr.mean_queue_m, abs=3.0)
        assert s.quality == pytest.approx(1.0)


def test_synthetic_levels_order_free_to_severe() -> None:
    free, _, _ = _run_scenario("free_flow")
    jam, _, _ = _run_scenario("congested")
    assert {s.traffic_state.value for s in free} == {"free"}
    assert {s.traffic_state.value for s in jam} == {"severe"}
    assert max(s.congestion_score for s in free) < min(s.congestion_score for s in jam)
