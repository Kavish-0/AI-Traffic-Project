"""Speed calculation and trajectory kinematics (distance / time on ground metres)."""

from __future__ import annotations

import numpy as np
import pytest

from traffic_intel.core.enums import ObjectClass, class_id_of
from traffic_intel.core.types import Tracks
from traffic_intel.tracking.trajectory import TrajectoryConfig, TrajectoryStore

FPS = 10.0


def feed(store: TrajectoryStore, positions: list[tuple[float, float]], *, tid: int = 1, t0: float = 1000.0,
         reliable: bool = True, cls: ObjectClass = ObjectClass.CAR, start_k: int = 0) -> None:
    for k, (x, y) in enumerate(positions, start=start_k):
        tracks = Tracks(
            track_id=np.array([tid]), xyxy=np.array([[0, 0, 10, 10]], dtype=np.float32),
            confidence=np.array([0.9], dtype=np.float32), class_id=np.array([class_id_of(cls)]),
        )
        store.update(t0 + k / FPS, tracks, np.array([[5, 10]], np.float32),
                     np.array([[x, y]], np.float32), np.array([reliable]))


def straight(speed_kmh: float, seconds: float, jitter_m: float = 0.0, seed: int = 0) -> list[tuple[float, float]]:
    rng = np.random.default_rng(seed)
    v = speed_kmh / 3.6
    n = int(seconds * FPS)
    return [(2.0 + rng.normal(0, jitter_m), 60.0 - v * k / FPS + rng.normal(0, jitter_m)) for k in range(n)]


@pytest.mark.parametrize("speed", [10.0, 36.0, 72.0])
def test_constant_speed_is_recovered(speed: float) -> None:
    store = TrajectoryStore()
    feed(store, straight(speed, 3.0))
    tr = store.get(1)
    assert tr is not None and tr.speed_kmh == pytest.approx(speed, rel=0.01)


def test_speed_robust_to_position_jitter() -> None:
    store = TrajectoryStore()
    feed(store, straight(36.0, 5.0, jitter_m=0.15))
    speeds = store.get(1).speed_samples[-20:]  # type: ignore[union-attr]
    assert np.mean(speeds) == pytest.approx(36.0, abs=2.0)


def test_distance_direction_and_heading() -> None:
    store = TrajectoryStore()
    feed(store, straight(36.0, 4.0))
    tr = store.get(1)
    assert tr is not None
    # 10 m/s for 3.9 s measured from first valid speed (after 0.8 s warm-up)
    assert tr.distance_m == pytest.approx(10.0 * (3.9 - 0.8), rel=0.05)
    np.testing.assert_allclose(tr.direction, [0, -1], atol=1e-3)
    assert tr.heading_deg(bearing_deg=0.0) == pytest.approx(180.0, abs=0.5)    # moving south


def test_young_track_has_no_speed() -> None:
    store = TrajectoryStore(TrajectoryConfig(speed_min_track_s=0.8))
    feed(store, straight(36.0, 0.5))
    assert store.get(1).speed_kmh is None  # type: ignore[union-attr]


def test_stopped_vehicle_duration_and_low_speed() -> None:
    store = TrajectoryStore()
    feed(store, straight(36.0, 2.0))
    last = straight(36.0, 2.0)[-1]
    feed(store, [last] * 60, start_k=20)          # stands still for 6 s
    tr = store.get(1)
    assert tr is not None and tr.speed_kmh is not None
    assert tr.speed_kmh < 1.0
    assert tr.stopped_duration_s == pytest.approx(6.0, abs=1.0)   # <=1 s detection lag
    assert tr.max_stopped_s >= tr.stopped_duration_s


def test_stopped_with_jitter_stays_below_threshold() -> None:
    """Box jitter on a stationary vehicle must not look like movement."""
    rng = np.random.default_rng(1)
    store = TrajectoryStore()
    feed(store, [(2 + rng.normal(0, 0.1), 20 + rng.normal(0, 0.1)) for _ in range(80)])
    assert store.get(1).speed_kmh < 3.0  # type: ignore[operator, union-attr]


def test_implausible_speed_rejected() -> None:
    store = TrajectoryStore(TrajectoryConfig(max_speed_kmh=160))
    feed(store, straight(400.0, 2.0))
    assert store.get(1).speed_kmh is None  # type: ignore[union-attr]


def test_unreliable_points_give_no_speed() -> None:
    store = TrajectoryStore()
    feed(store, straight(36.0, 3.0), reliable=False)
    assert store.get(1).speed_kmh is None  # type: ignore[union-attr]


def test_class_majority_vote() -> None:
    store = TrajectoryStore()
    feed(store, straight(30, 1.0), cls=ObjectClass.CAR)
    feed(store, straight(30, 0.2), cls=ObjectClass.TRUCK, start_k=10)
    assert store.get(1).object_class == "car"  # type: ignore[union-attr]


def test_lifecycle_expire_and_summary() -> None:
    store = TrajectoryStore(TrajectoryConfig(max_age_s=1.0))
    feed(store, straight(36.0, 2.0))
    assert store.expire(1000.0 + 2.5) == []
    done = store.expire(1000.0 + 3.5)
    assert [t.track_id for t in done] == [1] and store.get(1) is None
    s = done[0].to_summary("cam", "appr", np.array([0, -1], np.float32), set(), set())
    assert s.mean_speed_kmh == pytest.approx(36.0, rel=0.02)
    assert s.displacement_along_travel_m == pytest.approx(19.0, abs=1.0)
    assert len(s.path_ground_m) == len(s.path_image_px) >= 3


def test_tracked_object_view_has_trajectory() -> None:
    from datetime import UTC, datetime

    store = TrajectoryStore()
    feed(store, straight(36.0, 2.0))
    (obj,) = store.tracked_objects([1], datetime.now(UTC), tail=5)
    assert obj.class_name == "car" and len(obj.trajectory) == 5 and obj.center == (5.0, 5.0)
