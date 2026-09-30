from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from traffic_intel.core.config import PROJECT_ROOT, CalibrationConfig, CameraConfig, ZoneConfig
from traffic_intel.core.enums import ObjectClass, ZoneType, class_id_of
from traffic_intel.core.types import Detections, TrackedFrame, Tracks

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
T0 = datetime(2026, 9, 1, 8, 0, 0, tzinfo=UTC)


@pytest.fixture
def configs_dir() -> Path:
    return PROJECT_ROOT / "configs"


# --------------------------------------------------------------------------- #
# A simple, exactly-known camera: top-down-like view at 10 px per metre.
#   ground (x across, y metres upstream of the stop line) -> image (100 + 10x, 700 - 10y)
# Traffic travels towards -y, i.e. down the image, so box bottoms are front bumpers.
# --------------------------------------------------------------------------- #
PX_PER_M = 10.0


def g2i(x: float, y: float) -> tuple[float, float]:
    return 100.0 + PX_PER_M * x, 700.0 - PX_PER_M * y


def rect(x0: float, x1: float, y0: float, y1: float) -> list[tuple[float, float]]:
    return [g2i(x0, y1), g2i(x1, y1), g2i(x1, y0), g2i(x0, y0)]


def simple_camera(**overrides: object) -> CameraConfig:
    zones = [
        ZoneConfig(id="roi", type=ZoneType.ROI, points_px=rect(-1, 8, -2, 62)),
        ZoneConfig(id="lane_1", type=ZoneType.LANE, lane_index=1, direction=(0, -1), points_px=rect(0, 3.5, -2, 62)),
        ZoneConfig(id="lane_2", type=ZoneType.LANE, lane_index=2, direction=(0, -1), points_px=rect(3.5, 7, -2, 62)),
        ZoneConfig(id="entry", type=ZoneType.ENTRY_LINE, points_px=[g2i(-1, 50), g2i(8, 50)]),
        ZoneConfig(id="stop", type=ZoneType.EXIT_LINE, points_px=[g2i(-1, 0), g2i(8, 0)]),
        ZoneConfig(id="queue", type=ZoneType.QUEUE_ZONE, points_px=rect(0, 7, -2, 62)),
        ZoneConfig(id="loop", type=ZoneType.VIRTUAL_LOOP, points_px=rect(0, 7, 3, 8)),
    ]
    cfg = dict(
        id="cam_test", name="test", approach_id="junction_a_north", source="unused.mp4",
        process_fps=10.0, location=(77.59, 12.97),
        calibration=CalibrationConfig(
            image_points_px=[g2i(0, 0), g2i(7, 0), g2i(0, 60), g2i(7, 60), g2i(0, 30), g2i(7, 30)],
            ground_points_m=[(0, 0), (7, 0), (0, 60), (7, 60), (0, 30), (7, 30)],
            origin_lonlat=(77.59, 12.97), bearing_deg=0.0, image_size=(400, 800),
        ),
        zones=zones,
    )
    cfg.update(overrides)
    return CameraConfig(**cfg)  # type: ignore[arg-type]


LENGTH = {ObjectClass.CAR: 4.5, ObjectClass.BUS: 12.0, ObjectClass.MOTORCYCLE: 2.0,
          ObjectClass.TRUCK: 9.0, ObjectClass.BICYCLE: 1.8, ObjectClass.PEDESTRIAN: 0.5}


def box_for(x: float, front_y: float, cls: ObjectClass = ObjectClass.CAR, width: float = 1.8) -> list[float]:
    """Pixel box of a vehicle whose front bumper centre is at ground (x, front_y)."""
    u1, v2 = g2i(x - width / 2, front_y)
    u2, _ = g2i(x + width / 2, front_y)
    return [u1, v2 - PX_PER_M * LENGTH[cls], u2, v2]


def tracked_frame(
    k: int, objects: list[tuple[int, ObjectClass, float, float]], fps: float = 10.0, t0: datetime = T0
) -> TrackedFrame:
    """objects: (track_id, class, x, front_y). Frame k at t0 + k/fps."""
    ts = t0 + timedelta(seconds=k / fps)
    if objects:
        tracks = Tracks(
            track_id=np.array([o[0] for o in objects], dtype=np.int64),
            xyxy=np.array([box_for(o[2], o[3], o[1]) for o in objects], dtype=np.float32),
            confidence=np.full(len(objects), 0.9, dtype=np.float32),
            class_id=np.array([class_id_of(o[1]) for o in objects], dtype=np.int64),
        )
    else:
        tracks = Tracks()
    return TrackedFrame("cam_test", k, ts, (800, 400), Detections(), tracks)
