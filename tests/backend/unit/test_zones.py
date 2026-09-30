"""Zone geometry: polygons, line crossings, travel axis, ROI length, scaling."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import g2i, simple_camera

from traffic_intel.computer_vision.geometry import points_in_polygon, segments_intersect
from traffic_intel.computer_vision.homography import HomographyProjector
from traffic_intel.computer_vision.zones import CameraGeometry
from traffic_intel.core.enums import ZoneType


def geometry(**overrides: object) -> CameraGeometry:
    cam = simple_camera(**overrides)
    size = cam.calibration.image_size  # type: ignore[union-attr]
    return CameraGeometry(cam, size, HomographyProjector(cam.calibration, size))  # type: ignore[arg-type]


def test_points_in_polygon() -> None:
    sq = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], np.float32)
    assert points_in_polygon(np.array([[5, 5], [15, 5], [-1, -1]]), sq).tolist() == [True, False, False]


def test_segment_crossing_counts_once() -> None:
    a, b = np.array([0, 5]), np.array([10, 5])
    moves_p = np.array([[5, 0], [5, 4], [5, 5], [20, 0]])
    moves_q = np.array([[5, 4], [5, 5], [5, 6], [20, 10]])
    # approaching; landing on the line (counts); leaving from the line (not again); outside segment
    assert segments_intersect(moves_p, moves_q, a, b).tolist() == [False, True, False, False]


def test_travel_axis_stop_point_and_roi_length() -> None:
    geo = geometry()
    np.testing.assert_allclose(geo.travel_direction, [0, -1], atol=1e-6)
    np.testing.assert_allclose(geo.stop_point, [3.5, 0.0], atol=0.05)
    assert geo.lane_count == 2
    assert geo.anchor_reference == "front"            # traffic moves down the image, towards camera
    # ROI spans -2..62 m but the calibrated area is 0..60 (+15% margin) -> measured length in between
    assert 60.0 <= geo.roi_length_m <= 64.0
    np.testing.assert_allclose(geo.distance_upstream(np.array([[3.5, 20.0]])), [20.0], atol=0.05)


def test_travel_direction_from_lines_when_no_lanes() -> None:
    cam = simple_camera()
    zones = [z for z in cam.zones if z.type is not ZoneType.LANE]
    geo = geometry(zones=zones)
    np.testing.assert_allclose(geo.travel_direction, [0, -1], atol=1e-3)
    assert geo.lane_count == 1


def test_missing_travel_direction_is_an_error() -> None:
    cam = simple_camera()
    zones = [z for z in cam.zones if z.type in (ZoneType.ROI,)]
    with pytest.raises(ValueError, match="travel direction"):
        geometry(zones=zones)


def test_lane_assignment_and_loop() -> None:
    geo = geometry()
    pts = np.array([g2i(1.0, 10), g2i(5.0, 10), g2i(9.5, 10)], np.float32)
    assert geo.lane_of(pts).tolist() == [1, 2, -1]
    u, v = g2i(2, 5)
    assert geo.loop_occupied(np.array([[u - 9, v - 45, u + 9, v]], np.float32))
    u, v = g2i(2, 30)
    assert not geo.loop_occupied(np.array([[u - 9, v - 45, u + 9, v]], np.float32))


def test_zones_scale_with_processing_resolution() -> None:
    cam = simple_camera()
    half = (200, 400)
    proj = HomographyProjector(cam.calibration, half)  # type: ignore[arg-type]
    geo = CameraGeometry(cam, half, proj)
    u, v = g2i(1.0, 10)
    assert geo.lane_of(np.array([[u / 2, v / 2]], np.float32)).tolist() == [1]
    assert geo.roi_length_m == pytest.approx(geometry().roi_length_m, rel=1e-3)
