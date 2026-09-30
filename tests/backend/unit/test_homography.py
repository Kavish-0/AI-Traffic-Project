"""Coordinate conversion: pixels -> ground metres -> lon/lat."""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from traffic_intel.computer_vision.homography import EARTH_RADIUS_M, HomographyProjector
from traffic_intel.core.config import CalibrationConfig

GROUND = np.array([[0, 0], [7, 0], [0, 30], [7, 30], [0, 60], [7, 60]], dtype=np.float64)
# A strongly perspective ground->image mapping (road receding to a vanishing point).
H_TRUE = cv2.getPerspectiveTransform(
    np.float32([[0, 60], [7, 60], [7, 0], [0, 0]]), np.float32([[820, 420], [1100, 420], [1500, 1000], [400, 1000]])
)


def to_image(pts: np.ndarray) -> np.ndarray:
    p = np.hstack([pts, np.ones((len(pts), 1))]) @ H_TRUE.T
    return p[:, :2] / p[:, 2:3]


def calib(noise_px: float = 0.0, bearing: float = 0.0, size: tuple[int, int] = (1920, 1080)) -> CalibrationConfig:
    img = to_image(GROUND) + np.random.default_rng(0).normal(0, noise_px, (6, 2)) if noise_px else to_image(GROUND)
    return CalibrationConfig(
        image_points_px=[tuple(p) for p in img], ground_points_m=[tuple(p) for p in GROUND],
        origin_lonlat=(77.59, 12.97), bearing_deg=bearing, image_size=size,
    )


def test_exact_recovery_of_ground_coordinates() -> None:
    proj = HomographyProjector(calib())
    pts = np.array([[3.5, 12.0], [1.0, 45.0], [6.0, 2.5]])
    np.testing.assert_allclose(proj.image_to_ground(to_image(pts)), pts, atol=1e-3)
    assert proj.reprojection_error_m < 1e-3


def test_pixel_distance_is_not_metric_distance() -> None:
    """The same 10 px step is a few cm near the camera and metres far away."""
    proj = HomographyProjector(calib())
    near = proj.image_to_ground(np.array([[950, 990], [950, 1000]]))
    far = proj.image_to_ground(np.array([[960, 430], [960, 440]]))
    d_near = float(np.linalg.norm(near[1] - near[0]))
    d_far = float(np.linalg.norm(far[1] - far[0]))
    assert d_far > 10 * d_near


def test_ground_to_image_inverts() -> None:
    proj = HomographyProjector(calib())
    pts = np.array([[2.0, 10.0], [5.0, 55.0]])
    np.testing.assert_allclose(proj.image_to_ground(proj.ground_to_image(pts)), pts, atol=1e-3)


def test_frame_resize_rescales_calibration() -> None:
    """Calibration picked at 1920x1080, frames processed at 960x540."""
    full = HomographyProjector(calib())
    half = HomographyProjector(calib(), frame_size=(960, 540))
    pts = to_image(np.array([[3.0, 20.0]]))
    np.testing.assert_allclose(half.image_to_ground(pts / 2), full.image_to_ground(pts), atol=1e-3)


def test_noisy_calibration_reports_error_in_metres() -> None:
    proj = HomographyProjector(calib(noise_px=2.0))
    assert 0.01 < proj.reprojection_error_m < 1.0


def test_points_above_horizon_are_nan_and_outside_region() -> None:
    proj = HomographyProjector(calib())
    g = proj.image_to_ground(np.array([[960, 50]]))       # sky
    assert np.all(np.isnan(g))
    inside = proj.in_calibrated_region(np.array([[960, 700], [960, 50], [10, 1070]]))
    assert inside.tolist() == [True, False, False]


@pytest.mark.parametrize("bearing, east, north", [(0, 0, 1), (90, 1, 0), (180, 0, -1)])
def test_ground_to_lonlat_orientation(bearing: float, east: float, north: float) -> None:
    proj = HomographyProjector(calib(bearing=bearing))
    lon0, lat0 = 77.59, 12.97
    ll = proj.ground_to_lonlat(np.array([[0.0, 0.0], [0.0, 100.0]]))
    np.testing.assert_allclose(ll[0], [lon0, lat0], atol=1e-9)
    d_east = math.radians(ll[1, 0] - lon0) * EARTH_RADIUS_M * math.cos(math.radians(lat0))
    d_north = math.radians(ll[1, 1] - lat0) * EARTH_RADIUS_M
    assert d_east == pytest.approx(100 * east, abs=0.01)
    assert d_north == pytest.approx(100 * north, abs=0.01)


def test_x_axis_is_right_of_y_axis() -> None:
    proj = HomographyProjector(calib(bearing=0))   # +y north -> +x must be east
    ll = proj.ground_to_lonlat(np.array([[10.0, 0.0]]))
    assert ll[0, 0] > 77.59 and ll[0, 1] == pytest.approx(12.97)


def test_degenerate_calibration_rejected() -> None:
    bad = CalibrationConfig(
        image_points_px=[(0, 0), (1, 1), (2, 2), (3, 3)], ground_points_m=[(0, 0), (1, 1), (2, 2), (3, 3)],
        origin_lonlat=(0, 0), bearing_deg=0, image_size=(100, 100),
    )
    with pytest.raises(ValueError):
        HomographyProjector(bad)
