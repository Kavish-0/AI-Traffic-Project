"""Planar homography ground projector (default ``GroundProjector``).

pixel (u, v)  --H-->  ground (x, y) metres  --origin + bearing-->  lon/lat

Assumptions (see docs/TRAFFIC_STATE_ESTIMATION.md for the full list):
* The road inside the calibrated area is planar (no slope change, no flyover).
* Lens distortion is negligible, or frames were undistorted before detection
  and calibration points were picked on undistorted frames.
* The tracked point (bottom-centre of the box) lies on the road plane.
"""

from __future__ import annotations

import logging
import math

import cv2
import numpy as np

from traffic_intel.computer_vision.calibration import GROUND_PROJECTORS, GroundProjector
from traffic_intel.computer_vision.geometry import (
    as_points,
    convex_hull,
    expand_polygon,
    points_in_polygon,
    scale_points,
)
from traffic_intel.core.config import CalibrationConfig
from traffic_intel.core.types import BoolArray, FloatArray

log = logging.getLogger(__name__)

EARTH_RADIUS_M = 6_378_137.0


class HomographyProjector(GroundProjector):
    def __init__(
        self,
        calibration: CalibrationConfig,
        frame_size: tuple[int, int] | None = None,
        region_margin: float = 0.15,
        ransac_threshold_m: float | None = None,
        max_reprojection_error_m: float = 1.0,
    ) -> None:
        """
        Args:
            calibration: correspondences picked at ``calibration.image_size``.
            frame_size: (width, height) of the frames that will be projected;
                image points are rescaled if it differs from the calibration size.
            region_margin: how far (relative) outside the hull of the image
                calibration points projections are still considered reliable.
            ransac_threshold_m: use RANSAC with this inlier threshold (needs
                >= 5 points); None = plain least squares on all points.
            max_reprojection_error_m: log a warning above this error.
        """
        self.calibration = calibration
        self.frame_size = tuple(frame_size or calibration.image_size)
        self.image_points = scale_points(
            as_points(calibration.image_points_px), calibration.image_size, self.frame_size  # type: ignore[arg-type]
        )
        self.ground_points = as_points(calibration.ground_points_m)

        method, thresh = 0, 3.0
        if ransac_threshold_m is not None and len(self.image_points) >= 5:
            method, thresh = cv2.RANSAC, ransac_threshold_m
        H, _ = cv2.findHomography(self.image_points, self.ground_points, method, thresh)
        if H is None:
            raise ValueError("Homography could not be estimated (collinear or duplicate points?)")
        H = H.astype(np.float64)
        # H is defined up to scale *and sign*; fix the sign so the homogeneous
        # coordinate is positive on the road (w <= 0 then means "above the horizon").
        _, w = self._apply(H, self.image_points)
        if np.median(w) < 0:
            H = -H
        self.H: np.ndarray = H
        self.H_inv: np.ndarray = np.linalg.inv(self.H)

        self._region = expand_polygon(convex_hull(self.image_points), region_margin)
        self._error = float(
            np.mean(np.linalg.norm(self.image_to_ground(self.image_points) - self.ground_points, axis=1))
        )
        if self._error > max_reprojection_error_m:
            log.warning(
                "Calibration reprojection error %.2f m exceeds %.2f m - speeds will be unreliable",
                self._error,
                max_reprojection_error_m,
            )

        theta = math.radians(calibration.bearing_deg)
        # +y points at `bearing`; +x points at bearing + 90 deg (clockwise).
        self._rot = np.array(
            [[math.cos(theta), math.sin(theta)],      # east  = x*cos + y*sin
             [-math.sin(theta), math.cos(theta)]],    # north = -x*sin + y*cos
            dtype=np.float64,
        )
        self._lon0, self._lat0 = calibration.origin_lonlat

    # ------------------------------------------------------------------ #
    @staticmethod
    def _apply(H: np.ndarray, pts: FloatArray) -> tuple[np.ndarray, np.ndarray]:
        p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        homo = np.hstack([p, np.ones((len(p), 1))]) @ H.T
        w = homo[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            out = homo[:, :2] / w[:, None]
        return out, w

    def image_to_ground(self, points_px: FloatArray) -> FloatArray:
        out, w = self._apply(self.H, points_px)
        # w <= 0: point is above the horizon line -> no ground intersection.
        out[w <= 1e-12] = np.nan
        return out.astype(np.float32)

    def ground_to_image(self, points_m: FloatArray) -> FloatArray:
        out, w = self._apply(self.H_inv, points_m)
        out[w <= 1e-12] = np.nan
        return out.astype(np.float32)

    def ground_to_lonlat(self, points_m: FloatArray) -> FloatArray:
        p = np.asarray(points_m, dtype=np.float64).reshape(-1, 2)
        en = p @ self._rot.T                     # (east, north) metres
        lat = self._lat0 + np.degrees(en[:, 1] / EARTH_RADIUS_M)
        lon = self._lon0 + np.degrees(en[:, 0] / (EARTH_RADIUS_M * math.cos(math.radians(self._lat0))))
        return np.stack([lon, lat], axis=1).astype(np.float64)  # float64: float32 loses ~1 m

    @property
    def reprojection_error_m(self) -> float:
        return self._error

    def in_calibrated_region(self, points_px: FloatArray) -> BoolArray:
        pts = np.asarray(points_px, dtype=np.float32).reshape(-1, 2)
        _, w = self._apply(self.H, pts)
        return points_in_polygon(pts, self._region) & (w > 1e-12)

    def metres_per_pixel(self, point_px: FloatArray) -> float:
        """Local ground scale at a pixel (for annotation / tolerances)."""
        p = np.asarray(point_px, dtype=np.float32).reshape(1, 2)
        g = self.image_to_ground(np.vstack([p, p + np.array([[1.0, 0.0]]), p + np.array([[0.0, 1.0]])]))
        return float((np.linalg.norm(g[1] - g[0]) + np.linalg.norm(g[2] - g[0])) / 2.0)


@GROUND_PROJECTORS.register("homography")
def _create(**kwargs: object) -> HomographyProjector:
    return HomographyProjector(**kwargs)  # type: ignore[arg-type]
