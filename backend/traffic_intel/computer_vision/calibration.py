"""Camera calibration interface: image pixels -> ground metres -> WGS84.

Speed is computed from displacement in *ground* metres, never pixels. The
default implementation (Phase 2) fits a planar homography with
``cv2.findHomography`` from ``CalibrationConfig`` correspondences and reports
the reprojection error in metres so bad calibrations are visible.

Known limitation: the flat-road assumption breaks on slopes/flyovers and the
bottom-centre anchor point is biased for tall vehicles (bus, truck) seen at
oblique angles. Both are measured in Phase 2 validation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from traffic_intel.core.registry import Registry
from traffic_intel.core.types import BoolArray, FloatArray


class GroundProjector(ABC):
    @abstractmethod
    def image_to_ground(self, points_px: FloatArray) -> FloatArray:
        """(N, 2) pixels -> (N, 2) metres in the local road-aligned frame."""

    @abstractmethod
    def ground_to_lonlat(self, points_m: FloatArray) -> FloatArray:
        """(N, 2) local metres -> (N, 2) lon/lat (EPSG:4326)."""

    @property
    @abstractmethod
    def reprojection_error_m(self) -> float:
        """Mean error of the calibration correspondences, in metres."""

    def in_calibrated_region(self, points_px: FloatArray) -> BoolArray:
        """True where the projection is trustworthy (inside/near the area spanned
        by the calibration points). Default: everywhere."""
        return np.ones(len(points_px), dtype=bool)


GROUND_PROJECTORS: Registry[GroundProjector] = Registry("ground_projector")
GROUND_PROJECTORS.register_lazy("homography", "traffic_intel.computer_vision.homography")
