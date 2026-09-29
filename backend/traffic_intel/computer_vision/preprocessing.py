"""Frame preprocessors (applied in order before detection).

Resizing and fps sampling happen in the VideoSource (they define the
processing coordinate system). Preprocessors here must NOT move pixels:
masking and photometric changes only - except ``undistort``, which must then
also have been applied to the frames on which calibration/zones were drawn.

Configured in ``configs/pipeline.yaml -> preprocessors``; each factory
receives the ``camera`` config as an extra keyword argument.
"""

from __future__ import annotations

import cv2
import numpy as np

from traffic_intel.computer_vision.geometry import as_points, scale_points
from traffic_intel.computer_vision.interfaces import PREPROCESSORS, FramePreprocessor
from traffic_intel.core.config import CameraConfig
from traffic_intel.core.enums import ZoneType
from traffic_intel.core.types import Frame


class RoiMaskPreprocessor(FramePreprocessor):
    """Black out pixels outside the (dilated) ROI so the detector does not spend
    effort on - or hallucinate from - irrelevant areas. The feature engine still
    filters by ROI independently, so this is an optimisation, not a filter."""

    def __init__(self, camera: CameraConfig, padding_px: int = 40) -> None:
        self.camera = camera
        self.padding_px = padding_px
        self._mask: np.ndarray | None = None

    def _build(self, h: int, w: int) -> np.ndarray | None:
        roi = next((z for z in self.camera.zones if z.type is ZoneType.ROI), None)
        if roi is None:
            return None
        pts = scale_points(as_points(roi.points_px), self.camera.zone_reference_size() or (w, h), (w, h))
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts.round().astype(np.int32)], 255)
        if self.padding_px > 0:
            k = 2 * self.padding_px + 1
            mask = np.asarray(cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))), dtype=np.uint8)
        return mask

    def process(self, frame: Frame) -> Frame:
        h, w = frame.shape
        if self._mask is None or self._mask.shape != (h, w):
            self._mask = self._build(h, w)
        if self._mask is None:
            return frame
        frame.image = np.asarray(cv2.bitwise_and(frame.image, frame.image, mask=self._mask), dtype=np.uint8)
        return frame


class ClahePreprocessor(FramePreprocessor):
    """Contrast-limited histogram equalisation on luminance (night / haze)."""

    def __init__(self, camera: CameraConfig | None = None, clip_limit: float = 2.0, tile: int = 8,
                 only_if_mean_below: float | None = 90.0) -> None:
        self._clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
        self.only_if_mean_below = only_if_mean_below

    def process(self, frame: Frame) -> Frame:
        lab = cv2.cvtColor(frame.image, cv2.COLOR_BGR2LAB)
        l_chan = lab[:, :, 0]
        if self.only_if_mean_below is not None and float(l_chan.mean()) >= self.only_if_mean_below:
            return frame
        lab[:, :, 0] = self._clahe.apply(l_chan)
        frame.image = np.asarray(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), dtype=np.uint8)
        return frame


class UndistortPreprocessor(FramePreprocessor):
    """Lens undistortion from intrinsics (fx, fy, cx, cy at `image_size`) and
    OpenCV distortion coefficients (k1, k2, p1, p2[, k3])."""

    def __init__(self, camera: CameraConfig | None = None, *, camera_matrix: list[list[float]],
                 dist_coeffs: list[float], image_size: tuple[int, int]) -> None:
        self.K = np.asarray(camera_matrix, dtype=np.float64)
        self.D = np.asarray(dist_coeffs, dtype=np.float64)
        self.image_size = tuple(image_size)
        self._maps: tuple[np.ndarray, np.ndarray] | None = None
        self._for: tuple[int, int] | None = None

    def process(self, frame: Frame) -> Frame:
        h, w = frame.shape
        if self._for != (w, h):
            sx, sy = w / self.image_size[0], h / self.image_size[1]
            K = self.K.copy()
            K[0, :] *= sx
            K[1, :] *= sy
            self._maps = cv2.initUndistortRectifyMap(K, self.D, None, K, (w, h), cv2.CV_16SC2)
            self._for = (w, h)
        assert self._maps is not None
        frame.image = np.asarray(cv2.remap(frame.image, self._maps[0], self._maps[1], cv2.INTER_LINEAR), dtype=np.uint8)
        return frame


PREPROCESSORS.register("roi_mask")(RoiMaskPreprocessor)
PREPROCESSORS.register("clahe")(ClahePreprocessor)
PREPROCESSORS.register("undistort")(UndistortPreprocessor)
