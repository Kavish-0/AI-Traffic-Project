"""BoT-SORT-style tracker: ByteTrack association + camera motion compensation.

Implements the global-motion-compensation (GMC) part of BoT-SORT (Aharon et
al., 2022) with sparse optical flow: background corners (outside detection
boxes) are tracked between frames with pyramidal Lucas-Kanade, a partial
affine transform is fitted with RANSAC, and every track's Kalman state is
warped by it before association. This keeps ids stable for shaky poles,
PTZ drift or handheld footage. Appearance ReID is not included (not needed
for fixed traffic cameras; can be added behind the same interface).
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from traffic_intel.core.types import Detections, Frame
from traffic_intel.tracking.interfaces import TRACKERS
from traffic_intel.tracking.trackers.bytetrack import ByteTracker, STrack


class SparseOpticalFlowGMC:
    def __init__(self, downscale: float = 2.0, max_corners: int = 1000) -> None:
        self.downscale = max(1.0, downscale)
        self.max_corners = max_corners
        self._prev: np.ndarray | None = None

    def reset(self) -> None:
        self._prev = None

    def estimate(self, image: np.ndarray, boxes_xyxy: np.ndarray) -> np.ndarray:
        """2x3 affine mapping previous-frame pixels to current-frame pixels."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        h, w = gray.shape
        s = self.downscale
        small = cv2.resize(gray, (int(w / s), int(h / s))) if s > 1 else gray
        identity = np.eye(2, 3)
        prev, self._prev = self._prev, small
        if prev is None or prev.shape != small.shape:
            return identity

        mask = np.full(small.shape, 255, dtype=np.uint8)
        for x1, y1, x2, y2 in (boxes_xyxy / s).astype(int):
            mask[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)] = 0   # ignore moving objects
        pts = cv2.goodFeaturesToTrack(prev, self.max_corners, 0.01, 1, mask=mask, blockSize=3)
        if pts is None or len(pts) < 8:
            return identity
        nxt, status, _ = cv2.calcOpticalFlowPyrLK(prev, small, pts, None)  # type: ignore[call-overload]
        good = status.reshape(-1).astype(bool)
        if good.sum() < 8:
            return identity
        A, _ = cv2.estimateAffinePartial2D(pts[good], nxt[good], method=cv2.RANSAC)
        if A is None:
            return identity
        A[:, 2] *= s
        return np.asarray(A)


class BoTSORTTracker(ByteTracker):
    def __init__(self, gmc_downscale: float = 2.0, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.gmc = SparseOpticalFlowGMC(downscale=gmc_downscale)
        self.last_affine = np.eye(2, 3)

    def reset(self) -> None:
        super().reset()
        if hasattr(self, "gmc"):
            self.gmc.reset()

    def _compensate_motion(self, tracks: list[STrack], frame: Frame, dets: Detections) -> None:
        A = self.gmc.estimate(frame.image, dets.xyxy)
        self.last_affine = A
        if np.allclose(A, np.eye(2, 3), atol=1e-6):
            return
        R, t = A[:, :2], A[:, 2]
        R8 = np.eye(8)
        R8[0:2, 0:2] = R
        R8[4:6, 4:6] = R
        for trk in tracks:
            if trk.mean is None or trk.cov is None:
                continue
            mean = R8 @ trk.mean
            mean[:2] += t
            trk.mean = mean
            trk.cov = R8 @ trk.cov @ R8.T


@TRACKERS.register("botsort")
def _create(**kwargs: Any) -> BoTSORTTracker:  # noqa: ANN401
    return BoTSORTTracker(**kwargs)
