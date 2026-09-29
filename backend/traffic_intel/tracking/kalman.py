"""Constant-velocity Kalman filter in (cx, cy, aspect, height) box space.

Same formulation as SORT/DeepSORT/ByteTrack: noise scales with box height, so
far (small) and near (large) objects are handled consistently.
State: [cx, cy, a, h, vcx, vcy, va, vh]; one step = one processed frame.
"""

from __future__ import annotations

import numpy as np


class KalmanFilterXYAH:
    def __init__(self, std_weight_position: float = 1.0 / 20, std_weight_velocity: float = 1.0 / 160) -> None:
        self._F = np.eye(8)
        for i in range(4):
            self._F[i, 4 + i] = 1.0
        self._H = np.eye(4, 8)
        self._wp = std_weight_position
        self._wv = std_weight_velocity

    def initiate(self, xyah: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = np.r_[xyah, np.zeros(4)]
        h = xyah[3]
        std = [2 * self._wp * h, 2 * self._wp * h, 1e-2, 2 * self._wp * h,
               10 * self._wv * h, 10 * self._wv * h, 1e-5, 10 * self._wv * h]
        return mean, np.diag(np.square(std))

    def predict(self, mean: np.ndarray, cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = mean[3]
        std = [self._wp * h, self._wp * h, 1e-2, self._wp * h,
               self._wv * h, self._wv * h, 1e-5, self._wv * h]
        Q = np.diag(np.square(std))
        return self._F @ mean, self._F @ cov @ self._F.T + Q

    def update(self, mean: np.ndarray, cov: np.ndarray, xyah: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = mean[3]
        R = np.diag(np.square([self._wp * h, self._wp * h, 1e-1, self._wp * h]))
        S = self._H @ cov @ self._H.T + R
        PHt = cov @ self._H.T
        K = np.linalg.solve(S, PHt.T).T
        innovation = xyah - self._H @ mean
        return mean + K @ innovation, cov - K @ S @ K.T


def xyxy_to_xyah(b: np.ndarray) -> np.ndarray:
    w, h = b[2] - b[0], b[3] - b[1]
    return np.array([b[0] + w / 2, b[1] + h / 2, w / max(h, 1e-6), h], dtype=np.float64)


def xyah_to_xyxy(s: np.ndarray) -> np.ndarray:
    cx, cy, a, h = s[:4]
    w = a * h
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float64)
