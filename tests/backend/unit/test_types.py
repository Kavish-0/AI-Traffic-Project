from __future__ import annotations

import numpy as np

from traffic_intel.core.types import Detections, Tracks


def test_empty_defaults() -> None:
    assert len(Detections()) == 0
    assert len(Tracks()) == 0


def test_anchor_points_bottom_centre() -> None:
    t = Tracks(
        track_id=np.array([1]),
        xyxy=np.array([[10, 20, 30, 60]], dtype=np.float32),
        confidence=np.array([0.9], dtype=np.float32),
        class_id=np.array([0]),
    )
    np.testing.assert_allclose(t.anchor_points(), [[20.0, 60.0]])
