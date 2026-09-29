"""ByteTrack / BoT-SORT: identity persistence and lifecycle."""

from __future__ import annotations

from datetime import UTC, datetime

import cv2
import numpy as np

from traffic_intel.core.enums import ObjectClass, class_id_of
from traffic_intel.core.types import Detections, Frame
from traffic_intel.tracking.trackers.botsort import BoTSORTTracker, SparseOpticalFlowGMC
from traffic_intel.tracking.trackers.bytetrack import ByteTracker

BLANK = np.zeros((480, 640, 3), np.uint8)
CAR = class_id_of(ObjectClass.CAR)


def frame(k: int, image: np.ndarray = BLANK) -> Frame:
    return Frame("cam", k, datetime.now(UTC), image)


def dets(boxes: list[list[float]], conf: float | list[float] = 0.9, cls: int | list[int] = CAR) -> Detections:
    n = len(boxes)
    return Detections(
        np.array(boxes, np.float32).reshape(-1, 4),
        np.array(conf if isinstance(conf, list) else [conf] * n, np.float32),
        np.array(cls if isinstance(cls, list) else [cls] * n, np.int64),
    )


def box(x: float, y: float, w: float = 40, h: float = 30) -> list[float]:
    return [x, y, x + w, y + h]


def test_constant_motion_keeps_one_id() -> None:
    trk = ByteTracker(fps=10)
    ids = set()
    for k in range(30):
        out = trk.update(dets([box(10 + 5 * k, 100)]), frame(k))
        ids.update(out.track_id.tolist())
    assert ids == {1}


def test_new_track_needs_min_hits_except_first_frame() -> None:
    trk = ByteTracker(fps=10, min_hits=2)
    assert len(trk.update(dets([box(10, 10)]), frame(0))) == 1      # first frame: confirmed at once
    assert len(trk.update(dets([box(12, 10), box(300, 300)]), frame(1))) == 1   # new object tentative
    out = trk.update(dets([box(14, 10), box(302, 300)]), frame(2))
    assert len(out) == 2


def test_short_occlusion_keeps_id_long_occlusion_does_not() -> None:
    trk = ByteTracker(fps=10, track_buffer_s=1.0)
    for k in range(10):
        trk.update(dets([box(10 + 5 * k, 100)]), frame(k))
    for k in range(10, 15):                          # 0.5 s missed
        assert len(trk.update(dets([]), frame(k))) == 0
    out = trk.update(dets([box(10 + 5 * 15, 100)]), frame(15))
    assert out.track_id.tolist() == [1]
    for k in range(16, 40):                          # 2.4 s missed > buffer
        trk.update(dets([]), frame(k))
    trk.update(dets([box(300, 100)]), frame(40))
    out = trk.update(dets([box(305, 100)]), frame(41))
    assert out.track_id.tolist() != [1] and len(out) == 1


def test_low_score_detections_keep_track_alive() -> None:
    """ByteTrack's second association: a low-confidence frame does not break the track."""
    trk = ByteTracker(fps=10, track_high_thresh=0.5, track_low_thresh=0.1)
    for k in range(5):
        trk.update(dets([box(10 + 5 * k, 100)]), frame(k))
    out = trk.update(dets([box(35, 100)], conf=0.2), frame(5))
    assert out.track_id.tolist() == [1]


def test_crossing_objects_keep_identities() -> None:
    trk = ByteTracker(fps=10)
    history: dict[int, list[float]] = {}
    for k in range(40):
        a = box(10 + 10 * k, 100)            # moving right
        b = box(410 - 10 * k, 160)           # moving left, different row
        out = trk.update(dets([a, b]), frame(k))
        for i, tid in enumerate(out.track_id.tolist()):
            history.setdefault(tid, []).append(float(out.xyxy[i, 1]))
    assert len(history) == 2
    rows = sorted({round(np.mean(v)) for v in history.values()})
    assert rows == [100, 160]                # each id stayed on its own row


def test_class_is_confidence_weighted_vote() -> None:
    trk = ByteTracker(fps=10)
    truck = class_id_of(ObjectClass.TRUCK)
    for k in range(10):
        cls = truck if k in (3, 7) else CAR
        out = trk.update(dets([box(10 + 3 * k, 50)], cls=cls), frame(k))
    assert out.class_id.tolist() == [CAR]


def test_reset_never_reuses_ids() -> None:
    trk = ByteTracker(fps=10)
    trk.update(dets([box(10, 10)]), frame(0))
    trk.reset()
    out = trk.update(dets([box(10, 10)]), frame(0))
    assert out.track_id.tolist() == [2]


def _textured(shift: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    base = (rng.random((520, 700)) * 255).astype(np.uint8)
    base = cv2.GaussianBlur(base, (5, 5), 0)
    img = base[20 : 20 + 480, 20 + shift : 20 + shift + 640]
    return cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_GRAY2BGR)


def test_gmc_estimates_camera_translation() -> None:
    gmc = SparseOpticalFlowGMC(downscale=1.0)
    gmc.estimate(_textured(0), np.zeros((0, 4)))
    A = gmc.estimate(_textured(12), np.zeros((0, 4)))   # scene content moves 12 px left
    assert A[0, 2] == np.float64(A[0, 2])
    assert abs(A[0, 2] + 12) < 1.0 and abs(A[1, 2]) < 1.0


def test_botsort_survives_camera_jump() -> None:
    """A static object + sudden 25 px camera pan: BoT-SORT keeps the id."""
    trk = BoTSORTTracker(fps=10, gmc_downscale=1.0)
    for k in range(5):
        trk.update(dets([box(300, 200, 30, 30)]), frame(k, _textured(0)))
    out = trk.update(dets([box(275, 200, 30, 30)]), frame(5, _textured(25)))
    assert out.track_id.tolist() == [1]
