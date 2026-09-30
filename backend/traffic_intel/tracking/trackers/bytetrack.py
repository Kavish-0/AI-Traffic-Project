"""ByteTrack (Zhang et al., ECCV 2022), NumPy/SciPy implementation.

Why not wrap Ultralytics' tracker: this version has no torch dependency, can
be unit-tested frame by frame, exposes the track lifecycle explicitly, and is
the base for BoT-SORT-style camera-motion compensation (``botsort.py``).

Lifecycle
    NEW       created from an unmatched high-confidence detection (tentative,
              not output). Confirmed on its next match (``min_hits``), else removed.
    TRACKED   matched this frame -> output.
    LOST      unmatched; kept (Kalman-predicted) for ``track_buffer_s`` seconds
              and can be re-found with the same id.
    REMOVED   lost too long, or tentative and unmatched. Ids are never reused.

Association (per frame)
    1. high-score detections  <-> confirmed tracked + lost tracks   (IoU, fused with score)
    2. low-score detections   <-> remaining *tracked* tracks        (IoU)  - the "BYTE" step that
       keeps tracks alive through occlusion/blur when confidence dips
    3. remaining high dets    <-> tentative tracks                  (IoU)
    4. still-unmatched high dets above ``new_track_thresh`` start NEW tracks.

Matching is class-agnostic (YOLO often flickers car<->truck between frames);
each track's class is a confidence-weighted vote over its history.
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np
from scipy.optimize import linear_sum_assignment

from traffic_intel.core.types import Detections, Frame, Tracks
from traffic_intel.tracking.interfaces import TRACKERS, Tracker
from traffic_intel.tracking.kalman import KalmanFilterXYAH, xyah_to_xyxy, xyxy_to_xyah


class TrackState(IntEnum):
    NEW = 0
    TRACKED = 1
    LOST = 2
    REMOVED = 3


class STrack:
    def __init__(self, xyxy: np.ndarray, score: float, class_id: int) -> None:
        self.det_xyxy = np.asarray(xyxy, dtype=np.float64)
        self.score = float(score)
        self.class_votes: dict[int, float] = {int(class_id): float(score)}
        self.mean: np.ndarray | None = None
        self.cov: np.ndarray | None = None
        self.state = TrackState.NEW
        self.is_activated = False
        self.track_id = 0
        self.start_frame = 0
        self.frame_id = 0
        self.hits = 0

    # ---------------------------------------------------------------- #
    @property
    def class_id(self) -> int:
        return max(self.class_votes.items(), key=lambda kv: kv[1])[0]

    @property
    def kalman_xyxy(self) -> np.ndarray:
        return self.det_xyxy if self.mean is None else xyah_to_xyxy(self.mean)

    def _vote(self, class_id: int, score: float) -> None:
        self.class_votes[class_id] = self.class_votes.get(class_id, 0.0) + score

    def activate(self, kf: KalmanFilterXYAH, frame_id: int, track_id: int, confirm: bool) -> None:
        self.track_id = track_id
        self.mean, self.cov = kf.initiate(xyxy_to_xyah(self.det_xyxy))
        self.state = TrackState.TRACKED
        self.is_activated = confirm
        self.frame_id = self.start_frame = frame_id
        self.hits = 1

    def update(self, kf: KalmanFilterXYAH, det: STrack, frame_id: int, min_hits: int) -> None:
        assert self.mean is not None and self.cov is not None
        self.mean, self.cov = kf.update(self.mean, self.cov, xyxy_to_xyah(det.det_xyxy))
        self.det_xyxy = det.det_xyxy
        self.score = det.score
        self._vote(det.class_id, det.score)
        self.state = TrackState.TRACKED
        self.frame_id = frame_id
        self.hits += 1
        if self.hits >= min_hits:
            self.is_activated = True

    def predict(self, kf: KalmanFilterXYAH) -> None:
        assert self.mean is not None and self.cov is not None
        mean = self.mean.copy()
        if self.state != TrackState.TRACKED:
            mean[7] = 0.0  # do not keep growing/shrinking a box we cannot see
        self.mean, self.cov = kf.predict(mean, self.cov)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a = a[:, None, :]
    b = b[None, :, :]
    iw = np.clip(np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]), 0, None)
    ih = np.clip(np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]), 0, None)
    inter = iw * ih
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return np.asarray(inter / np.maximum(area_a + area_b - inter, 1e-9))


def match(cost: np.ndarray, thresh: float) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Hungarian assignment keeping only pairs with cost <= thresh."""
    n, m = cost.shape
    if n == 0 or m == 0:
        return [], list(range(n)), list(range(m))
    capped = np.where(cost > thresh, thresh + 1e3, cost)
    rows, cols = linear_sum_assignment(capped)
    pairs = [(int(r), int(c)) for r, c in zip(rows, cols, strict=True) if cost[r, c] <= thresh]
    mr = {r for r, _ in pairs}
    mc = {c for _, c in pairs}
    return pairs, [i for i in range(n) if i not in mr], [j for j in range(m) if j not in mc]


class ByteTracker(Tracker):
    def __init__(
        self,
        fps: float = 10.0,
        track_high_thresh: float = 0.5,
        track_low_thresh: float = 0.1,
        new_track_thresh: float = 0.6,
        track_buffer_s: float = 3.0,
        match_thresh: float = 0.8,
        second_match_thresh: float = 0.5,
        unconfirmed_match_thresh: float = 0.7,
        min_hits: int = 2,
        fuse_score: bool = True,
        output_box: str = "detection",   # "detection" (raw, unsmoothed) | "kalman"
        min_box_area: float = 0.0,
    ) -> None:
        self.fps = fps
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.new_track_thresh = new_track_thresh
        self.max_time_lost = max(1, int(round(track_buffer_s * fps)))
        self.match_thresh = match_thresh
        self.second_match_thresh = second_match_thresh
        self.unconfirmed_match_thresh = unconfirmed_match_thresh
        self.min_hits = max(1, min_hits)
        self.fuse_score = fuse_score
        self.output_box = output_box
        self.min_box_area = min_box_area
        self.kf = KalmanFilterXYAH()
        self._next_id = 1
        self.reset()

    def reset(self) -> None:
        # Ids keep increasing across resets so downstream trajectories never collide.
        self.frame_id = 0
        self.tracked: list[STrack] = []
        self.lost: list[STrack] = []

    # hook for camera-motion compensation (BoT-SORT)
    def _compensate_motion(self, tracks: list[STrack], frame: Frame, dets: Detections) -> None:
        return None

    def _cost(self, tracks: list[STrack], dets: list[STrack], fuse: bool) -> np.ndarray:
        if not tracks or not dets:
            return np.zeros((len(tracks), len(dets)))
        iou = iou_matrix(np.stack([t.kalman_xyxy for t in tracks]), np.stack([d.det_xyxy for d in dets]))
        if fuse:
            iou = iou * np.array([d.score for d in dets])[None, :]
        return 1.0 - iou

    def update(self, detections: Detections, frame: Frame) -> Tracks:
        self.frame_id += 1
        fid = self.frame_id
        scores = detections.confidence
        area = (detections.xyxy[:, 2] - detections.xyxy[:, 0]) * (detections.xyxy[:, 3] - detections.xyxy[:, 1])
        valid = area >= self.min_box_area
        high = valid & (scores >= self.track_high_thresh)
        low = valid & (scores > self.track_low_thresh) & ~high
        dets_high = [STrack(detections.xyxy[i], scores[i], detections.class_id[i]) for i in np.flatnonzero(high)]
        dets_low = [STrack(detections.xyxy[i], scores[i], detections.class_id[i]) for i in np.flatnonzero(low)]

        unconfirmed = [t for t in self.tracked if not t.is_activated]
        confirmed = [t for t in self.tracked if t.is_activated]
        pool = confirmed + self.lost
        for t in pool + unconfirmed:
            t.predict(self.kf)
        self._compensate_motion(pool + unconfirmed, frame, detections)

        # 1) high-score detections vs confirmed + lost tracks
        pairs, u_track, u_det = match(self._cost(pool, dets_high, self.fuse_score), self.match_thresh)
        for it, idet in pairs:
            pool[it].update(self.kf, dets_high[idet], fid, 1)   # confirmed tracks stay confirmed

        # 2) low-score detections vs still-unmatched *tracked* (not lost) tracks
        remaining = [pool[i] for i in u_track if pool[i].state == TrackState.TRACKED]
        pairs2, u_rem, _ = match(self._cost(remaining, dets_low, False), self.second_match_thresh)
        for it, idet in pairs2:
            remaining[it].update(self.kf, dets_low[idet], fid, 1)
        for it in u_rem:
            remaining[it].state = TrackState.LOST

        # 3) leftover high dets vs tentative tracks
        leftover = [dets_high[i] for i in u_det]
        pairs3, u_unconf, u_left = match(
            self._cost(unconfirmed, leftover, self.fuse_score), self.unconfirmed_match_thresh
        )
        for it, idet in pairs3:
            unconfirmed[it].update(self.kf, leftover[idet], fid, self.min_hits)
        for it in u_unconf:
            unconfirmed[it].state = TrackState.REMOVED

        # 4) new tentative tracks (confirmed immediately on the very first frame / min_hits == 1)
        new_tracks = []
        for idet in u_left:
            det = leftover[idet]
            if det.score < self.new_track_thresh:
                continue
            det.activate(self.kf, fid, self._next_id, confirm=(fid == 1 or self.min_hits == 1))
            self._next_id += 1
            new_tracks.append(det)

        # lifecycle bookkeeping
        for t in pool:
            if t.state == TrackState.LOST and fid - t.frame_id > self.max_time_lost:
                t.state = TrackState.REMOVED
        everything = pool + unconfirmed + new_tracks
        self.tracked = [t for t in everything if t.state == TrackState.TRACKED]
        self.lost = [t for t in everything if t.state == TrackState.LOST]
        self._drop_duplicates()

        out = [t for t in self.tracked if t.is_activated and t.frame_id == fid]
        if not out:
            return Tracks()
        boxes = [t.det_xyxy if self.output_box == "detection" else t.kalman_xyxy for t in out]
        return Tracks(
            track_id=np.array([t.track_id for t in out], dtype=np.int64),
            xyxy=np.stack(boxes).astype(np.float32),
            confidence=np.array([t.score for t in out], dtype=np.float32),
            class_id=np.array([t.class_id for t in out], dtype=np.int64),
        )

    def _drop_duplicates(self) -> None:
        """A lost track that overlaps a tracked one almost completely is a
        duplicate identity; keep the one that has existed longer."""
        if not self.tracked or not self.lost:
            return
        iou = iou_matrix(np.stack([t.kalman_xyxy for t in self.tracked]),
                         np.stack([t.kalman_xyxy for t in self.lost]))
        drop_lost: set[int] = set()
        drop_tracked: set[int] = set()
        for i, j in zip(*np.nonzero(iou > 0.85), strict=True):
            a, b = self.tracked[i], self.lost[j]
            if fid_age(a) >= fid_age(b):
                drop_lost.add(j)
            else:
                drop_tracked.add(i)
        self.tracked = [t for i, t in enumerate(self.tracked) if i not in drop_tracked]
        self.lost = [t for j, t in enumerate(self.lost) if j not in drop_lost]


def fid_age(t: STrack) -> int:
    return t.frame_id - t.start_frame


@TRACKERS.register("bytetrack")
def _create(**kwargs: object) -> ByteTracker:
    return ByteTracker(**kwargs)  # type: ignore[arg-type]
