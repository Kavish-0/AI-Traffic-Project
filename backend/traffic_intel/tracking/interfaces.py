"""Tracking interface (Module 1): per-frame detections -> persistent track IDs.

One tracker instance per camera (trackers are stateful). Implementations:
ByteTrack (default, motion-only, fast) and BoT-SORT (adds camera-motion
compensation + optional ReID; better for shaky/PTZ cameras).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from traffic_intel.core.registry import Registry
from traffic_intel.core.types import Detections, Frame, Tracks


class Tracker(ABC):
    @abstractmethod
    def update(self, detections: Detections, frame: Frame) -> Tracks:
        """Associate this frame's detections with existing tracks.

        ``frame`` is passed for trackers that need appearance or global motion
        compensation; motion-only trackers ignore the image.
        """

    @abstractmethod
    def reset(self) -> None:
        """Drop all state, e.g. after a stream reconnect or scene cut."""


TRACKERS: Registry[Tracker] = Registry("tracker")
TRACKERS.register_lazy("bytetrack", "traffic_intel.tracking.trackers.bytetrack")
TRACKERS.register_lazy("botsort", "traffic_intel.tracking.trackers.botsort")
