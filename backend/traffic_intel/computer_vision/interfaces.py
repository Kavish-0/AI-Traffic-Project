"""Perception interfaces (Module 1): video in, detections out.

Implementations live in ``sources/``, ``detectors/`` and are selected via the
registries below. Nothing outside ``computer_vision`` imports a concrete
detector (e.g. Ultralytics) - only ``Detector``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence

from traffic_intel.core.enums import ObjectClass
from traffic_intel.core.registry import Registry
from traffic_intel.core.types import Detections, Frame


class VideoSource(ABC):
    """Yields frames with *source* timestamps.

    Live streams (RTSP/HTTP) stamp frames with wall-clock UTC and drop stale
    frames when the consumer lags (latest-frame policy). File sources stamp
    frames with ``replay_start + pts`` and never drop, so offline runs are
    deterministic and reproducible.
    """

    camera_id: str

    @property
    @abstractmethod
    def fps(self) -> float: ...

    @property
    @abstractmethod
    def is_live(self) -> bool: ...

    @property
    @abstractmethod
    def frame_size(self) -> tuple[int, int]:
        """(width, height) of emitted frames, i.e. after the configured resize."""

    @abstractmethod
    def frames(self) -> Iterator[Frame]: ...

    @abstractmethod
    def close(self) -> None: ...


class FramePreprocessor(ABC):
    """Image processing before detection: undistortion, ROI masking, resizing,
    low-light enhancement (CLAHE), frame-rate sampling. Must not change the
    geometry the calibration assumes unless it also transforms coordinates
    back (``to_original``)."""

    @abstractmethod
    def process(self, frame: Frame) -> Frame: ...


class Detector(ABC):
    """Maps frames to detections in the project's ``ObjectClass`` taxonomy.

    Adapters are responsible for mapping model-specific class ids (e.g. COCO
    'person' -> PEDESTRIAN) and discarding classes we do not track.
    """

    @property
    @abstractmethod
    def classes(self) -> Sequence[ObjectClass]: ...

    @abstractmethod
    def detect(self, frames: Sequence[Frame]) -> list[Detections]:
        """Batch inference; returns one ``Detections`` per input frame."""

    def warmup(self) -> None:  # noqa: B027 - optional hook
        """Run a dummy inference so the first real frame is not slow."""


VIDEO_SOURCES: Registry[VideoSource] = Registry("video_source")
PREPROCESSORS: Registry[FramePreprocessor] = Registry("preprocessor")
DETECTORS: Registry[Detector] = Registry("detector")

# Implementations are declared lazily so heavy deps (torch) load only on use.
DETECTORS.register_lazy("yolo", "traffic_intel.computer_vision.detectors.yolo")
DETECTORS.register_lazy("precomputed", "traffic_intel.computer_vision.detectors.precomputed")
DETECTORS.register_lazy("synthetic", "traffic_intel.synthetic.sources")
VIDEO_SOURCES.register_lazy("opencv", "traffic_intel.computer_vision.sources.opencv_source")
VIDEO_SOURCES.register_lazy("synthetic", "traffic_intel.synthetic.sources")
for _name in ("roi_mask", "clahe", "undistort"):
    PREPROCESSORS.register_lazy(_name, "traffic_intel.computer_vision.preprocessing")
