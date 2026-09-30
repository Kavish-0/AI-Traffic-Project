"""Hot-path data types for the perception pipeline.

These are plain dataclasses holding NumPy arrays in struct-of-arrays layout.
They are created for *every processed frame*, so they deliberately avoid
Pydantic validation overhead. They never cross a process boundary or the API;
anything that does is converted to a model in ``core.schemas`` first.

Coordinate conventions
----------------------
* Image space: pixels of the *processed* frame (after any configured resize),
  origin top-left, ``xyxy`` = (x1, y1, x2, y2).
* Ground space: metres on the camera's local road plane, produced by the
  calibration homography. The frame is defined by the calibration points:
  ``+x`` is 90 deg clockwise from ``+y`` (i.e. "to the right" when facing +y),
  and ``bearing_deg`` gives the compass direction of ``+y``. The direction of
  travel is configured separately (``CameraConfig.travel_direction`` or lane
  directions) and need not equal +y.
* Geo space: WGS84 (EPSG:4326) lon/lat, only used for map/PostGIS output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import numpy.typing as npt

from traffic_intel.core.enums import class_from_id

FloatArray = npt.NDArray[np.float32]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]


@dataclass(slots=True)
class Frame:
    camera_id: str
    index: int                 # frame index in the source stream
    timestamp: datetime        # *source* time (UTC, tz-aware); see docs "Clock"
    image: npt.NDArray[np.uint8]  # H x W x 3, BGR

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.image.shape[0]), int(self.image.shape[1])


def _empty_boxes() -> FloatArray:
    return np.zeros((0, 4), dtype=np.float32)


def _empty_f() -> FloatArray:
    return np.zeros((0,), dtype=np.float32)


def _empty_i() -> IntArray:
    return np.zeros((0,), dtype=np.int64)


@dataclass(frozen=True, slots=True)
class Detection:
    """Per-object *view* of one row of ``Detections`` (for logging, tests, UIs).
    The pipeline itself works on the array form."""

    class_id: int
    confidence: float
    bbox: tuple[float, float, float, float]   # xyxy pixels
    timestamp: datetime

    @property
    def class_name(self) -> str:
        return class_from_id(self.class_id).value


@dataclass(slots=True)
class Detections:
    """All detections in one frame. ``class_id`` indexes ``OBJECT_CLASSES``
    (never raw model class ids; see ``computer_vision.class_mapping``)."""

    xyxy: FloatArray = field(default_factory=_empty_boxes)       # (N, 4)
    confidence: FloatArray = field(default_factory=_empty_f)     # (N,)
    class_id: IntArray = field(default_factory=_empty_i)         # (N,)

    def __len__(self) -> int:
        return int(self.xyxy.shape[0])

    def select(self, mask: BoolArray | IntArray) -> Detections:
        return Detections(self.xyxy[mask], self.confidence[mask], self.class_id[mask])

    def to_list(self, timestamp: datetime) -> list[Detection]:
        return [
            Detection(
                class_id=int(self.class_id[i]),
                confidence=float(self.confidence[i]),
                bbox=tuple(float(v) for v in self.xyxy[i]),  # type: ignore[arg-type]
                timestamp=timestamp,
            )
            for i in range(len(self))
        ]

    @staticmethod
    def concat(items: list[Detections]) -> Detections:
        if not items:
            return Detections()
        return Detections(
            np.concatenate([d.xyxy for d in items]).astype(np.float32),
            np.concatenate([d.confidence for d in items]).astype(np.float32),
            np.concatenate([d.class_id for d in items]).astype(np.int64),
        )


@dataclass(slots=True)
class Tracks:
    """Confirmed tracks in one frame (tracker output)."""

    track_id: IntArray = field(default_factory=_empty_i)         # (N,)
    xyxy: FloatArray = field(default_factory=_empty_boxes)       # (N, 4)
    confidence: FloatArray = field(default_factory=_empty_f)     # (N,)
    class_id: IntArray = field(default_factory=_empty_i)         # (N,)
    # Filled by the ground projector (None until projected):
    ground_xy: FloatArray | None = None                          # (N, 2) metres

    def __len__(self) -> int:
        return int(self.track_id.shape[0])

    def anchor_points(self) -> FloatArray:
        """Bottom-centre of each box: the point assumed to touch the road plane."""
        x = (self.xyxy[:, 0] + self.xyxy[:, 2]) / 2.0
        y = self.xyxy[:, 3]
        return np.stack([x, y], axis=1).astype(np.float32)

    def centers(self) -> FloatArray:
        """Geometric box centre (for drawing; NOT used for ground projection)."""
        return np.stack(
            [(self.xyxy[:, 0] + self.xyxy[:, 2]) / 2.0, (self.xyxy[:, 1] + self.xyxy[:, 3]) / 2.0],
            axis=1,
        ).astype(np.float32)

    def select(self, mask: BoolArray | IntArray) -> Tracks:
        return Tracks(
            self.track_id[mask],
            self.xyxy[mask],
            self.confidence[mask],
            self.class_id[mask],
            None if self.ground_xy is None else self.ground_xy[mask],
        )


@dataclass(slots=True)
class TrackedFrame:
    """Unit of work flowing from perception into the traffic feature engine."""

    camera_id: str
    frame_index: int
    timestamp: datetime
    image_shape: tuple[int, int]      # (height, width) of the processed frame
    detections: Detections
    tracks: Tracks
    processing_ms: dict[str, float] = field(default_factory=dict)  # per-stage latency
