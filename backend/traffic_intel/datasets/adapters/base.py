"""Dataset adapters: raw public-dataset formats -> the project's canonical representation.

The rest of the system never reads a raw dataset format. Three canonical forms, one per task
family, all built from types that already exist in the project:

    DetectionTrackingAdapter  -> SequenceInfo + AnnotatedFrame (ground truth as core ``Tracks``:
                                 track id, xyxy pixels, project ObjectClass ids) - the same objects
                                 the detector / tracker / trajectory store / feature engine use
    TimeSeriesAdapter         -> forecasting ``TimeSeriesDataset`` (long frame: series_id, timestamp
                                 UTC, columns from core/feature_schema.py - validated)
    IncidentAdapter           -> IncidentAnnotation intervals (dataset, video, start_s, end_s,
                                 source label) - NOT expanded to frames until an experiment needs it

Class names of a dataset are never assumed to equal YOLO/COCO names: each detection dataset
maps its own labels to ``core.enums.ObjectClass`` through the manifest's ``classes.class_map``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel

from traffic_intel.core.enums import ObjectClass, class_id_of
from traffic_intel.core.registry import Registry
from traffic_intel.core.types import Detections, Tracks
from traffic_intel.datasets.manifest import DatasetEntry, dataset_root, load_manifest, require, validate


class DatasetAdapter:
    """Base: knows its manifest entry and where the data lives; checks presence."""

    def __init__(self, entry: DatasetEntry | str, root: Path | str | None = None, **params: Any) -> None:
        self.entry = load_manifest().get(entry) if isinstance(entry, str) else entry
        self.root = Path(root) if root is not None else dataset_root(self.entry)
        self.params = {**self.entry.adapter.params, **self.entry.params, **params}

    @property
    def dataset_id(self) -> str:
        return self.entry.dataset_id

    def available(self) -> bool:
        return validate(self.entry, self.root).ok

    def check(self) -> Path:
        """Raise DatasetNotAvailableError (with download/licence instructions) if files are missing."""
        return require(self.entry, self.root)


# --------------------------------------------------------------------------- detection / tracking
@dataclass
class SequenceInfo:
    sequence_id: str
    fps: float
    frame_size: tuple[int, int] | None             # (width, height)
    n_frames: int
    media: Path | None = None                      # video file or image directory
    metadata: dict[str, Any] = field(default_factory=dict)   # camera id, scenario, time offset, weather ...


@dataclass
class AnnotatedFrame:
    """Ground truth of one frame, in the pipeline's own types."""

    sequence_id: str
    frame_index: int                               # 0-based index into the sequence
    time_s: float                                  # frame_index / fps
    tracks: Tracks                                 # confidence = 1.0; class ids = project ObjectClass ids
    ignore_regions: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), np.float32))  # xyxy
    unmapped: int = 0                              # boxes whose source class has no project class

    def as_detections(self) -> Detections:
        return Detections(self.tracks.xyxy, self.tracks.confidence, self.tracks.class_id)


class ClassMapper:
    """Source label -> project ObjectClass id, from the manifest's ``classes.class_map``."""

    def __init__(self, class_map: dict[str, str]) -> None:
        self.map = {k.lower(): class_id_of(ObjectClass(v)) for k, v in class_map.items()}

    def __call__(self, label: str) -> int | None:
        return self.map.get(str(label).lower())


def make_tracks(ids: list[int], boxes: list[tuple[float, float, float, float]], class_ids: list[int]) -> Tracks:
    n = len(ids)
    return Tracks(track_id=np.asarray(ids, np.int64), xyxy=np.asarray(boxes, np.float32).reshape(n, 4),
                  confidence=np.ones(n, np.float32), class_id=np.asarray(class_ids, np.int64))


class DetectionTrackingAdapter(DatasetAdapter, ABC):
    # official splits this adapter can load separately (params split=...); first = default
    splits: tuple[str, ...] = ("train",)

    def __init__(self, entry: DatasetEntry | str, root: Path | str | None = None, **params: Any) -> None:
        super().__init__(entry, root, **params)
        classes = self.entry.classes or {}
        self.classes = ClassMapper(params.get("class_map") or classes.get("class_map") or {})

    @abstractmethod
    def sequences(self) -> list[SequenceInfo]: ...

    @abstractmethod
    def frames(self, sequence_id: str) -> Iterator[AnnotatedFrame]: ...


# --------------------------------------------------------------------------- time series
class TimeSeriesAdapter(DatasetAdapter, ABC):
    @abstractmethod
    def load(self, **params: Any) -> Any:
        """-> forecasting.data.sources.TimeSeriesDataset (validated against core/feature_schema.py)."""


# --------------------------------------------------------------------------- incidents
class IncidentAnnotation(BaseModel):
    dataset_id: str
    video_id: str
    start_s: float
    end_s: float | None                      # None when the source gives only a start time
    source_label: str                        # verbatim from the dataset (e.g. "anomaly") - never re-labelled "accident"
    notes: str = ""

    def frame_interval(self, fps: float) -> tuple[int, int | None]:
        """Only when an experiment needs frames."""
        return int(self.start_s * fps), (None if self.end_s is None else int(self.end_s * fps))


class IncidentAdapter(DatasetAdapter, ABC):
    @abstractmethod
    def annotations(self) -> list[IncidentAnnotation]: ...

    @abstractmethod
    def videos(self) -> dict[str, Path]: ...


# --------------------------------------------------------------------------- registry
DATASET_ADAPTERS: Registry[DatasetAdapter] = Registry("dataset_adapter")
DATASET_ADAPTERS.register_lazy("ua_detrac", "traffic_intel.datasets.adapters.uadetrac")
DATASET_ADAPTERS.register_lazy("aicity_mtmc", "traffic_intel.datasets.adapters.aicity")
DATASET_ADAPTERS.register_lazy("aicity_anomaly", "traffic_intel.datasets.adapters.aicity_anomaly")
for _n in ("metr_la", "pems_bay", "pems08", "synthetic", "own_recordings"):
    DATASET_ADAPTERS.register_lazy(_n, "traffic_intel.datasets.adapters.timeseries")


def adapter_for(dataset_id: str, root: Path | str | None = None, **params: Any) -> DatasetAdapter:
    """Build the adapter named in the manifest entry of ``dataset_id``."""
    entry = load_manifest().get(dataset_id)
    if entry.adapter.type == "none":
        raise ValueError(f"dataset '{dataset_id}' has no adapter (it is consumed directly, e.g. weather)")
    factory = DATASET_ADAPTERS.get(entry.adapter.type)
    return factory(entry, root, **params)
