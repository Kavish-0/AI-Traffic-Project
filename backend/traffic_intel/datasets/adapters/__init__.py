"""Dataset adapters: raw public-dataset formats -> canonical project representations (see base.py)."""

from traffic_intel.datasets.adapters.base import (
    DATASET_ADAPTERS,
    AnnotatedFrame,
    DatasetAdapter,
    DetectionTrackingAdapter,
    IncidentAdapter,
    IncidentAnnotation,
    SequenceInfo,
    TimeSeriesAdapter,
    adapter_for,
)

__all__ = ["DATASET_ADAPTERS", "AnnotatedFrame", "DatasetAdapter", "DetectionTrackingAdapter", "IncidentAdapter",
           "IncidentAnnotation", "SequenceInfo", "TimeSeriesAdapter", "adapter_for"]
