"""Domain enumerations shared by every module.

Values are lowercase strings so they serialise cleanly to JSON, Postgres enums
and YAML configs.
"""

from __future__ import annotations

from enum import StrEnum


class ObjectClass(StrEnum):
    CAR = "car"
    MOTORCYCLE = "motorcycle"
    BUS = "bus"
    TRUCK = "truck"
    BICYCLE = "bicycle"
    PEDESTRIAN = "pedestrian"

    @property
    def is_vehicle(self) -> bool:
        return self is not ObjectClass.PEDESTRIAN


# Canonical class-id order used in all hot-path arrays (Detections.class_id,
# Tracks.class_id): class_id == OBJECT_CLASSES.index(cls). Detector adapters map
# model-specific ids into this order (computer_vision/class_mapping.py).
OBJECT_CLASSES: tuple[ObjectClass, ...] = tuple(ObjectClass)
VEHICLE_CLASS_IDS: frozenset[int] = frozenset(
    i for i, c in enumerate(OBJECT_CLASSES) if c.is_vehicle
)


# Boolean lookup: IS_VEHICLE[class_ids] is a fast vehicle mask for id arrays.
IS_VEHICLE: tuple[bool, ...] = tuple(c.is_vehicle for c in OBJECT_CLASSES)


def class_from_id(class_id: int) -> ObjectClass:
    return OBJECT_CLASSES[int(class_id)]


def class_id_of(cls: ObjectClass | str) -> int:
    return OBJECT_CLASSES.index(ObjectClass(cls))


# Typical footprint length in metres; used for bumper-to-bumper gaps, queue tail
# extent and space-occupancy. Overridable in feature_engine.params.vehicle_length_m.
DEFAULT_VEHICLE_LENGTH_M: dict[ObjectClass, float] = {
    ObjectClass.CAR: 4.5,
    ObjectClass.MOTORCYCLE: 2.0,
    ObjectClass.BUS: 12.0,
    ObjectClass.TRUCK: 9.0,
    ObjectClass.BICYCLE: 1.8,
    ObjectClass.PEDESTRIAN: 0.5,
}


# Passenger Car Unit factors (IRC:106 style defaults). Used to weight counts so
# that heterogeneous traffic (many motorcycles, few buses) is comparable.
# Overridable in configs/pipeline.yaml -> feature_engine.params.pcu.
DEFAULT_PCU: dict[ObjectClass, float] = {
    ObjectClass.CAR: 1.0,
    ObjectClass.MOTORCYCLE: 0.5,
    ObjectClass.BUS: 3.0,
    ObjectClass.TRUCK: 3.0,
    ObjectClass.BICYCLE: 0.4,
    ObjectClass.PEDESTRIAN: 0.0,
}


class CongestionLevel(StrEnum):
    FREE = "free"
    MODERATE = "moderate"
    HIGH = "high"
    SEVERE = "severe"


class ZoneType(StrEnum):
    ROI = "roi"                  # analysis region; everything outside is ignored
    LANE = "lane"                # one lane polygon with a travel direction
    ENTRY_LINE = "entry_line"    # crossing -> inflow
    EXIT_LINE = "exit_line"      # crossing -> outflow (usually the stop line)
    QUEUE_ZONE = "queue_zone"    # where queues are expected to form
    VIRTUAL_LOOP = "virtual_loop"  # occupancy measurement area


class AnomalyType(StrEnum):
    SUDDEN_SLOWDOWN = "sudden_slowdown"
    STOPPED_VEHICLE = "stopped_vehicle"
    WRONG_WAY = "wrong_way"
    ABNORMAL_TRAJECTORY = "abnormal_trajectory"
    SUDDEN_QUEUE = "sudden_queue"
    UNUSUAL_DENSITY = "unusual_density"
    STATISTICAL_OUTLIER = "statistical_outlier"   # Isolation Forest / AE / z-score / EWMA
    COLLISION_LIKE = "collision_like"             # converging tracks then both stop
    UNUSUAL_FLOW = "unusual_flow"                 # inflow/outflow imbalance (vehicles accumulating)


class AnomalyScope(StrEnum):
    TRACK = "track"      # about one or more specific tracked objects
    SERIES = "series"    # about the aggregated feature time-series of an approach


class Severity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class IncidentStatus(StrEnum):
    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    DISMISSED = "dismissed"   # operator feedback: false alarm -> training label


class ModelStatus(StrEnum):
    CANDIDATE = "candidate"
    PRODUCTION = "production"
    ARCHIVED = "archived"


class ModelTask(StrEnum):
    DETECTION = "detection"
    CONGESTION = "congestion"
    FORECASTING = "forecasting"
    ANOMALY = "anomaly"
    INCIDENT_FUSION = "incident_fusion"
