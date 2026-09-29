"""Intermediate data of the feature engine.

The engine reduces every processed frame to a ``FrameSample`` (only objects
inside the ROI) plus line-crossing events. Window metrics are then computed
from these by independent ``FeatureCalculator``s, so any metric can be
recomputed for any time range (fixed 60 s windows, rolling live window).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from traffic_intel.core.enums import (
    DEFAULT_PCU,
    DEFAULT_VEHICLE_LENGTH_M,
    IS_VEHICLE,
    OBJECT_CLASSES,
    ZoneType,
)
from traffic_intel.core.schemas import TrackSummary
from traffic_intel.core.types import BoolArray, FloatArray, IntArray
from traffic_intel.tracking.trajectory import TrajectoryConfig

_IS_VEHICLE = np.array(IS_VEHICLE, dtype=bool)


class FeatureEngineConfig(BaseModel):
    """``configs/pipeline.yaml -> feature_engine.params``."""

    model_config = ConfigDict(extra="forbid")

    window_s: float = Field(60.0, gt=0)            # fixed window (persisted / ML)
    live_window_s: float = Field(5.0, gt=0)        # rolling window for the live view
    trajectory: TrajectoryConfig = TrajectoryConfig()
    # stopped / queue definitions
    stopped_min_s: float = Field(2.0, ge=0)        # stopped = speed < trajectory.stopped_speed_kmh for >= this
    queue_speed_kmh: float = Field(7.0, ge=0)      # queued = slower than this (stopped or crawling)
    queue_head_max_m: float = Field(15.0, ge=0)    # first queued vehicle must be this close to the stop line
    queue_max_gap_m: float = Field(15.0, ge=0)     # max bumper gap between consecutive queued vehicles
    queue_view_limit_ratio: float = Field(0.9, gt=0, le=1)  # tail beyond this x ROI length -> "queue may exceed view"
    # density / occupancy / counting
    jam_density_veh_km_lane: float = Field(150.0, gt=0)  # in PCU if use_pcu
    use_pcu: bool = True
    pcu: dict[str, float] = {}                     # overrides of DEFAULT_PCU
    vehicle_length_m: dict[str, float] = {}        # overrides of DEFAULT_VEHICLE_LENGTH_M
    min_track_points: int = Field(3, ge=1)         # samples in ROI before a track is counted
    # ... and it must stay in the ROI *continuously* this long: anchors of distant vehicles
    # jitter across the far ROI edge (pixels = metres there) and would otherwise be counted.
    min_presence_s: float = Field(1.0, ge=0)
    min_speed_samples: int = Field(3, ge=1)        # else avg speed = None
    edge_margin_px: float = Field(4.0, ge=0)       # boxes touching the frame edge -> anchor unreliable
    lane_width_m: float = Field(3.5, gt=0)         # lane assignment fallback (no lane zones)
    # Part of the vehicle marked by the box anchor (bottom-centre): "auto" derives it from
    # the camera geometry (front bumper if traffic approaches the camera, else rear).
    anchor_reference: Literal["auto", "front", "rear", "center"] = "auto"
    speed_histogram_edges_kmh: list[float] = [0, 5, 10, 20, 30, 40, 50, 60, 80]
    calculators: list[str] | None = None           # None = all default calculators, in order

    def pcu_table(self) -> FloatArray:
        return np.array([self.pcu.get(c.value, DEFAULT_PCU[c]) for c in OBJECT_CLASSES], dtype=np.float32)

    def length_table(self) -> FloatArray:
        return np.array(
            [self.vehicle_length_m.get(c.value, DEFAULT_VEHICLE_LENGTH_M[c]) for c in OBJECT_CLASSES],
            dtype=np.float32,
        )


@dataclass(slots=True)
class FrameSample:
    """Objects inside the ROI at one processed frame. All arrays have length N."""

    t: float
    track_id: IntArray
    class_id: IntArray
    xyxy: FloatArray
    in_measure: BoolArray        # inside ROI AND the calibrated area (metric features)
    speed_kmh: FloatArray        # NaN = unknown
    stopped: BoolArray
    queued: BoolArray
    in_queue_zone: BoolArray
    lane: IntArray               # lane index, -1 = unknown
    along_m: FloatArray          # metres upstream of the stop line (NaN = unknown)
    lateral_m: FloatArray
    loop_occupied: bool

    @property
    def is_vehicle(self) -> BoolArray:
        return _IS_VEHICLE[self.class_id]


@dataclass(frozen=True, slots=True)
class CrossingEvent:
    t: float
    track_id: int
    class_id: int
    line_id: str
    kind: ZoneType               # ENTRY_LINE | EXIT_LINE
    forward: bool                # moved in the approach's travel direction


@dataclass(frozen=True, slots=True)
class GeometryFacts:
    roi_length_m: float
    lanes: int
    has_loop: bool
    has_entry_line: bool
    has_exit_line: bool
    anchor_reference: str = "center"     # front | rear | center


def tail_offset(anchor: str, length: FloatArray) -> FloatArray:
    """Metres from the anchor to the vehicle's *upstream* end (its tail in the queue)."""
    return {"front": length, "rear": 0.0 * length}.get(anchor, length / 2.0)


def head_offset(anchor: str, length: FloatArray) -> FloatArray:
    """Metres from the anchor to the vehicle's *downstream* end (its nose)."""
    return {"front": 0.0 * length, "rear": length}.get(anchor, length / 2.0)


@dataclass
class WindowData:
    start: float
    end: float
    samples: list[FrameSample]
    events: list[CrossingEvent]
    finished: list[TrackSummary]
    expected_frames: float
    geometry: GeometryFacts
    cfg: FeatureEngineConfig
    pcu: FloatArray
    lengths: FloatArray
    results: dict[str, object] = field(default_factory=dict)   # filled by calculators in order

    @property
    def duration_s(self) -> float:
        return self.end - self.start
