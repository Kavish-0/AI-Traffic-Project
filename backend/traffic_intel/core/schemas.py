"""Boundary data models (Pydantic v2).

Everything that crosses a module boundary at *window* rate (not frame rate),
goes onto the event bus, into the database layer, or out through the API is
one of these models. The frontend mirrors them in ``frontend/src/api/types.ts``.

Identifiers are human-readable slugs from the network config
(e.g. ``cam_01``, ``junction_a``, ``junction_a_north``) so logs, configs, the
DB and the UI all use the same keys.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from traffic_intel.core.enums import (
    AnomalyScope,
    AnomalyType,
    CongestionLevel,
    IncidentStatus,
    ObjectClass,
    Severity,
)


class Schema(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


# --------------------------------------------------------------------------- #
# Module 1 - perception export records (offline logging / datasets)
# --------------------------------------------------------------------------- #
class DetectionRecord(Schema):
    """One tracked detection, as written to Parquet for offline analysis."""

    camera_id: str
    timestamp: datetime
    frame_index: int
    track_id: int | None
    object_class: ObjectClass
    confidence: float = Field(ge=0.0, le=1.0)
    bbox_xyxy: tuple[float, float, float, float]
    ground_xy_m: tuple[float, float] | None = None


class TrackSummary(Schema):
    """A completed trajectory, emitted when a track is lost/finished."""

    camera_id: str
    approach_id: str
    track_id: int
    object_class: ObjectClass
    first_seen: datetime
    last_seen: datetime
    path_ground_m: list[tuple[float, float]]    # down-sampled polyline
    mean_speed_kmh: float | None
    max_stopped_s: float
    crossed_entry: bool
    crossed_exit: bool
    path_image_px: list[tuple[float, float]] = []   # anchor points, same sampling
    distance_m: float = 0.0                          # integrated smoothed speed
    displacement_along_travel_m: float | None = None  # >0 = moved in travel direction


# --------------------------------------------------------------------------- #
# Module 2 - traffic state (one row per approach per window)
# --------------------------------------------------------------------------- #
class TrafficFeatures(Schema):
    """Aggregated traffic measurements for one approach over one window.

    Definitions (see docs/TRAFFIC_STATE_ESTIMATION.md):
      density_veh_km     vehicles in ROI / (ROI length km * lanes), time-averaged
      density_ratio      density_veh_km / jam density (0..1+) - the "82%" figure
      occupancy_pct      % of window the virtual loop is covered by a vehicle
      flow_veh_h         exit-line crossings scaled to vehicles/hour
      avg_speed_kmh      space-mean speed: mean over (vehicle, instant) samples in the ROI
      queue_length_m     distance from stop line to the tail of the queue, time-averaged
    """

    approach_id: str
    camera_id: str
    window_start: datetime
    window_s: float
    vehicle_count: int                         # unique vehicle tracks seen in ROI
    count_by_class: dict[ObjectClass, int]
    pcu_count: float
    avg_speed_kmh: float | None                # None if no reliable speed samples
    speed_p85_kmh: float | None
    density_veh_km: float
    density_ratio: float
    occupancy_pct: float
    flow_veh_h: float
    inflow: int
    outflow: int
    queue_length_m: float
    stopped_vehicles: int
    avg_gap_m: float | None
    pedestrian_count: int
    quality: float = Field(ge=0.0, le=1.0, description="fraction of window with valid frames")
    median_speed_kmh: float | None = None
    # Per-vehicle mean-speed histogram, keys like "0-5", "5-10", ..., "80+".
    speed_distribution: dict[str, int] = {}
    # Completed-trajectory statistics (n, mean duration/distance/straightness, reverse moves).
    trajectory_stats: dict[str, float] = {}
    # Additional features from pluggable calculators (not yet promoted to columns).
    extra: dict[str, float] = {}


class ContextFeatures(Schema):
    timestamp: datetime
    hour: int
    day_of_week: int
    is_holiday: bool
    weather: str | None = None               # e.g. "clear", "rain", "fog"
    temperature_c: float | None = None
    precipitation_mm: float | None = None
    visibility_km: float | None = None


# --------------------------------------------------------------------------- #
# Module 3 - congestion
# --------------------------------------------------------------------------- #
class CongestionAssessment(Schema):
    approach_id: str
    timestamp: datetime
    level: CongestionLevel
    score: float = Field(ge=0.0, le=1.0)      # continuous severity for charts
    classifier: str                           # e.g. "rule_based", "xgboost"
    model_version: str | None = None
    explanation: list[str] = []               # human-readable reasons / rules fired


class TrafficState(Schema):
    """Flat 'what is happening now' record for one approach: measurements plus
    the congestion verdict. Output of the perception + traffic-state layer."""

    timestamp: datetime                        # end of the measurement window
    window_start: datetime
    window_s: float
    is_live: bool                              # True = rolling live window, False = fixed window
    camera_id: str
    approach_id: str
    junction_id: str
    road_id: str                               # road_segment_id

    vehicle_count: int
    car_count: int
    motorcycle_count: int
    bus_count: int
    truck_count: int
    bicycle_count: int
    pedestrian_count: int

    average_speed_kmh: float | None
    median_speed_kmh: float | None
    density_veh_km: float
    density_pct: float                         # density_ratio * 100
    occupancy_pct: float
    flow_veh_h: float
    inflow: int
    outflow: int
    queue_length_m: float
    stopped_vehicle_count: int

    traffic_state: CongestionLevel
    congestion_score: float
    explanation: list[str]
    quality: float


# --------------------------------------------------------------------------- #
# Module 4 - forecasting
# --------------------------------------------------------------------------- #
# Must equal core.feature_schema.FORECAST_TARGETS (checked by a test).
ForecastTarget = Literal["density_ratio", "avg_speed_kmh", "flow_veh_h", "occupancy_pct", "queue_length_m"]


class ForecastPoint(Schema):
    horizon_min: int                          # 5, 15, 30
    target_time: datetime
    target: ForecastTarget
    value: float
    lower: float | None = None                # prediction interval, if model supports it
    upper: float | None = None


class Forecast(Schema):
    approach_id: str
    issued_at: datetime
    model_name: str
    model_version: str
    points: list[ForecastPoint]
    predicted_level: dict[int, CongestionLevel] = {}   # horizon_min -> level


# --------------------------------------------------------------------------- #
# Module 5 - anomaly / incident
# --------------------------------------------------------------------------- #
class AnomalySignal(Schema):
    """Output of a single anomaly detector. Signals are evidence, not incidents."""

    approach_id: str
    camera_id: str | None
    timestamp: datetime
    type: AnomalyType
    scope: AnomalyScope
    detector: str                             # e.g. "ewma_speed", "isolation_forest"
    score: float = Field(ge=0.0, le=1.0)      # normalised strength of evidence
    track_ids: list[int] = []
    ground_xy_m: tuple[float, float] | None = None
    details: dict[str, Any] = {}


class IncidentAssessment(Schema):
    """Fused incident hypothesis. ``probability`` is a calibrated likelihood only
    once a fusion model has been fitted on labelled events; before that it is a
    heuristic score and the UI labels it as such (``calibrated=False``)."""

    incident_id: str | None = None
    approach_id: str
    timestamp: datetime
    probability: float = Field(ge=0.0, le=1.0)
    calibrated: bool = False
    severity: Severity
    status: IncidentStatus = IncidentStatus.OPEN
    lon: float | None = None
    lat: float | None = None
    location_description: str | None = None
    contributing: list[AnomalySignal] = []
    fusion_model: str = "weighted_logistic"
    # Incident intelligence (Phase 6). The type comes from the configurable taxonomy
    # (configs/anomaly.yaml -> incident_types), e.g. normal, sudden_slowdown, stopped_vehicle,
    # wrong_way, abnormal_queue, possible_incident.
    incident_type: str = "normal"
    label: str = ""                           # human label, e.g. "Possible incident"
    message: str = ""                         # always hedged, e.g. "Incident probability: 87% (uncertain)"
    junction_id: str | None = None
    road_id: str | None = None
    risk_level: Severity | None = None
    evidence: dict[str, float] = {}           # fused evidence per signal family (0..1)
    first_detected: datetime | None = None


# --------------------------------------------------------------------------- #
# Risk engine
# --------------------------------------------------------------------------- #
class RiskAssessment(Schema):
    approach_id: str
    timestamp: datetime
    risk_score: float = Field(ge=0.0, le=100.0)
    level: Severity
    components: dict[str, float]              # e.g. {"congestion": 0.8, "forecast_trend": 0.4, ...}
    explanation: list[str] = []
    reason_codes: list[str] = []              # machine-readable, e.g. SEVERE_CONGESTION_FORECAST
    junction_id: str | None = None
    road_id: str | None = None


# --------------------------------------------------------------------------- #
# Aggregated view (what the dashboard renders per road/approach)
# --------------------------------------------------------------------------- #
class ApproachSnapshot(Schema):
    approach_id: str
    junction_id: str
    road_segment_id: str
    timestamp: datetime
    features: TrafficFeatures | None
    congestion: CongestionAssessment | None
    forecast: Forecast | None
    incident: IncidentAssessment | None
    risk: RiskAssessment | None


# --------------------------------------------------------------------------- #
# Module 8 - what-if simulation
# --------------------------------------------------------------------------- #
class SimulationRequest(Schema):
    closed_segment_ids: list[str] = []
    capacity_factors: dict[str, float] = {}   # segment_id -> multiplier (e.g. 0.5 = lane closure)
    demand_factor: float = Field(default=1.0, gt=0.0)
    base_time: datetime | None = None         # None = latest observed state


class SegmentImpact(Schema):
    segment_id: str
    name: str = ""
    status: str = "open"                      # open | closed | reduced
    baseline_observed: bool = False           # baseline flow measured (camera) vs assumed
    baseline_flow_veh_h: float
    scenario_flow_veh_h: float
    flow_change_pct: float
    baseline_congestion_prob: float
    scenario_congestion_prob: float
    scenario_level: CongestionLevel


class JunctionImpact(Schema):
    junction_id: str
    name: str = ""
    baseline_congestion_prob: float
    scenario_congestion_prob: float


class SimulationResult(Schema):
    scenario_id: str
    request: SimulationRequest
    created_at: datetime
    impacts: list[SegmentImpact]
    unserved_demand_veh_h: float              # demand with no alternative path
    assumptions: list[str]
    junctions: list[JunctionImpact] = []
    summary: list[str] = []                   # e.g. "Road A (southbound): +23% flow"


# --------------------------------------------------------------------------- #
# Event bus envelope
# --------------------------------------------------------------------------- #
class BusMessage(Schema):
    topic: str
    timestamp: datetime
    source: str                               # producing component, e.g. "camera_worker:cam_01"
    payload: dict[str, Any]
