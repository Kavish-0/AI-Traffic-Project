"""API response models (the HTTP/WebSocket contract), separate from internal domain objects.

Services build these from the store, the configuration and the model registry. Anything that is
not available is represented explicitly (``status`` / ``message`` fields, ``None`` values) - never
by placeholder numbers.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

Level = Literal["free", "moderate", "high", "severe"]
RiskLevel = Literal["low", "medium", "high", "critical"]
SourceKind = Literal["file", "webcam", "stream", "synthetic", "unknown"]


class TrafficStateOut(BaseModel):
    """One approach's measurements (a 60 s window, or the rolling ~5 s live value)."""

    timestamp: datetime
    window_start: datetime
    window_s: float
    is_live: bool
    camera_id: str | None
    approach_id: str
    junction_id: str | None
    road_id: str | None
    vehicle_count: int
    car_count: int = 0
    motorcycle_count: int = 0
    bus_count: int = 0
    truck_count: int = 0
    bicycle_count: int = 0
    pedestrian_count: int = 0
    average_speed_kmh: float | None
    median_speed_kmh: float | None = None
    density_veh_km: float
    density_pct: float
    occupancy_pct: float
    flow_veh_h: float
    inflow: int
    outflow: int
    queue_length_m: float
    stopped_vehicle_count: int
    traffic_state: Level | None
    congestion_score: float | None
    explanation: list[str] = []
    quality: float


class ModelInfo(BaseModel):
    task: str
    name: str
    version: str
    trained_on: str | None = None             # dataset the model was trained on
    resolution_s: int | None = None
    required_history: int | None = None       # consecutive windows needed before a forecast
    trained_at: datetime | None = None
    metrics: dict[str, Any] = {}


class ForecastHorizon(BaseModel):
    horizon_min: int
    target_time: datetime
    level: Level | None
    values: dict[str, float]                  # target -> value (avg_speed_kmh, density_ratio, ...)


class ForecastOut(BaseModel):
    approach_id: str
    status: Literal["ok", "insufficient_history", "no_model", "stale", "no_data"]
    message: str
    model: ModelInfo | None = None
    issued_at: datetime | None = None
    current_level: Level | None = None
    horizons: list[ForecastHorizon] = []
    history_windows: int = 0


class ForecastPointOut(BaseModel):
    issued_at: datetime
    target_time: datetime
    value: float
    level: str | None


class ForecastVsActual(BaseModel):
    approach_id: str
    horizon_min: int
    target: str
    forecast: list[ForecastPointOut]
    actual: list[dict[str, Any]]              # {timestamp, value}


class AnomalyOut(BaseModel):
    id: int
    approach_id: str
    camera_id: str | None
    junction_id: str | None
    road_id: str | None
    timestamp: datetime
    type: str
    scope: str
    detector: str
    score: float = Field(description="strength of evidence 0..1 - not a probability of an incident")
    track_ids: list[int] | None
    lon: float | None
    lat: float | None
    details: dict[str, Any] | None
    incident_id: str | None


class IncidentOut(BaseModel):
    incident_id: str
    timestamp: datetime
    first_detected: datetime | None
    resolved_at: datetime | None
    approach_id: str
    junction_id: str | None
    road_id: str | None
    incident_type: str
    label: str
    probability: float = Field(description="fused incident score; a probability only when calibrated=true")
    max_probability: float
    calibrated: bool
    score_label: str                          # "Incident probability" or "Incident probability: UNCALIBRATED"
    severity: str
    risk_level: str | None
    status: str
    message: str
    evidence: dict[str, float]
    lon: float | None
    lat: float | None
    location_description: str | None = None
    operator_label: str | None
    operator_note: str | None
    contributing: list[dict[str, Any]]
    uncertainty_note: str


class RiskOut(BaseModel):
    approach_id: str
    timestamp: datetime
    junction_id: str | None
    road_id: str | None
    risk_score: float
    level: RiskLevel
    reason_codes: list[str]
    components: dict[str, float]
    explanation: list[str]
    incident_score: float | None = None


class CameraOut(BaseModel):
    id: str
    name: str
    approach_id: str
    junction_id: str | None
    source_kind: SourceKind
    synthetic: bool                           # test fixture: numbers are NOT real-world measurements
    configured: bool                          # present in configs/cameras/*.yaml
    lon: float | None
    lat: float | None
    status: Literal["online", "offline", "error", "never_started"]
    state: str | None = None
    last_frame_at: datetime | None = None
    updated_at: datetime | None = None
    processing_fps: float | None = None
    frames: int | None = None
    detector: str | None = None
    device: str | None = None
    error: str | None = None
    preview_available: bool = False


class ApproachOut(BaseModel):
    id: str
    name: str
    direction: str
    junction_id: str
    road_id: str
    cameras: list[str]
    level: Level | None = None
    risk_level: RiskLevel | None = None
    active_incident: str | None = None


class JunctionOut(BaseModel):
    id: str
    name: str
    lon: float
    lat: float
    signalized: bool
    approaches: list[ApproachOut]
    active_incidents: int = 0
    worst_level: Level | None = None


class RoadOut(BaseModel):
    id: str
    name: str
    from_node: str
    to_node: str
    lanes: int
    speed_limit_kmh: float
    capacity_veh_h: float | None
    importance: float | None
    approach_id: str | None                    # monitored roads have an approach with a camera
    junction_id: str | None
    monitored: bool
    level: Level | None = None
    stale: bool = True
    coordinates: list[list[float]]


class SnapshotOut(BaseModel):
    """Everything the road panel shows for one approach."""

    approach: ApproachOut
    road: RoadOut | None
    cameras: list[CameraOut]
    live: TrafficStateOut | None
    live_stale: bool
    window: TrafficStateOut | None
    window_stale: bool
    forecast: ForecastOut
    incident: IncidentOut | None
    incident_score: float | None               # latest fused score (also when no incident is open)
    risk: RiskOut | None
    synthetic: bool


class SummaryOut(BaseModel):
    timestamp: datetime | None
    vehicles: int | None                       # sum of latest live counts (fresh approaches only)
    avg_speed_kmh: float | None                # vehicle-weighted
    congested_approaches: int                  # HIGH or SEVERE
    monitored_approaches: int
    fresh_approaches: int
    active_incidents: int
    synthetic_approaches: int


class ComponentStatus(BaseModel):
    ok: bool
    status: str
    detail: str = ""


class SystemStatusOut(BaseModel):
    backend: ComponentStatus
    database: ComponentStatus
    pipeline: ComponentStatus
    forecast_model: ComponentStatus
    cameras: ComponentStatus
    weather: ComponentStatus
    websocket_clients: int
    version: str
    env: str
    models: list[ModelInfo] = []
    weather_now: dict[str, Any] | None = None
