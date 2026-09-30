"""Relational schema (SQLAlchemy 2.0): PostgreSQL + PostGIS in production, SQLite locally.

Portable column types (``database/types.py``): geometry is PostGIS ``geometry(..., 4326)`` when
the database has PostGIS and WKT text otherwise; JSON becomes the binary JSON type on PostgreSQL.
The same models therefore serve a zero-setup SQLite file, a plain PostgreSQL and PostGIS.

Three groups of tables:

* Static network & cameras  - small, mostly loaded from configs/ by a sync
  command; geometry in EPSG:4326.
* Time series               - one row per approach per 60 s window (or per
  forecast issue). High volume -> composite PKs on (approach_id, time) and
  monthly range partitioning added in the Phase 3 migration.
* Events                    - anomalies, incidents, simulations, model registry.

Frame-level detections are NOT stored here; they go to Parquet files
(data/interim) for offline work. See docs/ARCHITECTURE.md section 8.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy import Enum as SAEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship

from traffic_intel.core.enums import (
    AnomalyScope,
    AnomalyType,
    CongestionLevel,
    IncidentStatus,
    ModelStatus,
    ModelTask,
    Severity,
    ZoneType,
)
from traffic_intel.database.base import Base
from traffic_intel.database.types import BigIntPK, GeoType, JSONType, enum_type


def pg_enum(enum_cls: type[StrEnum], name: str) -> SAEnum:
    return enum_type(enum_cls, name)


TS = DateTime(timezone=True)


# =========================================================================== #
# Static network
# =========================================================================== #
class Junction(Base):
    __tablename__ = "junction"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    signalized: Mapped[bool] = mapped_column(Boolean, default=True)
    geom: Mapped[Any] = mapped_column(GeoType("POINT"))

    approaches: Mapped[list[Approach]] = relationship(back_populates="junction")


class RoadNode(Base):
    __tablename__ = "road_node"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    junction_id: Mapped[str | None] = mapped_column(ForeignKey("junction.id"))
    geom: Mapped[Any] = mapped_column(GeoType("POINT"))


class RoadSegment(Base):
    """Directed road-graph edge (also the polyline coloured on the map)."""

    __tablename__ = "road_segment"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    from_node_id: Mapped[str] = mapped_column(ForeignKey("road_node.id"))
    to_node_id: Mapped[str] = mapped_column(ForeignKey("road_node.id"))
    lanes: Mapped[int] = mapped_column(Integer)
    speed_limit_kmh: Mapped[float] = mapped_column(Float)
    free_flow_speed_kmh: Mapped[float] = mapped_column(Float)
    capacity_veh_h: Mapped[float] = mapped_column(Float)
    length_m: Mapped[float] = mapped_column(Float)   # computed with ST_Length(geom::geography)
    geom: Mapped[Any] = mapped_column(GeoType("LINESTRING"))


class Approach(Base):
    __tablename__ = "approach"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    junction_id: Mapped[str] = mapped_column(ForeignKey("junction.id"), index=True)
    road_segment_id: Mapped[str] = mapped_column(ForeignKey("road_segment.id"))
    name: Mapped[str] = mapped_column(String(200))
    direction: Mapped[str] = mapped_column(String(32))

    junction: Mapped[Junction] = relationship(back_populates="approaches")


class Camera(Base):
    __tablename__ = "camera"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    approach_id: Mapped[str] = mapped_column(ForeignKey("approach.id"), index=True)
    source_uri: Mapped[str] = mapped_column(Text)
    process_fps: Mapped[float] = mapped_column(Float)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    geom: Mapped[Any | None] = mapped_column(GeoType("POINT"))     # None: runtime camera without a surveyed location


class CameraCalibration(Base):
    """Versioned so historic observations stay tied to the calibration that
    produced their speeds."""

    __tablename__ = "camera_calibration"
    __table_args__ = (UniqueConstraint("camera_id", "version"),)

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    camera_id: Mapped[str] = mapped_column(ForeignKey("camera.id"))
    version: Mapped[int] = mapped_column(Integer)
    homography: Mapped[list[list[float]]] = mapped_column(JSONType)       # 3x3 image->ground
    image_points_px: Mapped[list[list[float]]] = mapped_column(JSONType)
    ground_points_m: Mapped[list[list[float]]] = mapped_column(JSONType)
    origin: Mapped[Any] = mapped_column(GeoType("POINT"))
    bearing_deg: Mapped[float] = mapped_column(Float)
    reprojection_error_m: Mapped[float] = mapped_column(Float)
    valid_from: Mapped[datetime] = mapped_column(TS)
    created_at: Mapped[datetime] = mapped_column(TS, server_default=func.now())


class CameraZone(Base):
    __tablename__ = "camera_zone"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    camera_id: Mapped[str] = mapped_column(ForeignKey("camera.id"), index=True)
    type: Mapped[ZoneType] = mapped_column(pg_enum(ZoneType, "zone_type"))
    points_px: Mapped[list[list[float]]] = mapped_column(JSONType)       # image space (no SRID)
    lane_index: Mapped[int | None] = mapped_column(Integer)
    direction: Mapped[list[float] | None] = mapped_column(JSONType)
    geom: Mapped[Any | None] = mapped_column(GeoType())  # projected footprint


# =========================================================================== #
# Time series
# =========================================================================== #
class TrafficObservation(Base):
    __tablename__ = "traffic_observation"

    approach_id: Mapped[str] = mapped_column(ForeignKey("approach.id"), primary_key=True)
    window_start: Mapped[datetime] = mapped_column(TS, primary_key=True)
    camera_id: Mapped[str] = mapped_column(ForeignKey("camera.id"))
    window_s: Mapped[float] = mapped_column(Float)
    vehicle_count: Mapped[int] = mapped_column(Integer)
    count_by_class: Mapped[dict[str, int]] = mapped_column(JSONType)
    pcu_count: Mapped[float] = mapped_column(Float)
    avg_speed_kmh: Mapped[float | None] = mapped_column(Float)
    speed_p85_kmh: Mapped[float | None] = mapped_column(Float)
    density_veh_km: Mapped[float] = mapped_column(Float)
    density_ratio: Mapped[float] = mapped_column(Float)
    occupancy_pct: Mapped[float] = mapped_column(Float)
    flow_veh_h: Mapped[float] = mapped_column(Float)
    inflow: Mapped[int] = mapped_column(Integer)
    outflow: Mapped[int] = mapped_column(Integer)
    queue_length_m: Mapped[float] = mapped_column(Float)
    stopped_vehicles: Mapped[int] = mapped_column(Integer)
    avg_gap_m: Mapped[float | None] = mapped_column(Float)
    pedestrian_count: Mapped[int] = mapped_column(Integer)
    quality: Mapped[float] = mapped_column(Float)
    calibration_version: Mapped[int | None] = mapped_column(Integer)
    median_speed_kmh: Mapped[float | None] = mapped_column(Float)
    speed_distribution: Mapped[dict[str, int]] = mapped_column(JSONType, default=dict)
    trajectory_stats: Mapped[dict[str, float]] = mapped_column(JSONType, default=dict)
    extra: Mapped[dict[str, float]] = mapped_column(JSONType, default=dict)   # pluggable-calculator features


class ContextObservation(Base):
    __tablename__ = "context_observation"

    region: Mapped[str] = mapped_column(String(64), primary_key=True)  # e.g. network name
    ts: Mapped[datetime] = mapped_column(TS, primary_key=True)
    is_holiday: Mapped[bool] = mapped_column(Boolean)
    weather: Mapped[str | None] = mapped_column(String(32))
    temperature_c: Mapped[float | None] = mapped_column(Float)
    precipitation_mm: Mapped[float | None] = mapped_column(Float)
    visibility_km: Mapped[float | None] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(32))


class CongestionStateRow(Base):
    __tablename__ = "congestion_state"

    approach_id: Mapped[str] = mapped_column(ForeignKey("approach.id"), primary_key=True)
    ts: Mapped[datetime] = mapped_column(TS, primary_key=True)
    classifier: Mapped[str] = mapped_column(String(64), primary_key=True)
    level: Mapped[CongestionLevel] = mapped_column(pg_enum(CongestionLevel, "congestion_level"))
    score: Mapped[float] = mapped_column(Float)
    model_version: Mapped[str | None] = mapped_column(String(64))
    explanation: Mapped[list[str]] = mapped_column(JSONType)


class ForecastRow(Base):
    """One row per (approach, issue time, model, horizon, target). Keeping the
    issue time lets us evaluate forecasts against what actually happened."""

    __tablename__ = "forecast"

    approach_id: Mapped[str] = mapped_column(ForeignKey("approach.id"), primary_key=True)
    issued_at: Mapped[datetime] = mapped_column(TS, primary_key=True)
    model_name: Mapped[str] = mapped_column(String(64), primary_key=True)
    horizon_min: Mapped[int] = mapped_column(Integer, primary_key=True)
    target: Mapped[str] = mapped_column(String(32), primary_key=True)
    model_version: Mapped[str] = mapped_column(String(64))
    target_time: Mapped[datetime] = mapped_column(TS, index=True)
    value: Mapped[float] = mapped_column(Float)
    lower: Mapped[float | None] = mapped_column(Float)
    upper: Mapped[float | None] = mapped_column(Float)
    level: Mapped[str | None] = mapped_column(String(16))          # predicted congestion level at the horizon


class RiskAssessmentRow(Base):
    __tablename__ = "risk_assessment"

    approach_id: Mapped[str] = mapped_column(ForeignKey("approach.id"), primary_key=True)
    ts: Mapped[datetime] = mapped_column(TS, primary_key=True)
    risk_score: Mapped[float] = mapped_column(Float)
    level: Mapped[Severity] = mapped_column(pg_enum(Severity, "severity"))
    components: Mapped[dict[str, float]] = mapped_column(JSONType)
    explanation: Mapped[list[str]] = mapped_column(JSONType)
    reason_codes: Mapped[list[str]] = mapped_column(JSONType, default=list)
    # fused incident score of the window (also when no incident is open) - for the score-vs-time chart
    incident_score: Mapped[float | None] = mapped_column(Float)


# =========================================================================== #
# Events
# =========================================================================== #
class AnomalyEvent(Base):
    __tablename__ = "anomaly_event"

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    approach_id: Mapped[str] = mapped_column(ForeignKey("approach.id"), index=True)
    camera_id: Mapped[str | None] = mapped_column(ForeignKey("camera.id"))
    ts: Mapped[datetime] = mapped_column(TS, index=True)
    type: Mapped[AnomalyType] = mapped_column(pg_enum(AnomalyType, "anomaly_type"))
    scope: Mapped[AnomalyScope] = mapped_column(pg_enum(AnomalyScope, "anomaly_scope"))
    detector: Mapped[str] = mapped_column(String(64))
    score: Mapped[float] = mapped_column(Float)
    track_ids: Mapped[list[int]] = mapped_column(JSONType, default=list)
    details: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)
    geom: Mapped[Any | None] = mapped_column(GeoType("POINT"))
    incident_id: Mapped[str | None] = mapped_column(ForeignKey("incident.id"), index=True)


class Incident(Base):
    __tablename__ = "incident"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)   # uuid
    approach_id: Mapped[str] = mapped_column(ForeignKey("approach.id"), index=True)
    status: Mapped[IncidentStatus] = mapped_column(pg_enum(IncidentStatus, "incident_status"))
    probability: Mapped[float] = mapped_column(Float)                 # latest
    max_probability: Mapped[float] = mapped_column(Float)
    calibrated: Mapped[bool] = mapped_column(Boolean, default=False)
    severity: Mapped[Severity] = mapped_column(pg_enum(Severity, "severity"))
    fusion_model: Mapped[str] = mapped_column(String(64))
    # Incident intelligence (Phase 6)
    junction_id: Mapped[str | None] = mapped_column(ForeignKey("junction.id"), index=True)
    road_id: Mapped[str | None] = mapped_column(ForeignKey("road_segment.id"))
    incident_type: Mapped[str] = mapped_column(String(32))          # configurable taxonomy id
    label: Mapped[str] = mapped_column(String(64))
    message: Mapped[str] = mapped_column(Text)                        # hedged text incl. probability
    risk_level: Mapped[Severity | None] = mapped_column(pg_enum(Severity, "severity"))
    evidence: Mapped[dict[str, float]] = mapped_column(JSONType, default=dict)
    contributing: Mapped[list[dict[str, Any]]] = mapped_column(JSONType, default=list)   # signals (evidence)
    location_description: Mapped[str | None] = mapped_column(Text)
    geom: Mapped[Any | None] = mapped_column(GeoType("POINT"))
    first_detected: Mapped[datetime] = mapped_column(TS)
    last_updated: Mapped[datetime] = mapped_column(TS, index=True)
    resolved_at: Mapped[datetime | None] = mapped_column(TS)
    # Operator feedback -> ground-truth labels for calibrating the fuser.
    operator_label: Mapped[str | None] = mapped_column(String(32))   # "true_incident" | "false_alarm" | ...
    operator_note: Mapped[str | None] = mapped_column(Text)


class ModelRegistryRow(Base):
    __tablename__ = "model_registry"
    __table_args__ = (UniqueConstraint("task", "name", "version"),)

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    task: Mapped[ModelTask] = mapped_column(pg_enum(ModelTask, "model_task"))
    name: Mapped[str] = mapped_column(String(64))
    version: Mapped[str] = mapped_column(String(64))
    framework: Mapped[str] = mapped_column(String(32))
    artifact_uri: Mapped[str] = mapped_column(Text)
    feature_schema_hash: Mapped[str] = mapped_column(String(32))
    metrics: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)
    params: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)
    status: Mapped[ModelStatus] = mapped_column(pg_enum(ModelStatus, "model_status"))
    trained_at: Mapped[datetime] = mapped_column(TS)


class SimulationScenario(Base):
    __tablename__ = "simulation_scenario"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(TS, server_default=func.now())
    request: Mapped[dict[str, Any]] = mapped_column(JSONType)
    result: Mapped[dict[str, Any]] = mapped_column(JSONType)


# =========================================================================== #
# Live runtime state (small, upserted)
# =========================================================================== #
class ApproachLiveState(Base):
    """Latest ~1 s rolling live value per approach (one row, upserted). The live stream itself is
    not a time series in the database - only 60 s windows are (``traffic_observation``)."""

    __tablename__ = "approach_live_state"

    approach_id: Mapped[str] = mapped_column(ForeignKey("approach.id"), primary_key=True)
    ts: Mapped[datetime] = mapped_column(TS)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONType)         # TrafficState (is_live = true)


class CameraStatusRow(Base):
    """Heartbeat of a camera worker (written by the pipeline runtime every few seconds)."""

    __tablename__ = "camera_status"

    camera_id: Mapped[str] = mapped_column(ForeignKey("camera.id"), primary_key=True)
    state: Mapped[str] = mapped_column(String(16))                    # starting | running | stopped | error
    run_id: Mapped[str | None] = mapped_column(String(32))
    updated_at: Mapped[datetime] = mapped_column(TS)
    last_frame_at: Mapped[datetime | None] = mapped_column(TS)       # source time of the last processed frame
    frames: Mapped[int] = mapped_column(BigInteger, default=0)
    processing_fps: Mapped[float | None] = mapped_column(Float)
    detector: Mapped[str | None] = mapped_column(String(64))
    device: Mapped[str | None] = mapped_column(String(128))
    source_kind: Mapped[str | None] = mapped_column(String(16))       # file | webcam | stream | synthetic
    preview_path: Mapped[str | None] = mapped_column(Text)           # latest annotated JPEG (low rate)
    error: Mapped[str | None] = mapped_column(Text)
    details: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)
