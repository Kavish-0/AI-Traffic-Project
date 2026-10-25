"""Configuration models and loaders.

Two layers:
* ``AppSettings`` - deployment settings from environment / ``.env``
  (DB URL, bus backend, paths). Never contains model hyper-parameters.
* YAML files under ``configs/`` - the *what*: road network, cameras,
  calibration, zones, pipeline component selection, rules. Validated into the
  models below at start-up so a typo fails fast instead of mid-stream.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from traffic_intel.core.enums import ZoneType
from traffic_intel.core.registry import ComponentSpec

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TI_", env_file=".env", extra="ignore")

    env: Literal["dev", "test", "prod"] = "dev"
    log_level: str = "INFO"
    log_format: Literal["text", "json"] = "text"   # json: one object per line (log collectors)
    config_dir: Path = PROJECT_ROOT / "configs"
    data_dir: Path = PROJECT_ROOT / "data"
    models_dir: Path = PROJECT_ROOT / "models"

    # SQLAlchemy URL. None = local SQLite file <data_dir>/db/traffic.db (zero setup). For PostgreSQL /
    # PostGIS: postgresql+psycopg://user:password@host:5432/dbname  (set it in .env - never commit secrets)
    database_url: str | None = None
    database_file: str = "traffic.db"           # SQLite file name under <data_dir>/db when database_url is unset
    # Which pipeline composition to run (configs/<file>). A pipeline file may set defaults for the settings
    # below in a top-level `settings:` block (e.g. pipeline_pems_bay.yaml: own database, 5-min freshness);
    # values given explicitly (environment / .env) always win.
    pipeline_file: str = "pipeline.yaml"
    # "memory" = single-process dev mode; "redis" = distributed workers.
    bus_backend: Literal["memory", "redis"] = "memory"
    redis_url: str = "redis://localhost:6379/0"

    api_cors_origins: list[str] = ["http://localhost:5173"]
    # Optional push channel (/ws/traffic + LiveHub), kept as a FUTURE interface. The dashboard polls the
    # REST API and does not need it; when disabled the route is not mounted and nothing polls for it.
    ws_enabled: bool = False
    ws_push_interval_s: float = 1.0
    # Freshness used by the API/dashboard: older data is shown as stale, never as current.
    live_stale_s: float = 10.0                  # 1 s live values
    window_stale_s: float = 180.0               # 60 s windows (3 missed windows)
    camera_offline_s: float = 15.0              # camera heartbeat

    @model_validator(mode="after")
    def _pipeline_defaults(self) -> AppSettings:
        path = self.config_dir / self.pipeline_file
        overrides = (load_yaml(path).get("settings") or {}) if path.is_file() else {}
        for key, value in overrides.items():
            if key not in type(self).model_fields or key in ("pipeline_file", "config_dir"):
                raise ValueError(f"{self.pipeline_file}: unknown or non-overridable setting '{key}'")
            if key not in self.model_fields_set:
                setattr(self, key, value)
        return self


@lru_cache
def get_settings() -> AppSettings:
    return AppSettings()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


# --------------------------------------------------------------------------- #
# Road network (Module 6/7/8) - configs/network/*.yaml
# --------------------------------------------------------------------------- #
LonLat = tuple[float, float]


class JunctionConfig(BaseModel):
    id: str
    name: str
    location: LonLat
    signalized: bool = True


class RoadNodeConfig(BaseModel):
    id: str
    location: LonLat
    junction_id: str | None = None


class RoadSegmentConfig(BaseModel):
    """Directed edge of the road graph."""

    id: str
    name: str
    from_node: str
    to_node: str
    geometry: list[LonLat] = Field(min_length=2)
    lanes: int = Field(ge=1)
    speed_limit_kmh: float = 50.0
    free_flow_speed_kmh: float | None = None   # defaults to speed limit
    capacity_veh_h: float | None = None        # effective capacity; default lanes x simulator capacity_per_lane
    importance: float | None = Field(None, ge=0, le=1)  # 0..1 for the risk engine (default from lanes/speed)


class ApproachConfig(BaseModel):
    """One road entering a junction; the unit every metric is reported for."""

    id: str
    junction_id: str
    road_segment_id: str
    name: str
    direction: str                             # e.g. "northbound"


class NetworkConfig(BaseModel):
    name: str
    # False when the coordinates are schematic, e.g. a public video whose recording site is not published: the
    # dashboard then draws the roads on a plain background instead of a real city map.
    geo_referenced: bool = True
    location_note: str | None = None
    junctions: list[JunctionConfig]
    nodes: list[RoadNodeConfig]
    segments: list[RoadSegmentConfig]
    approaches: list[ApproachConfig]

    @model_validator(mode="after")
    def _check_refs(self) -> NetworkConfig:
        nodes = {n.id for n in self.nodes}
        junctions = {j.id for j in self.junctions}
        segments = {s.id for s in self.segments}
        for s in self.segments:
            if s.from_node not in nodes or s.to_node not in nodes:
                raise ValueError(f"segment {s.id} references unknown node")
        for a in self.approaches:
            if a.junction_id not in junctions or a.road_segment_id not in segments:
                raise ValueError(f"approach {a.id} references unknown junction/segment")
        return self


# --------------------------------------------------------------------------- #
# Cameras - configs/cameras/*.yaml
# --------------------------------------------------------------------------- #
Point2 = tuple[float, float]


class CalibrationConfig(BaseModel):
    """Homography correspondences: >= 4 image pixels <-> ground metres.

    ``ground_points_m`` are in a local road-aligned frame (x across, y along
    travel). ``origin_lonlat`` + ``bearing_deg`` place that frame on the map.
    """

    image_points_px: list[Point2] = Field(min_length=4)
    ground_points_m: list[Point2] = Field(min_length=4)
    origin_lonlat: LonLat
    bearing_deg: float                           # direction of +y, clockwise from north
    image_size: tuple[int, int]                  # (width, height) the points were picked on

    @model_validator(mode="after")
    def _same_len(self) -> CalibrationConfig:
        if len(self.image_points_px) != len(self.ground_points_m):
            raise ValueError("image_points_px and ground_points_m must pair up")
        return self


class ZoneConfig(BaseModel):
    id: str
    type: ZoneType
    points_px: list[Point2] = Field(min_length=2)  # polygon, or 2-point line for *_LINE
    lane_index: int | None = None
    # For lanes: expected direction of travel in ground space (unit-ish vector).
    direction: Point2 | None = None


class CameraConfig(BaseModel):
    id: str
    name: str
    approach_id: str
    # File path (relative to project root or absolute), rtsp://..., http(s)://...,
    # "webcam:<index>", or "synthetic://<scenario>" (test/demo source).
    source: str
    source_type: Literal["auto", "file", "webcam", "stream", "synthetic"] = "auto"
    source_fps: float | None = None              # None = read from stream (override broken metadata)
    process_fps: float = Field(10.0, gt=0)       # frames/s actually analysed (sampling)
    # Processing resolution (width, height). Frames are resized right after decode;
    # zones/calibration are scaled automatically. None = native resolution.
    resolution: tuple[int, int] | None = None
    replay_start: str | None = None              # ISO time assigned to frame 0 for files
    realtime: bool = False                       # pace file playback at source speed
    loop: bool = False                           # restart file at EOF (demos)
    location: LonLat
    calibration: CalibrationConfig | None = None
    # Pixel coordinate system of `zones` (width, height). Defaults to
    # calibration.image_size, else the native source resolution.
    zones_image_size: tuple[int, int] | None = None
    zones: list[ZoneConfig] = []
    # Road geometry facts used by the feature engine:
    lanes: int | None = Field(None, ge=1)        # default: number of LANE zones, else 1
    travel_direction: Point2 | None = None       # ground-frame vector; default from lanes / lines
    roi_length_m: float | None = Field(None, gt=0)  # override the length measured from the ROI

    def zone_reference_size(self) -> tuple[int, int] | None:
        if self.zones_image_size is not None:
            return self.zones_image_size
        return self.calibration.image_size if self.calibration else None


# --------------------------------------------------------------------------- #
# Pipeline composition - configs/pipeline.yaml
# --------------------------------------------------------------------------- #
class OutputsConfig(BaseModel):
    """Where a camera worker writes its offline artefacts (paths relative to the project root)."""

    model_config = ConfigDict(extra="forbid")

    runs_dir: str = "runs"                       # <runs_dir>/<camera>/<run_id>/: states, annotated.mp4, summary
    parquet_dir: str = "data/interim"            # <parquet_dir>/<camera>/<date>/<run_id>/<dataset>-NNNNN.parquet
    record_detections: bool = True               # raw detector output per frame
    record_tracks: bool = True                   # tracked objects per frame (id, class, box, ground xy, speed)
    record_track_summaries: bool = True          # one row per finished trajectory
    parquet_flush_rows: int = Field(50_000, gt=0)
    annotate_video: bool = False                 # default for --annotate
    video_codec: str = "mp4v"                    # FourCC; "avc1" gives smaller files where available


class PipelineConfig(BaseModel):
    network_file: str
    camera_files: list[str]
    preprocessors: list[ComponentSpec] = []
    detector: ComponentSpec
    tracker: ComponentSpec
    ground_projector: ComponentSpec
    feature_engine: ComponentSpec
    congestion_classifier: ComponentSpec
    forecaster: ComponentSpec
    track_anomaly_detectors: list[ComponentSpec] = []
    series_anomaly_detectors: list[ComponentSpec] = []
    incident_fuser: ComponentSpec
    risk_engine: ComponentSpec
    simulator: ComponentSpec
    weather_provider: ComponentSpec
    calendar_provider: ComponentSpec
    outputs: OutputsConfig = OutputsConfig()
    settings: dict[str, Any] = {}                # AppSettings defaults for this composition (see AppSettings)


def load_pipeline_config(settings: AppSettings | None = None) -> PipelineConfig:
    """``settings.pipeline_file`` (default pipeline.yaml). ``extends: <file>`` takes every top-level key
    from that file and replaces the ones this file sets (e.g. pipeline_pems_bay.yaml reuses the camera
    stack of pipeline.yaml and only swaps network, forecaster and calendar)."""
    settings = settings or get_settings()
    raw = load_yaml(settings.config_dir / settings.pipeline_file)
    base = raw.pop("extends", None)
    if base:
        raw = {**load_yaml(settings.config_dir / base), **raw}
    return PipelineConfig.model_validate(raw)


def load_network(path: Path) -> NetworkConfig:
    return NetworkConfig.model_validate(load_yaml(path))


def load_camera(path: Path) -> CameraConfig:
    return CameraConfig.model_validate(load_yaml(path))


def resolve_path(path: str | Path) -> Path:
    """Relative paths in configs are relative to the project root."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p
