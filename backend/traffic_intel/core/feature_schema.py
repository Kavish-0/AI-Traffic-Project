"""Single source of truth for ML feature columns.

Both the offline dataset builder (``ai.training``, ``datasets/`` adapters) and the online
inference path build feature frames through these definitions, so a model can never be
trained on columns that production does not produce (train/serve skew). Every public
dataset adapter maps its raw fields ONTO these names (``validate_frame``) - there is no
second, dataset-specific schema.

A hash of the fitted preprocessing (which records the columns actually used) is stored with
every trained model artifact and checked at load time.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable

# Canonical long-format keys: one row per (series, window start).
KEY_COLUMNS: tuple[str, ...] = ("series_id", "timestamp")

# Per-window traffic measurements (from TrafficFeatures). A dataset provides a SUBSET
# (e.g. METR-LA / PEMS-BAY: speed only; PeMS08: flow, occupancy, speed; camera pipeline: all).
TRAFFIC_COLUMNS: tuple[str, ...] = (
    "vehicle_count",
    "pcu_count",
    "avg_speed_kmh",
    "median_speed_kmh",
    "speed_p85_kmh",
    "density_veh_km",
    "density_ratio",
    "occupancy_pct",
    "flow_veh_h",
    "inflow",
    "outflow",
    "queue_length_m",
    "stopped_vehicles",
    "avg_gap_m",
    "quality",
)

# Calendar context, derived from timestamps in LOCAL time (forecasting/data/context.py builds
# them; inference needs nothing but the clock). Groups are what training configs select.
CALENDAR_GROUPS: dict[str, list[str]] = {
    "hour": ["hour_sin", "hour_cos"],
    "day_of_week": ["dow_sin", "dow_cos", "is_weekend"],
    "holiday": ["is_holiday"],
    "rush_hour": ["is_rush_hour"],
}
CALENDAR_COLUMNS: tuple[str, ...] = tuple(c for cols in CALENDAR_GROUPS.values() for c in cols)

# Weather context (optional - a model may be traffic-only). weather_condition is categorical.
WEATHER_COLUMNS: tuple[str, ...] = ("temperature_c", "humidity_pct", "rain_mm", "visibility_km",
                                    "weather_condition")
WEATHER_CONDITIONS: tuple[str, ...] = ("clear", "cloudy", "fog", "drizzle", "rain", "snow", "thunderstorm")

CONTEXT_COLUMNS: tuple[str, ...] = CALENDAR_COLUMNS + WEATHER_COLUMNS

# Ground-truth / label columns a dataset may carry for evaluation only (never model inputs).
LABEL_COLUMNS: tuple[str, ...] = ("gt_incident", "gt_load_ratio")

FORECAST_TARGETS: tuple[str, ...] = ("density_ratio", "avg_speed_kmh", "flow_veh_h", "occupancy_pct",
                                     "queue_length_m")
FORECAST_HORIZONS_MIN: tuple[int, ...] = (5, 15, 30)
BASE_RESOLUTION_S: int = 60          # camera pipeline windows; public series may be coarser (e.g. 300 s)
DEFAULT_LOOKBACK_STEPS: int = 60     # 60 minutes of history per sample at 1-min resolution


class SchemaError(ValueError):
    """A frame does not follow the canonical feature schema."""


def validate_frame(columns: Iterable[str], *, source: str = "frame", require_traffic: bool = True) -> None:
    """Check that a canonical time-series frame uses only schema names.

    Allowed: KEY_COLUMNS (required), TRAFFIC_COLUMNS, CONTEXT_COLUMNS, LABEL_COLUMNS.
    Raises SchemaError naming the offending columns (e.g. a raw 'speed_mph' that an
    adapter forgot to convert)."""
    cols = list(columns)
    missing = [k for k in KEY_COLUMNS if k not in cols]
    if missing:
        raise SchemaError(f"{source}: missing key columns {missing}")
    allowed = set(KEY_COLUMNS) | set(TRAFFIC_COLUMNS) | set(CONTEXT_COLUMNS) | set(LABEL_COLUMNS)
    unknown = [c for c in cols if c not in allowed]
    if unknown:
        raise SchemaError(f"{source}: columns not in core/feature_schema.py: {unknown}")
    if require_traffic and not any(c in TRAFFIC_COLUMNS for c in cols):
        raise SchemaError(f"{source}: no traffic column from TRAFFIC_COLUMNS")


def schema_hash() -> str:
    payload = json.dumps(
        {
            "traffic": TRAFFIC_COLUMNS,
            "context": CONTEXT_COLUMNS,
            "targets": FORECAST_TARGETS,
            "resolution_s": BASE_RESOLUTION_S,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]
