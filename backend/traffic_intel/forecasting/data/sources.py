"""Dataset sources -> one canonical long-format time-series frame.

Every source returns a ``TimeSeriesDataset`` whose ``frame`` has columns:

    series_id, timestamp (UTC), <traffic columns ...>, <weather columns ...>

Traffic column names follow ``TrafficFeatures`` (``avg_speed_kmh``,
``flow_veh_h``, ``occupancy_pct``, ``density_ratio``, ``queue_length_m`` ...),
so models trained on any source speak the pipeline's vocabulary.

Sources
    pems08        real Caltrans freeway data (5-min, flow/occupancy/speed) + real weather
    metr_la       METR-LA speed (DCRNN release; place under data/raw/metr_la - see the dataset manifest)
    pems_bay      PEMS-BAY speed (DCRNN release; place under data/raw/pems_bay)
    synthetic     long 1-min series in the full TrafficFeatures schema (see synthetic_series.py)
    observations  the camera pipeline's own output (traffic_states.jsonl / TrafficFeatures parquet)

Public datasets are read through their adapters (``traffic_intel.datasets.adapters``), which
check that the files exist (clear error + download/licence instructions otherwise) and map raw
fields onto core/feature_schema.py.
"""

from __future__ import annotations

import glob
import json
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from traffic_intel.core.config import resolve_path

log = logging.getLogger(__name__)


@dataclass
class TimeSeriesDataset:
    name: str
    frame: pd.DataFrame
    resolution_s: int
    tz: str
    holiday_country: str | None
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def series_ids(self) -> list[str]:
        return sorted(self.frame["series_id"].unique())


def load_dataset(source: str, params: dict[str, Any] | None = None) -> TimeSeriesDataset:
    params = dict(params or {})
    if source == "pems08":
        return load_pems08(**params)
    if source == "synthetic":
        return load_synthetic(**params)
    if source == "observations":
        return load_observations(**params)
    if source in ("metr_la", "pems_bay", "own_recordings"):
        from traffic_intel.datasets.adapters import adapter_for

        root = params.pop("root", None)
        ds: TimeSeriesDataset = adapter_for(source, root).load(**params)  # type: ignore[attr-defined]
        return ds
    raise ValueError(f"unknown dataset source '{source}' (pems08 | metr_la | pems_bay | synthetic | observations)")


# --------------------------------------------------------------------------- #
def load_pems08(
    path: str = "data/external/pems08/pems08.npz",
    sensors: list[int] | None = None,
    every_k: int = 14,
    start_local: str = "2016-07-01 00:00",
    tz: str = "America/Los_Angeles",
    weather: dict[str, float] | None = None,
    holiday_country: str | None = "US",
) -> TimeSeriesDataset:
    """PeMS District 8: array (steps, sensors, [flow per 5 min, occupancy 0-1, speed mph]).

    Units are converted to the pipeline's: flow veh/h, occupancy %, speed km/h.
    Rows where flow AND occupancy are both 0 are detector outages -> NaN.
    """
    p = resolve_path(path)
    if not p.exists():
        raise FileNotFoundError(f"{p} missing - run: python -m traffic_intel.datasets --catalog timeseries "
                                "download pems08")
    data = np.load(p)["data"]
    n_steps, n_sensors, _ = data.shape
    ids = sensors if sensors is not None else list(range(0, n_sensors, every_k))
    times = pd.date_range(pd.Timestamp(start_local, tz=tz), periods=n_steps, freq="5min").tz_convert("UTC")
    frames = []
    for s in ids:
        flow, occ, spd = data[:, s, 0], data[:, s, 1], data[:, s, 2]
        outage = (flow == 0) & (occ == 0)
        df = pd.DataFrame({
            "series_id": f"pems08_{s:03d}",
            "timestamp": times,
            "flow_veh_h": np.where(outage, np.nan, flow * 12.0),
            "occupancy_pct": np.where(outage, np.nan, occ * 100.0),
            "avg_speed_kmh": np.where(outage, np.nan, spd * 1.609344),
        })
        frames.append(df)
    frame = pd.concat(frames, ignore_index=True)
    frame = _attach_weather(frame, weather, times)
    return TimeSeriesDataset(
        name="pems08", frame=frame, resolution_s=300, tz=tz, holiday_country=holiday_country,
        info={"sensors": ids, "file": str(p), "steps": n_steps, "outage_rows": int(frame["flow_veh_h"].isna().sum()),
              "provenance": "Caltrans PeMS District 8 via ASTGCN mirror (see configs/datasets/public_timeseries.yaml)",
              "time_origin_assumption": f"step 0 = {start_local} {tz}"},
    )


def load_synthetic(weather: dict[str, float] | None = None, holiday_country: str | None = "IN",
                   **cfg: Any) -> TimeSeriesDataset:
    from traffic_intel.forecasting.data.synthetic_series import SyntheticSeriesConfig, generate_series

    sc = SyntheticSeriesConfig(**cfg)
    times = pd.date_range(pd.Timestamp(sc.start).tz_convert("UTC"), periods=int(sc.days * 86400 / sc.resolution_s),
                          freq=f"{sc.resolution_s}s")
    hourly = _weather_hourly(weather, times)
    frame = generate_series(sc, hourly, holiday_country)
    return TimeSeriesDataset(
        name="synthetic", frame=frame, resolution_s=sc.resolution_s, tz=sc.tz, holiday_country=holiday_country,
        info={"generator": sc.model_dump(), "weather": "open-meteo archive" if hourly is not None else "synthetic"},
    )


def load_observations(paths: str | list[str] = "runs/*/*/traffic_states.jsonl", resolution_s: int = 60,
                      tz: str = "UTC", holiday_country: str | None = None,
                      weather: dict[str, float] | None = None) -> TimeSeriesDataset:
    """The camera pipeline's own output. Accepts TrafficState JSONL (window rows, not live)
    or TrafficFeatures Parquet; one series per approach."""
    files: list[str] = []
    for pat in [paths] if isinstance(paths, str) else paths:
        files += glob.glob(str(resolve_path(pat)))
    rows = []
    for f in files:
        if f.endswith(".parquet"):
            rows.extend(pd.read_parquet(f).to_dict("records"))
        else:
            with open(f, encoding="utf-8") as fh:
                rows.extend(json.loads(line) for line in fh if line.strip())
    if not rows:
        raise FileNotFoundError(f"no observations found for {paths}")
    frame = observations_frame(rows)
    frame = frame[frame["window_s"].round() >= resolution_s * 0.99].drop(columns=["window_s"])
    frame = frame.sort_values(["series_id", "timestamp"]).drop_duplicates(["series_id", "timestamp"])
    times = pd.DatetimeIndex(frame["timestamp"].unique()).sort_values()
    frame = _attach_weather(frame, weather, times)
    return TimeSeriesDataset(name="observations", frame=frame.reset_index(drop=True), resolution_s=resolution_s,
                             tz=tz, holiday_country=holiday_country, info={"files": files})


# TrafficState field -> TrafficFeatures column
_STATE_MAP = {"average_speed_kmh": "avg_speed_kmh", "median_speed_kmh": "median_speed_kmh",
              "stopped_vehicle_count": "stopped_vehicles"}


def observations_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """TrafficState / TrafficFeatures dicts -> canonical frame (one row per window)."""
    out = []
    for r in rows:
        if r.get("is_live"):
            continue
        d = {_STATE_MAP.get(k, k): v for k, v in r.items()}
        if "density_ratio" not in d and "density_pct" in d:
            d["density_ratio"] = d["density_pct"] / 100.0
        d["series_id"] = d.get("approach_id")
        d["timestamp"] = pd.Timestamp(d["window_start"]).tz_convert("UTC") if pd.Timestamp(
            d["window_start"]).tzinfo else pd.Timestamp(d["window_start"], tz="UTC")
        out.append(d)
    df = pd.DataFrame(out)
    keep = ["series_id", "timestamp", "window_s", "vehicle_count", "avg_speed_kmh", "density_veh_km", "density_ratio",
            "occupancy_pct", "flow_veh_h", "inflow", "outflow", "queue_length_m", "stopped_vehicles", "quality"]
    return df[[c for c in keep if c in df.columns]]


def _weather_hourly(weather: dict[str, float] | None, times: pd.DatetimeIndex) -> pd.DataFrame | None:
    if not weather:
        return None
    from traffic_intel.forecasting.data.context import fetch_weather

    start = (times.min() - pd.Timedelta(hours=1)).strftime("%Y-%m-%d")
    end = (times.max() + pd.Timedelta(hours=1)).strftime("%Y-%m-%d")
    try:
        return fetch_weather(weather["lat"], weather["lon"], start, end)
    except Exception as e:  # noqa: BLE001 - offline -> continue without weather
        log.warning("weather download failed (%s) - continuing without weather", e)
        return None


def _attach_weather(frame: pd.DataFrame, weather: dict[str, float] | None, times: pd.DatetimeIndex) -> pd.DataFrame:
    hourly = _weather_hourly(weather, times)
    if hourly is None:
        return frame
    from traffic_intel.forecasting.data.context import weather_to_grid

    grid = weather_to_grid(hourly, times)
    return frame.merge(grid, left_on="timestamp", right_index=True, how="left")
