"""Context features for forecasting: calendar (local time) and weather.

Calendar features are computed from timestamps (so inference needs nothing but
the clock). Weather comes from the Open-Meteo historical archive (hourly,
free, no key, CC BY 4.0), cached under ``data/raw/weather/`` (older caches in
``data/external/weather/`` are still read); the same provider
offers forecasts, so training and inference can share one source.
"""

from __future__ import annotations

import logging
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from traffic_intel.core.config import PROJECT_ROOT
from traffic_intel.core.feature_schema import CALENDAR_GROUPS, WEATHER_COLUMNS

__all__ = ["CALENDAR_GROUPS", "WEATHER_COLUMNS", "calendar_frame", "fetch_weather", "weather_to_grid", "wmo_condition"]

log = logging.getLogger(__name__)


def wmo_condition(code: float | int | None) -> str | None:
    """WMO weather interpretation code -> coarse condition category."""
    if code is None or (isinstance(code, float) and np.isnan(code)):
        return None
    c = int(code)
    if c == 0:
        return "clear"
    if c in (1, 2, 3):
        return "cloudy"
    if c in (45, 48):
        return "fog"
    if 51 <= c <= 57:
        return "drizzle"
    if 61 <= c <= 67 or 80 <= c <= 82:
        return "rain"
    if 71 <= c <= 77 or c in (85, 86):
        return "snow"
    if c >= 95:
        return "thunderstorm"
    return "cloudy"


def fetch_weather(lat: float, lon: float, start: str, end: str, *, cache_dir: Path | None = None,
                  timeout_s: float = 60.0) -> pd.DataFrame:
    """Hourly weather (UTC index) with columns WEATHER_COLUMNS. Cached as Parquet."""
    name = f"openmeteo_{lat:.3f}_{lon:.3f}_{start}_{end}.parquet"
    cache = (cache_dir or PROJECT_ROOT / "data" / "raw" / "weather") / name
    legacy = PROJECT_ROOT / "data" / "external" / "weather" / name
    for f in (cache, legacy):
        if f.exists():
            return pd.read_parquet(f)
    params: dict[str, str | float] = {
        "latitude": lat, "longitude": lon, "start_date": start, "end_date": end, "timezone": "UTC",
        "hourly": "temperature_2m,relative_humidity_2m,rain,weather_code,visibility",
    }
    r = httpx.get("https://archive-api.open-meteo.com/v1/archive", params=params, timeout=timeout_s)
    r.raise_for_status()
    h = r.json()["hourly"]
    df = pd.DataFrame({
        "temperature_c": h.get("temperature_2m"),
        "humidity_pct": h.get("relative_humidity_2m"),
        "rain_mm": h.get("rain"),
        # The archive API does not provide visibility (returns null) -> all-NaN column, dropped at fit.
        "visibility_km": [None if v is None else v / 1000.0 for v in h.get("visibility") or [None] * len(h["time"])],
        "weather_condition": [wmo_condition(c) for c in h.get("weather_code")],
    }, index=pd.to_datetime(h["time"], utc=True))
    df["visibility_km"] = pd.to_numeric(df["visibility_km"], errors="coerce")
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache)
    log.info("weather: %d hourly rows %s..%s cached at %s", len(df), start, end, cache)
    return df


def weather_to_grid(hourly: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Hourly weather -> the series grid: numeric columns interpolated in time,
    the categorical condition forward-filled."""
    full = hourly.reindex(hourly.index.union(index)).sort_index()
    num = [c for c in WEATHER_COLUMNS if c != "weather_condition" and c in full]
    full[num] = full[num].interpolate(method="time", limit_area="inside").ffill().bfill()
    full["weather_condition"] = full["weather_condition"].ffill().bfill()
    return full.reindex(index)


def calendar_frame(times: pd.DatetimeIndex, *, tz: str, holiday_country: str | None,
                   rush_hours: list[tuple[float, float]]) -> pd.DataFrame:
    """Cyclic time-of-day / day-of-week encodings and binary flags (local time)."""
    local = times.tz_convert(tz)
    minute = local.hour * 60 + local.minute
    dow = local.dayofweek
    hol = np.zeros(len(local), dtype=bool)
    if holiday_country:
        import holidays

        years = sorted(set(local.year))
        cal = holidays.country_holidays(holiday_country, years=years)
        hol = np.array([d in cal for d in local.date])
    hour_f = minute / 60.0
    rush = np.zeros(len(local), dtype=bool)
    for a, b in rush_hours:
        rush |= (hour_f >= a) & (hour_f < b)
    rush &= (dow < 5) & ~hol
    return pd.DataFrame({
        "hour_sin": np.sin(2 * np.pi * minute / 1440.0),
        "hour_cos": np.cos(2 * np.pi * minute / 1440.0),
        "dow_sin": np.sin(2 * np.pi * dow / 7.0),
        "dow_cos": np.cos(2 * np.pi * dow / 7.0),
        "is_weekend": (dow >= 5).astype(np.float32),
        "is_holiday": hol.astype(np.float32),
        "is_rush_hour": rush.astype(np.float32),
    }, index=times)

