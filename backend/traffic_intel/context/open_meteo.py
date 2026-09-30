"""Open-Meteo weather provider (``weather_provider.type: open_meteo``) - free, no API key, CC BY 4.0.

``observed``: current conditions from the forecast API (``current=``); ``forecast``: hourly values
interpolated to the requested times. Historical training data uses the SAME source (archive API,
forecasting/data/context.py), so live and training weather share definitions.
Network failures raise; callers treat weather as optional context.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import httpx

from traffic_intel.context.interfaces import WEATHER_PROVIDERS, WeatherProvider
from traffic_intel.core.schemas import ContextFeatures
from traffic_intel.forecasting.data.context import wmo_condition

URL = "https://api.open-meteo.com/v1/forecast"
VARS = "temperature_2m,relative_humidity_2m,rain,weather_code,visibility"


def _ctx(at: datetime, v: dict[str, Any], is_holiday: bool = False) -> ContextFeatures:
    vis = v.get("visibility")
    return ContextFeatures(timestamp=at, hour=at.hour, day_of_week=at.weekday(), is_holiday=is_holiday,
                           weather=wmo_condition(v.get("weather_code")), temperature_c=v.get("temperature_2m"),
                           precipitation_mm=v.get("rain"), visibility_km=None if vis is None else vis / 1000.0)


@WEATHER_PROVIDERS.register("open_meteo")
class OpenMeteoWeather(WeatherProvider):
    def __init__(self, timeout_s: float = 10.0, cache_ttl_s: float = 900.0, **_: object) -> None:
        self.timeout_s = timeout_s
        self.cache_ttl_s = cache_ttl_s
        self._cache: dict[tuple[float, float], tuple[float, ContextFeatures]] = {}

    def observed(self, lonlat: tuple[float, float], at: datetime) -> ContextFeatures:
        key = (round(lonlat[0], 3), round(lonlat[1], 3))
        hit = self._cache.get(key)
        if hit and at.timestamp() - hit[0] < self.cache_ttl_s:
            return hit[1]
        r = httpx.get(URL, params={"latitude": lonlat[1], "longitude": lonlat[0], "current": VARS,
                                   "timezone": "UTC"}, timeout=self.timeout_s)
        r.raise_for_status()
        cur = r.json()["current"]
        ctx = _ctx(datetime.fromisoformat(cur["time"]).replace(tzinfo=UTC), cur)
        self._cache[key] = (at.timestamp(), ctx)
        return ctx

    def forecast(self, lonlat: tuple[float, float], times: list[datetime]) -> list[ContextFeatures]:
        r = httpx.get(URL, params={"latitude": lonlat[1], "longitude": lonlat[0], "hourly": VARS,
                                   "timezone": "UTC", "forecast_days": 2}, timeout=self.timeout_s)
        r.raise_for_status()
        h = r.json()["hourly"]
        stamps = [datetime.fromisoformat(s).replace(tzinfo=UTC) for s in h["time"]]
        out = []
        for t in times:
            i = min(range(len(stamps)), key=lambda k: abs((stamps[k] - t).total_seconds()))
            out.append(_ctx(t, {k: h[k][i] for k in h if k != "time"}))
        return out
