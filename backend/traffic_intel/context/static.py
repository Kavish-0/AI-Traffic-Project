"""No weather (``weather_provider.type: none``): context is calendar-only; weather fields stay empty."""

from __future__ import annotations

from datetime import datetime

from traffic_intel.context.interfaces import WEATHER_PROVIDERS, WeatherProvider
from traffic_intel.core.schemas import ContextFeatures


@WEATHER_PROVIDERS.register("none")
class NoWeather(WeatherProvider):
    def __init__(self, **_: object) -> None:
        pass

    def observed(self, lonlat: tuple[float, float], at: datetime) -> ContextFeatures:
        return ContextFeatures(timestamp=at, hour=at.hour, day_of_week=at.weekday(), is_holiday=False)

    def forecast(self, lonlat: tuple[float, float], times: list[datetime]) -> list[ContextFeatures]:
        return [self.observed(lonlat, t) for t in times]
