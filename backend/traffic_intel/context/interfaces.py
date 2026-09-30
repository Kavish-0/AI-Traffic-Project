"""Contextual data providers: weather and calendar.

Weather default: Open-Meteo (free, no key, has historical archive so training
data and live inference use the same source). Calendar: the ``holidays``
package plus a local override file for city-specific events.
Providers must be cacheable and must never block the frame loop.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date, datetime

from traffic_intel.core.registry import Registry
from traffic_intel.core.schemas import ContextFeatures


class WeatherProvider(ABC):
    @abstractmethod
    def observed(self, lonlat: tuple[float, float], at: datetime) -> ContextFeatures: ...

    @abstractmethod
    def forecast(
        self, lonlat: tuple[float, float], times: list[datetime]
    ) -> list[ContextFeatures]: ...


class CalendarProvider(ABC):
    @abstractmethod
    def is_holiday(self, day: date) -> bool: ...


WEATHER_PROVIDERS: Registry[WeatherProvider] = Registry("weather_provider")
CALENDAR_PROVIDERS: Registry[CalendarProvider] = Registry("calendar_provider")
WEATHER_PROVIDERS.register_lazy("open_meteo", "traffic_intel.context.open_meteo")
WEATHER_PROVIDERS.register_lazy("none", "traffic_intel.context.static")
CALENDAR_PROVIDERS.register_lazy("holidays", "traffic_intel.context.calendar")
