"""Public-holiday calendar (``calendar_provider.type: holidays``) with an optional local override file
(YAML list of ISO dates, e.g. city events) - no hard-coded dates."""

from __future__ import annotations

from datetime import date
from functools import cached_property

from traffic_intel.context.interfaces import CALENDAR_PROVIDERS, CalendarProvider
from traffic_intel.core.config import load_yaml, resolve_path


@CALENDAR_PROVIDERS.register("holidays")
class HolidaysCalendar(CalendarProvider):
    def __init__(self, country: str = "IN", overrides_file: str | None = None, **_: object) -> None:
        self.country = country
        self.overrides_file = overrides_file

    @cached_property
    def _extra(self) -> set[date]:
        if not self.overrides_file:
            return set()
        data = load_yaml(resolve_path(self.overrides_file)) or {}
        return {date.fromisoformat(str(d)) for d in data.get("holidays", [])}

    def is_holiday(self, day: date) -> bool:
        import holidays

        return day in self._extra or day in holidays.country_holidays(self.country, years=[day.year])
