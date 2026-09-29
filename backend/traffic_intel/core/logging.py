"""Logging setup: readable text locally, one JSON object per line when ``TI_LOG_FORMAT=json``
(ready for log collectors in a future deployment). Components log events (startup, model loading,
camera state, errors) - never per-frame data."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

_TEXT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "process": record.process,
            "thread": record.threadName,
        }
        if record.exc_info:
            out["exc"] = self.formatException(record.exc_info)
        return json.dumps(out, ensure_ascii=False)


def configure_logging(level: str = "INFO", fmt: str | None = None) -> None:
    if fmt is None:
        from traffic_intel.core.config import get_settings

        fmt = get_settings().log_format
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter() if fmt == "json" else logging.Formatter(_TEXT))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    for noisy in ("httpx", "httpcore", "numexpr", "ultralytics", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
