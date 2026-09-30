"""Shared API dependencies (overridable in tests via ``app.dependency_overrides``).

Everything here is torch-free: the service context wraps the database store (SQLAlchemy), the
network/camera configuration and the NetworkX what-if simulator. Models run in the pipeline
runtime process, never in the API.
"""

from __future__ import annotations

import logging

from fastapi import HTTPException, Request

from traffic_intel.services.context import ServiceContext

log = logging.getLogger(__name__)


def get_ctx(request: Request) -> ServiceContext:
    """The API's service context; 503 with an explanation when the database is unavailable."""
    ctx = getattr(request.app.state, "ctx", None)
    if ctx is None:
        try:                                            # the database may have come up since startup
            ctx = ServiceContext.build()
            request.app.state.ctx, request.app.state.db_error = ctx, None
        except Exception as exc:  # noqa: BLE001
            request.app.state.db_error = f"{type(exc).__name__}: {exc}"
            log.warning("database unavailable: %s", request.app.state.db_error)
            raise HTTPException(503, f"Database unavailable ({request.app.state.db_error}). Check TI_DATABASE_URL "
                                     "(see .env.example) - without it a local SQLite file is used.") from exc
    return ctx
