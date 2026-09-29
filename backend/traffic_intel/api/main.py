"""FastAPI application factory.

    uvicorn traffic_intel.api.main:app --reload

The API process never imports torch/ultralytics: it reads the database (written by the pipeline
runtime, ``python -m traffic_intel.pipeline run``) through the service layer and serves it as REST.
The dashboard polls these endpoints. If the database is unreachable the API still starts and
reports it (``/api/v1/system/status``, HTTP 503 on data routes).

Optional, off by default (``TI_WS_ENABLED=true``): a push channel for future use
(LiveHub -> in-process bus -> /ws/traffic). Nothing in the current frontend depends on it.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from traffic_intel import __version__
from traffic_intel.api import websocket
from traffic_intel.api.live import LiveHub
from traffic_intel.api.routes import cameras, health, incidents, intelligence, models, network, simulation
from traffic_intel.core.config import get_settings
from traffic_intel.core.logging import configure_logging
from traffic_intel.pipeline.bus import InMemoryEventBus

API_PREFIX = "/api/v1"


log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    from traffic_intel.services.context import ServiceContext

    settings = get_settings()
    configure_logging(settings.log_level)
    app.state.bus = InMemoryEventBus()   # Phase 9: RedisStreamsEventBus when bus_backend == "redis"
    app.state.ctx, app.state.db_error, hub = None, None, None
    if getattr(app.state, "ctx_override", None) is not None:
        app.state.ctx = app.state.ctx_override
    else:
        try:
            app.state.ctx = await asyncio.to_thread(ServiceContext.build)
        except Exception as exc:  # noqa: BLE001 - start anyway; status endpoint reports it
            app.state.db_error = f"{type(exc).__name__}: {exc}"
            log.error("database unavailable at startup: %s", app.state.db_error)
    if app.state.ctx is not None:
        log.info("API started - database %s", app.state.ctx.store.url)
        if settings.ws_enabled:
            hub = asyncio.create_task(LiveHub(app.state.ctx, app.state.bus, settings.ws_push_interval_s).run())
    yield
    if hub is not None:
        hub.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await hub
    await app.state.bus.close()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="Urban Traffic & Incident Intelligence", version=__version__, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api_cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    for module in (health, network, intelligence, incidents, cameras, simulation, models):
        app.include_router(module.router, prefix=API_PREFIX)
    if settings.ws_enabled:                      # optional future push channel; the dashboard uses REST polling
        app.include_router(websocket.router)
    return app


app = create_app()
