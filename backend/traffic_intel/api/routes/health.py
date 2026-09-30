"""Liveness (/health) and the dashboard's system status (/system/status) from real signals."""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel

from traffic_intel import __version__
from traffic_intel.api.schemas import ComponentStatus, SystemStatusOut
from traffic_intel.core.config import get_settings
from traffic_intel.services import system

router = APIRouter(tags=["system"])


class HealthResponse(BaseModel):
    status: str
    version: str
    env: str
    bus_backend: str
    database: str


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    settings = get_settings()
    ctx = getattr(request.app.state, "ctx", None)
    db = "connected" if ctx is not None and ctx.store.ping() else "disconnected"
    return HealthResponse(status="ok", version=__version__, env=settings.env, bus_backend=settings.bus_backend,
                          database=db)


@router.get("/system/status")
def system_status(request: Request) -> SystemStatusOut:
    ctx = getattr(request.app.state, "ctx", None)
    if ctx is None:
        err = getattr(request.app.state, "db_error", None) or "not initialised"
        down = ComponentStatus(ok=False, status="unknown", detail="database unavailable")
        return SystemStatusOut(
            backend=ComponentStatus(ok=True, status="online", detail=f"traffic_intel {__version__}"),
            database=ComponentStatus(ok=False, status="disconnected", detail=err), pipeline=down,
            forecast_model=down, cameras=down, weather=down, websocket_clients=0, version=__version__,
            env=get_settings().env)
    return system.status(ctx)
