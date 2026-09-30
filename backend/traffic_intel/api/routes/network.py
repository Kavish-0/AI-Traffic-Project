"""Road network, junctions, roads and approaches (Modules 6 & 7)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from traffic_intel.api.deps import get_ctx
from traffic_intel.api.schemas import (
    ForecastOut,
    ForecastVsActual,
    JunctionOut,
    RoadOut,
    SnapshotOut,
    TrafficStateOut,
)
from traffic_intel.core.feature_schema import FORECAST_TARGETS
from traffic_intel.services import network as net
from traffic_intel.services import traffic
from traffic_intel.services.context import ServiceContext

router = APIRouter(tags=["network"])


@router.get("/network")
def network_geojson(ctx: ServiceContext = Depends(get_ctx)) -> dict[str, Any]:
    """GeoJSON FeatureCollection: roads (with the latest FRESH congestion level, risk, active incident),
    junctions and cameras - the map's single data source."""
    return net.network_geojson(ctx)


@router.get("/junctions")
def list_junctions(ctx: ServiceContext = Depends(get_ctx)) -> list[JunctionOut]:
    return net.junction_views(ctx)


@router.get("/junctions/{junction_id}")
def junction_detail(junction_id: str, ctx: ServiceContext = Depends(get_ctx)) -> JunctionOut:
    j = next((j for j in net.junction_views(ctx) if j.id == junction_id), None)
    if j is None:
        raise HTTPException(404, f"junction '{junction_id}' not in the network configuration")
    return j


@router.get("/roads")
def list_roads(ctx: ServiceContext = Depends(get_ctx)) -> list[RoadOut]:
    return net.road_views(ctx)


@router.get("/roads/{road_id}")
def road_detail(road_id: str, ctx: ServiceContext = Depends(get_ctx)) -> SnapshotOut:
    """A road's full panel (current state, forecast, incident, risk) via its monitored approach."""
    if road_id not in ctx.segments:
        raise HTTPException(404, f"road '{road_id}' not in the network configuration")
    a = ctx.approach_for_road(road_id)
    if a is None:
        raise HTTPException(404, f"road '{road_id}' is not monitored (no approach/camera configured)")
    return _snapshot(ctx, a.id)


def _snapshot(ctx: ServiceContext, approach_id: str) -> SnapshotOut:
    snap = traffic.snapshot(ctx, approach_id)
    if snap is None:
        raise HTTPException(404, f"approach '{approach_id}' not in the network configuration")
    return snap


@router.get("/approaches/{approach_id}/snapshot")
def approach_snapshot(approach_id: str, ctx: ServiceContext = Depends(get_ctx)) -> SnapshotOut:
    """Current state + forecast + incident + risk: what the road panel shows."""
    return _snapshot(ctx, approach_id)


@router.get("/approaches/{approach_id}/history")
def approach_history(approach_id: str, minutes: int = Query(120, ge=1, le=7 * 24 * 60),
                     ctx: ServiceContext = Depends(get_ctx)) -> list[TrafficStateOut]:
    """Closed 60 s windows (oldest first). Empty list = no data yet (not an error)."""
    if approach_id not in ctx.approaches:
        raise HTTPException(404, f"approach '{approach_id}' not in the network configuration")
    return traffic.history(ctx, approach_id, minutes)


@router.get("/approaches/{approach_id}/forecast")
def approach_forecast(approach_id: str, ctx: ServiceContext = Depends(get_ctx)) -> ForecastOut:
    """Latest forecast with an explicit status (ok / insufficient_history / no_model / stale / no_data)."""
    if approach_id not in ctx.approaches:
        raise HTTPException(404, f"approach '{approach_id}' not in the network configuration")
    return traffic.forecast(ctx, approach_id)


@router.get("/approaches/{approach_id}/forecast/history")
def forecast_vs_actual(approach_id: str, horizon_min: int = Query(15, ge=1), target: str = "avg_speed_kmh",
                       minutes: int = Query(180, ge=10, le=7 * 24 * 60),
                       ctx: ServiceContext = Depends(get_ctx)) -> ForecastVsActual:
    if approach_id not in ctx.approaches:
        raise HTTPException(404, f"approach '{approach_id}' not in the network configuration")
    if target not in FORECAST_TARGETS:
        raise HTTPException(422, f"target must be one of {list(FORECAST_TARGETS)}")
    return traffic.forecast_vs_actual(ctx, approach_id, horizon_min, target, minutes)
