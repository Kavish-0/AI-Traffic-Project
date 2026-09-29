"""Current traffic, KPIs, anomaly evidence and risk.

    GET /traffic/current                  latest closed-window state per approach
    GET /traffic/summary                  dashboard KPIs from FRESH live values only (null when none)
    GET /traffic/{approach_id}/history    closed windows of one approach
    GET /anomalies                        anomaly signals (evidence, not incidents), newest first
    GET /risk                             latest risk per approach, highest first
    GET /risk/{approach_id}/history       risk (and per-window incident score) time series

Every risk row carries its reason codes and the formula breakdown in ``explanation``.
"""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, Query

from traffic_intel.api.deps import get_ctx
from traffic_intel.api.schemas import AnomalyOut, RiskOut, SummaryOut, TrafficStateOut
from traffic_intel.core.enums import AnomalyType
from traffic_intel.services import incidents as svc
from traffic_intel.services import traffic
from traffic_intel.services.context import ServiceContext

router = APIRouter(tags=["intelligence"])


@router.get("/traffic/current")
def current_traffic(junction_id: str | None = None, ctx: ServiceContext = Depends(get_ctx)) -> list[TrafficStateOut]:
    return traffic.current(ctx, junction_id)


@router.get("/traffic/summary")
def traffic_summary(ctx: ServiceContext = Depends(get_ctx)) -> SummaryOut:
    return traffic.summary(ctx)


@router.get("/traffic/{approach_id}/history")
def traffic_history(approach_id: str, minutes: int = Query(120, ge=1, le=7 * 24 * 60),
                    ctx: ServiceContext = Depends(get_ctx)) -> list[TrafficStateOut]:
    return traffic.history(ctx, approach_id, minutes)


@router.get("/anomalies")
def anomalies(approach_id: str | None = None, junction_id: str | None = None, type: AnomalyType | None = None,
              since: datetime | None = None, min_score: float = Query(0.0, ge=0.0, le=1.0),
              limit: int = Query(200, ge=1, le=2000), ctx: ServiceContext = Depends(get_ctx)) -> list[AnomalyOut]:
    return svc.anomalies(ctx, approach_id=approach_id, junction_id=junction_id, type=type.value if type else None,
                         since=since, min_score=min_score, limit=limit)


@router.get("/risk")
def current_risk(approach_id: str | None = None, junction_id: str | None = None,
                 ctx: ServiceContext = Depends(get_ctx)) -> list[RiskOut]:
    return svc.latest_risk(ctx, approach_id, junction_id)


@router.get("/risk/{approach_id}/history")
def risk_history(approach_id: str, since: datetime | None = None, limit: int = Query(240, ge=1, le=5000),
                 ctx: ServiceContext = Depends(get_ctx)) -> list[RiskOut]:
    return svc.risk_history(ctx, approach_id, since, limit)
