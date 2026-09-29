"""Incidents (Module 5): hedged incident hypotheses with score, evidence and operator status.

An incident here is always a POSSIBLE incident for an operator to verify. Each item carries
``probability`` together with ``calibrated`` and ``score_label`` ("Incident probability: UNCALIBRATED"
until the fusion model is calibrated on labelled events), the fused ``evidence``, the contributing
signals and an ``uncertainty_note``. Operator feedback is stored as the label for future calibration.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from traffic_intel.api.deps import get_ctx
from traffic_intel.api.schemas import AnomalyOut, IncidentOut
from traffic_intel.core.enums import IncidentStatus
from traffic_intel.services import incidents as svc
from traffic_intel.services.context import ServiceContext

router = APIRouter(prefix="/incidents", tags=["incidents"])


class IncidentFeedback(BaseModel):
    status: Literal[IncidentStatus.ACKNOWLEDGED, IncidentStatus.RESOLVED, IncidentStatus.DISMISSED]
    label: Literal["true_incident", "false_alarm", "planned_event", "unsure"] | None = None
    note: str | None = None


@router.get("")
def list_incidents(status: IncidentStatus | None = None, approach_id: str | None = None,
                   junction_id: str | None = None, active_only: bool = False, since: datetime | None = None,
                   limit: int = Query(100, ge=1, le=1000), ctx: ServiceContext = Depends(get_ctx)) -> list[IncidentOut]:
    return svc.list_incidents(ctx, status=status.value if status else None, approach_id=approach_id,
                              junction_id=junction_id, active_only=active_only, since=since, limit=limit)


@router.get("/anomalies/recent")
def recent_anomalies(approach_id: str | None = None, limit: int = Query(100, ge=1, le=2000),
                     ctx: ServiceContext = Depends(get_ctx)) -> list[AnomalyOut]:
    """Alias of ``GET /anomalies`` (kept for the original API sketch)."""
    return svc.anomalies(ctx, approach_id=approach_id, limit=limit)


@router.get("/{incident_id}")
def get_incident(incident_id: str, ctx: ServiceContext = Depends(get_ctx)) -> IncidentOut:
    inc = svc.get_incident(ctx, incident_id)
    if inc is None:
        raise HTTPException(404, f"incident '{incident_id}' not found")
    return inc


@router.patch("/{incident_id}")
def update_incident(incident_id: str, feedback: IncidentFeedback,
                    ctx: ServiceContext = Depends(get_ctx)) -> IncidentOut:
    """Operator acknowledgement / resolution / dismissal (TRUE INCIDENT / FALSE ALARM labels). The labels
    are the ground truth for later calibration of the fusion model; the engine never overrides them."""
    inc = svc.feedback(ctx, incident_id, IncidentStatus(feedback.status), feedback.label, feedback.note)
    if inc is None:
        raise HTTPException(404, f"incident '{incident_id}' not found")
    return inc
