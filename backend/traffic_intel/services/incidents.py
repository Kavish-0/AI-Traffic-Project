"""Incidents, anomaly evidence and risk views.

The incident value is labelled as a PROBABILITY only when the fusion model is calibrated; until
then it is labelled "Incident probability: UNCALIBRATED" - the label travels with every incident.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from traffic_intel.api.schemas import AnomalyOut, IncidentOut, RiskOut
from traffic_intel.core.enums import IncidentStatus
from traffic_intel.services.context import ServiceContext


def view(row: dict[str, Any]) -> IncidentOut:
    return IncidentOut.model_validate({**row, "score_label": "Incident probability" if row["calibrated"]
                                       else "Incident probability: UNCALIBRATED"})


def risk_view(row: dict[str, Any]) -> RiskOut:
    return RiskOut.model_validate(row)


def list_incidents(ctx: ServiceContext, **kw: Any) -> list[IncidentOut]:
    return [view(r) for r in ctx.store.list_incidents(**kw)]


def get_incident(ctx: ServiceContext, incident_id: str) -> IncidentOut | None:
    r = ctx.store.get_incident(incident_id)
    return view(r) if r else None


def feedback(ctx: ServiceContext, incident_id: str, status: IncidentStatus, label: str | None,
             note: str | None) -> IncidentOut | None:
    """Operator decision; stored with the incident as a label for future calibration."""
    r = ctx.store.set_status(incident_id, status, label, note)
    return view(r) if r else None


def anomalies(ctx: ServiceContext, **kw: Any) -> list[AnomalyOut]:
    return [AnomalyOut.model_validate(r) for r in ctx.store.list_anomalies(**kw)]


def latest_risk(ctx: ServiceContext, approach_id: str | None = None, junction_id: str | None = None) -> list[RiskOut]:
    return [risk_view(r) for r in ctx.store.latest_risk(approach_id, junction_id)]


def risk_history(ctx: ServiceContext, approach_id: str, since: datetime | None, limit: int) -> list[RiskOut]:
    return [risk_view(r) for r in ctx.store.risk_history(approach_id, since, limit)]
