"""What-if simulation (Module 8): static re-assignment on the configured road graph.

Simulation / decision support - NOT a microscopic traffic simulator and not a guaranteed
real-world prediction (see ``simulation/static_assignment.py``); every result lists its
assumptions. Baseline flows are the latest fresh observations where available.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from traffic_intel.api.deps import get_ctx
from traffic_intel.core.schemas import SimulationRequest, SimulationResult
from traffic_intel.services import simulation as svc
from traffic_intel.services.context import ServiceContext

log = logging.getLogger(__name__)
router = APIRouter(prefix="/simulation", tags=["simulation"])


@router.post("/what-if")
def what_if(request: SimulationRequest, ctx: ServiceContext = Depends(get_ctx)) -> SimulationResult:
    """e.g. ``{"closed_segment_ids": ["road_b_wb"]}`` -> flow redistribution and congestion probabilities."""
    try:
        return svc.run(ctx, request)
    except ValueError as exc:
        log.warning("what-if rejected: %s", exc)
        raise HTTPException(422, str(exc)) from exc


@router.get("/scenarios/{scenario_id}")
def get_scenario(scenario_id: str, ctx: ServiceContext = Depends(get_ctx)) -> SimulationResult:
    res = svc.get(ctx, scenario_id)
    if res is None:
        raise HTTPException(404, f"scenario '{scenario_id}' not found")
    return res
