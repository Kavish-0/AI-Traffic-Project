"""What-if use-case: the existing NetworkX static assignment, with the latest OBSERVED flows as baseline."""

from __future__ import annotations

from typing import Any

from traffic_intel.core.schemas import SimulationRequest, SimulationResult, TrafficState
from traffic_intel.incidents.convert import features_from_state
from traffic_intel.services.context import ServiceContext


def baseline(ctx: ServiceContext) -> dict[str, Any]:
    """road_id -> latest fresh window features (only flow is used by the static model)."""
    out = {}
    for d in ctx.store.latest_traffic_states():
        if (ctx.age_s(d["timestamp"]) or 1e9) > ctx.settings.window_stale_s or d["traffic_state"] is None:
            continue
        st = TrafficState.model_validate({k: v for k, v in d.items() if k in TrafficState.model_fields})
        out[st.road_id] = features_from_state(st)
    return out


def run(ctx: ServiceContext, request: SimulationRequest) -> SimulationResult:
    result = ctx.simulator().what_if(request, baseline(ctx))
    ctx.store.add_simulation(result)
    return result


def get(ctx: ServiceContext, scenario_id: str) -> SimulationResult | None:
    row = ctx.store.get_simulation(scenario_id)
    return SimulationResult.model_validate(row) if row else None
