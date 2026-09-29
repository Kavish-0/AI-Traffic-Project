"""Traffic state, road snapshot, KPIs, history, forecasts (incl. forecast vs actual)."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from traffic_intel.api.schemas import (
    ForecastHorizon,
    ForecastOut,
    ForecastPointOut,
    ForecastVsActual,
    SnapshotOut,
    SummaryOut,
    TrafficStateOut,
)
from traffic_intel.services import incidents as incident_svc
from traffic_intel.services import models as model_svc
from traffic_intel.services import network as net
from traffic_intel.services.context import ServiceContext

# TrafficState field of each forecast target (for forecast-vs-actual)
ACTUAL_FIELD = {"avg_speed_kmh": "average_speed_kmh", "density_ratio": "density_pct", "flow_veh_h": "flow_veh_h",
                "occupancy_pct": "occupancy_pct", "queue_length_m": "queue_length_m"}


def _state(d: dict[str, Any] | None) -> TrafficStateOut | None:
    if not d:
        return None
    d = {k: v for k, v in d.items() if not k.startswith("_")}
    return TrafficStateOut.model_validate(d)


def current(ctx: ServiceContext, junction_id: str | None = None) -> list[TrafficStateOut]:
    return [s for s in (_state(d) for d in ctx.store.latest_traffic_states(junction_id)) if s]


def live(ctx: ServiceContext) -> dict[str, dict[str, Any]]:
    """approach_id -> latest live payload (with '_ts'), fresh or not."""
    return {p["approach_id"]: p for p in ctx.store.latest_live_states()}


def history(ctx: ServiceContext, approach_id: str, minutes: int = 120, limit: int = 1000) -> list[TrafficStateOut]:
    since = ctx.now() - timedelta(minutes=minutes)
    return [s for s in (_state(d) for d in ctx.store.traffic_history(approach_id, since, limit)) if s]


def summary(ctx: ServiceContext) -> SummaryOut:
    lv = live(ctx)
    wins = {s["approach_id"]: s for s in ctx.store.latest_traffic_states()}
    cams = net.camera_views(ctx)
    synth = {c.approach_id for c in cams if c.synthetic}
    vehicles, speed_w, n_speed, fresh, congested, latest = 0, 0.0, 0, 0, 0, None
    for aid in {a.id for a in ctx.network.approaches}:
        p = lv.get(aid)
        if p is not None and (ctx.age_s(p["_ts"]) or 1e9) <= ctx.settings.live_stale_s:
            fresh += 1
            vehicles += int(p["vehicle_count"])
            if p.get("average_speed_kmh") is not None and p["vehicle_count"]:
                speed_w += p["average_speed_kmh"] * p["vehicle_count"]
                n_speed += p["vehicle_count"]
            latest = max(latest, p["_ts"]) if latest else p["_ts"]
        w = wins.get(aid)
        if w and (ctx.age_s(w["timestamp"]) or 1e9) <= ctx.settings.window_stale_s and w["traffic_state"] in (
                "high", "severe"):
            congested += 1
    return SummaryOut(timestamp=latest, vehicles=vehicles if fresh else None,
                      avg_speed_kmh=round(speed_w / n_speed, 1) if n_speed else None, congested_approaches=congested,
                      monitored_approaches=len({c.approach_id for c in cams}), fresh_approaches=fresh,
                      active_incidents=len(ctx.store.list_incidents(active_only=True, limit=1000)),
                      synthetic_approaches=len(synth))


# --------------------------------------------------------------------------- forecasts
def forecast(ctx: ServiceContext, approach_id: str) -> ForecastOut:
    model = model_svc.production_forecaster(ctx)
    hist = ctx.store.traffic_history(approach_id, ctx.now() - timedelta(hours=3), 500)
    n = _consecutive_windows(hist)
    latest_state = hist[-1] if hist else None
    base = {"approach_id": approach_id, "model": model, "history_windows": n,
            "current_level": latest_state["traffic_state"] if latest_state else None}
    if model is None:
        return ForecastOut(status="no_model", message="Forecast unavailable - no PRODUCTION forecasting model in the "
                           "registry (train + promote one: python -m traffic_intel.ai.training ...).", **base)
    if not hist:
        return ForecastOut(status="no_data", message="Forecast unavailable - no traffic windows for this road yet.",
                           **base)
    fc = ctx.store.latest_forecast(approach_id)
    need = model.required_history or 0
    last_end = hist[-1]["timestamp"]
    if fc is None or (last_end - fc.issued_at).total_seconds() > 2 * 60:
        if n < need:
            return ForecastOut(status="insufficient_history", message=f"Forecast unavailable - insufficient "
                               f"historical data ({n}/{need} consecutive 1-min windows).", **base)
        return ForecastOut(status="stale", message="Forecast unavailable for the latest window (see pipeline log).",
                           **base)
    by_h: dict[int, dict[str, float]] = {}
    times: dict[int, datetime] = {}
    for p in fc.points:
        by_h.setdefault(p.horizon_min, {})[p.target] = round(p.value, 3)
        times[p.horizon_min] = p.target_time
    horizons = [ForecastHorizon(horizon_min=h, target_time=times[h], level=fc.predicted_level.get(h),
                                values=by_h[h]) for h in sorted(by_h)]
    return ForecastOut(status="ok", message=f"{fc.model_name} forecast issued {fc.issued_at:%H:%M:%S} UTC",
                       issued_at=fc.issued_at, horizons=horizons, **base)


def _consecutive_windows(hist: list[dict[str, Any]]) -> int:
    n = 0
    for a, b in zip(reversed(hist[:-1]), reversed(hist[1:]), strict=False):
        if abs((b["window_start"] - a["window_start"]).total_seconds() - a["window_s"]) > 1.5:
            break
        n += 1
    return n + (1 if hist else 0)


def forecast_vs_actual(ctx: ServiceContext, approach_id: str, horizon_min: int = 15, target: str = "avg_speed_kmh",
                       minutes: int = 180) -> ForecastVsActual:
    since = ctx.now() - timedelta(minutes=minutes)
    rows = ctx.store.forecast_history(approach_id, horizon_min, target, since)
    fc = [ForecastPointOut.model_validate(r) for r in rows]
    field = ACTUAL_FIELD.get(target, target)
    scale = 0.01 if target == "density_ratio" else 1.0
    actual = [{"timestamp": s["timestamp"], "value": None if s.get(field) is None else s[field] * scale}
              for s in ctx.store.traffic_history(approach_id, since, 1000)]
    return ForecastVsActual(approach_id=approach_id, horizon_min=horizon_min, target=target, forecast=fc,
                            actual=actual)


# --------------------------------------------------------------------------- snapshot
def snapshot(ctx: ServiceContext, approach_id: str) -> SnapshotOut | None:
    if approach_id not in ctx.approaches:
        return None
    cams = net.camera_views(ctx)
    appr = net.approach_views(ctx, cams)[approach_id]
    road = next((r for r in net.road_views(ctx) if r.id == appr.road_id), None)
    lv = live(ctx).get(approach_id)
    win = next((s for s in ctx.store.latest_traffic_states() if s["approach_id"] == approach_id), None)
    risk = ctx.store.latest_risk(approach_id=approach_id)
    incs = ctx.store.list_incidents(approach_id=approach_id, active_only=True, limit=1)
    mine = [c for c in cams if c.approach_id == approach_id]
    live_age = ctx.age_s(lv["_ts"]) if lv else None
    win_age = ctx.age_s(win["timestamp"]) if win else None
    return SnapshotOut(
        approach=appr, road=road, cameras=mine, live=_state(lv),
        live_stale=live_age is None or live_age > ctx.settings.live_stale_s,
        window=_state(win), window_stale=win_age is None or win_age > ctx.settings.window_stale_s,
        forecast=forecast(ctx, approach_id), incident=incident_svc.view(incs[0]) if incs else None,
        incident_score=risk[0].get("incident_score") if risk else None,
        risk=incident_svc.risk_view(risk[0]) if risk else None, synthetic=any(c.synthetic for c in mine))
