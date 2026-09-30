"""System status from real signals: database ping, camera heartbeats, model registry, data freshness."""

from __future__ import annotations

from traffic_intel import __version__
from traffic_intel.api.schemas import ComponentStatus, SystemStatusOut
from traffic_intel.services import models as model_svc
from traffic_intel.services import network as net
from traffic_intel.services.context import ServiceContext


def status(ctx: ServiceContext) -> SystemStatusOut:
    s = ctx.settings
    db_ok = ctx.store.ping()
    database = ComponentStatus(ok=db_ok, status="connected" if db_ok else "disconnected",
                               detail=f"{ctx.store.backend}: {ctx.store.url}")
    cams = net.camera_views(ctx) if db_ok else []
    online = [c for c in cams if c.status == "online"]
    errors = [c for c in cams if c.status == "error"]
    cameras = ComponentStatus(
        ok=bool(online) and not errors, status="online" if online else ("error" if errors else "offline"),
        detail=f"{len(online)}/{len(cams)} online" + (f"; errors: {', '.join(c.id for c in errors)}" if errors else ""))

    windows = ctx.store.latest_traffic_states() if db_ok else []
    newest = max((w["timestamp"] for w in windows), default=None)
    age = ctx.age_s(newest)
    running = bool(online) or (age is not None and age <= s.window_stale_s)
    pipeline = ComponentStatus(ok=running, status="running" if running else ("stale" if windows else "no_data"),
                               # replayed sources stamp start + source time; a system-clock step after the
                               # start can put that a few seconds ahead of now (same clamp as the dashboard)
                               detail=("last window " + (f"{max(age, 0.0):.0f} s ago" if age is not None else "never")
                                       + ("" if running else " - start it with: python -m traffic_intel.pipeline run")))

    model = model_svc.production_forecaster(ctx)
    fc_age = None
    if model is not None and windows:
        issued = [f.issued_at for w in windows if (f := ctx.store.latest_forecast(w["approach_id"])) is not None]
        fc_age = ctx.age_s(max(issued)) if issued else None
    serving = fc_age is not None and fc_age <= s.window_stale_s
    forecast_model = ComponentStatus(
        ok=model is not None, status=("loaded" if serving else "available") if model else "not_loaded",
        detail=(f"{model.name} {model.version} (trained on {model.trained_on})"
                + ("" if serving else " - no forecast issued recently (needs history / running pipeline)"))
        if model else "no PRODUCTION forecasting model in models/forecasting/")

    weather_now = ctx.store.latest_context(ctx.network.name) if db_ok else None
    w_age = ctx.age_s(weather_now["ts"]) if weather_now else None
    w_detail = (f"{weather_now['weather']} {weather_now['temperature_c']} C ({weather_now['source']})"
                if weather_now else "no weather observation (optional context)")
    weather = ComponentStatus(ok=weather_now is not None and (w_age or 0) < 3 * 3600,
                              status="available" if weather_now else "unavailable", detail=w_detail)
    return SystemStatusOut(
        backend=ComponentStatus(ok=True, status="online", detail=f"traffic_intel {__version__}"),
        database=database, pipeline=pipeline, forecast_model=forecast_model, cameras=cameras, weather=weather,
        websocket_clients=ctx.ws_clients, version=__version__, env=s.env,
        models=model_svc.list_models(ctx, production_only=True), weather_now=weather_now)
