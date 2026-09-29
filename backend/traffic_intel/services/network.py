"""Junctions, roads, approaches, cameras and the map GeoJSON.

Geometry and topology come from the network/camera configuration (source of truth, also synced
into the database); dynamic state (levels, risk, incidents, camera status) from the store.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from traffic_intel.api.schemas import ApproachOut, CameraOut, JunctionOut, RoadOut
from traffic_intel.services.context import ServiceContext

_LEVEL_ORDER = ["free", "moderate", "high", "severe"]


def _fresh_levels(ctx: ServiceContext) -> dict[str, dict[str, Any]]:
    """approach_id -> latest window state, only if not stale."""
    out = {}
    for st in ctx.store.latest_traffic_states():
        age = ctx.age_s(st["timestamp"])
        out[st["approach_id"]] = {**st, "stale": age is None or age > ctx.settings.window_stale_s}
    return out


def _risk_levels(ctx: ServiceContext) -> dict[str, dict[str, Any]]:
    return {r["approach_id"]: r for r in ctx.store.latest_risk()}


def _active_incidents(ctx: ServiceContext) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for inc in ctx.store.list_incidents(active_only=True, limit=500):
        out.setdefault(inc["approach_id"], inc)
    return out


# --------------------------------------------------------------------------- cameras
def camera_views(ctx: ServiceContext) -> list[CameraOut]:
    statuses = ctx.store.camera_statuses()
    db = ctx.store.cameras()
    ids = list(ctx.cameras) + [c for c in db if c not in ctx.cameras]
    appr = ctx.approaches
    out = []
    for cid in ids:
        cfg = ctx.cameras.get(cid)
        row = db.get(cid, {})
        st = statuses.get(cid)
        details = (st or {}).get("details") or {}
        approach_id = cfg.approach_id if cfg else row.get("approach_id", "")
        source = cfg.source if cfg else row.get("source", "")
        kind = (st or {}).get("source_kind") or (_kind(source, cfg.source_type if cfg else "auto"))
        age = ctx.age_s((st or {}).get("updated_at"))
        if st is None:
            status = "never_started"
        elif st["state"] == "error":
            status = "error"
        elif st["state"] == "running" and age is not None and age <= ctx.settings.camera_offline_s:
            status = "online"
        else:
            status = "offline"
        preview = (st or {}).get("preview_path")
        out.append(CameraOut(
            id=cid, name=(cfg.name if cfg else details.get("name") or row.get("name") or cid), approach_id=approach_id,
            junction_id=appr[approach_id].junction_id if approach_id in appr else None,
            source_kind=kind if kind in ("file", "webcam", "stream", "synthetic") else "unknown",
            synthetic=kind == "synthetic", configured=cfg is not None,
            lon=cfg.location[0] if cfg else row.get("lon"), lat=cfg.location[1] if cfg else row.get("lat"),
            status=status,
            state=(st or {}).get("state"), last_frame_at=(st or {}).get("last_frame_at"),
            updated_at=(st or {}).get("updated_at"), processing_fps=(st or {}).get("processing_fps"),
            frames=(st or {}).get("frames"), detector=(st or {}).get("detector"), device=(st or {}).get("device"),
            error=(st or {}).get("error"), preview_available=bool(preview and Path(preview).exists())))
    return out


def _kind(source: str, source_type: str) -> str:
    if source_type != "auto":
        return source_type
    if source.startswith("synthetic://"):
        return "synthetic"
    if source.startswith("webcam:"):
        return "webcam"
    return "stream" if "://" in source else "file"


def camera_view(ctx: ServiceContext, camera_id: str) -> CameraOut | None:
    return next((c for c in camera_views(ctx) if c.id == camera_id), None)


def preview_file(ctx: ServiceContext, camera_id: str) -> tuple[Path, datetime | None] | None:
    st = ctx.store.camera_statuses().get(camera_id)
    if not st or not st.get("preview_path"):
        return None
    p = Path(st["preview_path"])
    return (p, st.get("last_frame_at")) if p.exists() else None


# --------------------------------------------------------------------------- junctions / roads
def approach_views(ctx: ServiceContext, cams: list[CameraOut] | None = None) -> dict[str, ApproachOut]:
    cams = cams if cams is not None else camera_views(ctx)
    levels, risk, incs = _fresh_levels(ctx), _risk_levels(ctx), _active_incidents(ctx)
    out = {}
    for a in ctx.network.approaches:
        lv = levels.get(a.id)
        rk = risk.get(a.id)
        out[a.id] = ApproachOut(
            id=a.id, name=a.name, direction=a.direction, junction_id=a.junction_id, road_id=a.road_segment_id,
            cameras=[c.id for c in cams if c.approach_id == a.id],
            level=lv["traffic_state"] if lv and not lv["stale"] else None,
            risk_level=rk["level"] if rk and lv and not lv["stale"] else None,
            active_incident=incs[a.id]["incident_id"] if a.id in incs else None)
    return out


def junction_views(ctx: ServiceContext) -> list[JunctionOut]:
    appr = approach_views(ctx)
    out = []
    for j in ctx.network.junctions:
        mine = [a for a in appr.values() if a.junction_id == j.id]
        levels = [a.level for a in mine if a.level]
        out.append(JunctionOut(id=j.id, name=j.name, lon=j.location[0], lat=j.location[1], signalized=j.signalized,
                               approaches=mine, active_incidents=sum(1 for a in mine if a.active_incident),
                               worst_level=max(levels, key=_LEVEL_ORDER.index) if levels else None))
    return out


def road_views(ctx: ServiceContext) -> list[RoadOut]:
    levels = _fresh_levels(ctx)
    by_road = {a.road_segment_id: a for a in ctx.network.approaches}
    covered = {c["approach_id"] for c in ctx.store.cameras().values()} | {c.approach_id for c in ctx.cameras.values()}
    out = []
    for g in ctx.network.segments:
        a = by_road.get(g.id)
        lv = levels.get(a.id) if a else None
        out.append(RoadOut(
            id=g.id, name=g.name, from_node=g.from_node, to_node=g.to_node, lanes=g.lanes,
            speed_limit_kmh=g.speed_limit_kmh, capacity_veh_h=g.capacity_veh_h, importance=g.importance,
            approach_id=a.id if a else None, junction_id=a.junction_id if a else None,
            monitored=a is not None and a.id in covered,
            level=lv["traffic_state"] if lv and not lv["stale"] else None, stale=lv is None or lv["stale"],
            coordinates=[list(p) for p in g.geometry]))
    return out


# --------------------------------------------------------------------------- map
def network_geojson(ctx: ServiceContext) -> dict[str, Any]:
    """Roads (LineString, coloured by the latest FRESH level), junctions and cameras (Points)."""
    cams = camera_views(ctx)
    appr = approach_views(ctx, cams)
    risk = _risk_levels(ctx)
    feats: list[dict[str, Any]] = []
    for r in road_views(ctx):
        a = appr.get(r.approach_id) if r.approach_id else None
        feats.append({"type": "Feature", "id": r.id, "geometry": {"type": "LineString", "coordinates": r.coordinates},
                      "properties": {"kind": "road", "id": r.id, "name": r.name, "approach_id": r.approach_id,
                                     "junction_id": r.junction_id, "monitored": r.monitored,
                                     "level": r.level or "unknown", "stale": r.stale,
                                     "risk_level": a.risk_level if a else None,
                                     "risk_score": risk[a.id]["risk_score"] if a and a.risk_level else None,
                                     "incident": a.active_incident if a else None}})
    for inc in ctx.store.list_incidents(active_only=True, limit=200):
        g = ctx.segments.get(inc["road_id"] or "")
        if inc["lon"] is not None and inc["lat"] is not None:
            pt = [inc["lon"], inc["lat"]]
        elif g is not None:
            (x0, y0), (x1, y1) = g.geometry[0], g.geometry[-1]
            pt = [(x0 + x1) / 2, (y0 + y1) / 2]
        else:
            continue
        feats.append({"type": "Feature", "id": f"incident:{inc['incident_id']}",
                      "geometry": {"type": "Point", "coordinates": pt},
                      "properties": {"kind": "incident", "id": inc["incident_id"], "approach_id": inc["approach_id"],
                                     "incident_type": inc["incident_type"], "score": inc["probability"],
                                     "calibrated": inc["calibrated"]}})
    for j in ctx.network.junctions:
        feats.append({"type": "Feature", "id": j.id, "geometry": {"type": "Point", "coordinates": list(j.location)},
                      "properties": {"kind": "junction", "id": j.id, "name": j.name}})
    for c in cams:
        if c.lon is not None:
            feats.append({"type": "Feature", "id": c.id, "geometry": {"type": "Point", "coordinates": [c.lon, c.lat]},
                          "properties": {"kind": "camera", "id": c.id, "name": c.name, "status": c.status,
                                         "approach_id": c.approach_id, "synthetic": c.synthetic}})
    lons = [p[0] for g in ctx.network.segments for p in g.geometry]
    lats = [p[1] for g in ctx.network.segments for p in g.geometry]
    return {"type": "FeatureCollection", "features": feats,
            "bbox": [min(lons), min(lats), max(lons), max(lats)] if lons else None}
