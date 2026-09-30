"""LiveHub: database -> in-process bus -> WebSocket clients.

The pipeline runtime is a separate process (it loads torch; the API must not), so the API learns
about new data from the database: every ``ws_push_interval_s`` the hub reads what changed since
its last poll and publishes compact messages on the API's in-memory bus; ``api/websocket.py``
forwards them to subscribed browsers.

    traffic.update   one approach: live (~1 s) or window (60 s) values + traffic state, forecast
                     levels, incident score (+ calibrated flag), risk - only what the dashboard shows
    incident.update  an incident was opened / updated / resolved / acknowledged (IncidentOut)
    camera.status    all cameras' status (every few seconds)

Frame-level detections / tracks are never sent.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from traffic_intel.core.schemas import BusMessage
from traffic_intel.pipeline.bus import EventBus
from traffic_intel.services import incidents as incident_svc
from traffic_intel.services import network as net
from traffic_intel.services.context import ServiceContext

log = logging.getLogger(__name__)

TRAFFIC_UPDATE = "traffic.update"
INCIDENT_UPDATE = "incident.update"
CAMERA_STATUS = "camera.status"


def compact(state: dict[str, Any], *, is_live: bool, forecast: dict[int, str], risk: dict[str, Any] | None,
            incident: dict[str, Any] | None, synthetic: bool) -> dict[str, Any]:
    ts = state.get("_ts") or state["timestamp"]
    return {
        "approach_id": state["approach_id"], "junction_id": state.get("junction_id"), "road_id": state.get("road_id"),
        "timestamp": ts.isoformat() if isinstance(ts, datetime) else ts, "is_live": is_live,
        "vehicle_count": state["vehicle_count"], "average_speed": state.get("average_speed_kmh"),
        "density": state.get("density_pct"), "occupancy": state.get("occupancy_pct"), "flow": state.get("flow_veh_h"),
        "queue_length": state.get("queue_length_m"), "stopped": state.get("stopped_vehicle_count"),
        "traffic_state": state.get("traffic_state"),
        "forecast": {str(h): lvl for h, lvl in forecast.items()},
        "incident_score": risk.get("incident_score") if risk else None,
        "incident_calibrated": bool(incident and incident["calibrated"]),
        "incident_type": incident["incident_type"] if incident else None,
        "risk_level": risk["level"] if risk else None, "risk_score": risk["risk_score"] if risk else None,
        "synthetic": synthetic,
    }


LOOKBACK = timedelta(minutes=30)   # how far back a lagging camera's windows / incident updates are still pushed


def _latest(rows: list[dict[str, Any]], key: str, ts_key: str) -> dict[str, datetime]:
    out: dict[str, datetime] = {}
    for r in rows:
        _mark(out, r[key], r[ts_key])
    return out


def _mark(seen: dict[str, datetime], key: str, ts: datetime) -> None:
    if key not in seen or ts > seen[key]:
        seen[key] = ts


def _unseen(rows: list[dict[str, Any]], key: str, seen: dict[str, datetime], *, newer_only: bool,
            ts_key: str = "timestamp") -> list[dict[str, Any]]:
    """Rows not pushed yet: newer than the last one pushed for the same key (windows, incidents), or with a
    different timestamp (live states, which are replaced in place)."""
    return [r for r in rows if r[key] not in seen
            or (r[ts_key] > seen[r[key]] if newer_only else r[ts_key] != seen[r[key]])]


class LiveHub:
    def __init__(self, ctx: ServiceContext, bus: EventBus, interval_s: float = 1.0) -> None:
        self.ctx, self.bus, self.interval_s = ctx, bus, interval_s
        # What was already pushed, per approach / incident. Every camera writes its own timestamps and cameras
        # lag each other, so one global "newest timestamp" watermark would silently drop a slower camera's
        # updates. None = not seeded yet: the first poll records existing rows without replaying history.
        self._live_seen: dict[str, datetime] | None = None
        self._window_seen: dict[str, datetime] | None = None
        self._incident_seen: dict[str, datetime] | None = None
        self.last_cams = 0.0
        self._forecast: dict[str, dict[int, str]] = {}
        self._risk: dict[str, dict[str, Any]] = {}
        self._incident: dict[str, dict[str, Any]] = {}
        self._synthetic: set[str] = set()

    def _refresh(self) -> None:
        st = self.ctx.store
        self._risk = {r["approach_id"]: r for r in st.latest_risk()}
        self._incident = {}
        for i in st.list_incidents(active_only=True, limit=500):
            self._incident.setdefault(i["approach_id"], i)
        for a in self.ctx.network.approaches:
            fc = st.latest_forecast(a.id)
            self._forecast[a.id] = {h: lvl.value for h, lvl in fc.predicted_level.items()} if fc else {}

    def poll(self) -> list[BusMessage]:
        st, now = self.ctx.store, datetime.now(UTC)
        out: list[BusMessage] = []
        if now.timestamp() - self.last_cams >= 5.0:
            self.last_cams = now.timestamp()
            cams = net.camera_views(self.ctx)
            self._synthetic = {c.approach_id for c in cams if c.synthetic}
            out.append(self._msg(CAMERA_STATUS, {"cameras": [c.model_dump(mode="json") for c in cams]}))
        since = now - LOOKBACK
        all_windows = st.windows_since(since)
        all_incidents = st.incidents_updated_since(since)
        all_lives = st.latest_live_states()
        if self._window_seen is None or self._incident_seen is None or self._live_seen is None:
            # first poll: remember what is already stored instead of replaying it
            self._window_seen = _latest(all_windows, "approach_id", "timestamp")
            self._incident_seen = _latest(all_incidents, "incident_id", "timestamp")
            self._live_seen = _latest(all_lives, "approach_id", "_ts")
            if not self._risk:
                self._refresh()
            return out
        windows = _unseen(all_windows, "approach_id", self._window_seen, newer_only=True)
        incidents = _unseen(all_incidents, "incident_id", self._incident_seen, newer_only=True)
        # one row per approach (upserted in place): any change of its timestamp is a new live state
        lives = _unseen(all_lives, "approach_id", self._live_seen, newer_only=False, ts_key="_ts")
        if windows or incidents or not self._risk:
            self._refresh()
        for w in windows:
            _mark(self._window_seen, w["approach_id"], w["timestamp"])
            out.append(self._msg(TRAFFIC_UPDATE, self._compact(w, is_live=False)))
        for p in lives:
            self._live_seen[p["approach_id"]] = p["_ts"]
            out.append(self._msg(TRAFFIC_UPDATE, self._compact(p, is_live=True)))
        for i in incidents:
            _mark(self._incident_seen, i["incident_id"], i["timestamp"])
            out.append(self._msg(INCIDENT_UPDATE, incident_svc.view(i).model_dump(mode="json")))
        return out

    def _compact(self, state: dict[str, Any], *, is_live: bool) -> dict[str, Any]:
        aid = state["approach_id"]
        return compact(state, is_live=is_live, forecast=self._forecast.get(aid, {}), risk=self._risk.get(aid),
                       incident=self._incident.get(aid), synthetic=aid in self._synthetic)

    @staticmethod
    def _msg(topic: str, payload: dict[str, Any]) -> BusMessage:
        return BusMessage(topic=topic, timestamp=datetime.now(UTC), source="api.live_hub", payload=payload)

    async def run(self) -> None:
        log.info("live hub: polling the database every %.1f s", self.interval_s)
        while True:
            try:
                for m in await asyncio.to_thread(self.poll):
                    await self.bus.publish(m)
            except Exception:  # noqa: BLE001 - keep pushing; the dashboard shows the database state
                log.exception("live hub poll failed")
            await asyncio.sleep(self.interval_s)
