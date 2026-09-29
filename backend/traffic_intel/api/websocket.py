"""WebSocket gateway: bus -> browser.

Endpoints: ``/ws/traffic`` (dashboard) and ``/ws/live`` (same protocol).

Protocol (JSON):
  client -> server  {"action": "subscribe", "topics": ["traffic.update", ...],
                     "approach_ids": ["junction_a_north"] | null}
  server -> client  BusMessage  {"topic", "timestamp", "source", "payload"}

Default subscription: ``traffic.update`` (compact per-approach live/window values with forecast
levels, incident score and risk - see api/live.py), ``incident.update`` and ``camera.status``.
Messages come from the LiveHub (database changes) or any in-process publisher. The gateway only
reads from the bus - it never calls models - so the UI cannot slow down inference. Frame-level
detections are never sent.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from traffic_intel.api.live import CAMERA_STATUS, INCIDENT_UPDATE, TRAFFIC_UPDATE
from traffic_intel.core.config import get_settings
from traffic_intel.core.schemas import BusMessage
from traffic_intel.pipeline.bus import EventBus, Topics

log = logging.getLogger(__name__)
router = APIRouter()

DEFAULT_TOPICS = (TRAFFIC_UPDATE, INCIDENT_UPDATE, CAMERA_STATUS)
ALLOWED_TOPICS = frozenset({TRAFFIC_UPDATE, INCIDENT_UPDATE, CAMERA_STATUS, Topics.INCIDENT, Topics.RISK,
                            Topics.TRAFFIC_STATE, Topics.FEATURES_LIVE, Topics.ANOMALY_SIGNAL, Topics.FORECAST})


class _Subscription:
    def __init__(self) -> None:
        self.topics: set[str] = set(DEFAULT_TOPICS)
        self.approach_ids: set[str] | None = None
        self._last_live: dict[str, float] = {}

    def wants(self, msg: BusMessage, interval_s: float) -> bool:
        if msg.topic not in self.topics:
            return False
        approach = msg.payload.get("approach_id")
        if self.approach_ids is not None and approach is not None and approach not in self.approach_ids:
            return False
        if msg.topic == Topics.FEATURES_LIVE and approach is not None:
            now = time.monotonic()
            if now - self._last_live.get(approach, 0.0) < interval_s:
                return False
            self._last_live[approach] = now
        return True


@router.websocket("/ws/traffic")
@router.websocket("/ws/live")
async def live(ws: WebSocket) -> None:
    await ws.accept()
    ctx = getattr(ws.app.state, "ctx", None)
    if ctx is not None:
        ctx.ws_clients += 1
    bus: EventBus | None = getattr(ws.app.state, "bus", None)
    sub = _Subscription()
    interval = get_settings().ws_push_interval_s

    async def forward() -> None:
        if bus is None:
            return
        async for msg in bus.subscribe(*ALLOWED_TOPICS):
            if sub.wants(msg, interval):
                await ws.send_text(msg.model_dump_json())

    task = asyncio.create_task(forward())
    try:
        while True:
            req = await ws.receive_json()
            if req.get("action") == "subscribe":
                topics = set(req.get("topics") or DEFAULT_TOPICS) & ALLOWED_TOPICS
                sub.topics = topics or set(DEFAULT_TOPICS)
                ids = req.get("approach_ids")
                sub.approach_ids = set(ids) if ids else None
                await ws.send_json({"action": "subscribed", "topics": sorted(sub.topics),
                                    "approach_ids": sorted(sub.approach_ids) if sub.approach_ids else None})
    except WebSocketDisconnect:
        pass
    except Exception:  # noqa: BLE001
        log.exception("websocket connection error")
    finally:
        if ctx is not None:
            ctx.ws_clients -= 1
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
