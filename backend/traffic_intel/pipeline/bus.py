"""Event bus: the only channel between perception, analytics and the API.

* ``InMemoryEventBus`` - asyncio fan-out inside one process (dev mode,
  tests, offline replay). No external services needed.
* ``RedisStreamsEventBus`` - Phase 9, only when camera workers run as
  separate processes/machines. Redis is introduced *only* for that reason.

Messages are ``BusMessage`` envelopes (JSON-serialisable) so both backends
behave identically. Frame-rate data never goes on the bus.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import AsyncIterator

from traffic_intel.core.schemas import BusMessage


class Topics:
    FEATURES_LIVE = "traffic.features.live"        # ~1 Hz per approach, not persisted
    TRAFFIC_STATE = "traffic.state"                # TrafficState per closed window (and live)
    FEATURES_WINDOW = "traffic.features.window"    # 60 s windows, persisted
    TRACK_SUMMARY = "traffic.tracks.finished"
    ANOMALY_SIGNAL = "anomaly.signal"
    CONGESTION = "traffic.congestion"
    FORECAST = "traffic.forecast"
    INCIDENT = "incident.update"
    RISK = "risk.update"
    CAMERA_STATUS = "camera.status"                # heartbeat, fps, latency, errors


class EventBus(ABC):
    @abstractmethod
    async def publish(self, message: BusMessage) -> None: ...

    @abstractmethod
    def subscribe(self, *topics: str) -> AsyncIterator[BusMessage]: ...

    async def close(self) -> None:  # noqa: B027 - optional hook
        pass


class InMemoryEventBus(EventBus):
    """Bounded per-subscriber queues. A slow subscriber (e.g. a stalled
    WebSocket) loses its oldest messages instead of blocking producers."""

    def __init__(self, max_queue: int = 1000) -> None:
        self._max_queue = max_queue
        self._subscribers: dict[str, list[asyncio.Queue[BusMessage]]] = defaultdict(list)

    async def publish(self, message: BusMessage) -> None:
        for queue in list(self._subscribers.get(message.topic, [])):
            if queue.full():
                queue.get_nowait()
            queue.put_nowait(message)

    async def subscribe(self, *topics: str) -> AsyncIterator[BusMessage]:
        queue: asyncio.Queue[BusMessage] = asyncio.Queue(self._max_queue)
        for topic in topics:
            self._subscribers[topic].append(queue)
        try:
            while True:
                yield await queue.get()
        finally:
            for topic in topics:
                self._subscribers[topic].remove(queue)
