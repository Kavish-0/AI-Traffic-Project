"""What a camera worker may publish, and how it reaches the (async) event bus.

A camera worker runs in its own thread/process; frame-level data (detections,
tracks, trajectories) never leaves it (they go to Parquet/annotated video
only). The ONLY messages it publishes are the designed aggregates:

    traffic.features.live    TrafficState, rolling live window, about every live_tick_s (1 s)
    traffic.features.window  TrafficFeatures, one per closed 60 s window
    traffic.state            TrafficState,    one per closed 60 s window
    anomaly.signal           AnomalySignal,   track-level evidence (stopped / wrong-way / erratic vehicle);
                             rate-limited by the detectors (re-emitted at most every few seconds per track)

``WorkerPublisher`` enforces that whitelist, so a future change cannot leak
frame-rate data onto the bus by accident.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from concurrent.futures import Future
from datetime import UTC, datetime

from pydantic import BaseModel

from traffic_intel.core.schemas import BusMessage
from traffic_intel.pipeline.bus import EventBus, Topics

log = logging.getLogger(__name__)

WORKER_TOPICS: frozenset[str] = frozenset(
    {Topics.FEATURES_LIVE, Topics.FEATURES_WINDOW, Topics.TRAFFIC_STATE, Topics.ANOMALY_SIGNAL})


class Publisher(ABC):
    @abstractmethod
    def send(self, message: BusMessage) -> None: ...

    def close(self) -> None:  # noqa: B027 - optional hook
        pass


class NullPublisher(Publisher):
    def send(self, message: BusMessage) -> None:
        pass


class CollectingPublisher(Publisher):
    """Keeps messages in memory (tests, offline runs that want the published stream)."""

    def __init__(self) -> None:
        self.messages: list[BusMessage] = []

    def send(self, message: BusMessage) -> None:
        self.messages.append(message)


class BusPublisher(Publisher):
    """Thread-safe bridge from a worker thread to an asyncio ``EventBus`` running on ``loop``."""

    def __init__(self, bus: EventBus, loop: asyncio.AbstractEventLoop, timeout_s: float = 5.0) -> None:
        self.bus = bus
        self.loop = loop
        self.timeout_s = timeout_s
        self._pending: list[Future[None]] = []

    def send(self, message: BusMessage) -> None:
        fut = asyncio.run_coroutine_threadsafe(self.bus.publish(message), self.loop)
        self._pending = [f for f in self._pending if not f.done()] + [fut]

    def close(self) -> None:
        for fut in self._pending:
            try:
                fut.result(timeout=self.timeout_s)
            except Exception:  # noqa: BLE001
                log.exception("bus publish failed")


class WorkerPublisher:
    """Wraps a Publisher; only the designed aggregate topics are allowed."""

    def __init__(self, publisher: Publisher, source: str) -> None:
        self.publisher = publisher
        self.source = source
        self.counts: dict[str, int] = dict.fromkeys(WORKER_TOPICS, 0)

    def publish(self, topic: str, payload: BaseModel) -> None:
        if topic not in WORKER_TOPICS:
            raise ValueError(f"camera workers may not publish '{topic}' (allowed: {sorted(WORKER_TOPICS)})")
        self.publisher.send(BusMessage(
            topic=topic, timestamp=datetime.now(UTC), source=self.source,
            payload=payload.model_dump(mode="json"),
        ))
        self.counts[topic] += 1

    def close(self) -> None:
        self.publisher.close()
