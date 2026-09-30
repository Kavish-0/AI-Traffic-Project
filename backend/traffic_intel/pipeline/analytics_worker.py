"""Window-rate analytics loop: camera-worker aggregates in, incidents and risk out.

Single async task for the whole network (work per window is small):

    on ANOMALY_SIGNAL(approach):   buffer (+ persist) track-level evidence
    on FEATURES_WINDOW + TRAFFIC_STATE of the same (approach, window):
        engine.process_window(features, state)
            series detectors -> fusion (incident probability, type) -> forecast -> risk
        publish INCIDENT (when one is active or just resolved), RISK, FORECAST
        (series-level signals are persisted by the engine's store, not re-published)

The camera worker publishes features and state as two messages; they are paired by
(approach_id, window end). The heavy lifting lives in ``IntelligenceEngine`` so the same
code runs here, in the offline ``ingest`` CLI, in the scenarios and in the tests.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from traffic_intel.core.config import NetworkConfig, PipelineConfig
from traffic_intel.core.schemas import AnomalySignal, BusMessage, TrafficFeatures, TrafficState
from traffic_intel.incidents.engine import IntelligenceEngine, IntelligenceUpdate
from traffic_intel.pipeline.bus import EventBus, Topics

log = logging.getLogger(__name__)


class AnalyticsWorker:
    def __init__(self, network: NetworkConfig, pipeline: PipelineConfig, bus: EventBus, *,
                 engine: IntelligenceEngine | None = None, store: Any | None = None,
                 with_forecast: bool = True) -> None:
        self.network = network
        self.pipeline = pipeline
        self.bus = bus
        self.engine = engine or IntelligenceEngine.build(network, pipeline, store=store, with_forecast=with_forecast)
        self._pending_features: dict[tuple[str, datetime], TrafficFeatures] = {}
        self._pending_states: dict[tuple[str, datetime], TrafficState] = {}
        self.updates = 0

    async def handle(self, msg: BusMessage) -> IntelligenceUpdate | None:
        if msg.topic == Topics.ANOMALY_SIGNAL:
            self.engine.ingest_signals([AnomalySignal.model_validate(msg.payload)])
            return None
        if msg.topic == Topics.FEATURES_WINDOW:
            f = TrafficFeatures.model_validate(msg.payload)
            key = (f.approach_id, f.window_start + timedelta(seconds=f.window_s))
            self._pending_features[key] = f
        elif msg.topic == Topics.TRAFFIC_STATE:
            s = TrafficState.model_validate(msg.payload)
            if s.is_live:
                return None
            key = (s.approach_id, s.timestamp)
            self._pending_states[key] = s
        else:
            return None
        if key not in self._pending_features or key not in self._pending_states:
            return None
        update = self.engine.process_window(self._pending_features.pop(key), self._pending_states.pop(key))
        await self._publish(update)
        self.updates += 1
        return update

    async def _publish(self, u: IntelligenceUpdate) -> None:
        now = datetime.now(UTC)
        if u.incident is not None:
            await self.bus.publish(BusMessage(topic=Topics.INCIDENT, timestamp=now, source="analytics",
                                              payload=u.incident.model_dump(mode="json")))
        if u.risk is not None:
            await self.bus.publish(BusMessage(topic=Topics.RISK, timestamp=now, source="analytics",
                                              payload=u.risk.model_dump(mode="json")))
        if u.forecast is not None:
            await self.bus.publish(BusMessage(topic=Topics.FORECAST, timestamp=now, source="analytics",
                                              payload=u.forecast.model_dump(mode="json")))

    async def run(self) -> None:
        async for msg in self.bus.subscribe(Topics.FEATURES_WINDOW, Topics.TRAFFIC_STATE, Topics.ANOMALY_SIGNAL):
            try:
                await self.handle(msg)
            except Exception:  # noqa: BLE001 - one bad message must not stop the loop
                log.exception("analytics: failed to process %s", msg.topic)
