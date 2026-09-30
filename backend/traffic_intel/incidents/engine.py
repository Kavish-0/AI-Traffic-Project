"""Incident-intelligence engine: one closed window in -> signals, incident, risk out.

    TrafficFeatures (60 s window) + TrafficState (congestion verdict)
        -> history ring buffer per approach
        -> series detectors (sudden_slowdown, queue_growth, density_zscore, flow_imbalance, ewma, zscore,
           [isolation_forest, autoencoder])                              -> AnomalySignal (series)
        + buffered track signals from the camera worker (stopped_vehicle, wrong_way, trajectory_rules)
        -> incident fuser (weighted logistic, hysteresis, taxonomy)     -> IncidentAssessment | None
        -> forecaster (optional, PRODUCTION model)                      -> Forecast | None
        -> risk engine (weighted, documented)                           -> RiskAssessment
        -> TrafficStore (optional; database/store.py)

The engine holds only small in-memory state (last ``history_windows`` windows and
``signal_retention_s`` of signals per approach) and is independent of the bus, the
API and torch: the analytics worker, the scenario runner and the tests all drive
the same ``process_window`` method.
"""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from traffic_intel.anomaly.interfaces import (
    INCIDENT_FUSERS,
    SERIES_ANOMALY_DETECTORS,
    IncidentFuser,
    SeriesAnomalyDetector,
)
from traffic_intel.core.config import NetworkConfig, PipelineConfig
from traffic_intel.core.enums import CongestionLevel, IncidentStatus
from traffic_intel.core.schemas import (
    AnomalySignal,
    CongestionAssessment,
    Forecast,
    IncidentAssessment,
    RiskAssessment,
    TrafficFeatures,
    TrafficState,
)
from traffic_intel.incidents.convert import features_from_state
from traffic_intel.risk.interfaces import RISK_ENGINES, RiskEngine

__all__ = ["IntelligenceEngine", "IntelligenceUpdate", "features_from_state"]

log = logging.getLogger(__name__)

HISTORY_COLUMNS = ("avg_speed_kmh", "speed_p85_kmh", "median_speed_kmh", "density_veh_km", "density_ratio",
                   "occupancy_pct", "flow_veh_h", "inflow", "outflow", "queue_length_m", "stopped_vehicles",
                   "vehicle_count", "pcu_count", "quality")


@dataclass
class IntelligenceUpdate:
    approach_id: str
    timestamp: datetime
    signals: list[AnomalySignal] = field(default_factory=list)       # new series signals of this window
    incident: IncidentAssessment | None = None
    risk: RiskAssessment | None = None
    forecast: Forecast | None = None

    def summary(self) -> str:
        inc = self.incident
        inc_txt = "normal" if inc is None else f"{inc.incident_type} p={inc.probability:.0%} [{inc.status.value}]"
        r = self.risk
        risk_txt = "-" if r is None else f"{r.risk_score:.0f} {r.level.value} {','.join(r.reason_codes)}"
        sig = ",".join(sorted({s.type.value for s in self.signals})) or "-"
        return f"{self.timestamp:%H:%M:%S} {self.approach_id}: signals[{sig}] incident[{inc_txt}] risk[{risk_txt}]"


class _ApproachBuffer:
    def __init__(self, n: int) -> None:
        self.rows: deque[dict[str, Any]] = deque(maxlen=n)
        self.states: deque[TrafficState] = deque(maxlen=n)
        self.signals: list[AnomalySignal] = []
        self.incident: IncidentAssessment | None = None


class IntelligenceEngine:
    def __init__(self, network: NetworkConfig, *, series_detectors: Sequence[SeriesAnomalyDetector],
                 fuser: IncidentFuser, risk_engine: RiskEngine, forecaster: Any | None = None,
                 store: Any | None = None, history_windows: int = 60, signal_retention_s: float = 600.0) -> None:
        self.network = network
        self.series_detectors = list(series_detectors)
        self.fuser = fuser
        self.risk_engine = risk_engine
        self.forecaster = forecaster            # ForecastingService-like: .predict(history) -> .to_forecast()
        self.store = store                      # TrafficStore-like (database/store.py)
        self.history_windows = history_windows
        self.signal_retention = timedelta(seconds=signal_retention_s)
        self._buf: dict[str, _ApproachBuffer] = {}
        seg = {s.id: s for s in network.segments}
        self.meta: dict[str, dict[str, Any]] = {}
        for a in network.approaches:
            s = seg[a.road_segment_id]
            self.meta[a.id] = {"junction_id": a.junction_id, "road_id": a.road_segment_id,
                               "name": f"{a.name} ({a.direction})", "importance": s.importance,
                               "free_flow_kmh": s.free_flow_speed_kmh or s.speed_limit_kmh}
        if hasattr(self.fuser, "meta") and not self.fuser.meta:
            self.fuser.meta = self.meta

    @classmethod
    def build(cls, network: NetworkConfig, pipeline: PipelineConfig, *, store: Any | None = None,
              with_forecast: bool = True, **kw: Any) -> IntelligenceEngine:
        detectors = []
        for spec in pipeline.series_anomaly_detectors:
            try:
                detectors.append(SERIES_ANOMALY_DETECTORS.create(spec))
            except (FileNotFoundError, KeyError) as exc:   # learned detector not trained yet
                log.warning("series detector %s unavailable (%s) - skipped", spec.type, exc)
        forecaster = None
        if with_forecast:
            try:
                from traffic_intel.forecasting.service import ForecastingService
                forecaster = ForecastingService.from_registry(**pipeline.forecaster.params)
            except Exception as exc:  # noqa: BLE001 - forecasting is optional for incident intelligence
                log.warning("no forecasting model available (%s) - risk uses current state only", exc)
        return cls(network, series_detectors=detectors, fuser=INCIDENT_FUSERS.create(pipeline.incident_fuser),
                   risk_engine=RISK_ENGINES.create(pipeline.risk_engine), forecaster=forecaster, store=store, **kw)

    # ------------------------------------------------------------------ #
    def _b(self, approach_id: str) -> _ApproachBuffer:
        b = self._buf.get(approach_id)
        if b is None:
            b = self._buf[approach_id] = _ApproachBuffer(self.history_windows)
            if self.store is not None:
                b.incident = self.store.open_incident_for(approach_id)
        return b

    def ingest_signals(self, signals: Iterable[AnomalySignal]) -> None:
        """Track-level signals from camera workers (buffered until the next window is fused)."""
        sigs = list(signals)
        for s in sigs:
            self._b(s.approach_id).signals.append(s)
        if self.store is not None and sigs:
            self.store.add_anomalies(sigs)

    def history_frame(self, approach_id: str) -> pd.DataFrame:
        return pd.DataFrame(list(self._b(approach_id).rows))

    def process_window(self, features: TrafficFeatures, state: TrafficState | None = None,
                       congestion: CongestionAssessment | None = None) -> IntelligenceUpdate:
        aid = features.approach_id
        b = self._b(aid)
        end = features.window_start + timedelta(seconds=features.window_s)
        # numeric feature-engine extras too (mean_vehicles_in_region, queue_exceeds_view_fraction, ...)
        row = {"timestamp": end, **{k: v for k, v in features.extra.items() if isinstance(v, int | float)},
               **{c: getattr(features, c) for c in HISTORY_COLUMNS}}
        b.rows.append(row)
        if state is not None:
            b.states.append(state)
        if self.store is not None:
            self.store.add_window(features, state)
        if congestion is None and state is not None:
            congestion = CongestionAssessment(approach_id=aid, timestamp=state.timestamp, level=state.traffic_state,
                                              score=state.congestion_score, classifier="traffic_state",
                                              explanation=state.explanation)
        history = self.history_frame(aid)
        new: list[AnomalySignal] = []
        for d in self.series_detectors:
            try:
                new.extend(d.score(aid, history))
            except Exception:  # noqa: BLE001 - one broken detector must not stop the others
                log.exception("series detector %s failed on %s", getattr(d, "name", d), aid)
        b.signals.extend(new)
        b.signals = [s for s in b.signals if end - s.timestamp <= self.signal_retention]
        if self.store is not None and new:
            self.store.add_anomalies(new)

        level: CongestionLevel | None = congestion.level if congestion else None
        meta = self.meta.get(aid, {})
        kwargs = {"level": level} if _accepts(self.fuser.fuse, "level") else {}
        incident = self.fuser.fuse(aid, end, b.signals, features, {}, b.incident, **kwargs)

        forecast = self._forecast(aid, b)
        risk = self.risk_engine.assess(features, congestion, forecast, incident, signals=[
            s for s in b.signals if end - s.timestamp <= timedelta(seconds=120)],
            road_importance=meta.get("importance"), free_flow_kmh=meta.get("free_flow_kmh"),
            junction_id=meta.get("junction_id"), road_id=meta.get("road_id"))
        if incident is not None:
            incident = incident.model_copy(update={"risk_level": risk.level})
            if self.store is not None:
                incident = self.store.upsert_incident(incident)
            b.incident = None if incident.status in (IncidentStatus.RESOLVED, IncidentStatus.DISMISSED) else incident
        else:
            b.incident = None
        if self.store is not None:
            self.store.add_risk(risk, incident_score=getattr(self.fuser, "last_score", {}).get(aid))
            if forecast is not None:
                self.store.add_forecast(forecast)
        return IntelligenceUpdate(approach_id=aid, timestamp=end, signals=new, incident=incident, risk=risk,
                                  forecast=forecast)

    def _forecast(self, aid: str, b: _ApproachBuffer) -> Forecast | None:
        if self.forecaster is None or len(b.states) < getattr(self.forecaster, "required_history_steps", 1):
            return None
        try:
            fc: Forecast = self.forecaster.predict(list(b.states), approach_id=aid).to_forecast()
            return fc
        except Exception as exc:  # noqa: BLE001 - e.g. gaps in history; risk falls back to current state
            log.warning("forecast unavailable for %s: %s", aid, exc)
            return None


def _accepts(fn: Any, name: str) -> bool:
    import inspect
    try:
        return name in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False

