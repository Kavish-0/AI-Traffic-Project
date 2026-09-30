"""TrafficStore: the application's persistence layer, on the ORM models of ``database/models.py``.

Written by the pipeline runtime (camera workers + analytics engine) and read by the API. What is
stored (and what is not):

    network      junction, road_node, road_segment, approach, camera   <- synced from configs/*.yaml
    windows      traffic_observation (TrafficFeatures) + congestion_state (level, score, reasons)
    live         approach_live_state: the latest ~1 s live value per approach (upserted, not a series)
    forecasts    forecast (per issue time x horizon x target, with predicted level)
    events       anomaly_event, incident (with evidence, contributing signals, operator feedback)
    risk         risk_assessment (score, level, components, reason codes, per-window incident score)
    context      context_observation (weather)
    simulation   simulation_scenario (what-if requests + results)
    runtime      camera_status (heartbeat), model_registry (mirror of models/<task>/<name>/<version>)

Frame-level detections / tracks are NOT stored here - they stay in the camera worker and go to
Parquet (``data/interim``). Operator decisions win: an acknowledged or dismissed incident keeps its
status when the engine updates its probability; only ``resolved`` is set automatically.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Generator, Iterable
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from typing import Any, overload

from sqlalchemy import Select, delete, func, select, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from traffic_intel.core.config import CameraConfig, NetworkConfig
from traffic_intel.core.enums import AnomalyType, CongestionLevel, IncidentStatus, ModelStatus, Severity
from traffic_intel.core.schemas import (
    AnomalySignal,
    ContextFeatures,
    Forecast,
    ForecastPoint,
    IncidentAssessment,
    RiskAssessment,
    SimulationResult,
    TrafficFeatures,
    TrafficState,
)
from traffic_intel.database import models as m
from traffic_intel.database.base import Base
from traffic_intel.database.session import create_db_engine
from traffic_intel.database.types import line_wkt, point_coords, point_wkt

log = logging.getLogger(__name__)

_OPERATOR_STATUSES = {IncidentStatus.ACKNOWLEDGED, IncidentStatus.DISMISSED}
_ACTIVE = (IncidentStatus.OPEN, IncidentStatus.ACKNOWLEDGED)
_CLASSES = ("car", "motorcycle", "bus", "truck", "bicycle", "pedestrian")


@overload
def utc(dt: datetime) -> datetime: ...
@overload
def utc(dt: None) -> None: ...
def utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)


def _haversine_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    lon1, lat1, lon2, lat2 = map(math.radians, (*a, *b))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6_371_000 * math.asin(math.sqrt(h))


class TrafficStore:
    def __init__(self, url: str | None = None, *, engine: Engine | None = None, sync_config: bool = True) -> None:
        self.engine = engine or create_db_engine(url)
        Base.metadata.create_all(self.engine)
        self._session = sessionmaker(self.engine, expire_on_commit=False)
        if sync_config:
            from traffic_intel.core.config import get_settings, load_camera, load_network, load_pipeline_config

            s = get_settings()
            pipeline = load_pipeline_config(s)
            cams = [load_camera(p) for p in sorted((s.config_dir / "cameras").glob("*.yaml"))]
            self.sync_network(load_network(s.config_dir / pipeline.network_file), cams)

    @contextmanager
    def session(self) -> Generator[Session]:
        with self._session.begin() as s:
            yield s

    @property
    def url(self) -> str:
        """Connection URL without the password (logs, status API)."""
        return self.engine.url.render_as_string(hide_password=True)

    @property
    def backend(self) -> str:
        return self.engine.dialect.name

    def ping(self) -> bool:
        try:
            with self.engine.connect() as c:
                c.execute(text("SELECT 1"))
            return True
        except Exception:  # noqa: BLE001 - reported as "database unavailable"
            log.exception("database ping failed")
            return False

    # ------------------------------------------------------------------ network (config -> DB)
    def sync_network(self, network: NetworkConfig, cameras: Iterable[CameraConfig] = ()) -> None:
        with self.session() as s:
            for j in network.junctions:
                s.merge(m.Junction(id=j.id, name=j.name, signalized=j.signalized, geom=point_wkt(*j.location)))
            for n in network.nodes:
                s.merge(m.RoadNode(id=n.id, junction_id=n.junction_id, geom=point_wkt(*n.location)))
            for g in network.segments:
                length = sum(_haversine_m((a[0], a[1]), (b[0], b[1]))
                             for a, b in zip(g.geometry, g.geometry[1:], strict=False))
                s.merge(m.RoadSegment(
                    id=g.id, name=g.name, from_node_id=g.from_node, to_node_id=g.to_node, lanes=g.lanes,
                    speed_limit_kmh=g.speed_limit_kmh, free_flow_speed_kmh=g.free_flow_speed_kmh or g.speed_limit_kmh,
                    capacity_veh_h=g.capacity_veh_h or g.lanes * 900.0, length_m=round(length, 1),
                    geom=line_wkt(g.geometry)))
            for a in network.approaches:
                s.merge(m.Approach(id=a.id, junction_id=a.junction_id, road_segment_id=a.road_segment_id, name=a.name,
                                   direction=a.direction))
            known = {a.id for a in network.approaches}
            for c in cameras:
                if c.approach_id in known:
                    s.merge(m.Camera(id=c.id, name=c.name, approach_id=c.approach_id, source_uri=c.source,
                                     process_fps=c.process_fps, active=True, geom=point_wkt(*c.location)))

    def ensure_camera(self, s: Session, camera_id: str | None, approach_id: str, name: str | None = None,
                      source: str = "runtime") -> None:
        if camera_id and s.get(m.Camera, camera_id) is None:
            s.add(m.Camera(id=camera_id, name=name or camera_id, approach_id=approach_id, source_uri=source,
                           process_fps=0.0, active=True, geom=None))
            s.flush()

    # ------------------------------------------------------------------ windows / live
    def add_window(self, f: TrafficFeatures, st: TrafficState | None = None, classifier: str = "rule_based") -> None:
        with self.session() as s:
            self.ensure_camera(s, f.camera_id, f.approach_id)
            s.merge(m.TrafficObservation(
                approach_id=f.approach_id, window_start=utc(f.window_start), camera_id=f.camera_id, window_s=f.window_s,
                vehicle_count=f.vehicle_count, count_by_class={getattr(k, "value", k): v for k, v in
                                                              f.count_by_class.items()},
                pcu_count=f.pcu_count, avg_speed_kmh=f.avg_speed_kmh, speed_p85_kmh=f.speed_p85_kmh,
                density_veh_km=f.density_veh_km, density_ratio=f.density_ratio, occupancy_pct=f.occupancy_pct,
                flow_veh_h=f.flow_veh_h, inflow=f.inflow, outflow=f.outflow, queue_length_m=f.queue_length_m,
                stopped_vehicles=f.stopped_vehicles, avg_gap_m=f.avg_gap_m, pedestrian_count=f.pedestrian_count,
                quality=f.quality, calibration_version=None, median_speed_kmh=f.median_speed_kmh,
                speed_distribution=f.speed_distribution, trajectory_stats=f.trajectory_stats,
                extra={k: v for k, v in f.extra.items() if isinstance(v, int | float)}))
            if st is not None:
                s.merge(m.CongestionStateRow(approach_id=f.approach_id, ts=utc(f.window_start), classifier=classifier,
                                             level=st.traffic_state, score=st.congestion_score,
                                             model_version=None, explanation=list(st.explanation)))

    def upsert_live_state(self, st: TrafficState) -> None:
        with self.session() as s:
            self.ensure_camera(s, st.camera_id, st.approach_id)
            s.merge(m.ApproachLiveState(approach_id=st.approach_id, ts=utc(st.timestamp),
                                        payload=st.model_dump(mode="json")))

    def latest_live_states(self, since: datetime | None = None) -> list[dict[str, Any]]:
        with self.session() as s:
            q = select(m.ApproachLiveState)
            if since is not None:
                q = q.where(m.ApproachLiveState.ts > utc(since))
            return [{**r.payload, "_ts": utc(r.ts)} for r in s.scalars(q)]

    @staticmethod
    def _state_dicts(rows: list[tuple[Any, Any, Any]]) -> list[dict[str, Any]]:
        out = []
        for obs, cong, appr in rows:
            counts = obs.count_by_class or {}
            out.append({
                "timestamp": utc(obs.window_start) + timedelta(seconds=obs.window_s),
                "window_start": utc(obs.window_start), "window_s": obs.window_s, "is_live": False,
                "camera_id": obs.camera_id, "approach_id": obs.approach_id,
                "junction_id": appr.junction_id if appr else None, "road_id": appr.road_segment_id if appr else None,
                "vehicle_count": obs.vehicle_count, **{f"{c}_count": int(counts.get(c, 0)) for c in _CLASSES},
                "average_speed_kmh": obs.avg_speed_kmh, "median_speed_kmh": obs.median_speed_kmh,
                "speed_p85_kmh": obs.speed_p85_kmh, "density_veh_km": obs.density_veh_km,
                "density_pct": round(obs.density_ratio * 100.0, 2), "occupancy_pct": obs.occupancy_pct,
                "flow_veh_h": obs.flow_veh_h, "inflow": obs.inflow, "outflow": obs.outflow,
                "queue_length_m": obs.queue_length_m, "stopped_vehicle_count": obs.stopped_vehicles,
                "traffic_state": cong.level.value if cong else None, "congestion_score": cong.score if cong else None,
                "explanation": cong.explanation if cong else [], "quality": obs.quality})
        return out

    def _window_query(self) -> Select[tuple[m.TrafficObservation, m.CongestionStateRow, m.Approach]]:
        return (select(m.TrafficObservation, m.CongestionStateRow, m.Approach)
                .join(m.CongestionStateRow, (m.CongestionStateRow.approach_id == m.TrafficObservation.approach_id)
                      & (m.CongestionStateRow.ts == m.TrafficObservation.window_start), isouter=True)
                .join(m.Approach, m.Approach.id == m.TrafficObservation.approach_id, isouter=True))

    def latest_traffic_states(self, junction_id: str | None = None) -> list[dict[str, Any]]:
        with self.session() as s:
            latest = (select(m.TrafficObservation.approach_id, func.max(m.TrafficObservation.window_start).label("ws"))
                      .group_by(m.TrafficObservation.approach_id).subquery())
            q = self._window_query().join(latest, (latest.c.approach_id == m.TrafficObservation.approach_id)
                                          & (latest.c.ws == m.TrafficObservation.window_start))
            if junction_id:
                q = q.where(m.Approach.junction_id == junction_id)
            return self._state_dicts(list(s.execute(q.order_by(m.TrafficObservation.approach_id)).tuples()))

    def traffic_history(self, approach_id: str, since: datetime | None = None,
                        limit: int = 500) -> list[dict[str, Any]]:
        with self.session() as s:
            q = self._window_query().where(m.TrafficObservation.approach_id == approach_id)
            if since:
                q = q.where(m.TrafficObservation.window_start >= utc(since))
            rows = list(s.execute(q.order_by(m.TrafficObservation.window_start.desc()).limit(limit)).tuples())
            return self._state_dicts(rows[::-1])

    def windows_since(self, since: datetime) -> list[dict[str, Any]]:
        """Windows whose END is after ``since`` (for the WebSocket poller)."""
        with self.session() as s:
            q = self._window_query().where(m.TrafficObservation.window_start >= utc(since) - timedelta(minutes=5))
            rows = self._state_dicts(list(s.execute(q.order_by(m.TrafficObservation.window_start)).tuples()))
            return [r for r in rows if r["timestamp"] > utc(since)]

    # ------------------------------------------------------------------ anomalies
    def add_anomalies(self, signals: Iterable[AnomalySignal], incident_id: str | None = None) -> int:
        n = 0
        with self.session() as s:
            for sig in signals:
                self.ensure_camera(s, sig.camera_id, sig.approach_id)
                loc = sig.details.get("location") or {}
                s.add(m.AnomalyEvent(
                    approach_id=sig.approach_id, camera_id=sig.camera_id, ts=utc(sig.timestamp), type=sig.type,
                    scope=sig.scope, detector=sig.detector, score=sig.score, track_ids=list(sig.track_ids),
                    details=sig.model_dump(mode="json")["details"],
                    geom=point_wkt(loc["lon"], loc["lat"]) if "lon" in loc and "lat" in loc else None,
                    incident_id=incident_id))
                n += 1
        return n

    def list_anomalies(self, approach_id: str | None = None, junction_id: str | None = None, type: str | None = None,
                       since: datetime | None = None, min_score: float = 0.0, limit: int = 200) -> list[dict[str, Any]]:
        with self.session() as s:
            q = (select(m.AnomalyEvent, m.Approach).join(m.Approach, m.Approach.id == m.AnomalyEvent.approach_id,
                                                         isouter=True).where(m.AnomalyEvent.score >= min_score))
            if approach_id:
                q = q.where(m.AnomalyEvent.approach_id == approach_id)
            if junction_id:
                q = q.where(m.Approach.junction_id == junction_id)
            if type:
                q = q.where(m.AnomalyEvent.type == AnomalyType(type))
            if since:
                q = q.where(m.AnomalyEvent.ts >= utc(since))
            out = []
            for ev, appr in s.execute(q.order_by(m.AnomalyEvent.ts.desc(), m.AnomalyEvent.id.desc()).limit(limit)):
                lon, lat = point_coords(ev.geom)
                out.append({"id": ev.id, "approach_id": ev.approach_id, "camera_id": ev.camera_id,
                            "junction_id": appr.junction_id if appr else None,
                            "road_id": appr.road_segment_id if appr else None, "timestamp": utc(ev.ts),
                            "type": ev.type.value, "scope": ev.scope.value, "detector": ev.detector, "score": ev.score,
                            "track_ids": ev.track_ids, "lon": lon, "lat": lat, "details": ev.details,
                            "incident_id": ev.incident_id})
            return out

    # ------------------------------------------------------------------ incidents
    def upsert_incident(self, inc: IncidentAssessment) -> IncidentAssessment:
        """Insert or update; returns the stored view (operator status preserved)."""
        if inc.incident_id is None:
            raise ValueError("incident_id required")
        with self.session() as s:
            row = s.get(m.Incident, inc.incident_id)
            status = inc.status
            if row is not None and row.status in _OPERATOR_STATUSES and status is IncidentStatus.OPEN:
                status = row.status
            if row is None:
                row = m.Incident(id=inc.incident_id, max_probability=0.0, first_detected=utc(inc.first_detected
                                                                                          or inc.timestamp))
                s.add(row)
            row.approach_id, row.status, row.probability = inc.approach_id, status, inc.probability
            row.max_probability = max(row.max_probability or 0.0, inc.probability)
            row.calibrated, row.severity, row.fusion_model = inc.calibrated, inc.severity, inc.fusion_model
            row.junction_id, row.road_id = inc.junction_id, inc.road_id
            row.incident_type, row.label, row.message = inc.incident_type, inc.label, inc.message
            row.risk_level, row.evidence = inc.risk_level, dict(inc.evidence)
            row.contributing = [c.model_dump(mode="json") for c in inc.contributing]
            row.location_description = inc.location_description
            row.geom = point_wkt(inc.lon, inc.lat) if inc.lon is not None and inc.lat is not None else row.geom
            row.last_updated = utc(inc.timestamp)
            row.resolved_at = utc(inc.timestamp) if status is IncidentStatus.RESOLVED else row.resolved_at
            return inc.model_copy(update={"status": status})

    def _incident_view(self, r: m.Incident) -> dict[str, Any]:
        lon, lat = point_coords(r.geom)
        return {"incident_id": r.id, "timestamp": utc(r.last_updated), "first_detected": utc(r.first_detected),
                "resolved_at": utc(r.resolved_at), "approach_id": r.approach_id, "junction_id": r.junction_id,
                "road_id": r.road_id, "incident_type": r.incident_type, "label": r.label, "status": r.status.value,
                "probability": r.probability, "max_probability": r.max_probability, "calibrated": r.calibrated,
                "severity": r.severity.value, "risk_level": r.risk_level.value if r.risk_level else None,
                "message": r.message, "evidence": r.evidence or {}, "lon": lon, "lat": lat,
                "location_description": r.location_description, "operator_label": r.operator_label,
                "operator_note": r.operator_note, "contributing": r.contributing or [],
                "uncertainty_note": ("Probability is calibrated on labelled incidents" if r.calibrated else
                                     "Incident probability is UNCALIBRATED (an estimate from combined signals)")
                + "; this is a possible incident for an operator to verify, not a confirmed accident."}

    def get_incident(self, incident_id: str) -> dict[str, Any] | None:
        with self.session() as s:
            r = s.get(m.Incident, incident_id)
            return self._incident_view(r) if r else None

    def list_incidents(self, status: str | None = None, approach_id: str | None = None,
                       junction_id: str | None = None, active_only: bool = False, since: datetime | None = None,
                       limit: int = 100) -> list[dict[str, Any]]:
        with self.session() as s:
            q = select(m.Incident)
            if status:
                q = q.where(m.Incident.status == IncidentStatus(status))
            if active_only:
                q = q.where(m.Incident.status.in_(_ACTIVE))
            if approach_id:
                q = q.where(m.Incident.approach_id == approach_id)
            if junction_id:
                q = q.where(m.Incident.junction_id == junction_id)
            if since:
                q = q.where(m.Incident.last_updated >= utc(since))
            return [self._incident_view(r) for r in s.scalars(q.order_by(m.Incident.last_updated.desc()).limit(limit))]

    def open_incident_for(self, approach_id: str) -> IncidentAssessment | None:
        with self.session() as s:
            q = select(m.Incident).where(m.Incident.approach_id == approach_id, m.Incident.status.in_(_ACTIVE))
            r = s.scalars(q.order_by(m.Incident.last_updated.desc()).limit(1)).first()
            if r is None:
                return None
            v = self._incident_view(r)
        return IncidentAssessment(
            incident_id=v["incident_id"], approach_id=v["approach_id"], timestamp=v["timestamp"],
            probability=v["probability"], calibrated=v["calibrated"], severity=Severity(v["severity"]),
            status=IncidentStatus(v["status"]), lon=v["lon"], lat=v["lat"], incident_type=v["incident_type"],
            label=v["label"], message=v["message"], junction_id=v["junction_id"], road_id=v["road_id"],
            evidence=v["evidence"], first_detected=v["first_detected"],
            contributing=[AnomalySignal.model_validate(c) for c in v["contributing"]])

    def set_status(self, incident_id: str, status: IncidentStatus, operator_label: str | None = None,
                   note: str | None = None) -> dict[str, Any] | None:
        with self.session() as s:
            r = s.get(m.Incident, incident_id)
            if r is None:
                return None
            r.status = status
            if operator_label is not None:
                r.operator_label = operator_label
            if note is not None:
                r.operator_note = note
            if status in (IncidentStatus.RESOLVED, IncidentStatus.DISMISSED):
                r.resolved_at = datetime.now(UTC)
        log.info("incident %s -> %s (operator label %s)", incident_id, status.value, operator_label)
        return self.get_incident(incident_id)

    def incidents_updated_since(self, since: datetime) -> list[dict[str, Any]]:
        with self.session() as s:
            q = select(m.Incident).where(m.Incident.last_updated > utc(since)).order_by(m.Incident.last_updated)
            return [self._incident_view(r) for r in s.scalars(q)]

    # ------------------------------------------------------------------ risk
    def add_risk(self, r: RiskAssessment, incident_score: float | None = None) -> None:
        with self.session() as s:
            s.merge(m.RiskAssessmentRow(approach_id=r.approach_id, ts=utc(r.timestamp), risk_score=r.risk_score,
                                        level=r.level, components=r.components, explanation=r.explanation,
                                        reason_codes=r.reason_codes, incident_score=incident_score))

    def _risk_view(self, r: m.RiskAssessmentRow, appr: m.Approach | None) -> dict[str, Any]:
        return {"approach_id": r.approach_id, "timestamp": utc(r.ts), "risk_score": r.risk_score,
                "level": r.level.value, "reason_codes": r.reason_codes or [], "components": r.components or {},
                "explanation": r.explanation or [], "incident_score": r.incident_score,
                "junction_id": appr.junction_id if appr else None, "road_id": appr.road_segment_id if appr else None}

    def latest_risk(self, approach_id: str | None = None, junction_id: str | None = None) -> list[dict[str, Any]]:
        with self.session() as s:
            latest = (select(m.RiskAssessmentRow.approach_id, func.max(m.RiskAssessmentRow.ts).label("ts"))
                      .group_by(m.RiskAssessmentRow.approach_id).subquery())
            q = (select(m.RiskAssessmentRow, m.Approach)
                 .join(latest, (latest.c.approach_id == m.RiskAssessmentRow.approach_id)
                       & (latest.c.ts == m.RiskAssessmentRow.ts))
                 .join(m.Approach, m.Approach.id == m.RiskAssessmentRow.approach_id, isouter=True))
            if approach_id:
                q = q.where(m.RiskAssessmentRow.approach_id == approach_id)
            if junction_id:
                q = q.where(m.Approach.junction_id == junction_id)
            return [self._risk_view(r, a) for r, a in s.execute(q.order_by(m.RiskAssessmentRow.risk_score.desc()))]

    def risk_history(self, approach_id: str, since: datetime | None = None, limit: int = 500) -> list[dict[str, Any]]:
        with self.session() as s:
            q = (select(m.RiskAssessmentRow, m.Approach)
                 .join(m.Approach, m.Approach.id == m.RiskAssessmentRow.approach_id, isouter=True)
                 .where(m.RiskAssessmentRow.approach_id == approach_id))
            if since:
                q = q.where(m.RiskAssessmentRow.ts >= utc(since))
            rows = list(s.execute(q.order_by(m.RiskAssessmentRow.ts.desc()).limit(limit)))
            return [self._risk_view(r, a) for r, a in rows[::-1]]

    # ------------------------------------------------------------------ forecasts
    def add_forecast(self, fc: Forecast) -> None:
        with self.session() as s:
            for p in fc.points:
                level = fc.predicted_level.get(p.horizon_min)
                s.merge(m.ForecastRow(approach_id=fc.approach_id, issued_at=utc(fc.issued_at), model_name=fc.model_name,
                                      horizon_min=p.horizon_min, target=p.target, model_version=fc.model_version,
                                      target_time=utc(p.target_time), value=p.value, lower=p.lower, upper=p.upper,
                                      level=level.value if level else None))

    def latest_forecast(self, approach_id: str) -> Forecast | None:
        with self.session() as s:
            issued = s.scalar(select(func.max(m.ForecastRow.issued_at)).where(m.ForecastRow.approach_id == approach_id))
            if issued is None:
                return None
            rows = list(s.scalars(select(m.ForecastRow).where(m.ForecastRow.approach_id == approach_id,
                                                              m.ForecastRow.issued_at == issued)))
        levels = {r.horizon_min: CongestionLevel(r.level) for r in rows if r.level}
        return Forecast(approach_id=approach_id, issued_at=utc(issued), model_name=rows[0].model_name,
                        model_version=rows[0].model_version, predicted_level=levels,
                        points=[ForecastPoint(horizon_min=r.horizon_min, target_time=utc(r.target_time),
                                              target=r.target, value=r.value, lower=r.lower, upper=r.upper)
                                for r in sorted(rows, key=lambda r: (r.horizon_min, r.target))])

    def forecast_history(self, approach_id: str, horizon_min: int, target: str, since: datetime | None = None,
                         limit: int = 1000) -> list[dict[str, Any]]:
        with self.session() as s:
            q = select(m.ForecastRow).where(m.ForecastRow.approach_id == approach_id,
                                            m.ForecastRow.horizon_min == horizon_min, m.ForecastRow.target == target)
            if since:
                q = q.where(m.ForecastRow.target_time >= utc(since))
            rows = list(s.scalars(q.order_by(m.ForecastRow.target_time.desc()).limit(limit)))
            return [{"issued_at": utc(r.issued_at), "target_time": utc(r.target_time), "value": r.value,
                     "level": r.level, "model_name": r.model_name, "model_version": r.model_version}
                    for r in rows[::-1]]

    # ------------------------------------------------------------------ context / simulation
    def add_context(self, region: str, ctx: ContextFeatures, source: str) -> None:
        with self.session() as s:
            s.merge(m.ContextObservation(region=region, ts=utc(ctx.timestamp), is_holiday=ctx.is_holiday,
                                         weather=getattr(ctx.weather, "value", ctx.weather),
                                         temperature_c=ctx.temperature_c, precipitation_mm=ctx.precipitation_mm,
                                         visibility_km=ctx.visibility_km, source=source))

    def latest_context(self, region: str) -> dict[str, Any] | None:
        with self.session() as s:
            r = s.scalars(select(m.ContextObservation).where(m.ContextObservation.region == region)
                          .order_by(m.ContextObservation.ts.desc()).limit(1)).first()
            return None if r is None else {"ts": utc(r.ts), "weather": r.weather, "temperature_c": r.temperature_c,
                                           "precipitation_mm": r.precipitation_mm, "visibility_km": r.visibility_km,
                                           "is_holiday": r.is_holiday, "source": r.source}

    def add_simulation(self, res: SimulationResult) -> None:
        with self.session() as s:
            s.merge(m.SimulationScenario(id=res.scenario_id, created_at=utc(res.created_at),
                                         request=res.request.model_dump(mode="json"),
                                         result=res.model_dump(mode="json")))

    def get_simulation(self, scenario_id: str) -> dict[str, Any] | None:
        with self.session() as s:
            r = s.get(m.SimulationScenario, scenario_id)
            return r.result if r else None

    # ------------------------------------------------------------------ runtime status / models
    def upsert_camera_status(self, camera_id: str, approach_id: str, **fields: Any) -> None:
        with self.session() as s:
            self.ensure_camera(s, camera_id, approach_id)
            row = s.get(m.CameraStatusRow, camera_id) or m.CameraStatusRow(camera_id=camera_id, frames=0, details={})
            for k, v in fields.items():
                setattr(row, k, utc(v) if isinstance(v, datetime) else v)
            row.updated_at = datetime.now(UTC)
            s.merge(row)

    def cameras(self) -> dict[str, dict[str, Any]]:
        """Cameras known to the database (configured ones + runtime cameras such as synthetic scenes)."""
        with self.session() as s:
            out = {}
            for c in s.scalars(select(m.Camera)):
                lon, lat = point_coords(c.geom)
                out[c.id] = {"id": c.id, "name": c.name, "approach_id": c.approach_id, "source": c.source_uri,
                             "lon": lon, "lat": lat}
            return out

    def camera_statuses(self) -> dict[str, dict[str, Any]]:
        with self.session() as s:
            out = {}
            for r in s.scalars(select(m.CameraStatusRow)):
                out[r.camera_id] = {c.name: (utc(getattr(r, c.name)) if isinstance(getattr(r, c.name), datetime)
                                             else getattr(r, c.name)) for c in m.CameraStatusRow.__table__.columns}
            return out

    def sync_models(self, metas: Iterable[Any], production: dict[tuple[str, str], str]) -> int:
        """Mirror the file model registry (models/<task>/<name>/<version>/metadata.json)."""
        n = 0
        with self.session() as s:
            s.execute(delete(m.ModelRegistryRow))
            for md in metas:
                prod = production.get((md.task.value, md.name)) == md.version
                s.add(m.ModelRegistryRow(task=md.task, name=md.name, version=md.version, framework=md.framework,
                                         artifact_uri=f"models/{md.task.value}/{md.name}/{md.version}",
                                         feature_schema_hash=md.feature_schema_hash[:32], metrics=md.metrics,
                                         params=md.params, status=ModelStatus.PRODUCTION if prod else md.status,
                                         trained_at=utc(md.trained_at)))
                n += 1
        return n
