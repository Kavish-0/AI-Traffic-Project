"""Incident fusion: many weak, uncertain signals -> one hedged incident hypothesis.

Evidence families (configurable mapping from signal type -> family):

    symptoms   speed_drop, queue_growth, density_change, flow_imbalance
               (usually CONSEQUENCES of one cause - congestion - so they are combined with
               a noisy-OR into one strength S = 1 - prod(1 - e_f), not summed)
    specific   stopped_vehicle, trajectory_anomaly (wrong-way / erratic), anomaly_score
               (statistical / learned outliers)

Each family's evidence e_f = max over its signals in the last ``lookback_s`` of
score x (1 - age / lookback_s)  (older evidence fades linearly).

    z = bias + w_S * S + sum_f w_f * e_f + w_int * S * max(e_stopped, e_trajectory)
    incident_probability = 1 / (1 + exp(-z))

With the default weights (configs/anomaly.yaml) congestion alone (S ~ 1) gives
p ~ 0.18, one isolated specific cue p <= 0.38, and only symptoms PLUS a physical cue
(stopped vehicle / abnormal trajectory in the jam) exceed 0.6 - "do not rely on one
signal" is built into the functional form. The weights are HAND-SET (``calibrated =
False``) until fitted on labelled incidents (operator feedback); every output says so.

Incident type: the configurable taxonomy (``incident_types``) is evaluated in order;
the first rule whose conditions hold names the situation (e.g. possible_incident,
wrong_way, stopped_vehicle, abnormal_queue, sudden_slowdown, normal).

Lifecycle per approach: an incident opens when the type is not ``normal``, is updated
(probability, type, evidence; type can escalate) while it persists, and is resolved
after ``close_after_evaluations`` consecutive normal evaluations.
"""

from __future__ import annotations

import math
import uuid
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from traffic_intel.anomaly.interfaces import INCIDENT_FUSERS, IncidentFuser
from traffic_intel.core.config import get_settings, load_yaml
from traffic_intel.core.enums import CongestionLevel, IncidentStatus, Severity
from traffic_intel.core.schemas import AnomalySignal, IncidentAssessment, TrafficFeatures


class TypeRule(BaseModel):
    model_config = ConfigDict(extra="forbid")
    min_probability: float | None = None
    min_families: int | None = None            # families with evidence >= family_threshold
    family: str | None = None
    min_evidence: float = 0.0
    signal_type: str | None = None             # an AnomalyType that must be among the evidence


class IncidentTypeSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    label: str
    when: TypeRule = TypeRule()
    min_severity: Severity | None = None


class SeverityRule(BaseModel):
    min_probability: float = 0.0
    min_level: CongestionLevel | None = None


class FusionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lookback_s: float = 300.0
    bias: float = -4.0
    symptom_weight: float = 2.5
    interaction_weight: float = 3.0
    families: dict[str, list[str]]             # family -> AnomalyType values
    symptoms: list[str]                        # families combined by noisy-OR
    weights: dict[str, float]                  # specific families -> weight
    family_threshold: float = 0.3
    close_after_evaluations: int = Field(3, ge=1)
    incident_types: list[IncidentTypeSpec]
    severity: dict[Severity, SeverityRule] = {}


_LEVELS = [CongestionLevel.FREE, CongestionLevel.MODERATE, CongestionLevel.HIGH, CongestionLevel.SEVERE]
_SEV = [Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL]


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


class _ApproachState:
    def __init__(self) -> None:
        self.incident: IncidentAssessment | None = None
        self.quiet = 0


@INCIDENT_FUSERS.register("weighted_logistic")
class WeightedLogisticFuser(IncidentFuser):
    name = "weighted_logistic"

    def __init__(self, config_file: str | None = "anomaly.yaml", config: dict[str, Any] | None = None,
                 approach_meta: dict[str, dict[str, str]] | None = None) -> None:
        if config is None:
            path = Path(config_file or "anomaly.yaml")
            config = load_yaml(path if path.is_absolute() else get_settings().config_dir / path)
        self.cfg = FusionConfig.model_validate(config)
        self._family_of = {t: f for f, types in self.cfg.families.items() for t in types}
        self.meta = approach_meta or {}           # approach_id -> {junction_id, road_id}
        self._state: dict[str, _ApproachState] = {}
        self.last_score: dict[str, float] = {}       # latest fused score per approach (also when "normal")

    # ------------------------------------------------------------------ #
    def evidence(self, now: datetime, signals: Sequence[AnomalySignal]) -> tuple[dict[str, float], dict[str, set[str]]]:
        ev: dict[str, float] = dict.fromkeys(self.cfg.families, 0.0)
        types: dict[str, set[str]] = {f: set() for f in self.cfg.families}
        for s in signals:
            fam = self._family_of.get(s.type.value)
            age = (now - s.timestamp).total_seconds()
            if fam is None or age < -1 or age > self.cfg.lookback_s:
                continue
            e = s.score * (1.0 - max(age, 0.0) / self.cfg.lookback_s)
            if e > ev[fam]:
                ev[fam] = e
            if e > 0:
                types[fam].add(s.type.value)
        return ev, types

    def probability(self, ev: dict[str, float]) -> tuple[float, float]:
        """(incident probability, combined symptom strength S)."""
        S = 1.0
        for f in self.cfg.symptoms:
            S *= 1.0 - ev.get(f, 0.0)
        S = 1.0 - S
        z = self.cfg.bias + self.cfg.symptom_weight * S
        z += sum(w * ev.get(f, 0.0) for f, w in self.cfg.weights.items())
        physical = max(ev.get("stopped_vehicle", 0.0), ev.get("trajectory_anomaly", 0.0))
        z += self.cfg.interaction_weight * S * physical
        return _sigmoid(z), S

    def classify(self, p: float, ev: dict[str, float], types: dict[str, set[str]]) -> IncidentTypeSpec:
        present = {t for ts in types.values() for t in ts}
        n_fam = sum(1 for v in ev.values() if v >= self.cfg.family_threshold)
        for spec in self.cfg.incident_types:
            w = spec.when
            if w.min_probability is not None and p < w.min_probability:
                continue
            if w.min_families is not None and n_fam < w.min_families:
                continue
            if w.family is not None and ev.get(w.family, 0.0) < w.min_evidence:
                continue
            if w.signal_type is not None and w.signal_type not in present:
                continue
            return spec
        return self.cfg.incident_types[-1]

    def severity(self, p: float, level: CongestionLevel | None, spec: IncidentTypeSpec) -> Severity:
        sev = Severity.LOW
        for s in (Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM):
            rule = self.cfg.severity.get(s)
            if rule is None or p < rule.min_probability:
                continue
            if rule.min_level is not None and (level is None or _LEVELS.index(level) < _LEVELS.index(rule.min_level)):
                continue
            sev = s
            break
        if spec.min_severity is not None and _SEV.index(spec.min_severity) > _SEV.index(sev):
            sev = spec.min_severity
        return sev

    # ------------------------------------------------------------------ #
    def fuse(self, approach_id: str, now: datetime, signals: Sequence[AnomalySignal],
             features: TrafficFeatures | None, neighbour_features: dict[str, TrafficFeatures],
             open_incident: IncidentAssessment | None, level: CongestionLevel | None = None,
             location: tuple[float, float] | None = None) -> IncidentAssessment | None:
        st = self._state.setdefault(approach_id, _ApproachState())
        if open_incident is not None and st.incident is None:
            st.incident = open_incident
        ev, types = self.evidence(now, signals)
        p, S = self.probability(ev)
        self.last_score[approach_id] = round(p, 4)
        spec = self.classify(p, ev, types)
        normal = spec.id == self.cfg.incident_types[-1].id
        if normal:
            if st.incident is None:
                return None
            st.quiet += 1
            if st.quiet < self.cfg.close_after_evaluations:
                return st.incident.model_copy(update={"timestamp": now, "probability": round(p, 3)})
            resolved = st.incident.model_copy(update={
                "timestamp": now, "status": IncidentStatus.RESOLVED, "probability": round(p, 3),
                "message": f"Resolved: evidence subsided (incident probability now {p:.0%})."})
            st.incident, st.quiet = None, 0
            return resolved
        st.quiet = 0
        contributing = sorted(
            (s for s in signals if self._family_of.get(s.type.value) and 0 <= (now - s.timestamp).total_seconds()
             <= self.cfg.lookback_s), key=lambda s: -s.score)[:10]
        loc = location or _first_location(contributing)
        meta = self.meta.get(approach_id, {})
        prev = st.incident
        order = [t.id for t in self.cfg.incident_types]            # earlier = more serious
        type_id = spec.id
        if prev is not None and prev.incident_type in order and order.index(prev.incident_type) < order.index(spec.id):
            type_id = prev.incident_type      # within one incident the type may escalate, never silently downgrade
        label = next(t.label for t in self.cfg.incident_types if t.id == type_id)
        tspec = next(t for t in self.cfg.incident_types if t.id == type_id)
        sev = self.severity(p, level, tspec)
        evidence = {f: round(v, 3) for f, v in ev.items() if v > 0}
        evidence["symptom_strength"] = round(S, 3)
        inc = IncidentAssessment(
            incident_id=prev.incident_id if prev else str(uuid.uuid4()),
            approach_id=approach_id, timestamp=now, probability=round(p, 3), calibrated=False, severity=sev,
            status=prev.status if prev and prev.status is IncidentStatus.ACKNOWLEDGED else IncidentStatus.OPEN,
            lon=loc[0] if loc else None, lat=loc[1] if loc else None,
            location_description=meta.get("name"), contributing=contributing, fusion_model=self.name,
            incident_type=type_id, label=label, message=hedged_message(label, p, evidence),
            junction_id=meta.get("junction_id"), road_id=meta.get("road_id"), evidence=evidence,
            first_detected=prev.first_detected if prev and prev.first_detected else now,
        )
        st.incident = inc
        return inc


_FAMILY_TEXT = {"speed_drop": "abnormal slowdown", "queue_growth": "queue growing", "density_change":
                "unusual density", "flow_imbalance": "vehicles accumulating", "stopped_vehicle": "stopped vehicle",
                "trajectory_anomaly": "abnormal / wrong-way movement", "anomaly_score": "statistical outlier"}


def hedged_message(label: str, p: float, evidence: dict[str, float]) -> str:
    """Human text that always carries the probability and the uncertainty."""
    parts = [f"{_FAMILY_TEXT.get(k, k)} ({v:.2f})" for k, v in sorted(evidence.items(), key=lambda kv: -kv[1])
             if k != "symptom_strength" and v >= 0.2]
    ev = "; evidence: " + ", ".join(parts) if parts else ""
    return (f"{label} - incident probability {p:.0%} (uncalibrated estimate from combined signals; "
            f"not a confirmed accident - verify before acting){ev}.")



def _first_location(signals: Sequence[AnomalySignal]) -> tuple[float, float] | None:
    """lon/lat of the strongest signal that carries one (track-level signals do)."""
    for s in signals:
        loc = s.details.get("location") or {}
        if "lon" in loc and "lat" in loc:
            return float(loc["lon"]), float(loc["lat"])
    return None
