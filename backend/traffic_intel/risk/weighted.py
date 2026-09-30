"""Transparent weighted risk engine (``risk_engine.type: weighted``).

Components, each in [0, 1] (configs/risk.yaml):

    current_severity   congestion score of the latest window (rule classifier score)
    forecast_severity  worst predicted level over +5/+15/+30 min: free 0, moderate .35, high .65, severe 1
    incident           fused incident probability (0 if no active incident)
    queue              queue_length_m / min(queue_saturation_m, ROI length)
    speed              (1 - speed / free-flow speed) / 0.8, clipped
    density            density_ratio / density_saturation, clipped

    base  = sum(w_i * c_i) / sum(w_i)            over AVAILABLE components (missing -> weights renormalised)
    risk  = 100 * base * (1 - k + k * importance)   importance in [0, 1], k = importance_weight (0.3)

    i.e. a road of importance 1 keeps its score, importance 0 scales it by 0.7.

Level: low < 25 <= medium < 50 <= high < 75 <= critical. Safety floors (never lower
a level): a wrong-way signal -> at least HIGH; incident probability >= 0.85 -> at
least HIGH. Every component above its reason threshold adds a reason code and a
sentence, so the score is never opaque.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from traffic_intel.core.config import get_settings, load_yaml
from traffic_intel.core.enums import AnomalyType, CongestionLevel, Severity
from traffic_intel.core.schemas import (
    AnomalySignal,
    CongestionAssessment,
    Forecast,
    IncidentAssessment,
    RiskAssessment,
    TrafficFeatures,
)
from traffic_intel.risk.interfaces import RISK_ENGINES, RiskEngine

LEVEL_VALUE = {CongestionLevel.FREE: 0.0, CongestionLevel.MODERATE: 0.35, CongestionLevel.HIGH: 0.65,
               CongestionLevel.SEVERE: 1.0}
_ORDER = [Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL]


class RiskConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    weights: dict[str, float]
    importance_weight: float = 0.3
    default_importance: float = 0.5
    queue_saturation_m: float = 60.0
    density_saturation: float = 0.8
    default_free_flow_kmh: float = 50.0
    levels: dict[Severity, float]              # upper bounds
    reason_thresholds: dict[str, float] = {}
    signal_reason_min_score: float = 0.4


def _clip(x: float) -> float:
    return max(0.0, min(1.0, x))


@RISK_ENGINES.register("weighted")
class WeightedRiskEngine(RiskEngine):
    def __init__(self, config_file: str | None = "risk.yaml", config: dict[str, Any] | None = None) -> None:
        if config is None:
            path = Path(config_file or "risk.yaml")
            config = load_yaml(path if path.is_absolute() else get_settings().config_dir / path)
        self.cfg = RiskConfig.model_validate(config)

    def assess(self, features: TrafficFeatures | None, congestion: CongestionAssessment | None,
               forecast: Forecast | None, incident: IncidentAssessment | None, *,
               signals: Sequence[AnomalySignal] = (), road_importance: float | None = None,
               free_flow_kmh: float | None = None, junction_id: str | None = None,
               road_id: str | None = None) -> RiskAssessment:
        cfg, th = self.cfg, self.cfg.reason_thresholds
        comp: dict[str, float] = {}
        codes: list[str] = []
        why: list[str] = []

        def reason(code: str, text: str) -> None:
            if code not in codes:
                codes.append(code)
                why.append(text)

        cur_level = congestion.level if congestion else None
        if congestion is not None:
            comp["current_severity"] = _clip(congestion.score)
            if cur_level is CongestionLevel.SEVERE:
                reason("SEVERE_CONGESTION_NOW", "severe congestion now")
            elif cur_level is CongestionLevel.HIGH:
                reason("HIGH_CONGESTION_NOW", "high congestion now")
        if forecast is not None and forecast.predicted_level:
            worst_h, worst = max(forecast.predicted_level.items(), key=lambda kv: LEVEL_VALUE[kv[1]])
            comp["forecast_severity"] = LEVEL_VALUE[worst]
            if worst is CongestionLevel.SEVERE:
                reason("SEVERE_CONGESTION_FORECAST", f"severe congestion forecast within {worst_h} min")
            if cur_level is not None and LEVEL_VALUE[worst] > LEVEL_VALUE[cur_level]:
                reason("CONGESTION_WORSENING", f"congestion forecast to worsen ({cur_level.value} -> {worst.value})")
        p = incident.probability if incident is not None and incident.status.value in ("open", "acknowledged") else 0.0
        comp["incident"] = p
        if p >= th.get("incident_possible", 0.6):
            reason("POSSIBLE_INCIDENT", f"possible incident (probability {p:.0%}, uncalibrated)")
        elif p >= th.get("incident_elevated", 0.3):
            reason("ELEVATED_INCIDENT_PROBABILITY", f"elevated incident probability ({p:.0%})")
        if features is not None:
            roi = features.extra.get("roi_length_m") or cfg.queue_saturation_m
            comp["queue"] = _clip(features.queue_length_m / min(cfg.queue_saturation_m, roi))
            if comp["queue"] >= th.get("queue", 0.6):
                reason("LONG_QUEUE", f"long queue ({features.queue_length_m:.0f} m)")
            ff = free_flow_kmh or cfg.default_free_flow_kmh
            if features.avg_speed_kmh is not None:
                comp["speed"] = _clip((1.0 - features.avg_speed_kmh / ff) / 0.8)
                if comp["speed"] >= th.get("speed", 0.7):
                    reason("LOW_SPEED", f"low speed ({features.avg_speed_kmh:.0f} km/h vs free-flow {ff:.0f})")
            comp["density"] = _clip(features.density_ratio / cfg.density_saturation)
            if comp["density"] >= th.get("density", 0.7):
                reason("HIGH_DENSITY", f"high density ({features.density_ratio:.0%} of jam density)")
            if features.quality < th.get("data_quality", 0.5):
                reason("LOW_DATA_QUALITY", f"low data quality ({features.quality:.0%} of expected frames)")

        strongest: dict[AnomalyType, float] = {}
        for s in signals:
            strongest[s.type] = max(strongest.get(s.type, 0.0), s.score)
        smin = cfg.signal_reason_min_score
        for t, code, text in ((AnomalyType.SUDDEN_SLOWDOWN, "ABNORMAL_SLOWDOWN", "abnormal slowdown"),
                              (AnomalyType.STOPPED_VEHICLE, "STOPPED_VEHICLE", "stopped vehicle on the road"),
                              (AnomalyType.WRONG_WAY, "WRONG_WAY_VEHICLE", "wrong-way vehicle"),
                              (AnomalyType.SUDDEN_QUEUE, "QUEUE_INCREASING", "queue increasing abnormally"),
                              (AnomalyType.UNUSUAL_FLOW, "VEHICLES_ACCUMULATING", "vehicles accumulating"),
                              (AnomalyType.ABNORMAL_TRAJECTORY, "ABNORMAL_TRAJECTORY", "erratic vehicle movement")):
            if strongest.get(t, 0.0) >= (0.15 if t is AnomalyType.WRONG_WAY else smin):
                reason(code, text)

        used = {k: v for k, v in comp.items() if k in cfg.weights}
        wsum = sum(cfg.weights[k] for k in used)
        base = sum(cfg.weights[k] * v for k, v in used.items()) / wsum if wsum else 0.0
        imp = cfg.default_importance if road_importance is None else _clip(road_importance)
        k = cfg.importance_weight
        score = 100.0 * base * (1.0 - k + k * imp)
        level = next((lvl for lvl in _ORDER if score < cfg.levels[lvl]), Severity.CRITICAL)
        floors = []
        if "WRONG_WAY_VEHICLE" in codes:
            floors.append((Severity.HIGH, "wrong-way vehicle -> at least HIGH (safety floor)"))
        if p >= 0.85:
            floors.append((Severity.HIGH, "incident probability >= 85% -> at least HIGH (safety floor)"))
        for floor, text in floors:
            if _ORDER.index(floor) > _ORDER.index(level):
                level = floor
                why.append(text)
        if imp >= th.get("importance", 0.8) and base >= 0.4:
            reason("HIGH_IMPORTANCE_ROAD", f"important road (importance {imp:.1f})")
        comp["importance"] = imp
        breakdown = ", ".join(f"{n} {v:.2f}x{cfg.weights[n]}" for n, v in used.items())
        why.insert(0, f"risk = 100 x {base:.2f} x (1 - {k} + {k} x {imp:.2f}) = {score:.0f}  [{breakdown}]")
        ts = (features.window_start if features else None) or (congestion.timestamp if congestion else None) \
            or datetime.now(UTC)
        approach = (features.approach_id if features else None) or (congestion.approach_id if congestion else "") \
            or (incident.approach_id if incident else "")
        return RiskAssessment(approach_id=approach, timestamp=ts, risk_score=round(score, 1), level=level,
                              components={k2: round(v, 3) for k2, v in comp.items()}, explanation=why,
                              reason_codes=codes, junction_id=junction_id, road_id=road_id)
