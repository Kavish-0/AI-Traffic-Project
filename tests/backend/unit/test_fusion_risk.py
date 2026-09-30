"""Incident fusion (probability, taxonomy, lifecycle, hedged wording) and the transparent risk engine."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest
from conftest import T0

from traffic_intel.anomaly.fusion import WeightedLogisticFuser, hedged_message
from traffic_intel.core.config import get_settings, load_yaml
from traffic_intel.core.enums import AnomalyScope, AnomalyType, CongestionLevel, IncidentStatus, Severity
from traffic_intel.core.schemas import (
    AnomalySignal,
    CongestionAssessment,
    Forecast,
    IncidentAssessment,
    TrafficFeatures,
)
from traffic_intel.risk.weighted import WeightedRiskEngine


def sig(t: AnomalyType, score: float, dt_s: float = 0.0, **details: Any) -> AnomalySignal:
    return AnomalySignal(approach_id="junction_a_north", camera_id=None, timestamp=T0 + timedelta(seconds=dt_s),
                         type=t, scope=AnomalyScope.SERIES, detector=t.value, score=score, details=details)


# --------------------------------------------------------------------------- fusion
@pytest.fixture
def fuser() -> WeightedLogisticFuser:
    return WeightedLogisticFuser()


def test_no_single_signal_makes_an_incident(fuser: WeightedLogisticFuser) -> None:
    p_none, _ = fuser.probability({})
    p_congestion, S = fuser.probability({"speed_drop": 1.0, "queue_growth": 1.0, "density_change": 1.0})
    p_stopped, _ = fuser.probability({"stopped_vehicle": 1.0})
    p_both, _ = fuser.probability({"speed_drop": 0.9, "queue_growth": 0.8, "stopped_vehicle": 0.8})
    assert p_none < 0.05
    assert pytest.approx(1.0) == S and p_congestion < 0.25          # correlated symptoms: noisy-OR, not a sum
    for fam in ("stopped_vehicle", "trajectory_anomaly", "anomaly_score", "speed_drop"):
        assert fuser.probability({fam: 1.0})[0] < 0.4, fam           # any ONE cue alone stays below "possible"
    assert p_stopped < 0.6 < p_both                                  # symptoms + physical cue -> possible incident


def test_probability_is_monotone_in_every_family(fuser: WeightedLogisticFuser) -> None:
    base = {"speed_drop": 0.4, "stopped_vehicle": 0.3, "anomaly_score": 0.2}
    p0 = fuser.probability(base)[0]
    for fam in fuser.cfg.families:
        assert fuser.probability({**base, fam: min(1.0, base.get(fam, 0.0) + 0.3)})[0] >= p0


@pytest.mark.parametrize(("signals", "expected"), [
    ([], "normal"),
    ([(AnomalyType.SUDDEN_SLOWDOWN, 0.9)], "sudden_slowdown"),
    ([(AnomalyType.STOPPED_VEHICLE, 0.9)], "stopped_vehicle"),
    ([(AnomalyType.WRONG_WAY, 0.3)], "wrong_way"),
    ([(AnomalyType.SUDDEN_QUEUE, 0.9), (AnomalyType.SUDDEN_SLOWDOWN, 0.8)], "abnormal_queue"),
    ([(AnomalyType.SUDDEN_QUEUE, 1.0), (AnomalyType.SUDDEN_SLOWDOWN, 0.9), (AnomalyType.STOPPED_VEHICLE, 0.9)],
     "possible_incident"),
])
def test_incident_taxonomy(fuser: WeightedLogisticFuser, signals: list, expected: str) -> None:
    inc = fuser.fuse("junction_a_north", T0, [sig(t, s) for t, s in signals], None, {}, None)
    assert (inc.incident_type if inc else "normal") == expected


def test_taxonomy_is_configurable() -> None:
    cfg = load_yaml(get_settings().config_dir / "anomaly.yaml")
    cfg["incident_types"].insert(0, {"id": "suspected_breakdown", "label": "Suspected breakdown",
                                     "when": {"family": "stopped_vehicle", "min_evidence": 0.8}})
    f = WeightedLogisticFuser(config=cfg)
    inc = f.fuse("a", T0, [sig(AnomalyType.STOPPED_VEHICLE, 0.9)], None, {}, None)
    assert inc is not None and inc.incident_type == "suspected_breakdown" and inc.label == "Suspected breakdown"


def test_wording_never_claims_certainty(fuser: WeightedLogisticFuser) -> None:
    signals = [sig(AnomalyType.SUDDEN_QUEUE, 1.0), sig(AnomalyType.SUDDEN_SLOWDOWN, 1.0),
               sig(AnomalyType.STOPPED_VEHICLE, 1.0)]
    inc = fuser.fuse("a", T0, signals, None, {}, None)
    assert inc is not None and inc.calibrated is False and inc.probability < 1.0
    assert inc.label == "Possible incident" and "incident probability" in inc.message
    assert "not a confirmed accident" in inc.message and "uncalibrated" in inc.message
    assert "accident detected" not in inc.message.lower()
    assert "87%" in hedged_message("Possible incident", 0.87, {})


def test_lifecycle_same_id_escalates_never_downgrades_then_resolves(fuser: WeightedLogisticFuser) -> None:
    a = fuser.fuse("a", T0, [sig(AnomalyType.STOPPED_VEHICLE, 0.9)], None, {}, None)
    assert a is not None and a.incident_type == "stopped_vehicle" and a.status is IncidentStatus.OPEN
    strong = [sig(AnomalyType.STOPPED_VEHICLE, 0.9, 60), sig(AnomalyType.SUDDEN_QUEUE, 1.0, 60),
              sig(AnomalyType.SUDDEN_SLOWDOWN, 1.0, 60)]
    b = fuser.fuse("a", T0 + timedelta(seconds=60), strong, None, {}, a)
    assert b is not None and b.incident_id == a.incident_id and b.incident_type == "possible_incident"
    c = fuser.fuse("a", T0 + timedelta(seconds=120), [sig(AnomalyType.STOPPED_VEHICLE, 0.9, 120)], None, {}, b)
    assert c is not None and c.incident_type == "possible_incident"                 # no silent downgrade
    later = T0 + timedelta(hours=1)
    states = [fuser.fuse("a", later + timedelta(minutes=k), [], None, {}, c) for k in range(3)]
    assert [s.status for s in states] == [IncidentStatus.OPEN, IncidentStatus.OPEN, IncidentStatus.RESOLVED]
    assert fuser.fuse("a", later + timedelta(minutes=5), [], None, {}, None) is None


def test_old_evidence_fades(fuser: WeightedLogisticFuser) -> None:
    fresh = fuser.evidence(T0, [sig(AnomalyType.STOPPED_VEHICLE, 1.0)])[0]["stopped_vehicle"]
    old = fuser.evidence(T0 + timedelta(seconds=240), [sig(AnomalyType.STOPPED_VEHICLE, 1.0)])[0]["stopped_vehicle"]
    gone = fuser.evidence(T0 + timedelta(seconds=400), [sig(AnomalyType.STOPPED_VEHICLE, 1.0)])[0]["stopped_vehicle"]
    assert fresh == 1.0 and 0.0 < old < 0.5 and gone == 0.0


# --------------------------------------------------------------------------- risk
def feats(**kw: Any) -> TrafficFeatures:
    base: dict[str, Any] = dict(approach_id="junction_a_north", camera_id="c", window_start=T0, window_s=60.0,
                                vehicle_count=10, count_by_class={}, pcu_count=10.0, avg_speed_kmh=40.0,
                                speed_p85_kmh=45.0, density_veh_km=10.0, density_ratio=0.08, occupancy_pct=8.0,
                                flow_veh_h=600.0, inflow=10, outflow=10, queue_length_m=0.0, stopped_vehicles=0,
                                avg_gap_m=None, pedestrian_count=0, quality=1.0, extra={"roi_length_m": 60.0})
    base.update(kw)
    return TrafficFeatures(**base)


def congestion(level: CongestionLevel, score: float) -> CongestionAssessment:
    return CongestionAssessment(approach_id="junction_a_north", timestamp=T0, level=level, score=score,
                                classifier="rule_based")


def incident(p: float, status: IncidentStatus = IncidentStatus.OPEN) -> IncidentAssessment:
    return IncidentAssessment(approach_id="junction_a_north", timestamp=T0, probability=p, severity=Severity.HIGH,
                              status=status, incident_type="possible_incident")


@pytest.fixture
def risk() -> WeightedRiskEngine:
    return WeightedRiskEngine()


def test_risk_formula_is_the_documented_weighted_sum(risk: WeightedRiskEngine) -> None:
    f = feats(avg_speed_kmh=20.0, density_ratio=0.4, queue_length_m=30.0)
    r = risk.assess(f, congestion(CongestionLevel.HIGH, 0.6), None, incident(0.5), road_importance=1.0,
                    free_flow_kmh=50.0)
    c = {"current_severity": 0.6, "incident": 0.5, "queue": 30 / 60, "speed": (1 - 20 / 50) / 0.8,
         "density": 0.4 / 0.8}
    w = risk.cfg.weights
    base = sum(w[k] * v for k, v in c.items()) / sum(w[k] for k in c)          # forecast missing -> renormalised
    assert r.risk_score == pytest.approx(100 * base, abs=0.1)
    assert {k: r.components[k] for k in c} == pytest.approx(c, abs=1e-3)
    assert r.explanation[0].startswith("risk = 100 x")
    r_low_imp = risk.assess(f, congestion(CongestionLevel.HIGH, 0.6), None, incident(0.5), road_importance=0.0,
                            free_flow_kmh=50.0)
    assert r_low_imp.risk_score == pytest.approx(0.7 * r.risk_score, abs=0.2)


def test_risk_levels_and_reason_codes(risk: WeightedRiskEngine) -> None:
    calm = risk.assess(feats(), congestion(CongestionLevel.FREE, 0.05), None, None)
    assert calm.level is Severity.LOW and calm.risk_score < 25 and "POSSIBLE_INCIDENT" not in calm.reason_codes
    fc = Forecast(approach_id="junction_a_north", issued_at=T0, model_name="m", model_version="v", points=[],
                  predicted_level={5: CongestionLevel.HIGH, 15: CongestionLevel.SEVERE, 30: CongestionLevel.SEVERE})
    bad = risk.assess(feats(avg_speed_kmh=5.0, density_ratio=0.75, queue_length_m=58.0),
                      congestion(CongestionLevel.SEVERE, 0.95), fc, incident(0.9), road_importance=0.8,
                      signals=[sig(AnomalyType.STOPPED_VEHICLE, 0.9), sig(AnomalyType.SUDDEN_QUEUE, 0.8)])
    assert bad.level is Severity.CRITICAL and bad.risk_score >= 75
    for code in ("SEVERE_CONGESTION_NOW", "SEVERE_CONGESTION_FORECAST", "POSSIBLE_INCIDENT", "LONG_QUEUE",
                 "LOW_SPEED", "HIGH_DENSITY", "STOPPED_VEHICLE", "QUEUE_INCREASING", "HIGH_IMPORTANCE_ROAD"):
        assert code in bad.reason_codes, code
    worsening = risk.assess(feats(), congestion(CongestionLevel.MODERATE, 0.4), fc, None)
    assert "CONGESTION_WORSENING" in worsening.reason_codes


def test_safety_floors(risk: WeightedRiskEngine) -> None:
    ww = risk.assess(feats(), congestion(CongestionLevel.FREE, 0.05), None, None,
                     signals=[sig(AnomalyType.WRONG_WAY, 0.3)])
    assert ww.risk_score < 25 and ww.level is Severity.HIGH and "WRONG_WAY_VEHICLE" in ww.reason_codes
    assert any("safety floor" in e for e in ww.explanation)
    p = risk.assess(feats(), congestion(CongestionLevel.FREE, 0.05), None, incident(0.9))
    assert p.level is Severity.HIGH
    dismissed = risk.assess(feats(), congestion(CongestionLevel.FREE, 0.05), None,
                            incident(0.9, IncidentStatus.DISMISSED))
    assert dismissed.components["incident"] == 0.0 and dismissed.level is Severity.LOW


def test_low_data_quality_is_flagged(risk: WeightedRiskEngine) -> None:
    r = risk.assess(feats(quality=0.3), congestion(CongestionLevel.FREE, 0.05), None, None)
    assert "LOW_DATA_QUALITY" in r.reason_codes
