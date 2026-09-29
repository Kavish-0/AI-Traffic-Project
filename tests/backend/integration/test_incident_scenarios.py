"""The six synthetic incident scenarios, end to end through the real pipeline (tracking onwards):
trajectories -> feature engine -> track + series detectors -> fusion -> risk.

Checks LOGICAL behaviour (the right kind of situation is reported, with hedged probabilities and
explainable risk), not exact numbers."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from functools import cache
from pathlib import Path

import pandas as pd
import pytest

from traffic_intel.ai.model_store import FileModelStore
from traffic_intel.ai.training.anomaly import AnomalyExperimentConfig, run_anomaly_experiment
from traffic_intel.core.enums import AnomalyType, ModelTask, Severity
from traffic_intel.core.registry import ComponentSpec
from traffic_intel.core.schemas import BusMessage
from traffic_intel.incidents.scenarios import SCENARIOS, ScenarioResult, run_scenario
from traffic_intel.pipeline.bus import InMemoryEventBus, Topics

ORDER = [Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL]


@cache
def result(name: str) -> ScenarioResult:
    return run_scenario(name)


def _levels(r: ScenarioResult) -> list[Severity]:
    return [u.risk.level for u in r.after() if u.risk is not None]


def _codes(r: ScenarioResult) -> set[str]:
    return {c for u in r.after() if u.risk is not None for c in u.risk.reason_codes}


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_quiet_before_the_event_and_hedged_wording(name: str) -> None:
    r = result(name)
    assert len(r.before()) >= 9 and len(r.after()) >= 4
    assert r.max_probability(r.before()) < 0.2                  # 10 normal minutes: nothing reported
    for u in r.updates:
        if u.incident is not None:
            assert u.incident.calibrated is False and 0.0 <= u.incident.probability < 1.0
            assert "incident probability" in u.incident.message and "not a confirmed accident" in u.incident.message


def test_1_normal_traffic() -> None:
    r = result("normal")
    assert r.types_after() == [] and r.track_signals == 0
    assert set(_levels(r)) == {Severity.LOW}


def test_2_sudden_slowdown() -> None:
    r = result("sudden_slowdown")
    assert r.final_type() == "sudden_slowdown" and "possible_incident" not in r.types_after()
    assert r.max_probability(r.after()) < 0.6                  # a slowdown alone is not called an incident
    assert "ABNORMAL_SLOWDOWN" in _codes(r)
    first = r.after()[0]
    assert any(s.type is AnomalyType.SUDDEN_SLOWDOWN and s.details["drop_pct"] > 50 for s in first.signals)


def test_3_stopped_vehicle() -> None:
    r = result("stopped_vehicle")
    stopped = [s for s in r.signals if s.type is AnomalyType.STOPPED_VEHICLE]
    assert stopped and len({s.track_ids[0] for s in stopped}) == 1           # exactly the broken-down car
    s = stopped[0]
    assert s.details["stopped_duration_s"] >= 30 and 0 < s.details["confidence"] <= 1
    assert s.details["road_id"] == "road_a_sb" and {"lon", "lat"} <= set(s.details["location"])
    assert s.details["location"]["distance_upstream_of_stop_line_m"] == pytest.approx(37, abs=4)
    assert r.final_type() == "stopped_vehicle" and r.max_probability(r.after()) < 0.6
    assert "STOPPED_VEHICLE" in _codes(r)


def test_4_wrong_way() -> None:
    r = result("wrong_way")
    ww = [s for s in r.signals if s.type is AnomalyType.WRONG_WAY]
    assert ww and len({s.track_ids[0] for s in ww}) == 1
    assert r.final_type() == "wrong_way"
    assert Severity.HIGH in _levels(r) and "WRONG_WAY_VEHICLE" in _codes(r)   # safety floor, whatever the score


def test_5_queue_formation() -> None:
    r = result("queue_formation")
    assert r.final_type() == "abnormal_queue" and "possible_incident" not in r.types_after()
    assert not [s for s in r.signals if s.type is AnomalyType.STOPPED_VEHICLE]   # creeping queue != breakdown
    assert "QUEUE_INCREASING" in _codes(r)


def test_6_multiple_abnormal_signals() -> None:
    r = result("multiple_signals")
    assert r.final_type() == "possible_incident" and r.max_probability(r.after()) >= 0.8
    peak = max((u.incident for u in r.after() if u.incident), key=lambda i: i.probability)
    strong = [f for f, v in peak.evidence.items() if f != "symptom_strength" and v >= 0.3]
    assert len(strong) >= 3 and "stopped_vehicle" in strong
    assert peak.label == "Possible incident" and peak.lon is not None
    assert max(_levels(r), key=ORDER.index) in (Severity.HIGH, Severity.CRITICAL)
    assert {"POSSIBLE_INCIDENT", "STOPPED_VEHICLE", "QUEUE_INCREASING"} <= _codes(r)


def test_escalation_order_across_scenarios() -> None:
    """More (independent) evidence -> higher probability; one symptom family alone stays low."""
    p = {n: result(n).max_probability(result(n).after()) for n in SCENARIOS}
    assert p["normal"] < p["sudden_slowdown"] < p["multiple_signals"]
    assert p["stopped_vehicle"] < p["multiple_signals"] and p["queue_formation"] < p["multiple_signals"]


# --------------------------------------------------------------------------- analytics worker over the bus
async def test_analytics_worker_over_the_bus_matches_the_direct_engine() -> None:
    """Replay what the camera worker published (track signals, window features, window states - the
    latter two as SEPARATE messages, in either order) through the bus-driven analytics worker."""
    from traffic_intel.core.config import get_settings, load_network, load_pipeline_config
    from traffic_intel.pipeline.analytics_worker import AnalyticsWorker

    pipeline = load_pipeline_config()
    network = load_network(get_settings().config_dir / pipeline.network_file)
    bus = InMemoryEventBus()
    worker = AnalyticsWorker(network, pipeline, bus, with_forecast=False)
    r = result("multiple_signals")
    out: list[BusMessage] = []

    async def consume() -> None:
        async for m in bus.subscribe(Topics.INCIDENT, Topics.RISK):
            out.append(m)

    task = asyncio.create_task(consume())
    await asyncio.sleep(0)
    now = datetime.now(UTC)

    def msg(topic: str, model) -> BusMessage:  # noqa: ANN001
        return BusMessage(topic=topic, timestamp=now, source="replay", payload=model.model_dump(mode="json"))

    signals = sorted(r.signals, key=lambda s: s.timestamp)
    i = 0
    for k, (f, st) in enumerate(r.windows):
        while i < len(signals) and signals[i].timestamp <= st.timestamp:
            await worker.handle(msg(Topics.ANOMALY_SIGNAL, signals[i]))
            i += 1
        pair = [(Topics.FEATURES_WINDOW, f), (Topics.TRAFFIC_STATE, st)]
        first, second = pair[::-1] if k % 2 else pair
        assert await worker.handle(msg(*first)) is None                  # waits for its partner
        assert await worker.handle(msg(*second)) is not None
    await asyncio.sleep(0.05)
    task.cancel()
    assert worker.updates == len(r.windows)
    risk = [m for m in out if m.topic == Topics.RISK]
    incidents = [m for m in out if m.topic == Topics.INCIDENT]
    assert len(risk) == len(r.windows)
    assert incidents and incidents[-1].payload["incident_type"] == "possible_incident"
    direct = [u.risk.risk_score for u in r.updates if u.risk]
    assert [m.payload["risk_score"] for m in risk] == pytest.approx(direct, abs=0.2)


def test_learned_detectors_train_save_load(tmp_path: Path) -> None:
    cfg = AnomalyExperimentConfig(dataset={"days": 3, "seed": 5, "incident_rate_per_day": 1.0}, train_days=2,
                                  learned={"isolation_forest": {"n_estimators": 50, "max_samples": 256},
                                           "autoencoder": {"max_iter": 50}},
                                  statistical=[ComponentSpec(type="sudden_slowdown")])
    store = FileModelStore(tmp_path / "models")
    res = run_anomaly_experiment(cfg, experiments_root=tmp_path / "exp", store=store)
    m = pd.read_csv(Path(res.directory) / "metrics.csv").set_index("detector")
    assert {"isolation_forest", "autoencoder", "sudden_slowdown", "fused_probability"} <= set(m.index)
    assert ((m["false_alarms_per_h"] >= 0) & (m["pr_auc"].fillna(0) >= 0)).all()
    assert (Path(res.directory) / "report.md").exists()
    for kind in ("isolation_forest", "autoencoder"):
        det = store.load(ModelTask.ANOMALY, kind, res.experiment_id)
        assert det.q_normal > 0 and det.metadata is not None and det.metadata.metrics["detector"] == kind
