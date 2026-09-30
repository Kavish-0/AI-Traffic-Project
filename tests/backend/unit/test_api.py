"""REST + WebSocket API against a populated database (service layer -> store). No torch in the API."""

from __future__ import annotations

import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from test_store import features, state

from traffic_intel.api.live import LiveHub
from traffic_intel.api.main import create_app
from traffic_intel.core.config import get_settings
from traffic_intel.core.enums import CongestionLevel, IncidentStatus, Severity
from traffic_intel.core.schemas import IncidentAssessment, RiskAssessment
from traffic_intel.database.store import TrafficStore
from traffic_intel.pipeline.bus import InMemoryEventBus
from traffic_intel.services.context import ServiceContext


def _now() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


@pytest.fixture
def store(tmp_path: Path) -> TrafficStore:
    s = TrafficStore(f"sqlite:///{(tmp_path / 'api.db').as_posix()}")
    now = _now()
    for k in range(4):
        start = now - timedelta(minutes=4 - k)
        level = CongestionLevel.SEVERE if k == 3 else CongestionLevel.MODERATE
        s.add_window(features(start), state(start, level=level))
    s.upsert_live_state(state(now - timedelta(seconds=5), live=True, speed=11))
    s.upsert_camera_status("cam_x", "junction_a_north", state="running", frames=100, last_frame_at=now,
                           source_kind="synthetic", preview_path=None)
    s.add_risk(RiskAssessment(approach_id="junction_a_north", timestamp=now - timedelta(minutes=1), risk_score=72,
                              level=Severity.HIGH, components={"incident": 0.87}, explanation=["risk = ..."],
                              reason_codes=["POSSIBLE_INCIDENT"]), incident_score=0.87)
    s.upsert_incident(IncidentAssessment(
        incident_id="inc-1", approach_id="junction_a_north", timestamp=now, probability=0.87, severity=Severity.HIGH,
        incident_type="possible_incident", label="Possible incident", message="Possible incident - 87% (uncalibrated)",
        junction_id="junction_a", road_id="road_a_sb", evidence={"stopped_vehicle": 0.9}, first_detected=now))
    return s


@pytest.fixture
def client(store: TrafficStore):  # noqa: ANN201
    app = create_app()
    app.state.ctx_override = ServiceContext.build(store)
    with TestClient(app) as c:
        yield c


def test_network_junctions_roads_cameras(client: TestClient) -> None:
    geo = client.get("/api/v1/network").json()
    kinds = {f["properties"]["kind"] for f in geo["features"]}
    assert {"road", "junction", "camera", "incident"} <= kinds and len(geo["bbox"]) == 4
    road = next(f for f in geo["features"] if f["id"] == "road_a_sb")["properties"]
    assert road["level"] == "severe" and road["stale"] is False and road["incident"] == "inc-1"
    unmonitored = next(f for f in geo["features"] if f["id"] == "road_f_nb")["properties"]
    assert unmonitored["level"] == "unknown" and unmonitored["monitored"] is False      # never guessed
    j = client.get("/api/v1/junctions/junction_a").json()
    north = next(a for a in j["approaches"] if a["id"] == "junction_a_north")
    assert north["level"] == "severe" and "cam_x" in north["cameras"] and j["active_incidents"] == 1
    assert client.get("/api/v1/junctions/nope").status_code == 404
    roads = {r["id"]: r for r in client.get("/api/v1/roads").json()}
    assert roads["road_a_sb"]["monitored"] and not roads["road_g_sw"]["monitored"]
    assert client.get("/api/v1/roads/road_g_sw").status_code == 404                    # not monitored
    cams = {c["id"]: c for c in client.get("/api/v1/cameras").json()}
    assert cams["cam_x"]["status"] == "online" and cams["cam_x"]["synthetic"]
    assert cams["cam_01"]["status"] == "never_started"
    r = client.get("/api/v1/cameras/cam_x/frame.jpg")
    assert r.status_code == 404 and "Video stream unavailable" in r.json()["detail"]


def test_snapshot_forecast_history_summary(client: TestClient) -> None:
    snap = client.get("/api/v1/roads/road_a_sb").json()
    assert snap["approach"]["id"] == "junction_a_north" and snap["live"]["average_speed_kmh"] == 11
    assert snap["live_stale"] is False and snap["window"]["traffic_state"] == "severe"
    assert snap["incident"]["score_label"] == "Incident probability: UNCALIBRATED" and snap["incident_score"] == 0.87
    assert snap["risk"]["reason_codes"] == ["POSSIBLE_INCIDENT"] and snap["synthetic"] is True
    fc = snap["forecast"]
    assert fc["status"] == "insufficient_history" and "4/30" in fc["message"] and fc["horizons"] == []
    hist = client.get("/api/v1/approaches/junction_a_north/history?minutes=30").json()
    assert len(hist) == 4 and hist[0]["timestamp"] < hist[-1]["timestamp"]
    assert client.get("/api/v1/approaches/junction_a_east/history").json() == []           # empty, not an error
    fva = client.get("/api/v1/approaches/junction_a_north/forecast/history?horizon_min=15").json()
    assert fva["forecast"] == [] and len(fva["actual"]) == 4
    assert client.get("/api/v1/approaches/junction_a_north/forecast/history?target=bogus").status_code == 422
    s = client.get("/api/v1/traffic/summary").json()
    assert s["fresh_approaches"] == 1 and s["vehicles"] == 12 and s["avg_speed_kmh"] == 11
    assert s["congested_approaches"] == 1 and s["active_incidents"] == 1 and s["synthetic_approaches"] == 1
    assert len(client.get("/api/v1/risk/junction_a_north/history").json()) == 1


def test_incident_feedback_and_models(client: TestClient, store: TrafficStore) -> None:
    r = client.patch("/api/v1/incidents/inc-1", json={"status": "dismissed", "label": "false_alarm", "note": "shadow"})
    assert r.status_code == 200 and r.json()["operator_label"] == "false_alarm"
    assert store.get_incident("inc-1")["status"] == IncidentStatus.DISMISSED.value
    assert client.get("/api/v1/incidents?active_only=true").json() == []
    assert client.patch("/api/v1/incidents/missing", json={"status": "resolved"}).status_code == 404
    models = client.get("/api/v1/models?production_only=true").json()
    assert any(m["task"] == "forecasting" for m in models)


def test_what_if_uses_fresh_observed_baseline(client: TestClient) -> None:
    body = client.post("/api/v1/simulation/what-if", json={"closed_segment_ids": ["road_b_wb"]}).json()
    a = next(i for i in body["impacts"] if i["segment_id"] == "road_a_sb")
    assert a["baseline_observed"] is True and a["baseline_flow_veh_h"] == 720.0            # latest window flow
    assert any("NOT a microscopic" in x for x in body["assumptions"])
    assert client.get(f"/api/v1/simulation/scenarios/{body['scenario_id']}").status_code == 200


def test_system_status_reports_real_components(client: TestClient) -> None:
    s = client.get("/api/v1/system/status").json()
    assert s["database"]["ok"] and s["database"]["status"] == "connected"
    assert s["cameras"]["status"] == "online" and s["pipeline"]["status"] == "running"
    assert s["forecast_model"]["status"] in ("available", "loaded")
    assert client.get("/api/v1/health").json()["database"] == "connected"


@pytest.fixture
def ws_client(store: TrafficStore, monkeypatch: pytest.MonkeyPatch):  # noqa: ANN201
    """The optional push channel (TI_WS_ENABLED) - a future interface, not used by the dashboard."""
    monkeypatch.setenv("TI_WS_ENABLED", "true")
    get_settings.cache_clear()
    try:
        app = create_app()
        app.state.ctx_override = ServiceContext.build(store)
        with TestClient(app) as c:
            yield c
    finally:
        monkeypatch.delenv("TI_WS_ENABLED")
        get_settings.cache_clear()


def test_websocket_is_off_by_default(client: TestClient) -> None:
    """The dashboard polls REST; the push channel is not mounted (and LiveHub not started) unless enabled."""
    assert get_settings().ws_enabled is False
    assert not [r for r in client.app.routes if getattr(r, "path", "").startswith("/ws")]
    assert client.get("/api/v1/traffic/current").status_code == 200        # what the dashboard polls instead


def test_websocket_pushes_compact_traffic_updates(ws_client: TestClient, store: TrafficStore) -> None:
    with ws_client.websocket_connect("/ws/traffic") as ws:
        ws.send_json({"action": "subscribe", "topics": ["traffic.update"], "approach_ids": ["junction_a_south"]})
        # the connection starts on the default topics, so hub messages may arrive before the ack
        for _ in range(50):
            if ws.receive_json().get("action") == "subscribed":
                break
        else:
            pytest.fail("no subscription acknowledgement")
        time.sleep(0.2)
        store.upsert_live_state(state(_now(), approach="junction_a_south", live=True, speed=44))
        for _ in range(50):
            msg = ws.receive_json()
            if msg["payload"].get("is_live"):
                break
        assert msg["topic"] == "traffic.update"
        p = msg["payload"]
        assert p["approach_id"] == "junction_a_south" and p["is_live"] and p["average_speed"] == 44
        expected = {"timestamp", "junction_id", "road_id", "vehicle_count", "average_speed", "density", "occupancy",
                    "flow", "queue_length", "traffic_state", "forecast", "incident_score", "risk_level"}
        assert expected <= set(p)
        assert not {"xyxy", "bbox", "track_ids", "detections"} & set(p)                   # never frame-level


def test_live_hub_does_not_drop_a_lagging_camera(store: TrafficStore) -> None:
    """Cameras write their own timestamps: an update from a camera running behind another one must still be
    pushed (a single global 'newest timestamp' watermark used to drop it)."""
    hub = LiveHub(ServiceContext.build(store), InMemoryEventBus())
    hub.poll()                                                              # seeds; stored history is not replayed
    assert not [m for m in hub.poll() if m.topic == "traffic.update"]
    t = _now()
    store.upsert_live_state(state(t, approach="junction_a_north", live=True, speed=50))
    store.upsert_live_state(state(t - timedelta(seconds=3), approach="junction_a_south", live=True, speed=20))
    live = {m.payload["approach_id"]: m.payload["average_speed"] for m in hub.poll() if m.topic == "traffic.update"}
    assert live == {"junction_a_north": 50, "junction_a_south": 20}
    assert not [m for m in hub.poll() if m.topic == "traffic.update"]      # nothing new -> nothing pushed

    store.add_window(features(t), state(t))                                                        # north ahead
    store.add_window(features(t - timedelta(minutes=3), approach="junction_a_south"),
                     state(t - timedelta(minutes=3), approach="junction_a_south"))                  # south behind
    windows = [m.payload["approach_id"] for m in hub.poll() if m.topic == "traffic.update"]
    assert sorted(windows) == ["junction_a_north", "junction_a_south"]


def test_api_without_database_reports_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TI_DATABASE_URL", "postgresql+psycopg://nobody:x@127.0.0.1:1/none?connect_timeout=2")
    get_settings.cache_clear()
    try:
        with TestClient(create_app()) as c:
            assert c.get("/api/v1/health").json()["database"] == "disconnected"
            st = c.get("/api/v1/system/status").json()
            assert st["database"]["ok"] is False and st["backend"]["ok"] is True
            r = c.get("/api/v1/junctions")
            assert r.status_code == 503 and "Database unavailable" in r.json()["detail"]
    finally:
        monkeypatch.delenv("TI_DATABASE_URL")
        get_settings.cache_clear()


def test_api_process_stays_free_of_ml_frameworks() -> None:
    code = ("import sys, traffic_intel.api.main; "
            "print(','.join(m for m in ('torch', 'ultralytics', 'sklearn', 'xgboost') if m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == ""
