"""TrafficStore on the ORM models (SQLite here; the same models serve PostgreSQL / PostGIS)."""

from __future__ import annotations

import struct
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import get_args

import pytest

from traffic_intel.core.enums import (
    AnomalyScope,
    AnomalyType,
    CongestionLevel,
    IncidentStatus,
    ObjectClass,
    Severity,
)
from traffic_intel.core.feature_schema import FORECAST_TARGETS
from traffic_intel.core.schemas import (
    AnomalySignal,
    ContextFeatures,
    Forecast,
    ForecastPoint,
    ForecastTarget,
    IncidentAssessment,
    RiskAssessment,
    TrafficFeatures,
    TrafficState,
)
from traffic_intel.database.store import TrafficStore
from traffic_intel.database.types import wkb_to_wkt, wkt_coords

NOW = datetime.now(UTC).replace(microsecond=0)


def features(start: datetime, approach: str = "junction_a_north", speed: float = 30.0) -> TrafficFeatures:
    return TrafficFeatures(
        approach_id=approach, camera_id="cam_x", window_start=start, window_s=60.0, vehicle_count=12,
        count_by_class={ObjectClass.CAR: 10, ObjectClass.TRUCK: 2}, pcu_count=13.0, avg_speed_kmh=speed,
        speed_p85_kmh=speed + 5, density_veh_km=20.0, density_ratio=0.15, occupancy_pct=12.0, flow_veh_h=720.0,
        inflow=12, outflow=11, queue_length_m=8.0, stopped_vehicles=1, avg_gap_m=None, pedestrian_count=0,
        quality=1.0, extra={"roi_length_m": 60.0})


def state(start: datetime, approach: str = "junction_a_north", live: bool = False,
          level: CongestionLevel = CongestionLevel.MODERATE, speed: float = 30.0) -> TrafficState:
    return TrafficState(
        timestamp=start + timedelta(seconds=60), window_start=start, window_s=60.0 if not live else 5.0,
        is_live=live, camera_id="cam_x", approach_id=approach, junction_id="junction_a", road_id="road_a_sb",
        vehicle_count=12, car_count=10, motorcycle_count=0, bus_count=0, truck_count=2, bicycle_count=0,
        pedestrian_count=0, average_speed_kmh=speed, median_speed_kmh=speed, density_veh_km=20.0, density_pct=15.0,
        occupancy_pct=12.0, flow_veh_h=720.0, inflow=12, outflow=11, queue_length_m=8.0, stopped_vehicle_count=1,
        traffic_state=level, congestion_score=0.4, explanation=["moderate"], quality=1.0)


@pytest.fixture
def store(tmp_path: Path) -> TrafficStore:
    return TrafficStore(f"sqlite:///{(tmp_path / 't.db').as_posix()}")


def test_wkb_and_wkt_helpers() -> None:
    pt = b"\x01" + struct.pack("<I", 1) + struct.pack("<dd", 77.5, 12.9)
    assert wkb_to_wkt(pt) == "POINT(77.5 12.9)"
    ewkb_line = b"\x01" + struct.pack("<I", 2 | 0x20000000) + struct.pack("<I", 4326) + struct.pack("<I", 2) + \
        struct.pack("<dddd", 1, 2, 3, 4)
    assert wkb_to_wkt(ewkb_line) == "LINESTRING(1.0 2.0, 3.0 4.0)"
    poly = (b"\x01" + struct.pack("<I", 3) + struct.pack("<I", 1) + struct.pack("<I", 3)
            + struct.pack("<6d", 0, 0, 1, 0, 0, 0))
    assert wkb_to_wkt(poly).startswith("POLYGON((0.0 0.0")
    assert wkt_coords("LINESTRING(1 2, 3 4)") == [(1.0, 2.0), (3.0, 4.0)]
    assert wkt_coords(None) == []


def test_forecast_targets_match_feature_schema() -> None:
    assert set(get_args(ForecastTarget)) == set(FORECAST_TARGETS)


def test_network_sync_and_windows(store: TrafficStore) -> None:
    cams = store.cameras()
    assert "cam_01" in cams and cams["cam_01"]["lon"] is not None           # synced from configs/cameras
    for k in range(3):
        start = NOW - timedelta(minutes=3 - k)
        store.add_window(features(start, speed=30 - k), state(start, speed=30 - k))
    store.add_window(features(NOW - timedelta(minutes=1), "junction_a_east"),
                     state(NOW - timedelta(minutes=1), "junction_a_east"))
    latest = {s["approach_id"]: s for s in store.latest_traffic_states()}
    north = latest["junction_a_north"]
    assert north["average_speed_kmh"] == 28 and north["road_id"] == "road_a_sb"
    assert north["car_count"] == 10 and north["traffic_state"] == "moderate"
    hist = store.traffic_history("junction_a_north")
    assert [h["average_speed_kmh"] for h in hist] == [30, 29, 28]          # oldest first
    assert store.windows_since(NOW - timedelta(minutes=1, seconds=30))
    assert "cam_x" in store.cameras()                                       # runtime camera auto-registered


def test_live_forecast_risk_incident_roundtrip(store: TrafficStore) -> None:
    store.upsert_live_state(state(NOW - timedelta(seconds=5), live=True))
    store.upsert_live_state(state(NOW - timedelta(seconds=4), live=True, speed=12))
    live = store.latest_live_states()
    assert len(live) == 1 and live[0]["average_speed_kmh"] == 12 and live[0]["_ts"].tzinfo is not None

    fc = Forecast(approach_id="junction_a_north", issued_at=NOW, model_name="xgboost", model_version="v1",
                  points=[ForecastPoint(horizon_min=h, target_time=NOW + timedelta(minutes=h), target=t, value=v)
                          for h in (5, 15) for t, v in (("avg_speed_kmh", 20.0), ("occupancy_pct", 30.0))],
                  predicted_level={5: CongestionLevel.HIGH, 15: CongestionLevel.SEVERE})
    store.add_forecast(fc)
    back = store.latest_forecast("junction_a_north")
    assert back is not None and back.predicted_level == {5: CongestionLevel.HIGH, 15: CongestionLevel.SEVERE}
    assert {p.target for p in back.points} == {"avg_speed_kmh", "occupancy_pct"}
    assert len(store.forecast_history("junction_a_north", 15, "avg_speed_kmh")) == 1

    store.add_risk(RiskAssessment(approach_id="junction_a_north", timestamp=NOW, risk_score=55, level=Severity.HIGH,
                                  components={"incident": 0.4}, explanation=["x"], reason_codes=["LOW_SPEED"]),
                   incident_score=0.42)
    r = store.latest_risk()[0]
    assert r["incident_score"] == 0.42 and r["junction_id"] == "junction_a" and r["reason_codes"] == ["LOW_SPEED"]

    sig = AnomalySignal(approach_id="junction_a_north", camera_id="cam_x", timestamp=NOW,
                        type=AnomalyType.STOPPED_VEHICLE,
                        scope=AnomalyScope.TRACK, detector="stopped_vehicle", score=0.9, track_ids=[3],
                        details={"location": {"lon": 77.5946, "lat": 12.972}})
    store.add_anomalies([sig])
    a = store.list_anomalies(type="stopped_vehicle")[0]
    assert a["lon"] == pytest.approx(77.5946) and a["junction_id"] == "junction_a"

    inc = IncidentAssessment(incident_id="i1", approach_id="junction_a_north", timestamp=NOW, probability=0.7,
                             severity=Severity.HIGH, incident_type="stopped_vehicle", label="Stopped vehicle",
                             junction_id="junction_a", road_id="road_a_sb", evidence={"stopped_vehicle": 0.9},
                             contributing=[sig], first_detected=NOW)
    store.upsert_incident(inc)
    store.set_status("i1", IncidentStatus.ACKNOWLEDGED, "true_incident", "checked on CCTV")
    kept = store.upsert_incident(inc.model_copy(update={"probability": 0.9, "timestamp": NOW + timedelta(minutes=1)}))
    assert kept.status is IncidentStatus.ACKNOWLEDGED                       # operator decision wins
    row = store.get_incident("i1")
    assert row["max_probability"] == 0.9 and row["operator_label"] == "true_incident"
    assert row["contributing"][0]["track_ids"] == [3] and "UNCALIBRATED" in row["uncertainty_note"]
    assert store.open_incident_for("junction_a_north").contributing[0].type is AnomalyType.STOPPED_VEHICLE
    assert store.incidents_updated_since(NOW)
    resolved = {"status": IncidentStatus.RESOLVED, "timestamp": NOW + timedelta(minutes=2)}
    store.upsert_incident(inc.model_copy(update=resolved))
    assert store.open_incident_for("junction_a_north") is None


def test_runtime_status_context_models(store: TrafficStore) -> None:
    store.upsert_camera_status("cam_rt", "junction_a_south", state="running", frames=10, last_frame_at=NOW,
                               source_kind="synthetic", details={"name": "x"})
    st = store.camera_statuses()["cam_rt"]
    assert st["state"] == "running" and st["updated_at"].tzinfo is not None and st["source_kind"] == "synthetic"
    store.add_context("demo_city", ContextFeatures(timestamp=NOW, hour=1, day_of_week=2, is_holiday=False,
                                                   weather="rain", temperature_c=21.0), "open_meteo")
    assert store.latest_context("demo_city")["weather"] == "rain"
    from traffic_intel.ai.model_store import FileModelStore
    from traffic_intel.core.config import get_settings

    fs = FileModelStore(get_settings().models_dir)
    assert store.sync_models(fs.list(), {}) == len(fs.list())
    assert store.ping()
