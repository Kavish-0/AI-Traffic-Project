"""PEMS-BAY real-data mode: network generation, time mapping, sensor replay, speed-only congestion."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from traffic_intel.core.config import AppSettings, NetworkConfig, load_pipeline_config
from traffic_intel.core.schemas import TrafficFeatures
from traffic_intel.datasets.pems_bay_network import build_network, free_flow_speeds
from traffic_intel.pipeline.bus import Topics
from traffic_intel.pipeline.publish import CollectingPublisher
from traffic_intel.pipeline.sensor_worker import CAMERA_ID, SensorReplayConfig, SensorReplayWorker
from traffic_intel.traffic.congestion.rules import RuleBasedCongestionClassifier


def _meta() -> pd.DataFrame:
    # two stations each way on one freeway + one station on a crossing freeway 300 m away
    return pd.DataFrame({
        "fwy": [101, 101, 101, 101, 880], "dir": ["N", "N", "S", "S", "N"],
        "name": ["LARK AVE", "SB 85 rm-n-conn", "LARK AVE", "OLD IRONSIDES", "MAIN ST"],
        "lat": [37.30, 37.31, 37.30, 37.31, 37.3027], "lon": [-121.90, -121.90, -121.90, -121.90, -121.9],
        "abs_pm": [1.0, 2.0, 1.0, 2.0, 0.5], "lanes": [3, 4, 3, 3, 2], "type": ["ML"] * 5,
    }, index=pd.Index(["1", "2", "3", "4", "5"], name="sensor"))


def _network() -> NetworkConfig:
    return NetworkConfig.model_validate(build_network(_meta(), {"1": 100.0, "2": 95.0}))


def test_network_chains_stations_in_travel_order() -> None:
    net = _network()
    assert {a.id for a in net.approaches} == {f"pems_bay_{i}" for i in "12345"}
    seg = {s.id: s for s in net.segments}
    # northbound: station 1 then 2 (post mile ascending); southbound: 4 then 3
    assert seg["seg_pems_bay_2"].from_node == "n_pems_bay_1"
    assert seg["seg_pems_bay_3"].from_node == "n_pems_bay_4"
    assert seg["seg_pems_bay_1"].free_flow_speed_kmh == 100.0
    assert seg["seg_pems_bay_3"].free_flow_speed_kmh == 105.0          # no measurement -> speed limit
    assert {j.id for j in net.junctions} == {"pems_bay_us101_n", "pems_bay_us101_s", "pems_bay_i880_n"}
    # opposite carriageways are drawn apart, and connectors are only between different freeways
    assert net.segments[0].geometry != next(s for s in net.segments if s.id == "seg_pems_bay_3").geometry
    conns = [s for s in net.segments if s.id.startswith("conn_")]
    assert conns and all("assumed connector" in s.name for s in conns)
    assert not any("us101_n" in s.id and "us101_s" in s.id for s in conns)
    assert "Lark Ave" in seg["seg_pems_bay_1"].name and "SB 85" in seg["seg_pems_bay_2"].name


def test_free_flow_uses_training_period_only() -> None:
    idx = pd.date_range("2017-01-01", periods=10, freq="5min", tz="UTC")
    wide = pd.DataFrame({"a": [100.0] * 7 + [20.0] * 3, "b": [np.nan] * 10}, index=idx)
    ff = free_flow_speeds(wide, train_fraction=0.7)
    assert ff["a"] == 100.0 and ff["b"] == 105.0


def _worker(start_wall: datetime, pub: CollectingPublisher | None = None, **cfg: object) -> SensorReplayWorker:
    net = _network()
    c = SensorReplayConfig(**{"backfill_steps": 3, **cfg})
    idx = pd.date_range("2017-06-04", "2017-06-13", freq="5min", tz="America/Los_Angeles").tz_convert("UTC")
    speeds = pd.DataFrame({a.id: 100.0 for a in net.approaches}, index=idx)
    # station 1 crawls at 10 km/h on Monday 2017-06-05 at 08:00-08:05 local time
    speeds.loc[pd.Timestamp("2017-06-05 08:00", tz="America/Los_Angeles").tz_convert("UTC"), "pems_bay_1"] = 10.0
    speeds.loc[pd.Timestamp("2017-06-05 08:05", tz="America/Los_Angeles").tz_convert("UTC"), "pems_bay_2"] = np.nan
    pipeline = load_pipeline_config(AppSettings(pipeline_file="pipeline.yaml"))
    return SensorReplayWorker(c, net, pipeline, speeds, publisher=pub, clock=lambda: start_wall)


def test_wall_clock_maps_to_same_weekday_and_local_time_in_the_replay_week() -> None:
    # Monday 2026-09-28 08:00 PDT = 15:00 UTC  ->  Monday 2017-06-05 08:00 PDT
    w = _worker(datetime(2026, 9, 28, 15, 0, tzinfo=UTC))
    rec = w.recorded_time(datetime(2026, 9, 28, 15, 0, tzinfo=UTC))
    assert rec.tz_convert("America/Los_Angeles") == pd.Timestamp("2017-06-05 08:00", tz="America/Los_Angeles")
    sun = w.recorded_time(datetime(2026, 10, 4, 20, 30, tzinfo=UTC)).tz_convert("America/Los_Angeles")
    assert (sun.dayofweek, sun.hour, sun.minute) == (6, 13, 30)
    assert sun.date() == pd.Timestamp("2017-06-11").date()


def test_replay_publishes_speed_only_windows_and_skips_gaps() -> None:
    pub = CollectingPublisher()
    w = _worker(datetime(2026, 9, 28, 15, 12, tzinfo=UTC), pub)      # 08:12 PDT -> last closed window 08:05
    w.run(max_seconds=0)
    windows = [m for m in pub.messages if m.topic == Topics.TRAFFIC_STATE]
    feats = [m for m in pub.messages if m.topic == Topics.FEATURES_WINDOW]
    # 3 backfilled steps x 5 stations, minus the one missing reading
    assert len(windows) == len(feats) == 3 * 5 - 1
    slow = next(m.payload for m in windows if m.payload["approach_id"] == "pems_bay_1"
                and m.payload["window_start"].startswith("2026-09-28T15:00"))
    assert slow["average_speed_kmh"] == 10.0 and slow["camera_id"] == CAMERA_ID
    assert slow["traffic_state"] == "severe"            # 10 % of free flow, speed is the only indicator
    assert any("recorded Mon 2017-06-05 08:00" in e for e in slow["explanation"])
    f = next(m.payload for m in feats if m.payload["approach_id"] == "pems_bay_1")
    assert f["extra"]["speed_only"] == 1.0 and f["vehicle_count"] == 0
    assert w.status()["frames"] == 14


def test_live_values_repeat_the_latest_reading_with_the_current_time() -> None:
    now = datetime(2026, 9, 28, 15, 12, tzinfo=UTC)
    pub = CollectingPublisher()
    w = _worker(now, pub)
    w.publish_step(now - timedelta(minutes=12))          # 08:00 local: every station reported
    assert w.publish_live() == 5
    live = [m.payload for m in pub.messages if m.topic == Topics.FEATURES_LIVE]
    assert all(p["is_live"] and p["timestamp"].startswith("2026-09-28T15:12") for p in live)


def test_speed_only_features_ignore_unmeasured_indicators() -> None:
    clf = RuleBasedCongestionClassifier(rules_file="congestion_rules.yaml", free_flow_speeds={"x": 100.0})
    base = dict(approach_id="x", camera_id="c", window_start=datetime(2026, 1, 1, tzinfo=UTC), window_s=300.0,
                vehicle_count=0, count_by_class={}, pcu_count=0.0, avg_speed_kmh=20.0, speed_p85_kmh=None,
                density_veh_km=0.0, density_ratio=0.0, occupancy_pct=0.0, flow_veh_h=0.0, inflow=0, outflow=0,
                queue_length_m=0.0, stopped_vehicles=0, avg_gap_m=None, pedestrian_count=0, quality=1.0)
    mixed = clf.classify(TrafficFeatures(**base))
    only = clf.classify(TrafficFeatures(**base, extra={"speed_only": 1.0}))
    assert only.score == 1.0 and only.level.value == "severe"
    assert mixed.score < only.score                      # zeros for density etc. would dilute the verdict


def test_pipeline_file_extends_base_and_sets_mode_defaults(configs_dir) -> None:  # noqa: ANN001
    s = AppSettings(pipeline_file="pipeline_pems_bay.yaml")
    assert s.database_file == "traffic_pems_bay.db" and s.window_stale_s == 900
    assert AppSettings(pipeline_file="pipeline_pems_bay.yaml", window_stale_s=60).window_stale_s == 60
    p = load_pipeline_config(s)
    assert p.network_file == "network/pems_bay.yaml" and p.camera_files == []
    assert p.forecaster.params["model"].startswith("pems_bay_")
    assert p.tracker.type == load_pipeline_config(AppSettings()).tracker.type      # inherited from pipeline.yaml
    assert AppSettings().database_file == "traffic.db"


def test_replay_week_must_be_a_monday() -> None:
    with pytest.raises(ValueError, match="Monday"):
        _worker(datetime(2026, 9, 28, tzinfo=UTC), replay_week="2017-06-06")


def test_network_has_assumed_detours_and_a_closure_reroutes_onto_them() -> None:
    from traffic_intel.core.schemas import SimulationRequest
    from traffic_intel.simulation.static_assignment import StaticAssignmentSimulator

    net = _network()
    detours = [s for s in net.segments if s.id.startswith("detour_")]
    assert len(detours) == 2 * len(net.approaches)              # two halves (via a mid node) per segment
    assert all("assumed detour" in s.name and s.lanes == 1 for s in detours)
    sim = StaticAssignmentSimulator()
    sim.load_network(net)
    res = sim.what_if(SimulationRequest(closed_segment_ids=["seg_pems_bay_2"]), {})
    assert res.unserved_demand_veh_h == 0
    moved = {i.segment_id: i.scenario_flow_veh_h for i in res.impacts}
    assert moved["detour_pems_bay_2_a"] > 0 and moved["detour_pems_bay_2_b"] > 0
