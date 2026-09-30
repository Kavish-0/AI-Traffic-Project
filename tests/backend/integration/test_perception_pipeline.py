"""End-to-end: committed synthetic dataset -> full pipeline -> TrafficState (no GPU, no video, no DB)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import FIXTURES

from traffic_intel.computer_vision.detectors.precomputed import PrecomputedDetector, load_detection_table
from traffic_intel.core.config import get_settings, load_camera, load_network, load_pipeline_config
from traffic_intel.core.enums import CongestionLevel
from traffic_intel.core.schemas import TrafficState
from traffic_intel.pipeline.perception import PerceptionPipeline
from traffic_intel.pipeline.runner import run
from traffic_intel.synthetic.sources import SyntheticVideoSource

SMALL = FIXTURES / "synthetic_small"


def test_committed_fixture_is_reproducible() -> None:
    """The dataset in tests/fixtures is exactly what the seeded generator produces."""
    from traffic_intel.synthetic.scene import generate_scene

    scene = generate_scene("small")
    table = load_detection_table(SMALL / "detections.csv")
    for i in (0, 57, scene.n_frames - 1):
        np.testing.assert_allclose(table.get(i).xyxy, scene.detections[i].xyxy, atol=0.01)  # type: ignore[union-attr]
    truth = pd.read_csv(SMALL / "truth.csv")
    assert truth["frame"].max() == scene.n_frames - 1


def test_fixture_through_full_pipeline(tmp_path: Path) -> None:
    st = get_settings()
    pc = load_pipeline_config(st)
    net = load_network(st.config_dir / pc.network_file)
    cam = load_camera(SMALL / "camera.yaml")
    src = SyntheticVideoSource(cam, render=False)
    pipe = PerceptionPipeline.build(cam, pc, net, src.frame_size, src.fps,
                                    detector=PrecomputedDetector(path=str(SMALL / "detections.csv")))
    report = run(pipe, src, tmp_path, echo=False, write_live=True)

    # 20 s of data inside one 60 s window -> a single partial (20 s) window, flushed at the end
    assert len(report.states) == 1 and report.states[0].window_s == pytest.approx(20.0)
    s = report.states[0]
    assert isinstance(s, TrafficState) and not s.is_live
    assert (s.junction_id, s.road_id, s.approach_id) == ("junction_a", "road_a_sb", "junction_a_north")
    assert s.vehicle_count == s.car_count + s.motorcycle_count + s.bus_count + s.truck_count + s.bicycle_count
    assert s.vehicle_count > 5 and s.average_speed_kmh is not None and 0 < s.average_speed_kmh < 60
    assert s.traffic_state in set(CongestionLevel) and s.explanation
    assert s.quality == pytest.approx(1.0)

    lines = (tmp_path / "traffic_states.jsonl").read_text().splitlines()
    assert TrafficState.model_validate_json(lines[0]) == s
    live = (tmp_path / "live_states.jsonl").read_text().splitlines()
    assert len(live) == 20 and all(json.loads(x)["is_live"] for x in live)
    tracks = (tmp_path / "tracks.jsonl").read_text().splitlines()
    assert len(tracks) == report.tracks > 0
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["calibration_reprojection_error_m"] < 0.3 and summary["frames"] == 200
