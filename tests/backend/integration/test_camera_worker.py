"""Per-camera worker: publishes only 1 s live values + 60 s windows; frame-level data goes to
Parquet / annotated video only. Runs on the committed synthetic fixture (no GPU, no video file)."""

from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
from conftest import FIXTURES

from traffic_intel.computer_vision.detectors.precomputed import PrecomputedDetector, load_detection_table
from traffic_intel.core.config import PROJECT_ROOT, get_settings, load_camera, load_network, load_pipeline_config
from traffic_intel.core.schemas import BusMessage, TrafficFeatures, TrafficState
from traffic_intel.pipeline.bus import InMemoryEventBus, Topics
from traffic_intel.pipeline.camera_worker import CameraWorker
from traffic_intel.pipeline.publish import WORKER_TOPICS, BusPublisher, CollectingPublisher, WorkerPublisher

SMALL = FIXTURES / "synthetic_small"
DETECTIONS = str(SMALL / "detections.csv")


def _worker(tmp_path: Path, *, calibrated: bool = True, publisher=None, annotate: bool = True) -> CameraWorker:  # noqa: ANN001
    st = get_settings()
    pc = load_pipeline_config(st)
    pc = pc.model_copy(update={"outputs": pc.outputs.model_copy(update={
        "runs_dir": str(tmp_path / "runs"), "parquet_dir": str(tmp_path / "parquet")})})
    net = load_network(st.config_dir / pc.network_file)
    cam = load_camera(SMALL / "camera.yaml")
    if not calibrated:
        cam = cam.model_copy(update={"calibration": None, "zones_image_size": (1280, 720)})
    return CameraWorker.from_config(cam, pc, net, publisher=publisher, annotate_video=annotate,
                                    detector=PrecomputedDetector(path=DETECTIONS), run_id="test-run")


def test_worker_publishes_only_live_values_and_windows(tmp_path: Path) -> None:
    pub = CollectingPublisher()
    report = _worker(tmp_path, publisher=pub).run()
    topics = [m.topic for m in pub.messages]
    assert set(topics) <= WORKER_TOPICS
    live = [m for m in pub.messages if m.topic == Topics.FEATURES_LIVE]
    assert len(live) == 20                                # 20 s of footage, one live value per second
    assert all(TrafficState.model_validate(m.payload).is_live for m in live)
    windows = [TrafficFeatures.model_validate(m.payload) for m in pub.messages if m.topic == Topics.FEATURES_WINDOW]
    states = [TrafficState.model_validate(m.payload) for m in pub.messages if m.topic == Topics.TRAFFIC_STATE]
    assert len(windows) == len(states) == 1 and not states[0].is_live
    assert report.published == {Topics.FEATURES_LIVE: 20, Topics.FEATURES_WINDOW: 1, Topics.TRAFFIC_STATE: 1,
                                Topics.ANOMALY_SIGNAL: len(report.signals)}   # normal traffic: usually 0
    # nothing frame-level in any payload
    assert all("xyxy" not in str(m.payload) and "bbox" not in m.payload for m in pub.messages)


def test_worker_writes_parquet_and_annotated_video(tmp_path: Path) -> None:
    worker = _worker(tmp_path)
    report = worker.run()
    root = tmp_path / "parquet" / worker.camera.id / "2026-09-01" / "test-run"
    det = pd.read_parquet(root / "detections-00000.parquet")
    trk = pd.read_parquet(root / "tracks-00000.parquet")
    summ = pd.read_parquet(root / "track_summaries-00000.parquet")
    table = load_detection_table(DETECTIONS)
    assert len(det) == sum(len(d) for d in table.values())
    assert {"class_name", "confidence", "x1", "y2", "timestamp", "frame_index"} <= set(det.columns)
    assert {"track_id", "speed_kmh", "ground_x_m", "ground_y_m", "anchor_x"} <= set(trk.columns)
    assert trk.track_id.nunique() > 5 and trk.speed_kmh.notna().any()
    assert len(summ) == report.tracks and "path_ground_m" in summ.columns
    video = cv2.VideoCapture(str(report.annotated_video))
    assert int(video.get(cv2.CAP_PROP_FRAME_COUNT)) == report.frames == 200
    summary = (report.out_dir / "summary.json").read_text()  # type: ignore[operator]
    assert '"perception_only": false' in summary


def test_perception_only_mode_without_calibration(tmp_path: Path) -> None:
    pub = CollectingPublisher()
    worker = _worker(tmp_path, calibrated=False, publisher=pub, annotate=False)
    assert worker.pipe.perception_only
    report = worker.run()
    assert pub.messages == [] and report.states == []
    trk = pd.read_parquet(next((tmp_path / "parquet").rglob("tracks-*.parquet")))
    assert trk.ground_x_m.isna().all() and trk.speed_kmh.isna().all() and trk.track_id.nunique() > 5


def test_publisher_whitelist() -> None:
    wp = WorkerPublisher(CollectingPublisher(), source="t")
    with pytest.raises(ValueError, match="may not publish"):
        wp.publish(Topics.TRACK_SUMMARY, TrafficState.model_construct())
    with pytest.raises(ValueError):
        wp.publish("perception.detections", TrafficState.model_construct())


async def test_worker_thread_feeds_async_bus(tmp_path: Path) -> None:
    bus = InMemoryEventBus()
    loop = asyncio.get_running_loop()
    got: list[BusMessage] = []

    async def consume() -> None:
        async for m in bus.subscribe(*WORKER_TOPICS):
            got.append(m)
            if m.topic == Topics.TRAFFIC_STATE:
                return

    task = asyncio.create_task(consume())
    await asyncio.sleep(0)
    worker = _worker(tmp_path, publisher=BusPublisher(bus, loop), annotate=False)
    worker.start()
    await asyncio.wait_for(task, 60)
    worker.stop()
    assert {Topics.FEATURES_LIVE, Topics.FEATURES_WINDOW, Topics.TRAFFIC_STATE} <= {m.topic for m in got}


def test_replay_cli_end_to_end(tmp_path: Path) -> None:
    """The documented command, on the fixture (no detector, no GPU)."""
    out = tmp_path / "run"
    cmd = [sys.executable, "-m", "traffic_intel.pipeline", "replay", str(SMALL / "camera.yaml"),
           "--detections", DETECTIONS, "--annotate", "--no-record", "--out", str(out)]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT / "backend", timeout=300)
    assert res.returncode == 0, res.stderr[-2000:]
    assert "published" in res.stdout and (out / "annotated.mp4").exists() and (out / "summary.json").exists()


def test_api_process_does_not_import_torch() -> None:
    code = ("import sys, traffic_intel.api.main as m; m.create_app(); "
            "print(sorted(k for k in ('torch', 'ultralytics', 'cv2') if k in sys.modules))")
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120,
                         cwd=PROJECT_ROOT / "backend")
    assert res.returncode == 0, res.stderr
    assert "torch" not in res.stdout and "ultralytics" not in res.stdout


PUBLIC_VIDEO = PROJECT_ROOT / "data" / "raw" / "public" / "roboflow_vehicles.mp4"


@pytest.mark.gpu
@pytest.mark.cv
@pytest.mark.skipif(not PUBLIC_VIDEO.exists(), reason="public footage not downloaded")
def test_public_footage_on_real_detector(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("ultralytics")
    st = get_settings()
    pc = load_pipeline_config(st)
    pc = pc.model_copy(update={"outputs": pc.outputs.model_copy(update={
        "runs_dir": str(tmp_path / "runs"), "parquet_dir": str(tmp_path / "parquet")})})
    net = load_network(st.config_dir / pc.network_file)
    cam = load_camera(st.config_dir / "cameras" / "public_highway.yaml")
    worker = CameraWorker.from_config(cam, pc, net, annotate_video=False)
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert worker.status()["device"].startswith(expected)
    report = worker.run(max_seconds=4)
    trk = pd.read_parquet(next((tmp_path / "parquet").rglob("tracks-*.parquet")))
    assert report.frames >= 100 and trk.track_id.nunique() >= 2
    speeds = trk.groupby("track_id").speed_kmh.median().dropna()
    assert ((speeds > 40) & (speeds < 180)).mean() > 0.6     # plausible motorway speeds (approx. calibration)
    assert np.isfinite(trk.ground_y_m.dropna()).all()
