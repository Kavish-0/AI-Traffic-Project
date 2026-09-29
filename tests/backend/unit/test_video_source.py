"""Video input: file decoding, fps sampling, resizing, timestamps, preprocessing."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import pytest
from conftest import simple_camera

from traffic_intel.computer_vision.preprocessing import ClahePreprocessor, RoiMaskPreprocessor
from traffic_intel.computer_vision.sources import open_video_source
from traffic_intel.computer_vision.sources.opencv_source import OpenCVVideoSource, classify_source
from traffic_intel.core.types import Frame

START = datetime(2026, 9, 1, 8, 0, tzinfo=UTC)


@pytest.fixture(scope="module")
def video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """3 s of 30 fps 320x240 video; each frame's index is encoded in its brightness."""
    path = tmp_path_factory.mktemp("vid") / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (320, 240))
    for i in range(90):
        writer.write(np.full((240, 320, 3), i * 2, np.uint8))
    writer.release()
    return path


def test_classify_source() -> None:
    assert classify_source("webcam:0") == "webcam"
    assert classify_source("0") == "webcam"
    assert classify_source("rtsp://cam/stream") == "stream"
    assert classify_source("data/raw/a.mp4") == "file"
    assert classify_source("data/raw/a.mp4", "stream") == "stream"


def test_file_sampling_timestamps_and_resize(video: Path) -> None:
    src = OpenCVVideoSource("c", str(video), process_fps=10, resolution=(160, 120), replay_start=START)
    frames = list(src.frames())
    src.close()
    assert src.source_fps == pytest.approx(30.0) and src.fps == pytest.approx(10.0) and not src.is_live
    assert len(frames) == 30
    assert [f.index for f in frames[:4]] == [0, 3, 6, 9]
    assert frames[1].timestamp == START + timedelta(seconds=0.1)
    assert frames[0].image.shape == (120, 160, 3) and src.frame_size == (160, 120)
    assert abs(int(frames[2].image.mean()) - 12) <= 3          # frame 6 -> brightness 12
    assert src.stats.skipped_by_sampling == 60


def test_uneven_rate_ratio_stays_on_time_grid(video: Path) -> None:
    src = OpenCVVideoSource("c", str(video), process_fps=7, replay_start=START)
    ts = [(f.timestamp - START).total_seconds() for f in src.frames()]
    slots = [int(t * 7 + 1e-6) for t in ts]
    assert slots == sorted(set(slots)) and len(ts) == 21


def test_max_frames_and_loop(video: Path) -> None:
    src = OpenCVVideoSource("c", str(video), process_fps=30, replay_start=START, loop=True, max_frames=120)
    frames = list(src.frames())
    assert len(frames) == 120
    assert frames[95].timestamp == START + timedelta(seconds=95 / 30)       # time continues across loops


def test_from_camera_config(video: Path) -> None:
    cam = simple_camera(source=str(video), replay_start="2026-09-01T13:30:00+05:30", process_fps=5,
                        resolution=(160, 120))
    src = open_video_source(cam)
    first = next(iter(src.frames()))
    assert first.timestamp == datetime(2026, 9, 1, 8, 0, tzinfo=UTC)
    assert src.fps == 5 and src.frame_size == (160, 120)


def test_missing_file_is_a_clear_error() -> None:
    with pytest.raises(FileNotFoundError):
        OpenCVVideoSource("c", "does/not/exist.mp4", process_fps=10, replay_start=START)


def test_roi_mask_blacks_out_outside_roi() -> None:
    cam = simple_camera()
    pre = RoiMaskPreprocessor(cam, padding_px=0)
    out = pre.process(Frame("c", 0, START, np.full((800, 400, 3), 200, np.uint8)))
    assert out.image[400, 135].tolist() == [200, 200, 200]      # inside the road ROI
    assert out.image[400, 5].tolist() == [0, 0, 0]              # outside


def test_clahe_only_on_dark_frames() -> None:
    pre = ClahePreprocessor(only_if_mean_below=90)
    bright = np.full((50, 50, 3), 200, np.uint8)
    assert pre.process(Frame("c", 0, START, bright.copy())).image.tolist() == bright.tolist()
    rng = np.random.default_rng(0)
    dark = rng.integers(10, 40, (50, 50, 3), dtype=np.uint8)
    assert pre.process(Frame("c", 0, START, dark.copy())).image.std() > dark.std()
