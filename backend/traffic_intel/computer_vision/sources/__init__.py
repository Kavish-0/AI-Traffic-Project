"""VideoSource implementations and the factory that picks one per camera."""

from __future__ import annotations

from typing import Any

from traffic_intel.computer_vision.interfaces import VIDEO_SOURCES, VideoSource
from traffic_intel.core.config import CameraConfig


def open_video_source(camera: CameraConfig, **overrides: Any) -> VideoSource:
    """``synthetic://...`` (or source_type=synthetic) -> synthetic scene; anything else -> OpenCV."""
    kind = "synthetic" if (
        camera.source_type == "synthetic" or camera.source.startswith("synthetic://")
    ) else "opencv"
    return VIDEO_SOURCES.get(kind)(camera=camera, **overrides)
