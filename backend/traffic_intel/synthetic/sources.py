"""Synthetic camera config, video source and detector (``synthetic://<scenario>``).

    camera  = synthetic_camera("default")          # CameraConfig with calibration + zones
    source  = SyntheticVideoSource(camera)         # frames with correct timestamps (optionally rendered)
    detector= SyntheticDetector("default")         # noisy detections for each frame index

Scenes are generated once per scenario and cached.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from functools import lru_cache

import cv2
import numpy as np

from traffic_intel.computer_vision.interfaces import DETECTORS, VIDEO_SOURCES, Detector, VideoSource
from traffic_intel.core.config import CalibrationConfig, CameraConfig, ZoneConfig
from traffic_intel.core.enums import OBJECT_CLASSES, ObjectClass, ZoneType
from traffic_intel.core.types import Detections, Frame
from traffic_intel.synthetic.scene import SCENARIOS, SyntheticScene, generate_scene, project

_PRELOADED: dict[str, SyntheticScene] = {}


def register_scene(scenario: str, scene: SyntheticScene) -> None:
    """Make a pre-generated scene (e.g. loaded from the disk cache) available under ``scenario``."""
    _PRELOADED[scenario] = scene


@lru_cache(maxsize=16)
def get_scene(scenario: str = "default") -> SyntheticScene:
    return _PRELOADED.get(scenario) or generate_scene(SCENARIOS[scenario])


def scenario_from_uri(source: str) -> str:
    return source.split("://", 1)[1] if "://" in source else source or "default"


def synthetic_camera(scenario: str = "default", camera_id: str = "cam_synth", approach_id: str = "junction_a_north",
                     process_fps: float | None = None) -> CameraConfig:
    scene = get_scene(scenario)
    cfg = scene.cfg
    H = scene.H_ground_to_image
    W = cfg.lanes * cfg.lane_width_m

    def poly(x0: float, x1: float, a0: float, a1: float) -> list[tuple[float, float]]:
        pts = project(H, np.array([[x0, a1], [x1, a1], [x1, a0], [x0, a0]]))
        return [(round(float(x), 1), round(float(y), 1)) for x, y in pts]

    def line(x0: float, x1: float, a: float) -> list[tuple[float, float]]:
        pts = project(H, np.array([[x0, a], [x1, a]]))
        return [(round(float(x), 1), round(float(y), 1)) for x, y in pts]

    zones = [
        # ROI kept inside the calibrated, well-resolved area (0-60 m): far-field pixel jitter is
        # metres on the ground and makes presence counts flicker at the ROI edge.
        ZoneConfig(id="roi", type=ZoneType.ROI, points_px=poly(-3.0, W + 3.0, -3.0, 62.0)),
        *[
            ZoneConfig(id=f"lane_{i + 1}", type=ZoneType.LANE, lane_index=i + 1, direction=(0.0, -1.0),
                       points_px=poly(i * cfg.lane_width_m, (i + 1) * cfg.lane_width_m, -3.0, 62.0))
            for i in range(cfg.lanes)
        ],
        ZoneConfig(id="entry", type=ZoneType.ENTRY_LINE, points_px=line(-0.5, W + 0.5, 50.0)),
        ZoneConfig(id="stop_line", type=ZoneType.EXIT_LINE, points_px=line(-0.5, W + 0.5, 0.0)),
        ZoneConfig(id="queue", type=ZoneType.QUEUE_ZONE, points_px=poly(0.0, W, -2.0, 62.0)),
        ZoneConfig(id="loop", type=ZoneType.VIRTUAL_LOOP, points_px=poly(0.0, W, 3.0, 8.0)),
    ]
    return CameraConfig(
        id=camera_id,
        name=f"Synthetic ({scenario})",
        approach_id=approach_id,
        source=f"synthetic://{scenario}",
        source_type="synthetic",
        process_fps=process_fps or cfg.fps,
        replay_start=cfg.start.isoformat(),
        location=(77.5946, 12.9719),
        calibration=CalibrationConfig(
            image_points_px=scene.calibration_image_points,
            ground_points_m=scene.calibration_ground_points,
            origin_lonlat=(77.5946, 12.9721),
            bearing_deg=0.0,
            image_size=cfg.frame_size,
        ),
        zones=zones,
    )


_COLORS = {
    ObjectClass.CAR: (200, 160, 60), ObjectClass.MOTORCYCLE: (60, 180, 220), ObjectClass.BUS: (60, 60, 200),
    ObjectClass.TRUCK: (120, 90, 160), ObjectClass.BICYCLE: (80, 200, 80), ObjectClass.PEDESTRIAN: (230, 230, 230),
}


class SyntheticVideoSource(VideoSource):
    """Emits frames on the scene's timeline. ``render=True`` draws the road and
    objects (for annotated demo videos); otherwise one shared black image is used."""

    def __init__(self, camera: CameraConfig, render: bool = True, max_frames: int | None = None,
                 **_: object) -> None:
        self.camera_id = camera.id
        self.scene = get_scene(scenario_from_uri(camera.source))
        self.render = render
        self.max_frames = max_frames
        self._step = max(1, int(round(self.scene.cfg.fps / camera.process_fps)))
        w, h = self.scene.cfg.frame_size
        self._blank = np.zeros((h, w, 3), dtype=np.uint8)
        self._background = self._draw_background() if render else self._blank

    @property
    def fps(self) -> float:
        return self.scene.cfg.fps / self._step

    @property
    def is_live(self) -> bool:
        return False

    @property
    def frame_size(self) -> tuple[int, int]:
        return self.scene.cfg.frame_size

    def _draw_background(self) -> np.ndarray:
        img = np.full_like(self._blank, 70)
        cfg = self.scene.cfg
        H = self.scene.H_ground_to_image
        W = cfg.lanes * cfg.lane_width_m
        road = project(H, np.array([[0, 110], [W, 110], [W, -12], [0, -12]])).astype(np.int32)
        cv2.fillPoly(img, [road], (95, 95, 95))
        for i in range(1, cfg.lanes):
            x = i * cfg.lane_width_m
            for a in range(0, 110, 9):
                seg = project(H, np.array([[x, a], [x, a + 3]])).astype(np.int32)
                cv2.line(img, tuple(seg[0]), tuple(seg[1]), (230, 230, 230), 2)
        stop = project(H, np.array([[0, 0], [W, 0]])).astype(np.int32)
        cv2.line(img, tuple(stop[0]), tuple(stop[1]), (255, 255, 255), 4)
        return img

    def frames(self) -> Iterator[Frame]:
        for emitted, i in enumerate(range(0, self.scene.n_frames, self._step)):
            if self.max_frames is not None and emitted >= self.max_frames:
                return
            img = self._background
            if self.render:
                img = self._background.copy()
                for o in sorted(self.scene.truth[i], key=lambda o: -o.along):
                    if o.bbox is not None:
                        x1, y1, x2, y2 = (int(v) for v in o.bbox)
                        cv2.rectangle(img, (x1, y1), (x2, y2), _COLORS[o.cls], -1)
                        cv2.rectangle(img, (x1, y1), (x2, y2), (20, 20, 20), 1)
                if self.scene.signal_red[i]:
                    cv2.circle(img, (40, 40), 16, (0, 0, 255), -1)
                else:
                    cv2.circle(img, (40, 40), 16, (0, 200, 0), -1)
            yield Frame(self.camera_id, i, self.scene.timestamps[i], img)

    def close(self) -> None:
        pass


class SyntheticDetector(Detector):
    """Returns the scene's pre-generated noisy detections for ``Frame.index``."""

    def __init__(self, scenario: str = "default", **_: object) -> None:
        self.scene = get_scene(scenario)

    @property
    def classes(self) -> Sequence[ObjectClass]:
        return OBJECT_CLASSES

    def detect(self, frames: Sequence[Frame]) -> list[Detections]:
        return [self.scene.detections[f.index] if 0 <= f.index < self.scene.n_frames else Detections()
                for f in frames]


VIDEO_SOURCES.register("synthetic")(SyntheticVideoSource)
DETECTORS.register("synthetic")(SyntheticDetector)
