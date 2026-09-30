"""AI City Challenge multi-camera tracking (CityFlowV2, Track 1) adapter.

Layout (under the dataset root):

    <split>/<scenario>/<camera>/vdo.avi | vdo.mp4
    <split>/<scenario>/<camera>/gt/gt.txt          MOTChallenge: frame,id,left,top,width,height,1,-1,-1,-1
    <split>/<scenario>/<camera>/calibration.txt    "Homography matrix: h11 h12 h13;h21 h22 h23;h31 h32 h33"
    cam_timestamp/<scenario>.txt                   "<camera> <offset_s>" per line (optional)

Sequences are cameras (``<scenario>_<camera>``); frames are 1-based in gt.txt. Ground truth has
vehicle identities but no classes -> mapped via ``classes.class_map`` (default vehicle -> car).

``calibration_config()`` turns the camera's GPS homography into the project's
``CalibrationConfig`` (pixel <-> local metres east/north of a GPS origin, bearing 0), so the
existing homography projector, speed estimation and map work on real AI City cameras. The
direction of the matrix (GPS->image or image->GPS) and the GPS component order are parameters
(``homography_direction``, ``gps_order``) because they must be checked per release.
"""

from __future__ import annotations

import glob
import math
import re
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from traffic_intel.core.config import CalibrationConfig
from traffic_intel.datasets.adapters.base import (
    DATASET_ADAPTERS,
    AnnotatedFrame,
    DetectionTrackingAdapter,
    SequenceInfo,
    make_tracks,
)

EARTH_R = 6_371_000.0


@DATASET_ADAPTERS.register("aicity_mtmc")
class AICityAdapter(DetectionTrackingAdapter):
    def _cameras(self) -> dict[str, Path]:
        out = {}
        for gt in sorted(glob.glob(str(self.root / "*" / "S*" / "c*" / "gt" / "gt.txt"))):
            cam_dir = Path(gt).parent.parent
            out[f"{cam_dir.parent.name}_{cam_dir.name}"] = cam_dir
        return out

    def _offsets(self, scenario: str) -> dict[str, float]:
        f = self.root / "cam_timestamp" / f"{scenario}.txt"
        if not f.exists():
            return {}
        pairs = (ln.split() for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip())
        return {p[0]: float(p[1]) for p in pairs if len(p) >= 2}

    def _gt(self, cam_dir: Path) -> dict[int, list[tuple[int, tuple[float, float, float, float]]]]:
        rows: dict[int, list[tuple[int, tuple[float, float, float, float]]]] = defaultdict(list)
        for ln in (cam_dir / "gt" / "gt.txt").read_text(encoding="utf-8").splitlines():
            parts = [p for p in re.split(r"[,\s]+", ln.strip()) if p]
            if len(parts) < 6:
                continue
            f, tid, x, y, w, h = int(float(parts[0])), int(float(parts[1])), *map(float, parts[2:6])
            rows[f].append((tid, (x, y, x + w, y + h)))
        return rows

    def sequences(self) -> list[SequenceInfo]:
        fps = float(self.params.get("fps") or getattr(self.entry, "fps", 10) or 10)
        out = []
        for seq, cam_dir in self._cameras().items():
            scenario, cam = cam_dir.parent.name, cam_dir.name
            video = next(iter(sorted(cam_dir.glob("vdo.*"))), None)
            gt = self._gt(cam_dir)
            out.append(SequenceInfo(seq, fps, None, max(gt, default=0), video, {
                "split": cam_dir.parent.parent.name, "scenario": scenario, "camera": cam,
                "time_offset_s": self._offsets(scenario).get(cam, 0.0),
                "has_calibration": (cam_dir / "calibration.txt").exists()}))
        return out

    def frames(self, sequence_id: str) -> Iterator[AnnotatedFrame]:
        cam_dir = self._cameras()[sequence_id]
        fps = float(self.params.get("fps") or getattr(self.entry, "fps", 10) or 10)
        default_cls = self.classes("vehicle")
        gt = self._gt(cam_dir)
        for f in sorted(gt):
            ids = [r[0] for r in gt[f]]
            boxes = [r[1] for r in gt[f]]
            unmapped = 0 if default_cls is not None else len(ids)
            classes = [default_cls] * len(ids) if default_cls is not None else []
            if default_cls is None:
                ids, boxes = [], []
            k = f - 1
            yield AnnotatedFrame(sequence_id, k, k / fps, make_tracks(ids, boxes, classes),
                                 unmapped=unmapped)

    # ------------------------------------------------------------------ calibration
    def homography(self, sequence_id: str) -> np.ndarray:
        return parse_homography((self._cameras()[sequence_id] / "calibration.txt").read_text(encoding="utf-8"))

    def calibration_config(self, sequence_id: str, image_size: tuple[int, int],
                           grid: tuple[int, int] = (3, 2), region: tuple[float, float] = (0.45, 0.95)
                           ) -> CalibrationConfig:
        """Project CalibrationConfig from the camera's GPS homography.

        Picks a grid of pixels in the lower image (``region`` = fraction of the height, where the
        road is), maps them to GPS, and expresses them in metres east/north of their centroid."""
        H = self.homography(sequence_id)
        direction = self.params.get("homography_direction", "gps_to_image")
        img_to_gps = np.linalg.inv(H) if direction == "gps_to_image" else H
        w, h = image_size
        us = np.linspace(0.1 * w, 0.9 * w, grid[0])
        vs = np.linspace(region[0] * h, region[1] * h, grid[1])
        px = np.array([(u, v) for v in vs for u in us], dtype=np.float64)
        g = apply_h(img_to_gps, px)
        lat, lon = (g[:, 0], g[:, 1]) if self.params.get("gps_order", "lat_lon") == "lat_lon" else (g[:, 1], g[:, 0])
        lat0, lon0 = float(lat.mean()), float(lon.mean())
        east = np.radians(lon - lon0) * EARTH_R * math.cos(math.radians(lat0))
        north = np.radians(lat - lat0) * EARTH_R
        return CalibrationConfig(
            image_points_px=[(round(float(u), 2), round(float(v), 2)) for u, v in px],
            ground_points_m=[(round(float(x), 3), round(float(y), 3)) for x, y in zip(east, north, strict=True)],
            origin_lonlat=(lon0, lat0), bearing_deg=0.0, image_size=image_size)


def parse_homography(text: str) -> np.ndarray:
    m = re.search(r"Homography matrix:\s*([^\n]+)", text)
    if not m:
        raise ValueError("calibration.txt: no 'Homography matrix:' line")
    rows = [[float(x) for x in r.split()] for r in m.group(1).strip().split(";") if r.strip()]
    H = np.array(rows, dtype=np.float64)
    if H.shape != (3, 3):
        raise ValueError(f"calibration.txt: homography must be 3x3, got {H.shape}")
    return H


def apply_h(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    p = np.hstack([pts, np.ones((len(pts), 1))]) @ H.T
    return np.asarray(p[:, :2] / p[:, 2:3])
