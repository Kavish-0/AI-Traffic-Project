"""Per-camera zone geometry: ROI, lanes, count lines, queue zone, virtual loop.

Zones are configured in pixels of ``CameraConfig.zone_reference_size()`` and
rescaled here to the processing resolution, so changing ``resolution`` never
requires re-drawing zones.

Two regions matter:
* ``roi``                - configured analysis area; presence/counts use it.
* ``measurement_region`` - ROI intersected with the calibrated area of the
  homography; everything metric (speed, density, queue, gaps) uses it, and
  ``roi_length_m`` is measured on it, so counts-per-length stay consistent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from shapely.geometry import Polygon

from traffic_intel.computer_vision.calibration import GroundProjector
from traffic_intel.computer_vision.geometry import (
    as_points,
    points_in_polygon,
    scale_points,
    segments_intersect,
    unit,
)
from traffic_intel.core.config import CameraConfig
from traffic_intel.core.enums import ZoneType
from traffic_intel.core.types import BoolArray, FloatArray, IntArray

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class LineZone:
    id: str
    kind: ZoneType          # ENTRY_LINE | EXIT_LINE
    a: FloatArray           # image px
    b: FloatArray


@dataclass(frozen=True)
class LaneZone:
    id: str
    index: int
    polygon: FloatArray


class CameraGeometry:
    def __init__(
        self, camera: CameraConfig, frame_size: tuple[int, int], projector: GroundProjector
    ) -> None:
        self.camera = camera
        self.frame_size = frame_size
        self.projector = projector
        w, h = frame_size
        ref = camera.zone_reference_size() or frame_size

        full_frame = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        self.roi: FloatArray = full_frame
        self.lanes: list[LaneZone] = []
        self.lines: list[LineZone] = []
        self.queue_zones: list[FloatArray] = []
        self.loops: list[FloatArray] = []
        lane_dirs: list[FloatArray] = []

        for z in camera.zones:
            pts = scale_points(as_points(z.points_px), ref, frame_size)
            if z.type is ZoneType.ROI:
                self.roi = pts
            elif z.type is ZoneType.LANE:
                self.lanes.append(LaneZone(z.id, z.lane_index or len(self.lanes) + 1, pts))
                if z.direction is not None:
                    lane_dirs.append(unit(z.direction))
            elif z.type in (ZoneType.ENTRY_LINE, ZoneType.EXIT_LINE):
                if len(pts) != 2:
                    raise ValueError(f"zone {z.id}: lines need exactly 2 points")
                self.lines.append(LineZone(z.id, z.type, pts[0], pts[1]))
            elif z.type is ZoneType.QUEUE_ZONE:
                self.queue_zones.append(pts)
            elif z.type is ZoneType.VIRTUAL_LOOP:
                self.loops.append(pts)

        self.measurement_region = self._measurement_region()
        self.travel_direction = self._travel_direction(lane_dirs)          # ground unit vector
        self.stop_point = self._stop_point()                               # ground metres
        self.lane_count = camera.lanes or max(len(self.lanes), 1)
        self.roi_length_m = camera.roi_length_m or self._measure_roi_length()
        self.anchor_reference = self._anchor_reference()
        log.info(
            "camera %s: roi_length=%.1f m, lanes=%d, travel=(%.2f, %.2f), calib error=%.2f m",
            camera.id, self.roi_length_m, self.lane_count, *self.travel_direction,
            projector.reprojection_error_m,
        )

    # ------------------------------------------------------------------ #
    # construction helpers
    # ------------------------------------------------------------------ #
    def _measurement_region(self) -> FloatArray:
        region = getattr(self.projector, "_region", None)
        if region is None:
            return self.roi
        inter = Polygon(self.roi).buffer(0).intersection(Polygon(region).buffer(0))
        if inter.is_empty:
            log.warning("camera %s: ROI does not overlap the calibrated area", self.camera.id)
            return self.roi
        if inter.geom_type != "Polygon":
            inter = max(getattr(inter, "geoms", [inter]), key=lambda g: g.area)
        return np.asarray(inter.exterior.coords[:-1], dtype=np.float32)

    def _line_mid_ground(self, kind: ZoneType) -> FloatArray | None:
        for line in self.lines:
            if line.kind is kind:
                mid = ((line.a + line.b) / 2.0).reshape(1, 2)
                g = self.projector.image_to_ground(mid)[0]
                if np.all(np.isfinite(g)):
                    return np.asarray(g, dtype=np.float32)
        return None

    def _travel_direction(self, lane_dirs: list[FloatArray]) -> FloatArray:
        if self.camera.travel_direction is not None:
            return unit(self.camera.travel_direction)
        if lane_dirs:
            return unit(np.mean(lane_dirs, axis=0))
        entry, exit_ = self._line_mid_ground(ZoneType.ENTRY_LINE), self._line_mid_ground(ZoneType.EXIT_LINE)
        if entry is not None and exit_ is not None:
            return unit(exit_ - entry)
        raise ValueError(
            f"camera {self.camera.id}: cannot infer travel direction - set travel_direction, "
            "lane directions, or both an entry_line and an exit_line"
        )

    def _stop_point(self) -> FloatArray:
        exit_mid = self._line_mid_ground(ZoneType.EXIT_LINE)
        if exit_mid is not None:
            return exit_mid
        # No stop line: use the most downstream point of the measurement region.
        g = self.projector.image_to_ground(self.measurement_region)
        g = g[np.all(np.isfinite(g), axis=1)]
        return np.asarray(g[np.argmax(g @ self.travel_direction)], dtype=np.float32)

    def _anchor_reference(self) -> str:
        """Which part of a vehicle the box anchor (bottom-centre) marks.

        The bottom edge of a box is the part of the vehicle nearest the camera.
        If traffic moves *down* the image (towards the camera) that is the
        front bumper; if it moves up the image (away), the rear bumper.
        """
        to_img = getattr(self.projector, "ground_to_image", None)
        if to_img is None:
            return "center"
        p = np.stack([self.stop_point - self.travel_direction * 10.0, self.stop_point]).astype(np.float32)
        img = to_img(p)          # upstream point, stop point
        if not np.all(np.isfinite(img)):
            return "center"
        return "front" if img[1, 1] > img[0, 1] else "rear"

    def _measure_roi_length(self) -> float:
        g = self.projector.image_to_ground(self.measurement_region)
        g = g[np.all(np.isfinite(g), axis=1)]
        if len(g) < 3:
            raise ValueError(f"camera {self.camera.id}: ROI cannot be projected; set roi_length_m")
        along = g @ self.travel_direction
        length = float(along.max() - along.min())
        if length > 400:
            log.warning("camera %s: ROI is %.0f m long - check calibration or set roi_length_m",
                        self.camera.id, length)
        return length

    # ------------------------------------------------------------------ #
    # queries (vectorised)
    # ------------------------------------------------------------------ #
    def in_roi(self, points_px: FloatArray) -> BoolArray:
        return points_in_polygon(points_px, self.roi)

    def in_measurement_region(self, points_px: FloatArray) -> BoolArray:
        return points_in_polygon(points_px, self.measurement_region)

    def in_queue_zone(self, points_px: FloatArray) -> BoolArray:
        if not self.queue_zones:
            return self.in_measurement_region(points_px)
        mask = np.zeros(len(points_px), dtype=bool)
        for poly in self.queue_zones:
            mask |= points_in_polygon(points_px, poly)
        return mask

    def lane_of(self, points_px: FloatArray) -> IntArray:
        out = np.full(len(points_px), -1, dtype=np.int64)
        for lane in self.lanes:
            out[(out == -1) & points_in_polygon(points_px, lane.polygon)] = lane.index
        return out

    def distance_upstream(self, ground_xy: FloatArray) -> FloatArray:
        """Metres upstream of the stop point along the travel axis (>0 = before the stop line)."""
        return np.asarray((self.stop_point - ground_xy) @ self.travel_direction, dtype=np.float32)

    def lateral_offset(self, ground_xy: FloatArray) -> FloatArray:
        """Signed metres across the travel axis (right of travel = positive)."""
        d = ground_xy - self.stop_point
        u = self.travel_direction
        return np.asarray(d[:, 0] * u[1] - d[:, 1] * u[0], dtype=np.float32)

    def loop_occupied(self, boxes_xyxy: FloatArray) -> bool:
        """Any vehicle footprint (sampled along the box bottom edge) inside any loop."""
        if not self.loops or len(boxes_xyxy) == 0:
            return False
        xs = np.linspace(0.0, 1.0, 5, dtype=np.float32)
        x = boxes_xyxy[:, [0]] + (boxes_xyxy[:, [2]] - boxes_xyxy[:, [0]]) * xs[None, :]
        y = np.repeat(boxes_xyxy[:, [3]], len(xs), axis=1)
        pts = np.stack([x.ravel(), y.ravel()], axis=1)
        return any(bool(points_in_polygon(pts, poly).any()) for poly in self.loops)

    def line_crossings(
        self, prev_px: FloatArray, cur_px: FloatArray
    ) -> list[tuple[LineZone, BoolArray]]:
        """For each line, a mask of which movements prev->cur crossed it."""
        return [(line, segments_intersect(prev_px, cur_px, line.a, line.b)) for line in self.lines]

    @property
    def has_loop(self) -> bool:
        return bool(self.loops)

    def has_line(self, kind: ZoneType) -> bool:
        return any(line.kind is kind for line in self.lines)
