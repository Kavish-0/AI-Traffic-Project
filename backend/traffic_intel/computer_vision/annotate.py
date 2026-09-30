"""Debug/validation overlays: zones, tracks with speed and trajectory tails, HUD."""

from __future__ import annotations

import cv2
import numpy as np

from traffic_intel.computer_vision.zones import CameraGeometry
from traffic_intel.core.enums import CongestionLevel, ObjectClass
from traffic_intel.core.schemas import TrafficState
from traffic_intel.tracking.trajectory import TrackedObject

CLASS_BGR = {
    ObjectClass.CAR.value: (255, 200, 0), ObjectClass.MOTORCYCLE.value: (0, 220, 255),
    ObjectClass.BUS.value: (60, 60, 255), ObjectClass.TRUCK.value: (200, 120, 255),
    ObjectClass.BICYCLE.value: (80, 255, 80), ObjectClass.PEDESTRIAN.value: (255, 255, 255),
}
LEVEL_BGR = {
    CongestionLevel.FREE: (94, 197, 34), CongestionLevel.MODERATE: (8, 179, 234),
    CongestionLevel.HIGH: (22, 115, 249), CongestionLevel.SEVERE: (38, 38, 220),
}


def _poly(img: np.ndarray, pts: np.ndarray, color: tuple[int, int, int], thickness: int = 2) -> None:
    cv2.polylines(img, [np.asarray(pts).round().astype(np.int32)], True, color, thickness, cv2.LINE_AA)


def draw_zones(img: np.ndarray, geo: CameraGeometry) -> None:
    _poly(img, geo.roi, (255, 255, 0), 1)
    _poly(img, geo.measurement_region, (180, 180, 0), 1)
    for lane in geo.lanes:
        _poly(img, lane.polygon, (120, 120, 120), 1)
    for q in geo.queue_zones:
        _poly(img, q, (0, 140, 255), 1)
    for loop in geo.loops:
        _poly(img, loop, (255, 0, 255), 2)
    for line in geo.lines:
        color = (0, 255, 0) if line.kind.value == "entry_line" else (0, 0, 255)
        cv2.line(img, tuple(line.a.round().astype(int)), tuple(line.b.round().astype(int)), color, 2, cv2.LINE_AA)


def draw_ground_grid(img: np.ndarray, geo: CameraGeometry, step_m: float = 5.0, extent_m: float = 80.0) -> None:
    """Project a metric grid onto the image - the quickest visual calibration check."""
    proj = geo.projector
    to_img = getattr(proj, "ground_to_image", None)
    if to_img is None:
        return
    origin = geo.stop_point
    u = geo.travel_direction
    n = np.array([u[1], -u[0]])
    for a in np.arange(0, extent_m + 1e-6, step_m):
        pts = np.array([origin - u * a + n * s for s in np.linspace(-8, 8, 20)])
        p = to_img(pts.astype(np.float32))
        p = p[np.all(np.isfinite(p), axis=1)]
        if len(p) > 1:
            cv2.polylines(img, [p.round().astype(np.int32)], False, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(img, f"{a:.0f}m", tuple(p[-1].round().astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (0, 255, 255), 1, cv2.LINE_AA)


def draw_tracks(img: np.ndarray, objects: list[TrackedObject]) -> None:
    for o in objects:
        color = CLASS_BGR.get(o.class_name, (200, 200, 200))
        x1, y1, x2, y2 = (int(v) for v in o.bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        tail = np.array([[x, y] for _, x, y in o.trajectory], dtype=np.int32)
        if len(tail) > 1:
            cv2.polylines(img, [tail], False, color, 1, cv2.LINE_AA)
        speed = "--" if o.speed_kmh is None else f"{o.speed_kmh:.0f}"
        label = f"#{o.track_id} {o.class_name} {speed}km/h"
        if o.stopped_s >= 2:
            label += f" STOP {o.stopped_s:.0f}s"
        cv2.putText(img, label, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def draw_hud(img: np.ndarray, live: TrafficState | None, window: TrafficState | None) -> None:
    lines: list[tuple[str, tuple[int, int, int]]] = []
    for label, s in (("LIVE", live), ("LAST WINDOW", window)):
        if s is None:
            continue
        speed = "--" if s.average_speed_kmh is None else f"{s.average_speed_kmh:.0f}"
        lines.append((f"{label} {s.window_s:.0f}s: {s.traffic_state.value.upper()} ({s.congestion_score:.2f})",
                      LEVEL_BGR[s.traffic_state]))
        lines.append((f"  veh {s.vehicle_count}  speed {speed} km/h  dens {s.density_pct:.0f}%  "
                      f"occ {s.occupancy_pct:.0f}%  queue {s.queue_length_m:.0f} m  stopped {s.stopped_vehicle_count}"
                      f"  flow {s.flow_veh_h:.0f}/h", (255, 255, 255)))
    if not lines:
        return
    h = 22 * len(lines) + 10
    overlay = img.copy()
    cv2.rectangle(overlay, (0, img.shape[0] - h), (img.shape[1], img.shape[0]), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    for i, (text, color) in enumerate(lines):
        cv2.putText(img, text, (10, img.shape[0] - h + 22 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1,
                    cv2.LINE_AA)
