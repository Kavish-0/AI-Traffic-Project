"""Camera calibration check (headless - writes images instead of opening windows).

    python -m traffic_intel.computer_vision.calibration_tool cam_01 [--frame 0] [--source path] [--out dir]
    python -m traffic_intel.computer_vision.calibration_tool data/synthetic/default/camera.yaml

Writes to ``runs/calibration/<camera>/``:
    frame.png      raw frame at the processing resolution - open it in any image viewer
                   to read pixel coordinates for calibration points and zones
    overlay.png    zones + calibration points (with reprojection residuals) + projected metric grid
    birdseye.png   frame warped to a top-down metric view (10 px per metre); straight,
                   parallel lane lines with correct spacing = good calibration
and prints the reprojection error per point, ROI length, lanes, travel direction.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from traffic_intel.computer_vision import annotate
from traffic_intel.computer_vision.homography import HomographyProjector
from traffic_intel.computer_vision.sources import open_video_source
from traffic_intel.computer_vision.zones import CameraGeometry
from traffic_intel.core.config import PROJECT_ROOT, CameraConfig


def birdseye(img: np.ndarray, proj: HomographyProjector, geo: CameraGeometry, px_per_m: float = 10.0,
             margin_m: float = 5.0) -> np.ndarray:
    g = proj.image_to_ground(geo.measurement_region)
    g = g[np.all(np.isfinite(g), axis=1)]
    (xmin, ymin), (xmax, ymax) = g.min(axis=0) - margin_m, g.max(axis=0) + margin_m
    # ground -> output pixels (x right, y upstream = up)
    G2O = np.array([[px_per_m, 0, -xmin * px_per_m], [0, -px_per_m, ymax * px_per_m], [0, 0, 1]])
    size = (int((xmax - xmin) * px_per_m), int((ymax - ymin) * px_per_m))
    out = cv2.warpPerspective(img, G2O @ proj.H, size)
    for m in range(0, int(ymax - ymin) + 1, 10):                   # 10 m ticks
        y = int(size[1] - m * px_per_m)
        cv2.line(out, (0, y), (12, y), (0, 255, 255), 1)
        cv2.putText(out, f"{m + ymin:.0f}m", (14, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
    return out


def check(camera: CameraConfig, frame_index: int, out: Path) -> dict[str, object]:
    if camera.calibration is None:
        raise SystemExit(f"camera {camera.id} has no calibration section")
    src = open_video_source(camera)
    frame = None
    for f in src.frames():
        frame = f
        if f.index >= frame_index:
            break
    src.close()
    if frame is None:
        raise SystemExit("no frame could be read from the source")
    size = (frame.image.shape[1], frame.image.shape[0])
    proj = HomographyProjector(camera.calibration, size)
    geo = CameraGeometry(camera, size, proj)
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "frame.png"), frame.image)

    over = frame.image.copy()
    annotate.draw_zones(over, geo)
    annotate.draw_ground_grid(over, geo)
    residuals = []
    for (u, v), g in zip(proj.image_points, proj.ground_points, strict=True):
        back = proj.ground_to_image(g.reshape(1, 2))[0]
        err_m = float(np.linalg.norm(proj.image_to_ground(np.array([[u, v]]))[0] - g))
        residuals.append(err_m)
        cv2.circle(over, (int(u), int(v)), 6, (0, 0, 255), 2)
        cv2.line(over, (int(u), int(v)), (int(back[0]), int(back[1])), (0, 0, 255), 2)
        cv2.putText(over, f"({g[0]:.1f},{g[1]:.1f}) {err_m:.2f}m", (int(u) + 8, int(v) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out / "overlay.png"), over)
    cv2.imwrite(str(out / "birdseye.png"), birdseye(frame.image, proj, geo))
    return {
        "frame": frame.index,
        "frame_size": size,
        "reprojection_error_m": round(proj.reprojection_error_m, 3),
        "point_errors_m": [round(e, 3) for e in residuals],
        "roi_length_m": round(geo.roi_length_m, 1),
        "lanes": geo.lane_count,
        "travel_direction": [round(float(v), 3) for v in geo.travel_direction],
        "anchor_reference": geo.anchor_reference,
        "metres_per_pixel_at_stop_line": round(proj.metres_per_pixel(proj.ground_to_image(
            geo.stop_point.reshape(1, 2))[0]), 3),
        "output": str(out),
    }


def main() -> None:
    from traffic_intel.pipeline.__main__ import find_camera

    p = argparse.ArgumentParser(prog="traffic_intel.computer_vision.calibration_tool")
    p.add_argument("camera", help="camera id or path to a camera YAML")
    p.add_argument("--frame", type=int, default=0, help="source frame index to use")
    p.add_argument("--source", help="override the camera source")
    p.add_argument("--out")
    a = p.parse_args()
    cam = find_camera(a.camera)
    if a.source:
        local = Path(a.source)
        cam = cam.model_copy(update={"source": str(local.resolve()) if local.exists() else a.source,
                                     "source_type": "auto"})
    out = Path(a.out) if a.out else PROJECT_ROOT / "runs" / "calibration" / cam.id
    for k, v in check(cam, a.frame, out).items():
        print(f"{k:32s} {v}")


if __name__ == "__main__":
    main()
