"""Export a synthetic scene as a small, self-describing dataset.

    python -m traffic_intel.synthetic export --scenario small --out tests/fixtures/synthetic_small
    python -m traffic_intel.synthetic export --scenario default --out data/synthetic/default --video

Writes:
    camera.yaml       CameraConfig (calibration + zones) usable with `pipeline replay`
    detections.csv    noisy detections (precomputed-detector format)
    truth.csv         per-frame ground truth: frame, time, id, class, x_m, along_m, speed_kmh, bbox
    scene.json        generator config (seeded -> reproducible)
    video.mp4         rendered frames (only with --video)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from traffic_intel.computer_vision.detectors.precomputed import save_detection_table
from traffic_intel.synthetic.scene import SCENARIOS


def _config_path(path: Path) -> str:
    """Config paths are relative to the project root when possible."""
    from traffic_intel.core.config import PROJECT_ROOT

    path = path.resolve()
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def export(scenario: str, out: Path, video: bool = False) -> Path:
    from traffic_intel.synthetic.sources import SyntheticVideoSource, get_scene, synthetic_camera

    out.mkdir(parents=True, exist_ok=True)
    scene = get_scene(scenario)
    camera = synthetic_camera(scenario, camera_id=f"synth_{scenario}")
    video_path = out / "video.mp4"
    cam_dump = json.loads(camera.model_dump_json())
    cam_dump.update(source=_config_path(video_path) if video else f"synthetic://{scenario}",
                    source_type="file" if video else "synthetic")
    (out / "camera.yaml").write_text(yaml.safe_dump(cam_dump, sort_keys=False), encoding="utf-8")

    save_detection_table(out / "detections.csv", dict(enumerate(scene.detections)))
    rows = []
    for i, (ts, objs) in enumerate(zip(scene.timestamps, scene.truth, strict=True)):
        for o in objs:
            bb = o.bbox or (None, None, None, None)
            rows.append((i, ts.isoformat(), o.id, o.cls.value, round(o.x, 3), round(o.along, 3),
                         round(o.speed_kmh, 3), *(None if v is None else round(v, 2) for v in bb)))
    pd.DataFrame(rows, columns=["frame", "time", "id", "class", "x_m", "along_m", "speed_kmh",
                                "x1", "y1", "x2", "y2"]).to_csv(out / "truth.csv", index=False)
    (out / "scene.json").write_text(scene.cfg.model_dump_json(indent=2), encoding="utf-8")

    if video:
        import cv2

        src = SyntheticVideoSource(camera, render=True)
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter.fourcc(*"mp4v"), src.fps, src.frame_size)
        for frame in src.frames():
            writer.write(frame.image)
        writer.release()
    return out


def main() -> None:
    p = argparse.ArgumentParser(prog="traffic_intel.synthetic")
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--scenario", default="small", choices=sorted(SCENARIOS))
    e.add_argument("--out", required=True)
    e.add_argument("--video", action="store_true")
    args = p.parse_args()
    print(export(args.scenario, Path(args.out), args.video))


if __name__ == "__main__":
    main()
