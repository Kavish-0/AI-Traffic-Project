"""Process entry points.

    # Perception + traffic state on a configured camera (file, webcam or stream):
    python -m traffic_intel.pipeline replay cam_01 [--source path|webcam:0|rtsp://...]
                                                   [--max-seconds 120] [--annotate] [--live] [--realtime]
                                                   [--detections detections.csv]   # no detector needed
    # Same pipeline on a synthetic scene - no video, no GPU; prints ground truth next to estimates:
    python -m traffic_intel.pipeline synthetic [--scenario default|free_flow|congested] [--annotate]

    # Local full system: camera workers + analytics (anomaly, incidents, forecast, risk) -> database;
    # the API (uvicorn traffic_intel.api.main:app) serves it to the dashboard. Profiles: configs/runtime.yaml
    python -m traffic_intel.pipeline run [--profile demo|synthetic|public] [--max-seconds N]
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

from traffic_intel.core.config import (
    PROJECT_ROOT,
    CameraConfig,
    get_settings,
    load_camera,
    load_network,
    load_pipeline_config,
)
from traffic_intel.core.logging import configure_logging


def find_camera(camera_id: str) -> CameraConfig:
    if camera_id.endswith((".yaml", ".yml")):          # a camera YAML path
        return load_camera(Path(camera_id))
    settings = get_settings()
    pipeline = load_pipeline_config(settings)
    for rel in pipeline.camera_files:
        cam = load_camera(settings.config_dir / rel)
        if cam.id == camera_id:
            return cam
    for path in sorted((settings.config_dir / "cameras").glob("*.yaml")):
        cam = load_camera(path)
        if cam.id == camera_id:
            return cam
    raise SystemExit(f"camera '{camera_id}' not found in configs/cameras/")


def _out_dir(args: argparse.Namespace, name: str) -> Path:
    if args.out:
        return Path(args.out)
    return PROJECT_ROOT / "runs" / name / datetime.now().strftime("%Y%m%d-%H%M%S")


def cmd_replay(args: argparse.Namespace) -> None:
    from traffic_intel.pipeline.camera_worker import CameraWorker

    settings = get_settings()
    pipeline = load_pipeline_config(settings)
    network = load_network(settings.config_dir / pipeline.network_file)
    camera = find_camera(args.camera_id)
    updates: dict[str, object] = {}
    if args.source:
        # CLI paths are relative to the current directory (config paths: to the project root).
        local = Path(args.source)
        updates["source"] = str(local.resolve()) if local.exists() else args.source
        updates["source_type"] = "auto"
    if args.realtime:
        updates["realtime"] = True
    if updates:
        camera = camera.model_copy(update=updates)
    if args.device:
        det = pipeline.detector.model_copy(update={"params": {**pipeline.detector.params, "device": args.device}})
        pipeline = pipeline.model_copy(update={"detector": det})
    detector = None
    if args.detections:
        from traffic_intel.computer_vision.detectors.precomputed import PrecomputedDetector

        detector = PrecomputedDetector(path=str(Path(args.detections).resolve()))
    worker = CameraWorker.from_config(
        camera, pipeline, network, settings=settings,
        annotate_video=True if args.annotate else None, record=not args.no_record,
        out_dir=Path(args.out) if args.out else None, write_live_jsonl=args.live, echo=True, detector=detector,
    )
    st = worker.status()
    print(f"camera {camera.id} -> approach {camera.approach_id} | source {camera.source}")
    print(f"  {st['frame_size'][0]}x{st['frame_size'][1]} @ {st['source_fps']:.1f} fps | detector {st['detector']} "
          f"{st['model'] or ''} | device {st['device'] or 'n/a'}"
          + (" | PERCEPTION-ONLY (no calibration)" if st["perception_only"] else ""))
    print(f"  outputs: {worker.outputs.out_dir}")
    report = worker.run(max_seconds=args.max_seconds)
    st = worker.status()
    print(f"done: {report.frames} frames in {report.wall_s:.1f} s ({report.fps:.1f} frames/s), "
          f"{len(report.states)} windows, {report.tracks} finished tracks")
    print(f"  latency ms/frame: {st['latency_ms_mean']}")
    print(f"  published: {report.published}")
    if report.annotated_video:
        print(f"  annotated video: {report.annotated_video}")
    for name, files in report.parquet_files.items():
        print(f"  parquet {name}: {Path(files[0]).parent} ({len(files)} file(s))")
    print(f"  summary: {worker.outputs.out_dir / 'summary.json' if worker.outputs.out_dir else '-'}")


def cmd_synthetic(args: argparse.Namespace) -> None:
    import numpy as np

    from traffic_intel.pipeline.perception import PerceptionPipeline
    from traffic_intel.pipeline.runner import run
    from traffic_intel.synthetic.scene import truth_for_window
    from traffic_intel.synthetic.sources import SyntheticDetector, SyntheticVideoSource, synthetic_camera

    settings = get_settings()
    pipeline = load_pipeline_config(settings)
    network = load_network(settings.config_dir / pipeline.network_file)
    camera = synthetic_camera(args.scenario)
    source = SyntheticVideoSource(camera, render=args.annotate)
    pipe = PerceptionPipeline.build(camera, pipeline, network, source.frame_size, source.fps,
                                    detector=SyntheticDetector(args.scenario))
    out = _out_dir(args, f"synthetic_{args.scenario}")
    print(f"synthetic scenario '{args.scenario}': {source.scene.n_frames} frames; output {out}")
    report = run(pipe, source, out, annotate_video=args.annotate, write_live=args.live, record_root=None)

    projector, geometry, engine = pipe.projector, pipe.geometry, pipe.engine
    if projector is None or geometry is None or engine is None:
        raise SystemExit("synthetic camera has no calibration/zones - cannot compare with ground truth")

    def along_extent(poly: np.ndarray) -> tuple[float, float]:
        g = projector.image_to_ground(poly)
        a = geometry.distance_upstream(g[np.all(np.isfinite(g), axis=1)])
        return float(a.min()), float(a.max())

    region, roi = along_extent(geometry.measurement_region), along_extent(geometry.roi)
    print("\nestimate vs ground truth (full windows only):")
    print(f"{'window':8s} {'veh':>9s} {'speed km/h':>13s} {'outflow':>9s} {'queue m':>13s}")
    for s in report.states:
        if s.window_s < engine.cfg.window_s - 1e-6:  # type: ignore[attr-defined]
            continue
        tr = truth_for_window(source.scene, s.window_start, s.timestamp, region_along=region,
                              count_region_along=roi)
        sp = "--" if s.average_speed_kmh is None else f"{s.average_speed_kmh:.1f}"
        tsp = "--" if tr.mean_speed_kmh is None else f"{tr.mean_speed_kmh:.1f}"
        print(f"{s.window_start:%H:%M:%S} {s.vehicle_count:4d}/{tr.vehicles_present:<4d} {sp:>6s}/{tsp:<6s} "
              f"{s.outflow:4d}/{tr.outflow:<4d} {s.queue_length_m:6.1f}/{tr.mean_queue_m:<6.1f}")


def cmd_run(args: argparse.Namespace) -> None:
    from traffic_intel.pipeline.runtime import Runtime

    Runtime(args.profile, max_seconds=args.max_seconds).run()


def main() -> None:
    parser = argparse.ArgumentParser(prog="traffic_intel.pipeline")
    sub = parser.add_subparsers(dest="mode", required=True)

    p = sub.add_parser("replay", help="run perception + traffic state on a configured camera")
    p.add_argument("camera_id", help="camera id from configs/cameras/, or a path to a camera YAML")
    p.add_argument("--source", help="override the camera source (file, webcam:0, rtsp://...)")
    p.add_argument("--max-seconds", type=float, help="stop after this much source time")
    p.add_argument("--realtime", action="store_true", help="pace file playback at source speed")
    p.add_argument("--annotate", action="store_true", help="write annotated.mp4")
    p.add_argument("--live", action="store_true", help="also write rolling live states")
    p.add_argument("--no-record", action="store_true", help="do not write detection/track Parquet")
    p.add_argument("--device", help="override detector.params.device (auto, cuda, cuda:N, cpu)")
    p.add_argument("--detections", help="replay detections from a CSV instead of running the detector")
    p.add_argument("--out", help="output directory (default <outputs.runs_dir>/<camera>/<run_id>)")
    p.set_defaults(func=cmd_replay)

    s = sub.add_parser("synthetic", help="run the pipeline on a synthetic scene with ground truth")
    s.add_argument("--scenario", default="default", choices=["default", "free_flow", "congested", "small"])
    s.add_argument("--annotate", action="store_true")
    s.add_argument("--live", action="store_true")
    s.add_argument("--out")
    s.set_defaults(func=cmd_synthetic)

    r = sub.add_parser("run", help="local full system: cameras + analytics -> database (see configs/runtime.yaml)")
    r.add_argument("--profile", help="runtime profile (default: runtime.yaml default_profile)")
    r.add_argument("--max-seconds", type=float, help="stop each camera after this much source time")
    r.set_defaults(func=cmd_run)

    args = parser.parse_args()
    configure_logging(get_settings().log_level)
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    args.func(args)


if __name__ == "__main__":
    main()
