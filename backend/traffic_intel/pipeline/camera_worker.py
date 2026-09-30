"""Per-camera worker: source -> PerceptionPipeline -> outputs + aggregate publishing.

One worker per camera. It owns that camera's video source, (stateful) tracker,
trajectory store and feature engine. Per frame:

    frame -> preprocess -> Detector -> Tracker -> projection/trajectories -> feature engine
          |- annotated video frame            (optional, local file)
          |- annotated preview JPEG           (optional, overwritten every ``preview_interval_s``;
          |                                    the dashboard's low-rate camera view - never a frame stream)
          |- Parquet rows: detections, tracks (local files)
          '- every ~1 s: live TrafficState         -> publish traffic.features.live
    per 60 s window: TrafficFeatures + TrafficState -> publish traffic.features.window / traffic.state

Frame-level data never leaves the worker (see ``publish.WORKER_TOPICS``).

Run it blocking (``run()``, used by the replay CLI) or in a thread (``start()``
/ ``stop()``, used by the in-process dev mode with an asyncio bus).
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from traffic_intel.computer_vision import annotate
from traffic_intel.computer_vision.interfaces import Detector, VideoSource
from traffic_intel.core.config import (
    AppSettings,
    CameraConfig,
    NetworkConfig,
    PipelineConfig,
    get_settings,
    resolve_path,
)
from traffic_intel.core.schemas import AnomalySignal, TrackSummary, TrafficFeatures, TrafficState
from traffic_intel.pipeline.bus import Topics
from traffic_intel.pipeline.perception import PerceptionPipeline
from traffic_intel.pipeline.publish import NullPublisher, Publisher, WorkerPublisher
from traffic_intel.pipeline.sinks import JsonlSink, ParquetRecorder, Sinks, VideoSink
from traffic_intel.tracking.trajectory import TrackedObject

log = logging.getLogger(__name__)


@dataclass
class WorkerOutputs:
    out_dir: Path | None = None             # states JSONL, tracks JSONL, summary.json, annotated.mp4
    annotate_video: bool = False
    write_live_jsonl: bool = False
    parquet_root: Path | None = None        # None = no Parquet
    record_detections: bool = True
    record_tracks: bool = True
    record_track_summaries: bool = True
    parquet_flush_rows: int = 50_000
    video_codec: str = "mp4v"
    echo: bool = False                      # print one line per closed window
    preview_path: Path | None = None        # latest annotated frame (JPEG), overwritten atomically
    preview_interval_s: float = 1.0
    preview_max_width: int = 960


@dataclass
class RunReport:
    run_id: str = ""
    frames: int = 0
    wall_s: float = 0.0                      # set when the run ends
    started_perf: float | None = None        # perf_counter() at the start (live fps while running)
    states: list[TrafficState] = field(default_factory=list)
    signals: list[AnomalySignal] = field(default_factory=list)   # track-level anomaly evidence
    tracks: int = 0
    last_frame_at: datetime | None = None    # source time of the last processed frame
    error: str | None = None
    out_dir: Path | None = None
    annotated_video: Path | None = None
    parquet_files: dict[str, list[str]] = field(default_factory=dict)
    published: dict[str, int] = field(default_factory=dict)

    @property
    def fps(self) -> float:
        wall = self.wall_s or (time.perf_counter() - self.started_perf if self.started_perf is not None else 0.0)
        return self.frames / wall if wall > 0 else 0.0


def format_state(s: TrafficState) -> str:
    speed = "  --" if s.average_speed_kmh is None else f"{s.average_speed_kmh:4.1f}"
    return (f"{s.window_start:%H:%M:%S} {s.traffic_state.value.upper():9s} score {s.congestion_score:.2f} | "
            f"veh {s.vehicle_count:3d} (car {s.car_count} moto {s.motorcycle_count} bus {s.bus_count} "
            f"truck {s.truck_count} ped {s.pedestrian_count}) | speed {speed} km/h | dens {s.density_pct:5.1f}% | "
            f"occ {s.occupancy_pct:5.1f}% | flow {s.flow_veh_h:5.0f}/h in {s.inflow} out {s.outflow} | "
            f"queue {s.queue_length_m:5.1f} m | stopped {s.stopped_vehicle_count} | q {s.quality:.2f}")


def new_run_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


class CameraWorker:
    def __init__(
        self,
        pipe: PerceptionPipeline,
        source: VideoSource,
        *,
        publisher: Publisher | None = None,
        outputs: WorkerOutputs | None = None,
        run_id: str | None = None,
    ) -> None:
        self.pipe = pipe
        self.source = source
        self.camera = pipe.camera
        self.outputs = outputs or WorkerOutputs()
        self.run_id = run_id or new_run_id()
        self.publisher = WorkerPublisher(publisher or NullPublisher(), source=f"camera_worker:{self.camera.id}")
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.report = RunReport(run_id=self.run_id, out_dir=self.outputs.out_dir)

    # ------------------------------------------------------------------ #
    @classmethod
    def from_config(
        cls,
        camera: CameraConfig,
        pipeline: PipelineConfig,
        network: NetworkConfig,
        *,
        publisher: Publisher | None = None,
        settings: AppSettings | None = None,
        annotate_video: bool | None = None,
        record: bool = True,
        out_dir: Path | None = None,
        write_live_jsonl: bool = False,
        echo: bool = False,
        detector: Detector | None = None,
        run_id: str | None = None,
        source: VideoSource | None = None,
        preview_path: Path | None = None,
        preview_interval_s: float = 1.0,
    ) -> CameraWorker:
        """Build source + pipeline + outputs for one camera from configuration. ``source`` may be
        injected (e.g. wrapped for continuous live replay by the runtime)."""
        from traffic_intel.computer_vision.sources import open_video_source

        settings = settings or get_settings()
        run_id = run_id or new_run_id()
        source = source or open_video_source(camera)
        pipe = PerceptionPipeline.build(camera, pipeline, network, source.frame_size, source.fps, detector=detector)
        oc = pipeline.outputs
        outputs = WorkerOutputs(
            out_dir=out_dir or resolve_path(oc.runs_dir) / camera.id / run_id,
            annotate_video=oc.annotate_video if annotate_video is None else annotate_video,
            write_live_jsonl=write_live_jsonl,
            parquet_root=resolve_path(oc.parquet_dir) if record else None,
            record_detections=oc.record_detections,
            record_tracks=oc.record_tracks,
            record_track_summaries=oc.record_track_summaries,
            parquet_flush_rows=oc.parquet_flush_rows,
            video_codec=oc.video_codec,
            echo=echo,
            preview_path=preview_path,
            preview_interval_s=preview_interval_s,
        )
        return cls(pipe, source, publisher=publisher, outputs=outputs, run_id=run_id)

    # ------------------------------------------------------------------ #
    def start(self, max_seconds: float | None = None) -> threading.Thread:
        self._thread = threading.Thread(target=self.run, kwargs={"max_seconds": max_seconds},
                                        name=f"camera-{self.camera.id}", daemon=True)
        self._thread.start()
        return self._thread

    def stop(self, timeout: float | None = 10.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)

    def status(self) -> dict[str, Any]:
        stats = getattr(self.source, "stats", None)
        detector = self.pipe.detector
        info = getattr(detector, "device_info", None)
        return {
            "camera_id": self.camera.id,
            "run_id": self.run_id,
            "frames": self.report.frames,
            "processing_fps": round(self.report.fps, 2),
            "source_fps": self.source.fps,
            "is_live": self.source.is_live,
            "frame_size": list(self.source.frame_size),
            "latency_ms_mean": {k: round(v, 2) for k, v in self.pipe.latency_ms.items()},
            "dropped_by_lag": getattr(stats, "dropped_by_lag", 0),
            "detector": type(detector).__name__,
            "model": getattr(detector, "weights", None),
            "device": info.describe() if info is not None else None,
            "perception_only": self.pipe.perception_only,
            "published": dict(self.publisher.counts),
            "last_frame_at": self.report.last_frame_at,
            "error": self.report.error,
            "preview_path": str(o.preview_path) if (o := self.outputs).preview_path else None,
        }

    # ------------------------------------------------------------------ #
    def run(self, max_seconds: float | None = None) -> RunReport:
        o = self.outputs
        report = self.report
        t_start = time.perf_counter()
        report.started_perf = t_start
        first_ts = None
        last_window: TrafficState | None = None
        last_live: TrafficState | None = None
        last_preview = 0.0
        with Sinks() as sinks:
            states_sink = tracks_sink = live_sink = anomaly_sink = None
            if o.out_dir is not None:
                o.out_dir.mkdir(parents=True, exist_ok=True)
                states_sink = sinks.add(JsonlSink(o.out_dir / "traffic_states.jsonl"))
                tracks_sink = sinks.add(JsonlSink(o.out_dir / "tracks.jsonl"))
                anomaly_sink = sinks.add(JsonlSink(o.out_dir / "anomalies.jsonl"))
                if o.write_live_jsonl:
                    live_sink = sinks.add(JsonlSink(o.out_dir / "live_states.jsonl"))
            recorder: ParquetRecorder | None = None
            if o.parquet_root is not None and (o.record_detections or o.record_tracks or o.record_track_summaries):
                recorder = sinks.add(ParquetRecorder(
                    o.parquet_root, self.camera.id, self.run_id, detections=o.record_detections,
                    tracks=o.record_tracks, track_summaries=o.record_track_summaries,
                    flush_rows=o.parquet_flush_rows))
            video: VideoSink | None = None
            if o.annotate_video and o.out_dir is not None:
                video = sinks.add(VideoSink(o.out_dir / "annotated.mp4", self.source.fps, self.source.frame_size,
                                            o.video_codec))
                report.annotated_video = video.path

            def emit_windows(features: list[TrafficFeatures], states: list[TrafficState],
                             tracks: list[TrackSummary]) -> None:
                nonlocal last_window
                for f in features:
                    self.publisher.publish(Topics.FEATURES_WINDOW, f)
                for s in states:
                    self.publisher.publish(Topics.TRAFFIC_STATE, s)
                    report.states.append(s)
                    last_window = s
                    if states_sink:
                        states_sink.write(s)
                    if o.echo:
                        print(format_state(s), flush=True)
                for tr in tracks:
                    if tracks_sink:
                        tracks_sink.write(tr)
                if recorder:
                    recorder.add_track_summaries(tracks)
                report.tracks += len(tracks)

            try:
                # One dummy inference first, so CUDA/cuDNN initialisation does not distort latency stats.
                self.pipe.detector.warmup()
                for frame in self.source.frames():
                    if self._stop.is_set():
                        break
                    if first_ts is None:
                        first_ts = frame.timestamp
                    if max_seconds is not None and (frame.timestamp - first_ts).total_seconds() > max_seconds:
                        break
                    res = self.pipe.process(frame)
                    report.frames += 1
                    report.last_frame_at = frame.timestamp
                    for sig in res.signals:
                        self.publisher.publish(Topics.ANOMALY_SIGNAL, sig)
                        report.signals.append(sig)
                        if anomaly_sink:
                            anomaly_sink.write(sig)
                    emit_windows(res.window_features, res.window_states, res.finished_tracks)
                    if res.live_state is not None:
                        last_live = res.live_state
                        self.publisher.publish(Topics.FEATURES_LIVE, res.live_state)
                        if live_sink:
                            live_sink.write(res.live_state)
                    preview_due = o.preview_path is not None and time.monotonic() - last_preview >= o.preview_interval_s
                    objects = self.pipe.tracked_objects(res) if (recorder or video or preview_due) else []
                    if recorder:
                        recorder.add_frame(res.tracked, objects)
                    if video:
                        img = frame.image.copy()
                        if self.pipe.geometry is not None:
                            annotate.draw_zones(img, self.pipe.geometry)
                        annotate.draw_tracks(img, objects)
                        annotate.draw_hud(img, last_live, last_window)
                        video.write(img)
                    if preview_due:
                        last_preview = time.monotonic()
                        self._write_preview(frame.image, objects, last_live, last_window)
            except KeyboardInterrupt:
                log.warning("%s: interrupted - flushing partial window", self.camera.id)
            finally:
                self.source.close()
                feats, states, tracks = self.pipe.flush()
                emit_windows(feats, states, tracks)
                self.publisher.close()
            report.wall_s = time.perf_counter() - t_start
            if recorder:
                recorder.close()
                report.parquet_files = {k: [str(p) for p in v] for k, v in recorder.files.items() if v}
        report.published = dict(self.publisher.counts)
        if o.out_dir is not None:
            self._write_summary(o.out_dir)
        return report

    def _write_preview(self, image: Any, objects: list[TrackedObject], live: TrafficState | None,  # noqa: ANN401
                       window: TrafficState | None) -> None:
        """Annotated JPEG (boxes, class, track id, speed, zones, HUD), replaced atomically."""
        import cv2

        o = self.outputs
        assert o.preview_path is not None
        img = image.copy()
        if self.pipe.geometry is not None:
            annotate.draw_zones(img, self.pipe.geometry)
        annotate.draw_tracks(img, objects)
        annotate.draw_hud(img, live, window)
        h, w = img.shape[:2]
        if w > o.preview_max_width:
            img = cv2.resize(img, (o.preview_max_width, int(h * o.preview_max_width / w)))
        o.preview_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = o.preview_path.with_suffix(".tmp.jpg")
        if cv2.imwrite(str(tmp), img, [cv2.IMWRITE_JPEG_QUALITY, 80]):
            try:
                tmp.replace(o.preview_path)
            except OSError:                      # reader holds the file (Windows): skip this preview
                log.debug("preview for %s busy - skipped", self.camera.id)

    def _write_summary(self, out_dir: Path) -> None:
        r = self.report
        summary = {
            **self.status(),
            "wall_s": round(r.wall_s, 2),
            "windows": len(r.states),
            "tracks_finished": r.tracks,
            "annotated_video": str(r.annotated_video) if r.annotated_video else None,
            "parquet_files": r.parquet_files,
            "calibration_reprojection_error_m": (
                round(self.pipe.projector.reprojection_error_m, 3) if self.pipe.projector else None),
            "roi_length_m": round(self.pipe.geometry.roi_length_m, 1) if self.pipe.geometry else None,
            "lanes": self.pipe.geometry.lane_count if self.pipe.geometry else None,
            "states": [json.loads(s.model_dump_json()) for s in r.states],
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
