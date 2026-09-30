"""Compatibility wrapper: run an already-built PerceptionPipeline through a CameraWorker.

Prefer ``CameraWorker.from_config`` (configuration-driven). This helper is kept
for callers that construct the pipeline and source themselves (tests, synthetic mode).
"""

from __future__ import annotations

from pathlib import Path

from traffic_intel.computer_vision.interfaces import VideoSource
from traffic_intel.pipeline.camera_worker import CameraWorker, RunReport, WorkerOutputs, format_state
from traffic_intel.pipeline.perception import PerceptionPipeline
from traffic_intel.pipeline.publish import Publisher

__all__ = ["RunReport", "format_state", "run"]


def run(
    pipe: PerceptionPipeline,
    source: VideoSource,
    out_dir: Path,
    *,
    annotate_video: bool = False,
    write_live: bool = False,
    record_root: Path | None = None,
    max_seconds: float | None = None,
    echo: bool = True,
    publisher: Publisher | None = None,
) -> RunReport:
    outputs = WorkerOutputs(out_dir=out_dir, annotate_video=annotate_video, write_live_jsonl=write_live,
                            parquet_root=record_root, echo=echo)
    return CameraWorker(pipe, source, publisher=publisher, outputs=outputs).run(max_seconds=max_seconds)
