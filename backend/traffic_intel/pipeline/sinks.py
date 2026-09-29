"""Output sinks of a camera worker: JSONL, Parquet (detections/tracks/trajectories), annotated video."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import TypeVar

import cv2
import numpy as np
import pandas as pd
from pydantic import BaseModel

from traffic_intel.core.enums import class_from_id
from traffic_intel.core.schemas import TrackSummary
from traffic_intel.core.types import TrackedFrame
from traffic_intel.tracking.trajectory import TrackedObject


class JsonlSink:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._fh = path.open("w", encoding="utf-8")

    def write(self, item: BaseModel) -> None:
        self._fh.write(item.model_dump_json() + "\n")

    def close(self) -> None:
        self._fh.close()


class ParquetRecorder:
    """Frame-level perception data -> Parquet, for offline datasets.

    Layout: ``<root>/<camera>/<YYYY-MM-DD>/<run_id>/<dataset>-NNNNN.parquet`` with datasets
      detections      raw detector output per frame (class, confidence, box)
      tracks          tracked objects per frame (track id, voted class, box, anchor,
                      ground position, speed, stopped time, track age)
      track_summaries one row per finished trajectory (paths as lists)
    Buffers are flushed every ``flush_rows`` rows, when the date changes, and on close.
    This data stays inside the camera worker: it is never published on the bus.
    """

    DETECTION_COLUMNS = ["camera_id", "run_id", "frame_index", "timestamp", "class_id", "class_name",
                         "confidence", "x1", "y1", "x2", "y2"]
    TRACK_COLUMNS = ["camera_id", "run_id", "frame_index", "timestamp", "track_id", "class_name", "confidence",
                     "x1", "y1", "x2", "y2", "anchor_x", "anchor_y", "ground_x_m", "ground_y_m", "speed_kmh",
                     "stopped_s", "age_s"]

    def __init__(self, root: Path, camera_id: str, run_id: str, *, detections: bool = True, tracks: bool = True,
                 track_summaries: bool = True, flush_rows: int = 50_000) -> None:
        self.root = root
        self.camera_id = camera_id
        self.run_id = run_id
        self.enabled = {"detections": detections, "tracks": tracks, "track_summaries": track_summaries}
        self.flush_rows = flush_rows
        self._rows: dict[str, list[tuple[object, ...] | dict[str, object]]] = {k: [] for k in self.enabled}
        self._part: dict[str, int] = dict.fromkeys(self.enabled, 0)
        self._day: str | None = None
        self.files: dict[str, list[Path]] = {k: [] for k in self.enabled}
        self.rows_written: dict[str, int] = dict.fromkeys(self.enabled, 0)

    def _check_day(self, ts: datetime) -> None:
        day = ts.date().isoformat()
        if self._day is not None and day != self._day:
            self.flush()
        self._day = day

    def add_frame(self, tracked: TrackedFrame, objects: list[TrackedObject]) -> None:
        self._check_day(tracked.timestamp)
        ts, fi = tracked.timestamp, tracked.frame_index
        if self.enabled["detections"]:
            d = tracked.detections
            for i in range(len(d)):
                x1, y1, x2, y2 = (float(v) for v in d.xyxy[i])
                cid = int(d.class_id[i])
                self._rows["detections"].append((self.camera_id, self.run_id, fi, ts, cid, class_from_id(cid).value,
                                                 float(d.confidence[i]), x1, y1, x2, y2))
        if self.enabled["tracks"]:
            for o in objects:
                gx, gy = o.ground_xy if o.ground_xy is not None else (None, None)
                self._rows["tracks"].append((self.camera_id, self.run_id, fi, ts, o.track_id, o.class_name,
                                             o.confidence, *o.bbox, *o.anchor, gx, gy, o.speed_kmh, o.stopped_s,
                                             o.age_s))
        if any(len(r) >= self.flush_rows for r in self._rows.values()):
            self.flush()

    def add_track_summaries(self, summaries: list[TrackSummary]) -> None:
        if not self.enabled["track_summaries"]:
            return
        for s in summaries:
            row = s.model_dump()
            row["object_class"] = s.object_class.value
            row["run_id"] = self.run_id
            self._rows["track_summaries"].append(row)

    def flush(self) -> None:
        if self._day is None:
            return
        out_dir = self.root / self.camera_id / self._day / self.run_id
        for name, rows in self._rows.items():
            if not rows:
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            if name == "detections":
                df = pd.DataFrame(rows, columns=self.DETECTION_COLUMNS)
            elif name == "tracks":
                df = pd.DataFrame(rows, columns=self.TRACK_COLUMNS)
            else:
                df = pd.DataFrame(rows)
            path = out_dir / f"{name}-{self._part[name]:05d}.parquet"
            df.to_parquet(path, index=False)
            self.files[name].append(path)
            self.rows_written[name] += len(df)
            self._part[name] += 1
            rows.clear()

    def close(self) -> None:
        self.flush()


class VideoSink:
    def __init__(self, path: Path, fps: float, size: tuple[int, int], codec: str = "mp4v") -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.frames = 0
        self._writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*codec), fps, size)
        if not self._writer.isOpened():
            raise OSError(f"cannot open video writer for {path}")

    def write(self, img: np.ndarray) -> None:
        self._writer.write(img)
        self.frames += 1

    def close(self) -> None:
        self._writer.release()


_Sink = TypeVar("_Sink", "JsonlSink", "ParquetRecorder", "VideoSink")


class Sinks:
    """Context manager closing everything it holds."""

    def __init__(self) -> None:
        self.items: list[JsonlSink | ParquetRecorder | VideoSink] = []

    def add(self, sink: _Sink) -> _Sink:
        self.items.append(sink)
        return sink

    def __enter__(self) -> Sinks:
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None,
                 tb: TracebackType | None) -> None:
        for s in self.items:
            s.close()
