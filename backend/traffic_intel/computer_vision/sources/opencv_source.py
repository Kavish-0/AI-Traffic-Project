"""OpenCV video source: files, webcams and network streams.

Timestamps
    file    : replay_start + frame_index / source_fps   (deterministic)
    webcam  : UTC wall clock at capture
    stream  : UTC wall clock at capture

Sampling (``process_fps``)
    Frames are emitted on a fixed time grid ``t0, t0 + 1/process_fps, ...``;
    the first decoded frame at or after each grid point is emitted. For files
    this is exact and deterministic (no frames are dropped by lag). For live
    sources a background thread keeps only the newest frame, so a slow
    consumer lowers the effective rate instead of accumulating latency;
    skipped frames are counted in ``stats``.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from traffic_intel.computer_vision.interfaces import VIDEO_SOURCES, VideoSource
from traffic_intel.core.config import CameraConfig, resolve_path
from traffic_intel.core.types import Frame

log = logging.getLogger(__name__)

SourceKind = Literal["file", "webcam", "stream"]


def classify_source(source: str, declared: str = "auto") -> SourceKind:
    if declared in ("file", "webcam", "stream"):
        return declared  # type: ignore[return-value]
    s = source.strip().lower()
    if s.startswith("webcam:") or s.isdigit():
        return "webcam"
    if s.startswith(("rtsp://", "rtmp://", "http://", "https://", "udp://", "tcp://")):
        return "stream"
    return "file"


@dataclass
class SourceStats:
    decoded: int = 0
    emitted: int = 0
    skipped_by_sampling: int = 0
    dropped_by_lag: int = 0          # live only: frames overwritten before use
    read_errors: int = 0
    reconnects: int = 0
    started_at: float = field(default_factory=time.monotonic)


class OpenCVVideoSource(VideoSource):
    def __init__(
        self,
        camera_id: str,
        source: str,
        process_fps: float,
        *,
        source_type: str = "auto",
        source_fps: float | None = None,
        resolution: tuple[int, int] | None = None,
        replay_start: datetime | None = None,
        realtime: bool = False,
        loop: bool = False,
        reconnect_max_s: float = 30.0,
        max_frames: int | None = None,
        prefetch: int = 8,
    ) -> None:
        self.camera_id = camera_id
        self.source = source
        self.kind: SourceKind = classify_source(source, source_type)
        self.process_fps = float(process_fps)
        self.resolution = tuple(resolution) if resolution else None
        self.realtime = realtime
        self.loop = loop
        self.reconnect_max_s = reconnect_max_s
        self.max_frames = max_frames
        self.prefetch = prefetch            # file sources: frames decoded ahead in a background thread
        self.stats = SourceStats()
        self._closed = False

        self._cap = self._open()
        native_fps = self._cap.get(cv2.CAP_PROP_FPS) or 0.0
        if source_fps:
            self._source_fps = float(source_fps)
        elif 1.0 <= native_fps <= 240.0:
            self._source_fps = float(native_fps)
        else:
            self._source_fps = 30.0
            log.warning("%s: source fps unknown (%.1f); assuming 30 - set source_fps", camera_id, native_fps)
        if self.process_fps > self._source_fps:
            log.info("%s: process_fps %.1f > source fps %.1f; every frame will be used",
                     camera_id, self.process_fps, self._source_fps)

        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.native_size = (w, h)
        self._frame_size = self.resolution or self.native_size

        if self.kind == "file":
            if replay_start is None:
                replay_start = datetime.now(UTC).replace(microsecond=0)
                log.warning("%s: replay_start not set; using now (%s) - hour/day features will "
                            "not match the recording", camera_id, replay_start.isoformat())
            self.replay_start: datetime | None = replay_start
        else:
            self.replay_start = None

        self._latest: tuple[np.ndarray, datetime, int] | None = None
        self._latest_lock = threading.Condition()
        self._reader: threading.Thread | None = None

    # ------------------------------------------------------------------ #
    @classmethod
    def from_camera(cls, camera: CameraConfig, **overrides: object) -> OpenCVVideoSource:
        replay_start = datetime.fromisoformat(camera.replay_start) if camera.replay_start else None
        if replay_start is not None and replay_start.tzinfo is None:
            replay_start = replay_start.replace(tzinfo=UTC)
        kwargs: dict[str, object] = dict(
            source_type=camera.source_type,
            source_fps=camera.source_fps,
            resolution=camera.resolution,
            replay_start=replay_start,
            realtime=camera.realtime,
            loop=camera.loop,
        )
        kwargs.update(overrides)
        return cls(camera.id, camera.source, camera.process_fps, **kwargs)  # type: ignore[arg-type]

    def _open(self) -> cv2.VideoCapture:
        if self.kind == "webcam":
            index = int(self.source.split(":", 1)[-1])
            backend = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") and _is_windows() else cv2.CAP_ANY
            cap = cv2.VideoCapture(index, backend)
        elif self.kind == "file":
            path = resolve_path(self.source)
            if not Path(path).exists():
                raise FileNotFoundError(f"{self.camera_id}: video file not found: {path}")
            cap = cv2.VideoCapture(str(path))
        else:
            cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise ConnectionError(f"{self.camera_id}: cannot open source {self.source!r}")
        return cap

    # ------------------------------------------------------------------ #
    @property
    def fps(self) -> float:
        """Effective emitted frame rate."""
        return min(self.process_fps, self._source_fps)

    @property
    def source_fps(self) -> float:
        return self._source_fps

    @property
    def is_live(self) -> bool:
        return self.kind != "file"

    @property
    def frame_size(self) -> tuple[int, int]:
        return self._frame_size  # type: ignore[return-value]

    def _resize(self, img: np.ndarray) -> np.ndarray:
        if self.resolution and (img.shape[1], img.shape[0]) != self.resolution:
            interp = cv2.INTER_AREA if img.shape[1] > self.resolution[0] else cv2.INTER_LINEAR
            return cv2.resize(img, self.resolution, interpolation=interp)
        return img

    def frames(self) -> Iterator[Frame]:
        gen = self._file_frames() if self.kind == "file" else self._live_frames()
        if self.kind == "file" and self.prefetch > 0:
            gen = self._prefetched(gen)
        for frame in gen:
            if self.max_frames is not None and self.stats.emitted >= self.max_frames:
                self._closed = True
                return
            self.stats.emitted += 1
            yield frame

    def _prefetched(self, gen: Iterator[Frame]) -> Iterator[Frame]:
        """Decode files in a background thread (bounded queue) so decoding overlaps inference.
        Every frame is still delivered, in order - replays stay deterministic."""
        q: queue.Queue[Frame | BaseException | None] = queue.Queue(maxsize=self.prefetch)

        def produce() -> None:
            try:
                for f in gen:
                    while not self._closed:
                        try:
                            q.put(f, timeout=0.2)
                            break
                        except queue.Full:
                            continue
                    if self._closed:
                        return
            except BaseException as e:  # noqa: BLE001 - re-raised in the consumer
                q.put(e)
                return
            q.put(None)

        threading.Thread(target=produce, name=f"decode-{self.camera_id}", daemon=True).start()
        while True:
            item = q.get()
            if item is None:
                return
            if isinstance(item, BaseException):
                raise item
            yield item

    # ---------------------------- files ------------------------------- #
    def _file_frames(self) -> Iterator[Frame]:
        assert self.replay_start is not None
        offset_s = 0.0           # accumulated duration of previous loops
        wall0 = time.monotonic()
        while not self._closed:
            index = 0
            last_slot = -1
            last_t = 0.0
            while not self._closed:
                t = index / self._source_fps
                # Emit the first frame in each 1/process_fps slot of source time.
                slot = int(np.floor(t * self.process_fps + 1e-6))
                if slot <= last_slot:                       # skipped by sampling: grab without retrieve
                    if not self._cap.grab():
                        break
                    self.stats.decoded += 1
                    self.stats.skipped_by_sampling += 1
                    last_t = t
                    index += 1
                    continue
                ok, img = self._cap.read()
                if not ok:
                    break
                self.stats.decoded += 1
                last_t = t
                index += 1
                last_slot = slot
                if self.realtime:
                    delay = (offset_s + t) - (time.monotonic() - wall0)
                    if delay > 0:
                        time.sleep(delay)
                yield Frame(
                    camera_id=self.camera_id,
                    index=index - 1,
                    timestamp=self.replay_start + timedelta(seconds=offset_s + t),
                    image=self._resize(img),
                )
            if not self.loop or self._closed:
                return
            offset_s += last_t + 1.0 / self._source_fps
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # ---------------------------- live -------------------------------- #
    def _reader_loop(self) -> None:
        backoff = 1.0
        index = 0
        while not self._closed:
            ok, img = self._cap.read()
            if not ok:
                self.stats.read_errors += 1
                log.warning("%s: read failed; reconnecting in %.0f s", self.camera_id, backoff)
                self._cap.release()
                time.sleep(backoff)
                backoff = min(backoff * 2, self.reconnect_max_s)
                try:
                    self._cap = self._open()
                    self.stats.reconnects += 1
                except ConnectionError:
                    continue
                continue
            backoff = 1.0
            self.stats.decoded += 1
            with self._latest_lock:
                if self._latest is not None:
                    self.stats.dropped_by_lag += 1
                self._latest = (img, datetime.now(UTC), index)
                self._latest_lock.notify()
            index += 1

    def _live_frames(self) -> Iterator[Frame]:
        self._reader = threading.Thread(target=self._reader_loop, name=f"src-{self.camera_id}", daemon=True)
        self._reader.start()
        step = 1.0 / self.process_fps
        next_due = time.monotonic()
        while not self._closed:
            now = time.monotonic()
            if now < next_due:
                time.sleep(next_due - now)
            with self._latest_lock:
                while self._latest is None and not self._closed:
                    self._latest_lock.wait(timeout=1.0)
                if self._closed or self._latest is None:
                    return
                img, ts, idx = self._latest
                self._latest = None
            next_due = max(next_due + step, time.monotonic())
            yield Frame(camera_id=self.camera_id, index=idx, timestamp=ts, image=self._resize(img))

    def close(self) -> None:
        self._closed = True
        with self._latest_lock:
            self._latest_lock.notify_all()
        if self._reader is not None:
            self._reader.join(timeout=2.0)
        self._cap.release()


def _is_windows() -> bool:
    import os

    return os.name == "nt"


@VIDEO_SOURCES.register("opencv")
def _create(camera: CameraConfig, **overrides: object) -> OpenCVVideoSource:
    return OpenCVVideoSource.from_camera(camera, **overrides)
