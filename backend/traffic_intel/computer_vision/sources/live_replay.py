"""Continuous live replay of a finite source (video file or synthetic scene).

The local runtime runs recorded footage as if it were a live camera: frames are paced at source
speed, the source restarts at its end, and every frame is stamped with WALL-CLOCK time (start of
the run + elapsed source time). Windows, forecasts and incidents therefore carry current
timestamps, and the dashboard's "last updated" is honest. Frame indices are kept from the
underlying source (detectors that replay per-frame data, e.g. synthetic or precomputed, rely on them).

Webcams and network streams are already live and are never wrapped.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import replace
from datetime import UTC, datetime, timedelta

from traffic_intel.computer_vision.interfaces import VideoSource
from traffic_intel.core.types import Frame


class LiveReplaySource(VideoSource):
    def __init__(self, source: VideoSource, *, loop: bool = True, realtime: bool = True,
                 start: datetime | None = None) -> None:
        self.inner = source
        self.camera_id = source.camera_id
        self.loop = loop
        self.realtime = realtime
        # None: anchored to the wall clock when the FIRST frame is read (pacing starts then too), so
        # time spent building other cameras before this worker starts is not a permanent lag
        self.start = start
        self.loops = 0
        self._closed = False

    @property
    def fps(self) -> float:
        return self.inner.fps

    @property
    def frame_size(self) -> tuple[int, int]:
        return self.inner.frame_size

    @property
    def is_live(self) -> bool:
        return True

    @property
    def stats(self) -> object:
        return getattr(self.inner, "stats", None)

    def frames(self) -> Iterator[Frame]:
        wall0 = time.monotonic()
        start = self.start if self.start is not None else datetime.now(UTC)
        offset = 0.0                                   # source seconds of the previous loops
        step = 1.0 / max(self.inner.fps, 1e-6)
        while not self._closed:
            first: datetime | None = None
            t_rel = 0.0
            for f in self.inner.frames():
                if self._closed:
                    return
                first = first or f.timestamp
                t_rel = (f.timestamp - first).total_seconds()
                t = offset + t_rel
                if self.realtime:
                    delay = t - (time.monotonic() - wall0)
                    if delay > 0:
                        time.sleep(delay)
                yield replace(f, timestamp=start + timedelta(seconds=t))
            if not self.loop or first is None:
                return
            offset += t_rel + step
            self.loops += 1
            rewind = getattr(self.inner, "rewind", None)   # a file source must seek back before re-iterating
            if rewind is not None:
                rewind()

    def close(self) -> None:
        self._closed = True
        close = getattr(self.inner, "close", None)
        if close is not None:
            close()
