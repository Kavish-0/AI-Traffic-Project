"""Replay of recorded loop-detector speeds (PEMS-BAY) as live road sensors.

The real-data counterpart of a camera worker: instead of video it has 325 real stations' 5-min average
speeds, and it publishes the same aggregates on the bus (TrafficFeatures + TrafficState per closed window,
live TrafficState every few seconds), so the analytics worker, database, API and dashboard are unchanged.

Time mapping (what the dashboard shows as "now"): the current wall-clock time in the network's time zone
(America/Los_Angeles) is mapped to the SAME weekday and time of day in ``replay_week`` (a week of the
chronological TEST period the forecaster never trained on). Monday 14:35 PT today shows the reading of
Monday 14:35 PT in that week, stamped with today's time, so rush hours, calendar features and forecasts
line up with the clock. The recorded time is kept in ``extra["recorded_at"]`` (epoch seconds).

What is real: every speed value. What is not measured: vehicle counts, density, occupancy, queues -
published as 0 with ``extra["speed_only"] = 1`` so the congestion rules use speed only and the
dashboard shows them as "not measured". There is no video and no track-level evidence.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from traffic_intel.core.config import NetworkConfig, PipelineConfig
from traffic_intel.core.schemas import TrafficFeatures, TrafficState
from traffic_intel.pipeline.bus import Topics
from traffic_intel.pipeline.publish import NullPublisher, Publisher, WorkerPublisher
from traffic_intel.traffic.state import build_traffic_state

log = logging.getLogger(__name__)

CAMERA_ID = "pems_bay_sensors"        # one pseudo-camera row for the whole detector network


class SensorReplayConfig(BaseModel):
    """``sensors:`` block of a runtime profile (configs/runtime.yaml)."""

    model_config = ConfigDict(extra="forbid")
    dataset: str = "pems_bay"
    replay_week: str = "2017-06-05"           # Monday (local) of the replayed week; must be in the test split
    timezone: str = "America/Los_Angeles"
    step_s: int = 300                        # the dataset's resolution
    backfill_steps: int = 24                 # history published at start (forecasts need their look-back)
    live_tick_s: float = 5.0
    approach_prefix: str = "pems_bay_"       # approach id = prefix + station id (configs/network/pems_bay.yaml)


class SensorReplayWorker:
    def __init__(self, cfg: SensorReplayConfig, network: NetworkConfig, pipeline: PipelineConfig,
                 speeds: pd.DataFrame, *, publisher: Publisher | None = None,
                 processed: Callable[[], int] | None = None, clock: Callable[[], datetime] | None = None) -> None:
        """``speeds``: wide matrix, UTC DatetimeIndex, one column per approach id, km/h, NaN = missing.
        ``processed``: windows the analytics side has finished (back-pressure during the backfill burst)."""
        from traffic_intel.traffic.congestion.rules import RuleBasedCongestionClassifier

        self.cfg = cfg
        self.approaches = {a.id: a for a in network.approaches if a.id in speeds.columns}
        if not self.approaches:
            raise ValueError("no network approach has replay data (check approach_prefix / network file)")
        self.speeds = speeds[list(self.approaches)]
        seg = {s.id: s for s in network.segments}
        ff = {a.id: seg[a.road_segment_id].free_flow_speed_kmh or seg[a.road_segment_id].speed_limit_kmh
              for a in self.approaches.values()}
        params = dict(pipeline.congestion_classifier.params)
        self.classifier = RuleBasedCongestionClassifier(rules_file=params.get("rules_file", "congestion_rules.yaml"),
                                                        free_flow_speeds=ff)
        self.publisher = WorkerPublisher(publisher or NullPublisher(), source="sensor_replay:pems_bay")
        self.processed = processed
        self.clock = clock or (lambda: datetime.now(UTC))
        self.tz = cfg.timezone
        self.week0 = pd.Timestamp(cfg.replay_week, tz=self.tz).normalize()
        if self.week0.dayofweek != 0:
            raise ValueError(f"replay_week {cfg.replay_week} is not a Monday")
        self._stop = threading.Event()
        self._latest: dict[str, tuple[datetime, TrafficFeatures, TrafficState]] = {}
        self.windows = 0
        self.last_window_end: datetime | None = None
        self.last_recorded: datetime | None = None
        self.error: str | None = None
        self.run_id = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")

    # ------------------------------------------------------------------ time mapping
    def recorded_time(self, wall: datetime) -> pd.Timestamp:
        """Wall-clock instant -> the instant with the same weekday / local time in the replay week (UTC)."""
        local = pd.Timestamp(wall).tz_convert(self.tz)
        offset = local.tz_localize(None) - local.tz_localize(None).normalize() + timedelta(days=local.dayofweek)
        rec = (self.week0.tz_localize(None) + offset).tz_localize(self.tz, nonexistent="shift_forward",
                                                                   ambiguous=False)
        return rec.tz_convert("UTC")

    def _window_end(self, now: datetime) -> datetime:
        ts = int(now.timestamp()) // self.cfg.step_s * self.cfg.step_s
        return datetime.fromtimestamp(ts, UTC)

    # ------------------------------------------------------------------ publishing
    def build_window(self, approach_id: str, window_start: datetime) -> tuple[TrafficFeatures, TrafficState] | None:
        rec = self.recorded_time(window_start)
        try:
            v = self.speeds.at[rec, approach_id]
        except KeyError:
            return None
        if v is None or not np.isfinite(v):
            return None                                       # missing reading = gap (like a lost window)
        f = TrafficFeatures(
            approach_id=approach_id, camera_id=CAMERA_ID, window_start=window_start, window_s=float(self.cfg.step_s),
            vehicle_count=0, count_by_class={}, pcu_count=0.0, avg_speed_kmh=round(float(v), 2),
            speed_p85_kmh=None, density_veh_km=0.0, density_ratio=0.0, occupancy_pct=0.0, flow_veh_h=0.0,
            inflow=0, outflow=0, queue_length_m=0.0, stopped_vehicles=0, avg_gap_m=None, pedestrian_count=0,
            quality=1.0, median_speed_kmh=None,
            extra={"speed_only": 1.0, "recorded_at": rec.timestamp()})
        a = self.classifier.classify(f)
        a.explanation.append(f"PEMS-BAY loop detector, reading recorded {rec.tz_convert(self.tz):%a %Y-%m-%d %H:%M} "
                             "local time (replayed)")
        return f, build_traffic_state(f, a, self.approaches[approach_id])

    def publish_step(self, window_start: datetime) -> int:
        n = 0
        for aid in self.approaches:
            built = self.build_window(aid, window_start)
            if built is None:
                continue
            f, st = built
            self.publisher.publish(Topics.FEATURES_WINDOW, f)
            self.publisher.publish(Topics.TRAFFIC_STATE, st)
            self._latest[aid] = (st.timestamp, f, st)
            n += 1
        self.windows += n
        self.last_window_end = window_start + timedelta(seconds=self.cfg.step_s)
        self.last_recorded = self.recorded_time(window_start).to_pydatetime()
        return n

    def publish_live(self) -> int:
        now = self.clock()
        n = 0
        for end, _f, st in self._latest.values():
            if (now - end).total_seconds() > 2 * self.cfg.step_s:
                continue                                      # reading too old to be shown as live
            self.publisher.publish(Topics.FEATURES_LIVE, st.model_copy(update={
                "is_live": True, "timestamp": now, "window_start": now - timedelta(seconds=self.cfg.step_s)}))
            n += 1
        return n

    def _wait_for_analytics(self, target: int, timeout_s: float = 120.0) -> None:
        if self.processed is None:
            return
        t0 = time.monotonic()
        while self.processed() < target and not self._stop.is_set() and time.monotonic() - t0 < timeout_s:
            time.sleep(0.2)

    # ------------------------------------------------------------------ main loop
    def run(self, max_seconds: float | None = None) -> None:
        t_start = time.monotonic()
        step = timedelta(seconds=self.cfg.step_s)
        end = self._window_end(self.clock())
        # backfill: the last `backfill_steps` closed windows, oldest first, one step in flight at a time
        for k in range(self.cfg.backfill_steps, 0, -1):
            if self._stop.is_set():
                return
            before = self.windows
            self.publish_step(end - k * step)
            self._wait_for_analytics(before)
        log.info("sensor replay: %d stations, backfilled %d windows; replaying %s week of %s",
                 len(self.approaches), self.windows, self.cfg.dataset, self.week0.date())
        last_live = 0.0
        while not self._stop.is_set():
            if max_seconds is not None and time.monotonic() - t_start > max_seconds:
                break
            now = self.clock()
            if self._window_end(now) > end:
                end = self._window_end(now)
                self.publish_step(end - step)
            if time.monotonic() - last_live >= self.cfg.live_tick_s:
                self.publish_live()
                last_live = time.monotonic()
            self._stop.wait(0.5)
        self.publisher.close()

    def status(self) -> dict[str, Any]:
        return {"run_id": self.run_id, "last_frame_at": self.last_window_end, "frames": self.windows,
                "processing_fps": None, "detector": f"{len(self.approaches)} loop detectors (PEMS-BAY replay)",
                "device": None, "preview_path": None, "error": self.error, "model": None,
                "recorded_at": self.last_recorded.isoformat() if self.last_recorded else None}


def load_replay_speeds(cfg: SensorReplayConfig, approach_ids: list[str]) -> pd.DataFrame:
    """The replay week (+ one day before it for the backfill) as a wide km/h matrix keyed by approach id."""
    from traffic_intel.datasets.adapters import adapter_for

    week0 = pd.Timestamp(cfg.replay_week, tz=cfg.timezone)
    sensors = [a.removeprefix(cfg.approach_prefix) for a in approach_ids if a.startswith(cfg.approach_prefix)]
    adapter: Any = adapter_for(cfg.dataset)                    # a time-series adapter (has .load)
    ds = adapter.load(sensors=sensors, weather=False, holiday_country=None,
                                       start=str((week0 - pd.Timedelta(days=1)).tz_localize(None)),
                                       end=str((week0 + pd.Timedelta(days=8)).tz_localize(None)))
    wide = ds.frame.pivot(index="timestamp", columns="series_id", values="avg_speed_kmh")
    wide.columns = [cfg.approach_prefix + c.removeprefix(f"{cfg.dataset}_") for c in wide.columns]
    return wide
