"""Local full-system runtime (``python -m traffic_intel.pipeline run``).

    camera workers (threads)  --in-memory bus-->  analytics worker  --> TrafficStore (database)
      video -> YOLO -> tracker        live values (1 s)   anomaly detection               |
      -> features -> 60 s windows     windows (60 s)      incident fusion, forecast, risk |
      -> preview JPEG (1/s)           anomaly signals                                     v
                                                                  API process (no torch) reads it,
                                                                  WebSocket pushes to the dashboard

Only aggregates travel: per-frame detections and tracks stay in the camera worker (Parquet if
``record_parquet``). This is the existing single-process dev mode of docs/ARCHITECTURE.md with the
database as the hand-over to the API; Redis Streams and separate camera processes remain future scope.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator

from traffic_intel.core.config import (
    CameraConfig,
    NetworkConfig,
    PipelineConfig,
    get_settings,
    load_camera,
    load_network,
    load_pipeline_config,
    load_yaml,
    resolve_path,
)
from traffic_intel.core.schemas import TrafficState
from traffic_intel.pipeline.bus import InMemoryEventBus, Topics
from traffic_intel.pipeline.sensor_worker import SensorReplayConfig

log = logging.getLogger(__name__)


class CameraRef(BaseModel):
    model_config = ConfigDict(extra="forbid")
    camera: str | None = None                # id from configs/cameras/*.yaml
    synthetic: str | None = None             # synthetic scenario (test fixture)
    camera_id: str | None = None
    approach_id: str | None = None
    name: str | None = None
    duration_s: float | None = None
    seed: int | None = None
    arrival_rate_veh_h_per_lane: float | None = None
    process_fps: float | None = None         # override the camera's analysed frames/s (e.g. CPU-only profiles)

    @model_validator(mode="after")
    def _one_source(self) -> CameraRef:
        if bool(self.camera) == bool(self.synthetic):
            raise ValueError("a runtime camera entry needs exactly one of `camera` or `synthetic`")
        if self.synthetic and not (self.camera_id and self.approach_id):
            raise ValueError("synthetic camera entries need `camera_id` and `approach_id` (no defaults)")
        return self


class Profile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    description: str = ""
    cameras: list[CameraRef] = []
    sensors: SensorReplayConfig | None = None    # recorded road-sensor data replayed as live (sensor_worker.py)
    pipeline_file: str | None = None             # the profile needs this TI_PIPELINE_FILE (checked at start)

    @model_validator(mode="after")
    def _has_source(self) -> Profile:
        if not self.cameras and self.sensors is None:
            raise ValueError("a runtime profile needs `cameras` and/or `sensors`")
        return self


class LiveReplay(BaseModel):
    realtime: bool = True
    loop: bool = True
    wall_clock: bool = True


class PreviewCfg(BaseModel):
    enabled: bool = True
    interval_s: float = 1.0
    max_width: int = 960
    dir: str = "runs/live_preview"


class WeatherCfg(BaseModel):
    enabled: bool = True
    interval_s: float = 900.0


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    default_profile: str
    profiles: dict[str, Profile]
    live_replay: LiveReplay = LiveReplay()
    record_parquet: bool = False
    heartbeat_s: float = 2.0
    live_persist_s: float = 1.0
    forecast: bool = True
    preview: PreviewCfg = PreviewCfg()
    weather: WeatherCfg = WeatherCfg()


def load_runtime_config(path: Path | None = None) -> RuntimeConfig:
    return RuntimeConfig.model_validate(load_yaml(path or get_settings().config_dir / "runtime.yaml"))


@dataclass
class WorkerHandle:
    camera: CameraConfig
    source_kind: str
    worker: Any = None
    thread: threading.Thread | None = None
    error: str | None = None
    state: str = "starting"
    info: dict[str, Any] = field(default_factory=dict)


def _find_camera(camera_id: str) -> CameraConfig:
    for path in sorted((get_settings().config_dir / "cameras").glob("*.yaml")):
        cam = load_camera(path)
        if cam.id == camera_id:
            return cam
    raise ValueError(f"camera '{camera_id}' not found in configs/cameras/")


def _scene_config(ref: CameraRef) -> Any:
    from traffic_intel.synthetic.scene import SCENARIOS

    base = SCENARIOS[ref.synthetic or "default"]
    upd = {k: v for k, v in (("duration_s", ref.duration_s), ("seed", ref.seed),
                             ("arrival_rate_veh_h_per_lane", ref.arrival_rate_veh_h_per_lane)) if v is not None}
    return base.model_copy(update={"name": f"rt_{ref.camera_id}", **upd})


def prepare_synthetic_scenes(refs: list[CameraRef]) -> None:
    """Register (cached / parallel-generated) scenes for the profile's synthetic cameras."""
    from traffic_intel.synthetic.cache import load_or_generate
    from traffic_intel.synthetic.scene import SCENARIOS
    from traffic_intel.synthetic.sources import register_scene

    cfgs = [_scene_config(r) for r in refs if r.synthetic]
    for c in cfgs:
        SCENARIOS[c.name] = c
    for name, scene in load_or_generate(cfgs).items():
        register_scene(name, scene)


def _synthetic(ref: CameraRef) -> tuple[CameraConfig, Any]:
    from traffic_intel.synthetic.scene import SCENARIOS
    from traffic_intel.synthetic.sources import SyntheticDetector, synthetic_camera

    name = f"rt_{ref.camera_id}"
    SCENARIOS.setdefault(name, _scene_config(ref))
    assert ref.camera_id and ref.approach_id                      # enforced by CameraRef validation
    cam = synthetic_camera(name, camera_id=ref.camera_id, approach_id=ref.approach_id)
    cam = cam.model_copy(update={"name": ref.name or cam.name})
    return cam, SyntheticDetector(name)


class Runtime:
    def __init__(self, profile: str | None = None, *, cfg: RuntimeConfig | None = None,
                 pipeline: PipelineConfig | None = None, store: Any = None, max_seconds: float | None = None) -> None:
        from traffic_intel.database.store import TrafficStore

        self.cfg = cfg or load_runtime_config()
        self.profile_name = profile or self.cfg.default_profile
        if self.profile_name not in self.cfg.profiles:
            raise ValueError(f"unknown runtime profile '{self.profile_name}' ({', '.join(self.cfg.profiles)})")
        self.profile = self.cfg.profiles[self.profile_name]
        s = get_settings()
        if pipeline is None and self.profile.pipeline_file and self.profile.pipeline_file != s.pipeline_file:
            raise ValueError(
                f"profile '{self.profile_name}' runs with {self.profile.pipeline_file}, but TI_PIPELINE_FILE is "
                f"{s.pipeline_file}. Set it for BOTH the pipeline and the API, e.g. in PowerShell: "
                f"$env:TI_PIPELINE_FILE=\"{self.profile.pipeline_file}\"")
        self.pipeline = pipeline or load_pipeline_config(s)
        self.network: NetworkConfig = load_network(s.config_dir / self.pipeline.network_file)
        self.store = store or TrafficStore()
        self.max_seconds = max_seconds
        self.handles: list[WorkerHandle] = []
        self.sensors: Any = None                      # SensorReplayWorker when the profile has `sensors`
        self.sensor_thread: threading.Thread | None = None
        self.sensor_state = "starting"
        self._stop = threading.Event()
        self._last_live: dict[str, float] = {}

    # ------------------------------------------------------------------ workers
    def _build(self, ref: CameraRef, loop: asyncio.AbstractEventLoop, bus: InMemoryEventBus) -> WorkerHandle:
        from traffic_intel.computer_vision.sources import open_video_source
        from traffic_intel.computer_vision.sources.live_replay import LiveReplaySource
        from traffic_intel.pipeline.camera_worker import CameraWorker
        from traffic_intel.pipeline.publish import BusPublisher

        detector = None
        if ref.synthetic:
            camera, detector = _synthetic(ref)
            kind = "synthetic"
        else:
            camera = _find_camera(ref.camera or "")
            if ref.process_fps is not None:
                camera = camera.model_copy(update={"process_fps": ref.process_fps})
            kind = camera.source_type if camera.source_type != "auto" else (
                "webcam" if camera.source.startswith("webcam:") else
                "stream" if "://" in camera.source else "file")
        handle = WorkerHandle(camera, kind)
        preview = (resolve_path(self.cfg.preview.dir) / f"{camera.id}.jpg") if self.cfg.preview.enabled else None
        if kind == "synthetic":
            from traffic_intel.synthetic.sources import SyntheticVideoSource

            src: Any = SyntheticVideoSource(camera, render=self.cfg.preview.enabled)
        else:
            src = open_video_source(camera.model_copy(update={"realtime": False, "loop": False}))
        lr = self.cfg.live_replay
        if kind in ("file", "synthetic") and (lr.loop or lr.realtime or lr.wall_clock):
            # wall_clock: stamped from the moment this camera's first frame is read
            src = LiveReplaySource(src, loop=lr.loop, realtime=lr.realtime,
                                   start=None if lr.wall_clock else datetime.now(UTC))
        self.store.sync_network(self.network, [camera])
        handle.worker = CameraWorker.from_config(
            camera, self.pipeline, self.network, publisher=BusPublisher(bus, loop), record=self.cfg.record_parquet,
            detector=detector, source=src, preview_path=preview, preview_interval_s=self.cfg.preview.interval_s)
        handle.worker.outputs.preview_max_width = self.cfg.preview.max_width
        return handle

    def _run_worker(self, h: WorkerHandle) -> None:
        h.state = "running"
        try:
            h.worker.run(max_seconds=self.max_seconds)
            h.state = "stopped"
        except Exception as exc:  # noqa: BLE001 - reported to the dashboard, never swallowed silently
            h.state, h.error = "error", f"{type(exc).__name__}: {exc}"
            log.exception("camera %s failed", h.camera.id)
        finally:
            self._heartbeat(h)

    def _heartbeat(self, h: WorkerHandle) -> None:
        st: dict[str, Any] = h.worker.status() if h.worker is not None else {}
        try:
            self.store.upsert_camera_status(
                h.camera.id, h.camera.approach_id, state=h.state, run_id=st.get("run_id"),
                last_frame_at=st.get("last_frame_at"), frames=int(st.get("frames", 0)),
                processing_fps=st.get("processing_fps"), detector=st.get("detector"), device=st.get("device"),
                source_kind=h.source_kind, preview_path=st.get("preview_path"), error=h.error or st.get("error"),
                details={"name": h.camera.name, "source": h.camera.source if h.source_kind != "stream" else "stream",
                         "model": st.get("model"), "latency_ms_mean": st.get("latency_ms_mean"),
                         "perception_only": st.get("perception_only"), "profile": self.profile_name,
                         "frame_size": st.get("frame_size")})
        except Exception:  # noqa: BLE001
            log.exception("heartbeat for %s failed", h.camera.id)

    # ------------------------------------------------------------------ async tasks
    async def _persist_live(self, bus: InMemoryEventBus) -> None:
        async for msg in bus.subscribe(Topics.FEATURES_LIVE):
            st = TrafficState.model_validate(msg.payload)
            now = datetime.now(UTC).timestamp()
            if now - self._last_live.get(st.approach_id, 0.0) < self.cfg.live_persist_s:
                continue
            self._last_live[st.approach_id] = now
            try:
                await asyncio.to_thread(self.store.upsert_live_state, st)
            except Exception:  # noqa: BLE001
                log.exception("live state persist failed for %s", st.approach_id)

    async def _heartbeats(self) -> None:
        while not self._stop.is_set():
            for h in self.handles:
                await asyncio.to_thread(self._heartbeat, h)
            if self.sensors is not None:
                await asyncio.to_thread(self._sensor_heartbeat)
            await asyncio.sleep(self.cfg.heartbeat_s)

    def _sensor_heartbeat(self) -> None:
        from traffic_intel.pipeline.sensor_worker import CAMERA_ID

        w = self.sensors
        st = w.status()
        try:
            self.store.upsert_camera_status(
                CAMERA_ID, next(iter(w.approaches)), state=self.sensor_state, run_id=st["run_id"],
                last_frame_at=st["last_frame_at"], frames=int(st["frames"]), processing_fps=None,
                detector=st["detector"], device=None, source_kind="sensor_replay", preview_path=None,
                error=w.error, details={"name": f"{w.cfg.dataset.upper().replace('_', '-')} loop detectors "
                                                "(recorded, replayed)",
                                        "source": w.cfg.dataset, "profile": self.profile_name,
                                        "replay_week": w.cfg.replay_week, "recorded_at": st["recorded_at"],
                                        "stations": len(w.approaches)})
        except Exception:  # noqa: BLE001
            log.exception("heartbeat for sensor replay failed")

    def _run_sensors(self) -> None:
        self.sensor_state = "running"
        try:
            self.sensors.run(max_seconds=self.max_seconds)
            self.sensor_state = "stopped"
        except Exception as exc:  # noqa: BLE001 - reported to the dashboard
            self.sensor_state, self.sensors.error = "error", f"{type(exc).__name__}: {exc}"
            log.exception("sensor replay failed")
        finally:
            self._sensor_heartbeat()

    def _build_sensors(self, loop: asyncio.AbstractEventLoop, bus: InMemoryEventBus, analytics: Any) -> None:
        from traffic_intel.pipeline.publish import BusPublisher
        from traffic_intel.pipeline.sensor_worker import SensorReplayWorker, load_replay_speeds

        cfg = self.profile.sensors
        assert cfg is not None
        speeds = load_replay_speeds(cfg, [a.id for a in self.network.approaches])
        self.sensors = SensorReplayWorker(cfg, self.network, self.pipeline, speeds,
                                          publisher=BusPublisher(bus, loop), processed=lambda: analytics.updates)

    async def _weather(self) -> None:
        from traffic_intel.context.interfaces import WEATHER_PROVIDERS

        provider = WEATHER_PROVIDERS.create(self.pipeline.weather_provider)
        j = self.network.junctions[0]
        while not self._stop.is_set():
            try:
                ctx = await asyncio.to_thread(provider.observed, (j.location[0], j.location[1]), datetime.now(UTC))
                await asyncio.to_thread(self.store.add_context, self.network.name, ctx,
                                        self.pipeline.weather_provider.type)
                log.info("weather: %s %.1f C", ctx.weather, ctx.temperature_c or float("nan"))
            except Exception as exc:  # noqa: BLE001 - weather is optional context
                log.warning("weather unavailable (%s) - continuing without it", exc)
            await asyncio.sleep(self.cfg.weather.interval_s)

    def _sync_models(self) -> None:
        from traffic_intel.ai.model_store import FileModelStore

        fs = FileModelStore(get_settings().models_dir)
        metas = fs.list()
        prod = {}
        for md in metas:
            with contextlib.suppress(FileNotFoundError):
                prod[(md.task.value, md.name)] = fs.resolve_version(md.task, md.name, "production")
        n = self.store.sync_models(metas, prod)
        log.info("model registry: %d model versions mirrored to the database", n)

    # ------------------------------------------------------------------ main
    async def run_async(self) -> None:
        from traffic_intel.pipeline.analytics_worker import AnalyticsWorker

        loop = asyncio.get_running_loop()
        bus = InMemoryEventBus(max_queue=5000)
        log.info("runtime profile '%s': %d camera(s); database %s", self.profile_name, len(self.profile.cameras),
                 self.store.url)
        await asyncio.to_thread(self._sync_models)
        analytics = AnalyticsWorker(self.network, self.pipeline, bus, store=self.store, with_forecast=self.cfg.forecast)
        if analytics.engine.forecaster is None:
            log.warning("forecasting disabled or no PRODUCTION model - dashboard will show 'forecast unavailable'")
        else:
            log.info("forecasting model loaded: %s", analytics.engine.forecaster.model_info)
        tasks = [asyncio.create_task(analytics.run(), name="analytics"),
                 asyncio.create_task(self._persist_live(bus), name="persist_live"),
                 asyncio.create_task(self._heartbeats(), name="heartbeats")]
        if self.cfg.weather.enabled and self.network.geo_referenced:
            tasks.append(asyncio.create_task(self._weather(), name="weather"))
        elif self.cfg.weather.enabled:
            log.info("weather off: network %s has no real location (geo_referenced: false)", self.network.name)
        await asyncio.sleep(0)
        await asyncio.to_thread(prepare_synthetic_scenes, self.profile.cameras)
        for ref in self.profile.cameras:
            try:
                h = await asyncio.to_thread(self._build, ref, loop, bus)
            except Exception as exc:  # noqa: BLE001 - e.g. missing video file / no YOLO weights
                cam_id = ref.camera or ref.camera_id or "?"
                log.error("camera %s not started: %s", cam_id, exc)
                self._report_unstartable(ref, exc)
                continue
            h.thread = threading.Thread(target=self._run_worker, args=(h,), name=f"camera-{h.camera.id}", daemon=True)
            self.handles.append(h)
            h.thread.start()
            log.info("camera %s (%s) -> approach %s started", h.camera.id, h.source_kind, h.camera.approach_id)
        if self.profile.sensors is not None:
            try:
                await asyncio.to_thread(self._build_sensors, loop, bus, analytics)
                self.sensor_thread = threading.Thread(target=self._run_sensors, name="sensor-replay", daemon=True)
                self.sensor_thread.start()
            except Exception as exc:  # noqa: BLE001 - e.g. dataset not on disk
                log.error("sensor replay not started: %s", exc)

        def alive() -> bool:
            return any(t is not None and t.is_alive()
                       for t in [*(h.thread for h in self.handles), self.sensor_thread])

        try:
            while not self._stop.is_set() and alive():
                await asyncio.sleep(0.5)
        finally:
            self.stop()
            for h in self.handles:
                if h.thread:
                    await asyncio.to_thread(h.thread.join, 15)
            if self.sensor_thread:
                await asyncio.to_thread(self.sensor_thread.join, 15)
            await asyncio.sleep(0.5)                      # let analytics drain the last windows
            for t in tasks:
                t.cancel()
            log.info("runtime stopped")

    def _report_unstartable(self, ref: CameraRef, exc: Exception) -> None:
        try:
            cam = _find_camera(ref.camera) if ref.camera else None
            if cam is not None:
                self.store.upsert_camera_status(cam.id, cam.approach_id, state="error", source_kind="file",
                                                error=f"not started: {type(exc).__name__}: {exc}",
                                                details={"name": cam.name, "profile": self.profile_name})
        except Exception:  # noqa: BLE001
            log.exception("could not record camera start failure")

    def stop(self) -> None:
        self._stop.set()
        for h in self.handles:
            if h.worker is not None:
                h.worker._stop.set()
        if self.sensors is not None:
            self.sensors._stop.set()

    def run(self) -> None:
        try:
            asyncio.run(self.run_async())
        except KeyboardInterrupt:
            log.info("interrupted")
            self.stop()

