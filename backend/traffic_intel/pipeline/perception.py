"""Per-camera perception + traffic-state pipeline (synchronous, one camera).

    frame -> preprocess -> detect -> track -> ground projection
          -> feature engine (trajectories, samples) -> windows -> congestion -> TrafficState
          -> track anomaly detectors (stopped vehicle, wrong-way, erratic) -> AnomalySignal

All stages are built from ``configs/pipeline.yaml`` through the registries, so
every model is replaceable; any stage can also be injected (tests, synthetic
data, a detector shared by several cameras).

This class does no I/O: callers (CLI replay, CameraWorker) decide where
results go.
"""

from __future__ import annotations

import dataclasses
import inspect
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from traffic_intel.anomaly.interfaces import TRACK_ANOMALY_DETECTORS, TrackAnomalyDetector, TrackContext
from traffic_intel.computer_vision.calibration import GROUND_PROJECTORS, GroundProjector
from traffic_intel.computer_vision.interfaces import DETECTORS, PREPROCESSORS, Detector, FramePreprocessor
from traffic_intel.computer_vision.zones import CameraGeometry
from traffic_intel.core.config import ApproachConfig, CameraConfig, NetworkConfig, PipelineConfig
from traffic_intel.core.registry import ComponentSpec
from traffic_intel.core.schemas import AnomalySignal, TrackSummary, TrafficFeatures, TrafficState
from traffic_intel.core.types import Detections, Frame, TrackedFrame
from traffic_intel.tracking.interfaces import TRACKERS, Tracker
from traffic_intel.tracking.trajectory import TrackedObject, TrajectoryConfig, TrajectoryStore
from traffic_intel.traffic.interfaces import (
    CONGESTION_CLASSIFIERS,
    FEATURE_ENGINES,
    CongestionClassifier,
    TrafficFeatureEngine,
)
from traffic_intel.traffic.state import build_traffic_state

log = logging.getLogger(__name__)


@dataclass
class PerceptionResult:
    frame: Frame
    tracked: TrackedFrame
    window_features: list[TrafficFeatures] = field(default_factory=list)
    window_states: list[TrafficState] = field(default_factory=list)
    live_state: TrafficState | None = None
    finished_tracks: list[TrackSummary] = field(default_factory=list)
    signals: list[AnomalySignal] = field(default_factory=list)      # track-level anomaly evidence


class PerceptionPipeline:
    def __init__(
        self,
        camera: CameraConfig,
        approach: ApproachConfig,
        *,
        detector: Detector,
        tracker: Tracker,
        projector: GroundProjector | None,
        geometry: CameraGeometry | None,
        feature_engine: TrafficFeatureEngine | None,
        classifier: CongestionClassifier | None,
        live_classifier: CongestionClassifier | None = None,
        trajectory_store: TrajectoryStore | None = None,
        preprocessors: Sequence[FramePreprocessor] = (),
        live_tick_s: float = 1.0,
        track_detectors: Sequence[TrackAnomalyDetector] = (),
        road_id: str | None = None,
    ) -> None:
        self.camera = camera
        self.approach = approach
        self.detector = detector
        self.tracker = tracker
        self.projector = projector
        self.geometry = geometry
        self.engine = feature_engine
        self.classifier = classifier
        self.live_classifier = live_classifier or classifier
        self.preprocessors = list(preprocessors)
        self.live_tick_s = live_tick_s
        # Perception-only mode (camera not calibrated yet): detection, tracking, trajectories in
        # pixels, annotation and Parquet still work; no metric features, no windows, no live values.
        self.perception_only = feature_engine is None
        self.store: TrajectoryStore = (
            getattr(feature_engine, "store", None) or trajectory_store or TrajectoryStore()
        )
        self._last_live: float | None = None
        # Track anomaly detectors need metric geometry, so they only run on calibrated cameras.
        self.track_detectors = list(track_detectors) if geometry is not None else []
        self.track_ctx = TrackContext(
            camera_id=camera.id, approach_id=approach.id, junction_id=approach.junction_id,
            road_id=road_id or approach.road_segment_id, geometry=geometry, store=self.store,
        ) if geometry is not None else None
        self.latency_ms: dict[str, float] = {}      # running mean per stage
        self.frames_processed = 0

    # ------------------------------------------------------------------ #
    @classmethod
    def build(
        cls,
        camera: CameraConfig,
        pipeline: PipelineConfig,
        network: NetworkConfig,
        frame_size: tuple[int, int],
        fps: float,
        *,
        detector: Detector | None = None,
        tracker: Tracker | None = None,
    ) -> PerceptionPipeline:
        approach = next((a for a in network.approaches if a.id == camera.approach_id), None)
        if approach is None:
            raise ValueError(f"camera {camera.id}: approach '{camera.approach_id}' not in network {network.name}")
        segment = next(s for s in network.segments if s.id == approach.road_segment_id)
        free_flow = {approach.id: segment.free_flow_speed_kmh or segment.speed_limit_kmh}

        preprocessors = [PREPROCESSORS.create(spec, camera=camera) for spec in pipeline.preprocessors]
        detector = detector or DETECTORS.create(pipeline.detector)
        tracker = tracker or TRACKERS.create(pipeline.tracker, fps=fps)
        fe_params = dict(pipeline.feature_engine.params)
        live_tick_s = float(fe_params.pop("live_tick_s", 1.0))
        if camera.calibration is None:
            log.warning(
                "camera %s has no calibration -> PERCEPTION-ONLY mode: detections, tracks, annotated video "
                "and Parquet are produced, but no metric features, live values or 60 s windows "
                "(add `calibration:` to the camera YAML; docs/TRAFFIC_STATE_ESTIMATION.md section 8)",
                camera.id,
            )
            traj_cfg = TrajectoryConfig(**fe_params.get("trajectory", {}))
            return cls(
                camera, approach, detector=detector, tracker=tracker, projector=None, geometry=None,
                feature_engine=None, classifier=None, preprocessors=preprocessors, live_tick_s=live_tick_s,
                trajectory_store=TrajectoryStore(traj_cfg),
            )
        projector = GROUND_PROJECTORS.create(
            pipeline.ground_projector, calibration=camera.calibration, frame_size=frame_size
        )
        geometry = CameraGeometry(camera, frame_size, projector)
        engine = FEATURE_ENGINES.create(
            ComponentSpec(type=pipeline.feature_engine.type, params=fe_params),
            camera=camera, geometry=geometry, fps=fps,
        )
        classifier = _make_classifier(pipeline.congestion_classifier, free_flow, stateful=True)
        live_classifier = _make_classifier(pipeline.congestion_classifier, free_flow, stateful=False)
        track_detectors = [TRACK_ANOMALY_DETECTORS.create(spec) for spec in pipeline.track_anomaly_detectors]
        return cls(
            camera, approach, detector=detector, tracker=tracker, projector=projector, geometry=geometry,
            feature_engine=engine, classifier=classifier, live_classifier=live_classifier,
            preprocessors=preprocessors, live_tick_s=live_tick_s, track_detectors=track_detectors,
        )

    # ------------------------------------------------------------------ #
    def _time(self, stage: str, start: float) -> float:
        now = time.perf_counter()
        ms = (now - start) * 1000.0
        prev = self.latency_ms.get(stage)
        self.latency_ms[stage] = ms if prev is None else prev + (ms - prev) / (self.frames_processed + 1)
        return now

    def process(self, frame: Frame) -> PerceptionResult:
        t0 = time.perf_counter()
        # Preprocessors work on a shallow copy: the detector sees e.g. the ROI-masked image,
        # while the caller's frame (annotation, recording) keeps the original pixels.
        seen = dataclasses.replace(frame)
        for p in self.preprocessors:
            seen = p.process(seen)
        t1 = self._time("preprocess", t0)
        detections = self.detector.detect([seen])[0]
        self._time("detect", t1)
        return self.process_detections(frame, detections)

    def process_detections(self, frame: Frame, detections: Detections) -> PerceptionResult:
        """Entry point when detection ran elsewhere (e.g. batched across cameras)."""
        t0 = time.perf_counter()
        tracks = self.tracker.update(detections, frame)
        t1 = self._time("track", t0)
        if self.projector is not None:
            tracks.ground_xy = self.projector.image_to_ground(tracks.anchor_points())
        tracked = TrackedFrame(
            camera_id=self.camera.id, frame_index=frame.index, timestamp=frame.timestamp,
            image_shape=frame.shape, detections=detections, tracks=tracks,
        )
        if self.engine is None:
            t = frame.timestamp.timestamp()
            self.store.update(t, tracks, tracks.anchor_points(), None, np.zeros(len(tracks), dtype=bool))
            self.store.expire(t)
            self._time("features", t1)
            tracked.processing_ms = dict(self.latency_ms)
            self.frames_processed += 1
            return PerceptionResult(frame=frame, tracked=tracked)
        self.engine.update(tracked)
        signals: list[AnomalySignal] = []
        if self.track_detectors and self.track_ctx is not None:
            t_a = time.perf_counter()
            for d in self.track_detectors:
                signals.extend(d.update(tracked, self.track_ctx))
            self._time("anomaly", t_a)
        windows = self.engine.close_windows(frame.timestamp)
        result = PerceptionResult(frame=frame, tracked=tracked, window_features=windows, signals=signals)
        result.window_states = [self._state(f, live=False) for f in windows]
        t = frame.timestamp.timestamp()
        if self._last_live is None or t - self._last_live >= self.live_tick_s:
            self._last_live = t
            result.live_state = self._state(self.engine.live_features(frame.timestamp), live=True)
        result.finished_tracks = self.engine.finished_tracks()
        self._time("features", t1)
        tracked.processing_ms = dict(self.latency_ms)
        self.frames_processed += 1
        return result

    def flush(self) -> tuple[list[TrafficFeatures], list[TrafficState], list[TrackSummary]]:
        if self.engine is None:
            self.store.flush()
            return [], [], []
        feats = self.engine.flush()
        return feats, [self._state(f, live=False) for f in feats], self.engine.finished_tracks()

    def _state(self, features: TrafficFeatures, *, live: bool) -> TrafficState:
        clf = self.live_classifier if live else self.classifier
        assert clf is not None
        return build_traffic_state(features, clf.classify(features), self.approach, live=live)

    def tracked_objects(self, result: PerceptionResult, tail: int = 30) -> list[TrackedObject]:
        return self.store.tracked_objects(result.tracked.tracks.track_id.tolist(), result.frame.timestamp, tail)


def _make_classifier(spec: ComponentSpec, free_flow: dict[str, float], *, stateful: bool) -> CongestionClassifier:
    """Classifiers receive ``free_flow_speeds`` and ``stateful`` if they accept them."""
    params: dict[str, Any] = dict(spec.params)
    factory = CONGESTION_CLASSIFIERS.get(spec.type)
    sig = inspect.signature(factory)
    accepts_any = any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values())
    for name, value in (("free_flow_speeds", free_flow), ("stateful", stateful)):
        if accepts_any or name in sig.parameters:
            params.setdefault(name, value)
    return factory(**params)
