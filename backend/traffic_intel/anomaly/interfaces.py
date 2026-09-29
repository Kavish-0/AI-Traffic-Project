"""Anomaly & incident interfaces (Module 5).

Three layers, each producing ``AnomalySignal`` evidence:

1. ``TrackAnomalyDetector``  - streaming, per frame, per camera. Explicit
   behaviours: stopped vehicle outside queue zone, wrong-way, erratic or
   off-lane trajectory, converging-then-stopped tracks (collision-like).
2. ``SeriesAnomalyDetector`` - per closed window, per approach. Statistical
   (robust z-score vs time-of-day profile, EWMA control charts) and learned
   (Isolation Forest, autoencoder reconstruction error).
3. ``IncidentFuser`` - combines recent signals (and neighbouring approaches'
   state) into one ``IncidentAssessment`` with a probability, severity and
   location. It never claims certainty; it estimates likelihood.

Signal-cycle awareness: at signalised junctions a queue that builds every
red phase is normal. Series detectors compare against same-time-of-day
baselines and use windows >= one signal cycle; track detectors exclude
vehicles stopped inside QUEUE_ZONE while the queue is discharging normally.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

import pandas as pd

from traffic_intel.ai.model_store import TrainableModel
from traffic_intel.core.registry import Registry
from traffic_intel.core.schemas import (
    AnomalySignal,
    IncidentAssessment,
    TrafficFeatures,
)
from traffic_intel.core.types import TrackedFrame

if TYPE_CHECKING:
    from traffic_intel.computer_vision.zones import CameraGeometry
    from traffic_intel.tracking.trajectory import TrajectoryStore


@dataclass
class TrackContext:
    """What a track-level detector may use besides the frame: the camera's zone geometry
    (lanes with configured directions, queue zone, projector) and the trajectory store with
    per-track kinematics (speed, direction, stopped duration) already computed by the pipeline."""

    camera_id: str
    approach_id: str
    junction_id: str | None
    road_id: str | None
    geometry: CameraGeometry
    store: TrajectoryStore


class TrackAnomalyDetector(ABC):
    name: str

    @abstractmethod
    def update(self, tracked: TrackedFrame, ctx: TrackContext) -> list[AnomalySignal]:
        """Called every processed frame; returns new signals (usually [])."""


class SeriesAnomalyDetector(ABC):
    name: str

    @abstractmethod
    def score(self, approach_id: str, history: pd.DataFrame) -> list[AnomalySignal]:
        """Score the NEWEST window of ``history`` against the windows before it.

        ``history``: one row per closed window of one approach, oldest first, columns
        ``timestamp`` + TrafficFeatures numeric fields (avg_speed_kmh, density_ratio,
        occupancy_pct, queue_length_m, flow_veh_h, inflow, outflow, stopped_vehicles,
        quality). Detectors are stateless between calls: all state is in ``history``
        and in fitted parameters, so results are reproducible from stored windows."""


class LearnedSeriesAnomalyDetector(SeriesAnomalyDetector, TrainableModel):
    """Isolation Forest / autoencoder: fitted offline on 'normal' periods."""

    @abstractmethod
    def fit(self, train: pd.DataFrame) -> None: ...


class IncidentFuser(ABC):
    name: str

    @abstractmethod
    def fuse(
        self,
        approach_id: str,
        now: datetime,
        signals: Sequence[AnomalySignal],
        features: TrafficFeatures | None,
        neighbour_features: dict[str, TrafficFeatures],
        open_incident: IncidentAssessment | None,
    ) -> IncidentAssessment | None:
        """Return the current incident hypothesis for the approach, or None.

        Must apply hysteresis: open above ``p_open``, close only below
        ``p_close`` for N windows, and update rather than duplicate
        ``open_incident``.
        """


TRACK_ANOMALY_DETECTORS: Registry[TrackAnomalyDetector] = Registry("track_anomaly_detector")
SERIES_ANOMALY_DETECTORS: Registry[SeriesAnomalyDetector] = Registry("series_anomaly_detector")
INCIDENT_FUSERS: Registry[IncidentFuser] = Registry("incident_fuser")

for _name in ("stopped_vehicle", "wrong_way", "trajectory_rules"):
    TRACK_ANOMALY_DETECTORS.register_lazy(_name, "traffic_intel.anomaly.detectors.track_rules")
for _name in ("sudden_slowdown", "queue_growth", "density_zscore", "flow_imbalance", "ewma", "zscore"):
    SERIES_ANOMALY_DETECTORS.register_lazy(_name, "traffic_intel.anomaly.detectors.statistical")
SERIES_ANOMALY_DETECTORS.register_lazy("isolation_forest", "traffic_intel.anomaly.detectors.learned")
SERIES_ANOMALY_DETECTORS.register_lazy("autoencoder", "traffic_intel.anomaly.detectors.learned")
INCIDENT_FUSERS.register_lazy("weighted_logistic", "traffic_intel.anomaly.fusion")
