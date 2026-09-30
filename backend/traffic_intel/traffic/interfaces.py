"""Traffic state interfaces (Modules 2 & 3).

``TrafficFeatureEngine`` is the bridge from frame-rate perception to
window-rate analytics. It is stateful per camera/approach: it keeps short
per-track histories in ground coordinates, computes speeds, zone crossings
and occupancy, and closes fixed windows.

``CongestionClassifier`` turns one window (plus context) into a level. The
rule-based baseline is the reference every ML classifier is compared to.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

from traffic_intel.core.registry import Registry
from traffic_intel.core.schemas import (
    CongestionAssessment,
    ContextFeatures,
    TrackSummary,
    TrafficFeatures,
)
from traffic_intel.core.types import TrackedFrame


class TrafficFeatureEngine(ABC):
    @abstractmethod
    def update(self, tracked: TrackedFrame) -> None:
        """Ingest one frame. ``tracked.tracks.ground_xy`` must be populated."""

    @abstractmethod
    def live_features(self, now: datetime) -> TrafficFeatures:
        """Rolling features over the last few seconds for the live dashboard
        (not persisted, not used for ML)."""

    @abstractmethod
    def close_windows(self, until: datetime) -> list[TrafficFeatures]:
        """Emit every fixed window (default 60 s) that ended at or before
        ``until``. Called on each frame; returns [] most of the time."""

    @abstractmethod
    def finished_tracks(self) -> list[TrackSummary]:
        """Trajectories whose tracks were lost since the last call."""

    def flush(self) -> list[TrafficFeatures]:
        """End of stream: finish open trajectories, emit the final partial window."""
        return []


class CongestionClassifier(ABC):
    name: str

    @abstractmethod
    def classify(
        self, features: TrafficFeatures, context: ContextFeatures | None = None
    ) -> CongestionAssessment: ...


FEATURE_ENGINES: Registry[TrafficFeatureEngine] = Registry("feature_engine")
CONGESTION_CLASSIFIERS: Registry[CongestionClassifier] = Registry("congestion_classifier")

FEATURE_ENGINES.register_lazy("default", "traffic_intel.traffic.features.engine")
CONGESTION_CLASSIFIERS.register_lazy("rule_based", "traffic_intel.traffic.congestion.rules")
# Learned congestion classifiers (random_forest / xgboost vs the rule baseline) are Phase 7 - not implemented,
# so they are deliberately NOT registered (a registry must never advertise components that cannot load).
