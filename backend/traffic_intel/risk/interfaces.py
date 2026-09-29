"""Risk engine interface.

Combines what is happening now, what is forecast and what looks abnormal into one
explainable score per approach (0..100) with a level and reason codes. The default
implementation (``risk/weighted.py``) is a documented weighted sum - see
docs/INCIDENT_INTELLIGENCE.md for the formula.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from traffic_intel.core.registry import Registry
from traffic_intel.core.schemas import (
    AnomalySignal,
    CongestionAssessment,
    Forecast,
    IncidentAssessment,
    RiskAssessment,
    TrafficFeatures,
)


class RiskEngine(ABC):
    @abstractmethod
    def assess(
        self,
        features: TrafficFeatures | None,
        congestion: CongestionAssessment | None,
        forecast: Forecast | None,
        incident: IncidentAssessment | None,
        *,
        signals: Sequence[AnomalySignal] = (),
        road_importance: float | None = None,
        free_flow_kmh: float | None = None,
        junction_id: str | None = None,
        road_id: str | None = None,
    ) -> RiskAssessment: ...


RISK_ENGINES: Registry[RiskEngine] = Registry("risk_engine")
RISK_ENGINES.register_lazy("weighted", "traffic_intel.risk.weighted")
