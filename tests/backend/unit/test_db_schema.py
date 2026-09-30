from __future__ import annotations

from traffic_intel.database import models  # noqa: F401
from traffic_intel.database.base import Base


def test_expected_tables() -> None:
    expected = {
        "junction", "road_node", "road_segment", "approach", "camera",
        "camera_calibration", "camera_zone", "traffic_observation",
        "context_observation", "congestion_state", "forecast", "risk_assessment",
        "anomaly_event", "incident", "model_registry", "simulation_scenario",
    }
    assert expected <= set(Base.metadata.tables)
