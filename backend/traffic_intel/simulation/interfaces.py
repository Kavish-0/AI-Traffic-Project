"""What-if simulation interface (Module 8).

A deliberately simplified *static* model on a small NetworkX DiGraph built
from ``NetworkConfig``: nodes = road nodes/junctions, edges = directed road
segments with capacity and free-flow travel time. It is NOT a microscopic or
city-scale simulator.

Planned method (Phase 8): take observed edge flows as baseline; remove closed
edges / scale capacities; reassign the displaced flow over alternative paths
using BPR travel-time functions and the method of successive averages; map
resulting volume/capacity ratios to congestion probability with the same
thresholds as the congestion classifier. Assumptions are returned with every
result.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from traffic_intel.core.config import NetworkConfig
from traffic_intel.core.registry import Registry
from traffic_intel.core.schemas import SimulationRequest, SimulationResult, TrafficFeatures


class TrafficSimulator(ABC):
    @abstractmethod
    def load_network(self, network: NetworkConfig) -> None: ...

    @abstractmethod
    def what_if(
        self, request: SimulationRequest, baseline: dict[str, TrafficFeatures]
    ) -> SimulationResult:
        """``baseline`` maps road_segment_id -> latest observed features."""


SIMULATORS: Registry[TrafficSimulator] = Registry("simulator")
SIMULATORS.register_lazy("static_assignment", "traffic_intel.simulation.static_assignment")
