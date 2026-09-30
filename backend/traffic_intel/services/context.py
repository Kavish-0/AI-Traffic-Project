"""Shared application context for the service layer (API use-cases).

Holds what every use-case needs - the store (database), the road network and camera
configuration (configs/*.yaml are the source of truth) and settings. Built once per API process;
nothing here imports torch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from traffic_intel.core.config import (
    AppSettings,
    CameraConfig,
    NetworkConfig,
    get_settings,
    load_camera,
    load_network,
    load_pipeline_config,
)
from traffic_intel.database.store import TrafficStore

if TYPE_CHECKING:
    from traffic_intel.simulation.interfaces import TrafficSimulator


@dataclass
class ServiceContext:
    store: TrafficStore
    network: NetworkConfig
    cameras: dict[str, CameraConfig]
    settings: AppSettings
    ws_clients: int = 0
    _simulator: TrafficSimulator | None = field(default=None, repr=False)

    @classmethod
    def build(cls, store: TrafficStore | None = None) -> ServiceContext:
        s = get_settings()
        network = load_network(s.config_dir / load_pipeline_config(s).network_file)
        known = {a.id for a in network.approaches}
        # only cameras on this network (a mode without video, e.g. pipeline_pems_bay.yaml, lists none)
        cams = {c.id: c for c in (load_camera(p) for p in sorted((s.config_dir / "cameras").glob("*.yaml")))
                if c.approach_id in known}
        return cls(store=store or TrafficStore(), network=network, cameras=cams, settings=s)

    # ------------------------------------------------------------------ lookups
    @property
    def approaches(self) -> dict[str, Any]:
        return {a.id: a for a in self.network.approaches}

    @property
    def segments(self) -> dict[str, Any]:
        return {g.id: g for g in self.network.segments}

    @property
    def junctions(self) -> dict[str, Any]:
        return {j.id: j for j in self.network.junctions}

    def approach_for_road(self, road_id: str) -> Any | None:
        return next((a for a in self.network.approaches if a.road_segment_id == road_id), None)

    def simulator(self) -> TrafficSimulator:
        if self._simulator is None:
            from traffic_intel.simulation.interfaces import SIMULATORS

            self._simulator = SIMULATORS.create(load_pipeline_config(self.settings).simulator)
            self._simulator.load_network(self.network)
        return self._simulator

    @staticmethod
    def now() -> datetime:
        return datetime.now(UTC)

    def age_s(self, ts: datetime | None) -> float | None:
        return None if ts is None else (self.now() - ts).total_seconds()
