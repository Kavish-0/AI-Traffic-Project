"""Light conversions shared by the engine, the CLI and the API (no pandas / model imports)."""

from __future__ import annotations

from traffic_intel.core.enums import ObjectClass
from traffic_intel.core.schemas import TrafficFeatures, TrafficState


def features_from_state(s: TrafficState) -> TrafficFeatures:
    """Rebuild window features from a TrafficState (e.g. a recorded ``traffic_states.jsonl``).

    TrafficState is the flat published record; fields it does not carry (p85 speed, gaps,
    feature-engine extras) are left empty, so detectors that need them simply skip."""
    counts = {ObjectClass.CAR: s.car_count, ObjectClass.MOTORCYCLE: s.motorcycle_count, ObjectClass.BUS: s.bus_count,
              ObjectClass.TRUCK: s.truck_count, ObjectClass.BICYCLE: s.bicycle_count,
              ObjectClass.PEDESTRIAN: s.pedestrian_count}
    return TrafficFeatures(
        approach_id=s.approach_id, camera_id=s.camera_id, window_start=s.window_start, window_s=s.window_s,
        vehicle_count=s.vehicle_count, count_by_class=counts, pcu_count=float(s.vehicle_count),
        avg_speed_kmh=s.average_speed_kmh, speed_p85_kmh=None, density_veh_km=s.density_veh_km,
        density_ratio=s.density_pct / 100.0, occupancy_pct=s.occupancy_pct, flow_veh_h=s.flow_veh_h,
        inflow=s.inflow, outflow=s.outflow, queue_length_m=s.queue_length_m, stopped_vehicles=s.stopped_vehicle_count,
        avg_gap_m=None, pedestrian_count=s.pedestrian_count, quality=s.quality, median_speed_kmh=s.median_speed_kmh)
