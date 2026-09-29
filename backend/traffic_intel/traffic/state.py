"""Assemble the flat ``TrafficState`` output from features + congestion verdict."""

from __future__ import annotations

from datetime import timedelta

from traffic_intel.core.config import ApproachConfig
from traffic_intel.core.enums import ObjectClass
from traffic_intel.core.schemas import CongestionAssessment, TrafficFeatures, TrafficState


def build_traffic_state(
    features: TrafficFeatures,
    congestion: CongestionAssessment,
    approach: ApproachConfig,
    *,
    live: bool = False,
) -> TrafficState:
    c = features.count_by_class
    return TrafficState(
        timestamp=features.window_start + timedelta(seconds=features.window_s),
        window_start=features.window_start,
        window_s=features.window_s,
        is_live=live,
        camera_id=features.camera_id,
        approach_id=features.approach_id,
        junction_id=approach.junction_id,
        road_id=approach.road_segment_id,
        vehicle_count=features.vehicle_count,
        car_count=c.get(ObjectClass.CAR, 0),
        motorcycle_count=c.get(ObjectClass.MOTORCYCLE, 0),
        bus_count=c.get(ObjectClass.BUS, 0),
        truck_count=c.get(ObjectClass.TRUCK, 0),
        bicycle_count=c.get(ObjectClass.BICYCLE, 0),
        pedestrian_count=features.pedestrian_count,
        average_speed_kmh=features.avg_speed_kmh,
        median_speed_kmh=features.median_speed_kmh,
        density_veh_km=features.density_veh_km,
        density_pct=round(features.density_ratio * 100.0, 1),
        occupancy_pct=features.occupancy_pct,
        flow_veh_h=features.flow_veh_h,
        inflow=features.inflow,
        outflow=features.outflow,
        queue_length_m=features.queue_length_m,
        stopped_vehicle_count=features.stopped_vehicles,
        traffic_state=congestion.level,
        congestion_score=congestion.score,
        explanation=congestion.explanation,
        quality=features.quality,
    )
