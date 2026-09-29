"""What-if static assignment on the demo network (NetworkX + BPR + MSA)."""

from __future__ import annotations

from pathlib import Path

import pytest

from traffic_intel.core.config import load_network
from traffic_intel.core.schemas import SimulationRequest, TrafficFeatures
from traffic_intel.simulation.static_assignment import StaticAssignmentSimulator


@pytest.fixture
def sim(configs_dir: Path) -> StaticAssignmentSimulator:
    s = StaticAssignmentSimulator()
    s.load_network(load_network(configs_dir / "network" / "demo_network.yaml"))
    return s


def _by_id(res) -> dict:  # noqa: ANN001
    return {i.segment_id: i for i in res.impacts}


def test_closing_road_b_redistributes_to_alternatives(sim: StaticAssignmentSimulator) -> None:
    res = sim.what_if(SimulationRequest(closed_segment_ids=["road_b_wb"]), {})
    imp = _by_id(res)
    assert imp["road_b_wb"].status == "closed" and imp["road_b_wb"].scenario_flow_veh_h == 0.0
    assert imp["road_a_sb"].flow_change_pct > 5 and imp["road_c_nb"].flow_change_pct > 5     # both alternatives
    assert imp["road_d_eb"].flow_change_pct == 0.0                                          # unrelated arm
    assert res.unserved_demand_veh_h == 0.0
    # conservation: everything that used to arrive at junction A via B still arrives (via A or C)
    into_a = ("road_a_sb", "road_b_wb", "road_c_nb", "road_d_eb")
    assert sum(imp[s].scenario_flow_veh_h for s in into_a) == pytest.approx(
        sum(imp[s].baseline_flow_veh_h for s in into_a), abs=0.5)
    j = {x.junction_id: x for x in res.junctions}["junction_a"]
    assert j.scenario_congestion_prob > j.baseline_congestion_prob
    assert any(line.startswith("Road A (southbound): +") for line in res.summary)
    assert any("NOT a microscopic" in a for a in res.assumptions)


def test_lane_closure_diverts_only_part_of_the_flow(sim: StaticAssignmentSimulator) -> None:
    res = sim.what_if(SimulationRequest(capacity_factors={"road_a_sb": 0.5}), {})
    a = _by_id(res)["road_a_sb"]
    assert a.status == "reduced" and 0 < a.scenario_flow_veh_h <= a.baseline_flow_veh_h
    assert a.scenario_congestion_prob > a.baseline_congestion_prob     # same traffic, half the capacity


def test_observed_baseline_and_unserved_demand(sim: StaticAssignmentSimulator) -> None:
    obs = TrafficFeatures.model_construct(flow_veh_h=1500.0)
    res = sim.what_if(SimulationRequest(closed_segment_ids=["dev_highway_approach_seg"]),
                      {"dev_highway_approach_seg": obs})
    imp = _by_id(res)["dev_highway_approach_seg"]
    assert imp.baseline_observed and imp.baseline_flow_veh_h == 1500.0
    assert res.unserved_demand_veh_h == pytest.approx(1500.0)          # no alternative path exists
    assert any("unserved" in s for s in res.summary)


def test_demand_factor_and_validation(sim: StaticAssignmentSimulator) -> None:
    base = _by_id(sim.what_if(SimulationRequest(), {}))
    more = _by_id(sim.what_if(SimulationRequest(demand_factor=1.5), {}))
    assert more["road_a_sb"].baseline_flow_veh_h == pytest.approx(1.5 * base["road_a_sb"].baseline_flow_veh_h)
    assert more["road_a_sb"].scenario_congestion_prob > base["road_a_sb"].scenario_congestion_prob
    with pytest.raises(ValueError):
        sim.what_if(SimulationRequest(closed_segment_ids=["no_such_road"]), {})
    ps = [sim.congestion_probability(x) for x in (0.2, 0.6, 0.85, 1.0, 1.3)]
    assert ps == sorted(ps) and ps[2] == pytest.approx(0.5)
