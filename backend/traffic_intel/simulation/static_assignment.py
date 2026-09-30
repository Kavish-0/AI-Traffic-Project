"""Static what-if assignment on the configured road graph (``simulator.type: static_assignment``).

THIS IS NOT A MICROSCOPIC (or mesoscopic / dynamic) TRAFFIC SIMULATOR. It moves hourly
flows between directed road segments with textbook macroscopic formulas so that an
operator can ask "roughly where does the traffic go if Road B closes?". Individual
vehicles, signal timing, queue spill-back, turning restrictions and route-choice
behaviour are not modelled. Every result carries this list of assumptions.

Method
------
1. Graph: nodes = road nodes / junctions, edges = directed segments with
   length (from geometry), free-flow time t0 = length / free-flow speed,
   capacity c = capacity_veh_h or lanes x capacity_per_lane.
2. Baseline flow per edge: the latest OBSERVED flow (camera features of the
   approach on that segment) or, for unobserved edges, ``assumed_vc x c``.
   All flows are multiplied by the request's ``demand_factor``.
3. Affected edges (closed, or capacity multiplied by a factor < 1): the edge's
   flow is treated as demand between its two end nodes and re-assigned over
   {the edge itself if still open} U {k shortest alternative paths} with the
   Method of Successive Averages (MSA): each iteration loads each OD pair
   all-or-nothing onto its currently cheapest path (BPR link times, background
   flow of the other edges included) and averages with step 1/n.
       BPR:  t = t0 * (1 + alpha * (v / c) ** beta)        alpha 0.15, beta 4
   Demand with no alternative path is reported as ``unserved_demand_veh_h``.
4. Congestion probability per edge = logistic(k * (v/c - x0))   (k 10, x0 0.85:
   v/c 0.85 -> 50 %). Junction probability = flow-weighted mean over the edges
   entering the junction.
"""

from __future__ import annotations

import math
import uuid
from datetime import UTC, datetime
from itertools import islice
from typing import Any

import networkx as nx

from traffic_intel.core.config import NetworkConfig
from traffic_intel.core.enums import CongestionLevel
from traffic_intel.core.schemas import (
    JunctionImpact,
    SegmentImpact,
    SimulationRequest,
    SimulationResult,
    TrafficFeatures,
)
from traffic_intel.simulation.interfaces import SIMULATORS, TrafficSimulator

ASSUMPTIONS = [
    "Static macroscopic assignment on a small road graph - NOT a microscopic or dynamic traffic simulator.",
    "Only the flow of closed / reduced segments is re-routed; it is treated as demand between that "
    "segment's end nodes (true origins/destinations of trips are unknown).",
    "Drivers choose among the k shortest alternative paths by BPR travel time (MSA equilibrium); "
    "no turning restrictions, signal timing, spill-back or behavioural adaptation (e.g. trips cancelled).",
    "Unobserved segments use an assumed baseline volume/capacity ratio; capacities come from lanes x "
    "capacity_per_lane unless configured.",
    "Congestion probability is a logistic function of volume/capacity (not calibrated on local data); "
    "treat the numbers as relative indications, not predictions.",
]


def _haversine_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    lon1, lat1, lon2, lat2 = map(math.radians, (*a, *b))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6_371_000 * math.asin(math.sqrt(h))


def _level(p: float) -> CongestionLevel:
    return (CongestionLevel.FREE if p < 0.25 else CongestionLevel.MODERATE if p < 0.5
            else CongestionLevel.HIGH if p < 0.75 else CongestionLevel.SEVERE)


@SIMULATORS.register("static_assignment")
class StaticAssignmentSimulator(TrafficSimulator):
    def __init__(self, bpr_alpha: float = 0.15, bpr_beta: float = 4.0, msa_iterations: int = 50,
                 k_paths: int = 3, capacity_per_lane: float = 900.0, assumed_vc: float = 0.5,
                 congestion_k: float = 10.0, congestion_x0: float = 0.85, report_min_change_pct: float = 5.0) -> None:
        self.alpha, self.beta = bpr_alpha, bpr_beta
        self.iterations, self.k_paths = msa_iterations, k_paths
        self.capacity_per_lane, self.assumed_vc = capacity_per_lane, assumed_vc
        self.k, self.x0 = congestion_k, congestion_x0
        self.report_min_change_pct = report_min_change_pct
        self.graph: nx.DiGraph | None = None
        self.network: NetworkConfig | None = None

    # ------------------------------------------------------------------ #
    def load_network(self, network: NetworkConfig) -> None:
        g = nx.DiGraph()
        for n in network.nodes:
            g.add_node(n.id, junction_id=n.junction_id)
        for s in network.segments:
            length = sum(_haversine_m((a[0], a[1]), (b[0], b[1]))
                         for a, b in zip(s.geometry, s.geometry[1:], strict=False))
            ff = s.free_flow_speed_kmh or s.speed_limit_kmh
            g.add_edge(s.from_node, s.to_node, id=s.id, name=s.name, length_m=length,
                       t0_s=max(length, 1.0) / (ff / 3.6),
                       capacity=s.capacity_veh_h or s.lanes * self.capacity_per_lane)
        self.graph, self.network = g, network
        self._edge_by_id = {d["id"]: (u, v) for u, v, d in g.edges(data=True)}
        self._junction_names = {j.id: j.name for j in network.junctions}

    def congestion_probability(self, vc: float) -> float:
        return 1.0 / (1.0 + math.exp(-self.k * (vc - self.x0)))

    def _time(self, d: dict[str, Any], flow: float, cap: float) -> float:
        if cap <= 0:
            return math.inf
        return float(d["t0_s"] * (1.0 + self.alpha * (flow / cap) ** self.beta))

    # ------------------------------------------------------------------ #
    def what_if(self, request: SimulationRequest, baseline: dict[str, TrafficFeatures]) -> SimulationResult:
        if self.graph is None:
            raise RuntimeError("load_network() first")
        g = self.graph
        unknown = [e for e in [*request.closed_segment_ids, *request.capacity_factors] if e not in self._edge_by_id]
        if unknown:
            raise ValueError(f"unknown segment id(s): {unknown}")

        cap0 = {d["id"]: d["capacity"] for _, _, d in g.edges(data=True)}
        cap = dict(cap0)
        for sid, f in request.capacity_factors.items():
            cap[sid] = cap0[sid] * max(0.0, f)
        for sid in request.closed_segment_ids:
            cap[sid] = 0.0
        observed = {sid for sid in cap0 if sid in baseline and baseline[sid].flow_veh_h is not None}
        base = {sid: (baseline[sid].flow_veh_h if sid in observed else self.assumed_vc * cap0[sid])
                * request.demand_factor for sid in cap0}

        affected = [sid for sid in cap0 if cap[sid] < cap0[sid]]
        flow = dict(base)
        ods: list[tuple[float, list[list[str]]]] = []
        unserved = 0.0
        for sid in affected:
            u, v = self._edge_by_id[sid]
            demand = base[sid]
            flow[sid] = 0.0                                   # re-assigned below
            paths = [[sid]] if cap[sid] > 0 else []
            h = g.copy()
            for cid in affected:
                if cap[cid] <= 0 or cid == sid:
                    h.remove_edge(*self._edge_by_id[cid])
            try:
                for p in islice(nx.shortest_simple_paths(h, u, v, weight="t0_s"), self.k_paths + 1):
                    ids = [h.edges[a, b]["id"] for a, b in zip(p, p[1:], strict=False)]
                    if ids != [sid]:
                        paths.append(ids)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                pass
            paths = paths[: self.k_paths + (1 if cap[sid] > 0 else 0)]
            if not paths:
                unserved += demand
                continue
            ods.append((demand, paths))

        # MSA over the displaced OD demand; background = baseline flow of every other edge
        assigned = [[0.0] * len(paths) for _, paths in ods]
        for n in range(1, self.iterations + 1):
            load = dict(flow)
            for (_, paths), x in zip(ods, assigned, strict=True):
                for p, q in zip(paths, x, strict=True):
                    for e in p:
                        load[e] += q
            for i, (demand, paths) in enumerate(ods):
                costs = [sum(self._time(g.edges[self._edge_by_id[e]], load[e], cap[e]) for e in p) for p in paths]
                best = min(range(len(paths)), key=costs.__getitem__)
                target = [demand if j == best else 0.0 for j in range(len(paths))]
                assigned[i] = [x + (t - x) / n for x, t in zip(assigned[i], target, strict=True)]
        for (_, paths), x in zip(ods, assigned, strict=True):
            for p, q in zip(paths, x, strict=True):
                for e in p:
                    flow[e] += q

        impacts, summary = [], []
        prob_b, prob_s = {}, {}
        for _, _, d in g.edges(data=True):
            sid = d["id"]
            pb = self.congestion_probability(base[sid] / cap0[sid])
            ps = 1.0 if cap[sid] <= 0 else self.congestion_probability(flow[sid] / cap[sid])
            prob_b[sid], prob_s[sid] = pb, (0.0 if cap[sid] <= 0 else ps)
            change = 100.0 * (flow[sid] - base[sid]) / base[sid] if base[sid] > 0 else 0.0
            status = "closed" if cap[sid] <= 0 else "reduced" if cap[sid] < cap0[sid] else "open"
            impacts.append(SegmentImpact(
                segment_id=sid, name=d["name"], status=status, baseline_observed=sid in observed,
                baseline_flow_veh_h=round(base[sid], 1), scenario_flow_veh_h=round(flow[sid], 1),
                flow_change_pct=round(change, 1), baseline_congestion_prob=round(pb, 3),
                scenario_congestion_prob=round(prob_s[sid], 3),
                scenario_level=CongestionLevel.FREE if status == "closed" else _level(ps)))
            if status == "closed":
                summary.append(f"{d['name']}: CLOSED ({base[sid]:.0f} veh/h displaced)")
            elif abs(change) >= self.report_min_change_pct:
                summary.append(f"{d['name']}: {change:+.0f}% flow, "
                               f"congestion probability {pb:.0%} -> {prob_s[sid]:.0%}")

        junctions = []
        assert self.network is not None  # set together with self.graph in load_network()
        for j in self.network.junctions:
            nodes = [n for n, nd in g.nodes(data=True) if nd.get("junction_id") == j.id]
            inc = [d["id"] for n in nodes for _, _, d in g.in_edges(n, data=True)]
            if not inc:
                continue

            def weighted(fl: dict[str, float], pr: dict[str, float], edges: list[str] = inc) -> float:
                w = sum(fl[e] for e in edges)
                return sum(fl[e] * pr[e] for e in edges) / w if w > 0 else 0.0

            jb, js = weighted(base, prob_b), weighted(flow, prob_s)
            junctions.append(JunctionImpact(junction_id=j.id, name=j.name, baseline_congestion_prob=round(jb, 3),
                                            scenario_congestion_prob=round(js, 3)))
            if abs(js - jb) >= 0.05:
                summary.append(f"{j.name}: congestion probability {jb:.0%} -> {js:.0%}")
        if unserved > 0:
            summary.append(f"{unserved:.0f} veh/h have no alternative path (unserved)")
        if not summary:
            summary.append("No material change predicted by the static model.")
        assumptions = [*ASSUMPTIONS, f"Baseline observed on {len(observed)} of {len(cap0)} segments; "
                       f"others assume v/c = {self.assumed_vc}."]
        return SimulationResult(scenario_id=str(uuid.uuid4()), request=request, created_at=datetime.now(UTC),
                                impacts=impacts, unserved_demand_veh_h=round(unserved, 1),
                                assumptions=assumptions, junctions=junctions, summary=summary)
