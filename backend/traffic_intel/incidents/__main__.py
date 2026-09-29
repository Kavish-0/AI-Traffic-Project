"""Incident-intelligence CLI.

    python -m traffic_intel.incidents scenarios [--name stopped_vehicle] [--store]
        run the synthetic scenarios through the real pipeline and print their timelines
        (--store also writes states / anomalies / incidents / risk to the intelligence DB,
        so the API has data to show)
    python -m traffic_intel.incidents ingest runs/<camera>/<run_id> [--no-forecast]
        replay a recorded camera run (traffic_states.jsonl + anomalies.jsonl) through the engine
        and store the results
    python -m traffic_intel.incidents what-if --close road_b_wb [--capacity road_a_sb=0.5] [--demand 1.1]
        static re-assignment on the configured network (baseline = latest stored states)
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

from traffic_intel.core.config import (
    NetworkConfig,
    PipelineConfig,
    get_settings,
    load_network,
    load_pipeline_config,
)


def _network() -> tuple[NetworkConfig, PipelineConfig]:
    pipeline = load_pipeline_config()
    return load_network(get_settings().config_dir / pipeline.network_file), pipeline


def cmd_scenarios(args: argparse.Namespace) -> None:
    from traffic_intel.database.store import TrafficStore
    from traffic_intel.incidents.scenarios import SCENARIOS, run_scenario

    network, pipeline = _network()
    store = TrafficStore(args.db) if args.store else None
    names = [args.name] if args.name else list(SCENARIOS)
    rows = []
    for day, name in enumerate(names):          # one day per scenario so stored timelines do not overlap
        start = datetime.fromisoformat("2026-09-01T08:00:00+05:30") + timedelta(days=day)
        res = run_scenario(name, network=network, pipeline=pipeline, store=store, start=start)
        print(f"\n=== {name}: {res.spec.description} (event at +{res.spec.event_at_s:.0f} s; "
              f"expected: {res.spec.expected_type})")
        for u in res.updates:
            print("  " + u.summary())
        last = next((u.incident for u in reversed(res.after()) if u.incident is not None), None)
        if last is not None:
            print(f"  -> {last.message}")
        peak = res.peak_risk(res.after())
        peak_risk = peak.risk if peak else None
        rows.append((name, res.spec.expected_type, res.final_type(), res.max_probability(res.before()),
                     res.max_probability(res.after()), peak_risk.risk_score if peak_risk else 0.0,
                     peak_risk.level.value if peak_risk else "-"))
    print("\nscenario           expected           reported           p(before) p(after) peak risk")
    for r in rows:
        print(f"{r[0]:<18} {r[1]:<18} {r[2]:<18} {r[3]:>8.0%} {r[4]:>8.0%} {r[5]:>5.0f} {r[6]}")


def cmd_ingest(args: argparse.Namespace) -> None:
    from traffic_intel.core.schemas import AnomalySignal, TrafficState
    from traffic_intel.database.store import TrafficStore
    from traffic_intel.incidents.convert import features_from_state
    from traffic_intel.incidents.engine import IntelligenceEngine

    run_dir = Path(args.run_dir)
    network, pipeline = _network()
    store = TrafficStore(args.db)
    engine = IntelligenceEngine.build(network, pipeline, store=store, with_forecast=not args.no_forecast)
    states = [TrafficState.model_validate_json(line) for line in
              (run_dir / "traffic_states.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    anomalies_file = run_dir / "anomalies.jsonl"
    signals = [AnomalySignal.model_validate_json(line) for line in
               anomalies_file.read_text(encoding="utf-8").splitlines() if line.strip()] \
        if anomalies_file.exists() else []
    signals.sort(key=lambda s: s.timestamp)
    i = 0
    for st in sorted(states, key=lambda s: s.timestamp):
        batch = []
        while i < len(signals) and signals[i].timestamp <= st.timestamp:
            batch.append(signals[i])
            i += 1
        engine.ingest_signals(batch)
        print(engine.process_window(features_from_state(st), st).summary())
    print(f"ingested {len(states)} windows, {len(signals)} track signals into {store.engine.url}")


def cmd_what_if(args: argparse.Namespace) -> None:
    """Same use-case as POST /api/v1/simulation/what-if (services.simulation): latest fresh observations
    as baseline, result stored in the database."""
    from traffic_intel.core.schemas import SimulationRequest
    from traffic_intel.database.store import TrafficStore
    from traffic_intel.services import simulation
    from traffic_intel.services.context import ServiceContext

    factors = {k: float(v) for k, v in (c.split("=", 1) for c in args.capacity)}
    req = SimulationRequest(closed_segment_ids=args.close, capacity_factors=factors, demand_factor=args.demand)
    res = simulation.run(ServiceContext.build(TrafficStore(args.db)), req)
    print("\n".join(res.summary))
    print("\nAssumptions:\n- " + "\n- ".join(res.assumptions))
    if args.json:
        print(json.dumps(res.model_dump(mode="json"), indent=2))


def main() -> None:
    p = argparse.ArgumentParser(prog="traffic_intel.incidents")
    p.add_argument("--db", default=None, help="SQLAlchemy URL (default: TI_DATABASE_URL or data/db/traffic.db)")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scenarios")
    s.add_argument("--name", default=None)
    s.add_argument("--store", action="store_true")
    s.set_defaults(fn=cmd_scenarios)
    g = sub.add_parser("ingest")
    g.add_argument("run_dir")
    g.add_argument("--no-forecast", action="store_true")
    g.set_defaults(fn=cmd_ingest)
    w = sub.add_parser("what-if")
    w.add_argument("--close", nargs="*", default=[])
    w.add_argument("--capacity", nargs="*", default=[], help="segment=factor, e.g. road_a_sb=0.5")
    w.add_argument("--demand", type=float, default=1.0)
    w.add_argument("--json", action="store_true")
    w.set_defaults(fn=cmd_what_if)
    args = p.parse_args()
    get_settings()
    args.fn(args)


if __name__ == "__main__":
    main()
