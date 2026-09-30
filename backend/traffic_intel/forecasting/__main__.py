"""Forecasting inference CLI.

    # history = TrafficState / TrafficFeatures JSONL (e.g. a camera run's traffic_states.jsonl)
    python -m traffic_intel.forecasting predict --history runs/<cam>/<run>/traffic_states.jsonl

    # worked example on held-out data: history up to --at from the experiment's dataset,
    # forecast, and the actual values that followed
    python -m traffic_intel.forecasting example --config configs/training/forecasting.yaml \
        --approach junction_a_north --at 2025-07-08T08:00:00+05:30

Common options: --model auto|<name> (default: PRODUCTION model), --version production|latest|<id>
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from traffic_intel.core.config import get_settings
from traffic_intel.core.logging import configure_logging

if TYPE_CHECKING:
    from traffic_intel.forecasting.service import ForecastingService


def _service(a: argparse.Namespace) -> ForecastingService:
    from traffic_intel.forecasting.service import ForecastingService

    return ForecastingService.from_registry(model=a.model, version=a.version)


def cmd_predict(a: argparse.Namespace) -> None:
    svc = _service(a)
    rows = [json.loads(line) for line in Path(a.history).read_text(encoding="utf-8").splitlines() if line.strip()]
    res = svc.predict(rows, approach_id=a.approach)
    print(res.model_dump_json(indent=2))


def cmd_example(a: argparse.Namespace) -> None:
    from traffic_intel.ai.training.experiment import load_experiment_config
    from traffic_intel.forecasting.data.sources import load_dataset

    svc = _service(a)
    cfg = load_experiment_config(Path(a.config))
    ds = load_dataset(cfg.dataset.source, cfg.dataset.params)
    f = ds.frame[ds.frame["series_id"] == a.approach].sort_values("timestamp")
    at = pd.Timestamp(a.at).tz_convert("UTC")
    hist = f[(f["timestamp"] <= at) & (f["timestamp"] > at - pd.Timedelta(seconds=ds.resolution_s * (
        svc.required_history_steps + 5)))]
    weather = [c for c in ("temperature_c", "humidity_pct", "rain_mm", "visibility_km", "weather_condition")
               if c in hist]
    res = svc.predict(hist.drop(columns=weather), context=hist[["timestamp", *weather]])
    out = json.loads(res.model_dump_json())
    actual = {}
    for h in res.horizons_min:
        row = f[f["timestamp"] == res.issued_at + pd.Timedelta(minutes=h)]
        if len(row):
            actual[f"{h}_min"] = {t: round(float(row.iloc[0][t]), 3) for t in res.predictions[f"{h}_min"]}
    out["actual_future (held-out, for comparison)"] = actual
    print(json.dumps(out, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(prog="traffic_intel.forecasting")
    p.add_argument("--model", default="auto")
    p.add_argument("--version", default="production")
    sub = p.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("predict")
    pr.add_argument("--history", required=True)
    pr.add_argument("--approach")
    pr.set_defaults(func=cmd_predict)
    ex = sub.add_parser("example")
    ex.add_argument("--config", default=str(get_settings().config_dir / "training" / "forecasting.yaml"))
    ex.add_argument("--approach", default="junction_a_north")
    ex.add_argument("--at", required=True, help="forecast origin (ISO time with offset), inside the test period")
    ex.set_defaults(func=cmd_example)
    a = p.parse_args()
    configure_logging("WARNING")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    a.func(a)


if __name__ == "__main__":
    main()
