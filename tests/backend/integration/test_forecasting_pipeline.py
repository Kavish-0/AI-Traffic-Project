"""End-to-end: experiment (train + evaluate + registry + tracking) -> forecasting service.
Small offline synthetic dataset, CPU only (no network, no GPU)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from traffic_intel.ai.model_store import FileModelStore
from traffic_intel.ai.training.experiment import ExperimentConfig, run_experiment
from traffic_intel.core.enums import CongestionLevel, ModelStatus, ModelTask
from traffic_intel.core.schemas import Forecast
from traffic_intel.forecasting.data.sources import load_dataset
from traffic_intel.forecasting.service import ForecastingService, InsufficientHistoryError

DATASET = {"source": "synthetic",
           "params": {"days": 4, "seed": 3, "holiday_country": None,
                      "approaches": [{"series_id": "junction_a_north", "lanes": 2}]}}
CFG = {
    "name": "test_tiny", "seed": 0, "dataset": DATASET,
    "features": {"dynamic": ["avg_speed_kmh", "density_ratio", "occupancy_pct", "queue_length_m", "flow_veh_h"],
                 "targets": ["avg_speed_kmh", "density_ratio", "occupancy_pct", "queue_length_m", "flow_veh_h"],
                 "weather": ["rain_mm", "weather_condition"], "calendar": ["hour", "day_of_week", "rush_hour"],
                 "lags": [1, 2, 5], "rolling": [5]},
    "horizons_min": [5, 15, 30], "split": {"train": 0.6, "val": 0.2, "test": 0.2}, "train_stride": 3,
    "primary_target": "avg_speed_kmh", "plots": {"days": 0.25, "start_offset_days": 0.1},
    "models": [
        {"name": "persistence", "type": "persistence"},
        {"name": "linear", "type": "linear"},
        {"name": "xgboost", "type": "xgboost", "params": {"n_estimators": 40, "device": "cpu"}},
        {"name": "lstm", "type": "lstm", "params": {"lookback_steps": 10, "hidden": 16, "layers": 1,
                                                     "max_epochs": 1, "device": "cpu"}},
    ],
}


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory):  # noqa: ANN201
    root = tmp_path_factory.mktemp("fc")
    store = FileModelStore(root / "models")
    cfg = ExperimentConfig.model_validate(CFG)
    res = run_experiment(cfg, experiments_root=root / "experiments", store=store)
    return root, store, res


def test_experiment_record_and_registry(trained) -> None:  # noqa: ANN001
    root, store, res = trained
    d = Path(res.directory)
    for f in ("config.yaml", "dataset.json", "models.json", "metrics_test.csv", "metrics_val.csv", "comparison.md"):
        assert (d / f).exists(), f
    assert (d / "plots" / "rmse_by_horizon_avg_speed_kmh.png").exists()
    info = json.loads((d / "dataset.json").read_text())
    p = info["periods"]
    assert p["train"]["end"] == p["val"]["start"] and p["val"]["end"] == p["test"]["start"]   # chronological
    m = pd.read_csv(d / "metrics_test.csv")
    assert set(m["model"]) == {"persistence", "linear", "xgboost", "lstm"}
    assert set(m["horizon_min"]) == {5, 15, 30} and m[["MAE", "RMSE", "R2"]].notna().all().all()
    assert "Model selection" in (d / "comparison.md").read_text()
    for name in ("linear", "xgboost", "lstm"):
        vdir = root / "models" / "forecasting" / name / res.experiment_id
        assert {"params.json", "preprocessor.json", "metadata.json"} <= {f.name for f in vdir.iterdir()}
        meta = store.metadata(ModelTask.FORECASTING, name, res.experiment_id)
        assert meta.train_data["periods"] == p and meta.params["type"] == name and meta.metrics["test"]


def test_experiments_are_not_overwritten(trained) -> None:  # noqa: ANN001
    root, store, first = trained
    cfg = ExperimentConfig.model_validate({**CFG, "models": CFG["models"][:2]})
    import time

    time.sleep(1.1)                                    # experiment ids have 1-second resolution
    second = run_experiment(cfg, experiments_root=root / "experiments", store=store)
    assert second.experiment_id != first.experiment_id
    assert Path(first.directory).exists() and Path(second.directory).exists()
    idx = pd.read_csv(root / "experiments" / "index.csv")
    assert set(idx["experiment_id"]) == {first.experiment_id, second.experiment_id}
    assert len(store.list(ModelTask.FORECASTING)) == 4 + 2


def _history(store_root: Path, minutes: int, until: str | None = None) -> list[dict]:  # noqa: ARG001
    ds = load_dataset("synthetic", DATASET["params"])
    f = ds.frame.dropna(subset=["avg_speed_kmh"]).tail(minutes)
    rows = []
    for r in f.to_dict("records"):
        rows.append({"approach_id": r["series_id"], "window_start": r["timestamp"].isoformat(), "window_s": 60.0,
                     "is_live": False, "average_speed_kmh": r["avg_speed_kmh"],
                     "density_pct": r["density_ratio"] * 100, "occupancy_pct": r["occupancy_pct"],
                     "queue_length_m": r["queue_length_m"], "flow_veh_h": r["flow_veh_h"],
                     "stopped_vehicle_count": r["stopped_vehicles"], "traffic_state": "moderate"})
    return rows


def test_forecasting_service_end_to_end(trained) -> None:  # noqa: ANN001
    root, store, res = trained
    store.promote(ModelTask.FORECASTING, "xgboost", res.experiment_id)
    assert store.metadata(ModelTask.FORECASTING, "xgboost", res.experiment_id).status is ModelStatus.PRODUCTION
    svc = ForecastingService.from_registry(store=store)
    assert svc.model_info["name"] == "xgboost" and svc.horizons_min == [5, 15, 30]
    out = svc.predict(_history(root, 60))
    assert out.approach_id == "junction_a_north" and out.current_state is CongestionLevel.MODERATE
    assert set(out.forecast) == {"5_min", "15_min", "30_min"}
    assert all(isinstance(v, CongestionLevel) for v in out.forecast.values())
    for key in out.forecast:
        p = out.predictions[key]
        assert set(p) == {"avg_speed_kmh", "density_ratio", "occupancy_pct", "queue_length_m", "flow_veh_h"}
        assert p["avg_speed_kmh"] >= 0 and 0 <= p["occupancy_pct"] <= 100
        assert out.explanations[key]
    assert any("weather" in n for n in out.notes)            # no weather context was supplied
    fc = out.to_forecast()
    assert isinstance(fc, Forecast) and {pt.horizon_min for pt in fc.points} == {5, 15, 30}


def test_service_rejects_insufficient_history(trained) -> None:  # noqa: ANN001
    root, store, res = trained
    svc = ForecastingService.from_registry(model="lstm", version=res.experiment_id, store=store)
    with pytest.raises(InsufficientHistoryError):
        svc.predict(_history(root, 3))
    rows = _history(root, 60)
    del rows[52:57]                    # 5-minute hole (> max_gap_fill_steps) inside the model's input window
    with pytest.raises(InsufficientHistoryError):
        svc.predict(rows)
    rows = _history(root, 60)
    del rows[20:30]                    # a hole older than the input window does not matter
    assert svc.predict(rows).forecast
