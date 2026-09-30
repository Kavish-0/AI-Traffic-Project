"""The METR-LA experiment configs run end to end through the adapter -> preprocessor -> chronological
split -> models -> experiment record, on a METR-LA-format fixture (the real file is placed by hand).
Traffic-only and traffic+context variants are separate experiments that differ only in context."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from traffic_intel.ai.model_store import FileModelStore
from traffic_intel.ai.training.experiment import ModelEntry, load_experiment_config, run_experiment
from traffic_intel.core.config import PROJECT_ROOT
from traffic_intel.core.feature_schema import CALENDAR_COLUMNS

pytest.importorskip("tables")


def _metr_la_fixture(root: Path, days: int = 14) -> None:
    """Daily-periodic speeds (morning dip) for 4 sensors, naive local 5-min index, mph, zeros = missing."""
    idx = pd.date_range("2012-04-02 00:00", periods=days * 288, freq="5min")
    h = idx.hour.to_numpy() + idx.minute.to_numpy() / 60
    rng = np.random.default_rng(3)
    cols = {}
    for k, sid in enumerate(["717445", "717446", "717447", "717448"]):
        dip = 25 * np.exp(-((h - 8.0 - 0.3 * k) / 1.2) ** 2) * (idx.dayofweek.to_numpy() < 5)
        cols[sid] = 62 - dip + rng.normal(0, 1.5, len(idx))
    df = pd.DataFrame(cols, index=idx)
    df.iloc[100:103, 0] = 0.0
    root.mkdir(parents=True, exist_ok=True)
    df.to_hdf(root / "metr-la.h5", key="df")


@pytest.mark.parametrize("variant", ["traffic_only", "context"])
def test_metr_la_experiment_variants(tmp_path: Path, variant: str) -> None:
    _metr_la_fixture(tmp_path / "metr_la")
    cfg = load_experiment_config(PROJECT_ROOT / "configs" / "training" / f"forecasting_metr_la_{variant}.yaml")
    params = {**cfg.dataset.params, "root": str(tmp_path / "metr_la"), "every_k": 1, "weather": False}
    cfg = cfg.model_copy(update={
        "dataset": cfg.dataset.model_copy(update={"params": params}),
        "models": [ModelEntry(name="persistence", type="persistence"),
                   ModelEntry(name="linear", type="linear", params={"alpha": 0.0})],
        "features": cfg.features.model_copy(update={"weather": []}),     # weather needs the network; calendar stays
    })
    res = run_experiment(cfg, experiments_root=tmp_path / "exp", store=FileModelStore(tmp_path / "models"))
    out = Path(res.directory)
    info = (out / "dataset.json").read_text(encoding="utf-8")
    assert "metr_la_717445" in info
    test = pd.read_csv(out / "metrics_test.csv")
    assert set(test["horizon_min"]) == {5, 15, 30} and set(test["model"]) == {"persistence", "linear"}
    lin = test[test["model"] == "linear"].set_index("horizon_min")["RMSE"]
    per = test[test["model"] == "persistence"].set_index("horizon_min")["RMSE"]
    assert (lin <= per * 1.05).all()                       # learned baseline at least as good as persistence
    pre = (Path(tmp_path / "models" / "forecasting" / "linear" / res.experiment_id) / "preprocessor.json").read_text()
    uses_calendar = any(c in pre for c in CALENDAR_COLUMNS)
    assert uses_calendar is (variant == "context")          # the ONLY difference between the two experiments
