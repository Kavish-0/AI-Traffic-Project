"""Every forecaster: common interface, shapes, save/load, baselines, metrics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from traffic_intel.ai.evaluation.metrics import mae, metric_table, r2, rmse, summary_table
from traffic_intel.core.registry import ComponentSpec
from traffic_intel.forecasting.data.preprocess import FeatureSpec, Preprocessor
from traffic_intel.forecasting.data.windows import Period, build_blocks, make_windows
from traffic_intel.forecasting.interfaces import FORECASTERS, forecaster_class

T0 = pd.Timestamp("2026-03-02 00:00", tz="UTC")
TINY = {
    "persistence": {}, "seasonal_naive": {}, "linear": {"alpha": 0.0},
    "random_forest": {"n_estimators": 10, "max_depth": 6, "n_jobs": 1},
    "xgboost": {"n_estimators": 30, "device": "cpu"},
    "lstm": {"lookback_steps": 8, "hidden": 16, "layers": 1, "max_epochs": 2, "device": "cpu"},
    "gru": {"lookback_steps": 8, "hidden": 16, "layers": 1, "max_epochs": 2, "device": "cpu"},
    "transformer": {"lookback_steps": 8, "d_model": 16, "heads": 2, "layers": 1, "ff": 32, "max_epochs": 2,
                    "device": "cpu"},
}


@pytest.fixture(scope="module")
def data():  # noqa: ANN201
    rng = np.random.default_rng(0)
    n = 3000
    i = np.arange(n)
    daily = np.sin(2 * np.pi * i / 1440)
    df = pd.DataFrame({"series_id": "s", "timestamp": T0 + pd.to_timedelta(i, unit="min"),
                       "flow_veh_h": 800 + 400 * daily + rng.normal(0, 20, n),
                       "avg_speed_kmh": 40 - 10 * daily + rng.normal(0, 1, n)})
    spec = FeatureSpec(dynamic=["flow_veh_h", "avg_speed_kmh"], targets=["flow_veh_h", "avg_speed_kmh"],
                       calendar=["hour"], lags=[1, 2, 5], rolling=[3])
    pre = Preprocessor(spec, resolution_s=60, tz="UTC", holiday_country=None)
    reg = pre.regularize(df)
    pre.fit(reg[reg["timestamp"] < T0 + pd.Timedelta(minutes=2000)])
    blocks = build_blocks(pre.transform(reg), pre)
    mk = lambda a, b: make_windows(blocks, Period("p", T0 + pd.Timedelta(minutes=a), T0 + pd.Timedelta(minutes=b)),  # noqa: E731
                                   lookback=8, horizon_steps=[5, 15, 30], pre=pre, season_steps=1440)
    return pre, mk(0, 2000), mk(2000, 2500), mk(2500, 3000)


@pytest.mark.parametrize("name", list(TINY))
def test_model_fit_predict_save_load(name: str, data, tmp_path: Path) -> None:  # noqa: ANN001
    pre, train, val, test = data
    model = FORECASTERS.create(ComponentSpec(type=name, params=TINY[name]))
    info = model.fit(train, val)
    pred = model.predict(test)
    assert pred.shape == (len(test), 3, 2) and np.isfinite(pred).all()
    assert isinstance(info, dict)
    model.save(tmp_path / name)
    clone = forecaster_class(name).load(tmp_path / name)
    np.testing.assert_allclose(clone.predict(test), pred, rtol=1e-4, atol=1e-4)


def test_persistence_and_seasonal_naive_definitions(data) -> None:  # noqa: ANN001
    _, _, _, test = data
    pers = FORECASTERS.create(ComponentSpec(type="persistence")).predict(test)
    np.testing.assert_array_equal(pers[:, 2], test.last())
    sn = FORECASTERS.create(ComponentSpec(type="seasonal_naive")).predict(test)
    np.testing.assert_array_equal(sn, test.seasonal())


def test_learned_model_beats_persistence_on_predictable_signal(data) -> None:  # noqa: ANN001
    pre, train, val, test = data
    y = pre.inverse_targets(test.y())
    lin = FORECASTERS.create(ComponentSpec(type="linear"))
    lin.fit(train, val)
    p_lin = pre.inverse_targets(lin.predict(test))
    p_per = pre.inverse_targets(FORECASTERS.create(ComponentSpec(type="persistence")).predict(test))
    assert rmse(y[:, 2, 1], p_lin[:, 2, 1]) < rmse(y[:, 2, 1], p_per[:, 2, 1])   # speed at +30 min


def test_delta_target_transform_roundtrip(data) -> None:  # noqa: ANN001
    _, train, _, _ = data
    lin = FORECASTERS.create(ComponentSpec(type="linear"))
    flat = lin._targets(train)
    np.testing.assert_allclose(lin._levels(flat, train), train.y(), atol=1e-5)
    with pytest.raises(ValueError):
        FORECASTERS.create(ComponentSpec(type="linear", params={"target_transform": "log"}))


def test_metrics_known_values() -> None:
    y, p = np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 4.0])
    assert mae(y, p) == pytest.approx(1 / 3)
    assert rmse(y, p) == pytest.approx(np.sqrt(1 / 3))
    assert r2(y, p) == pytest.approx(0.5)
    assert r2(y, y) == 1.0 and np.isnan(r2(np.ones(3), p))
    Y = np.stack([y, y], axis=1)[:, :, None]                       # [N=3, H=2, K=1]
    long = metric_table(Y, {"persistence": Y + 1.0, "m": Y + 0.5}, [5, 15], ["x"])
    assert set(long.columns) >= {"model", "target", "horizon_min", "MAE", "RMSE", "R2", "skill_vs_persistence"}
    m = long[(long.model == "m") & (long.horizon_min == 5)].iloc[0]
    assert pytest.approx(0.5) == m.RMSE and m.skill_vs_persistence == pytest.approx(0.5)
    s = summary_table(long)
    assert list(s.columns[:2]) == ["target", "model"] and len(s) == 2
