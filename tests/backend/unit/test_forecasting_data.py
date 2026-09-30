"""Forecasting data pipeline: preprocessing, missing values, calendar, chronological split, windows."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from traffic_intel.forecasting.data.context import calendar_frame, wmo_condition
from traffic_intel.forecasting.data.preprocess import FeatureSpec, Preprocessor
from traffic_intel.forecasting.data.windows import (
    Period,
    SplitConfig,
    build_blocks,
    chronological_split,
    make_windows,
    tabular_matrix,
)

T0 = pd.Timestamp("2026-03-02 00:00", tz="UTC")     # a Monday


def frame(n: int = 600, series: tuple[str, ...] = ("a",), drop: list[int] | None = None) -> pd.DataFrame:
    """Two traffic columns with a known pattern: flow = step index, speed = 50 - step/100."""
    rows = []
    for s in series:
        for i in range(n):
            if drop and i in drop:
                continue
            rows.append({"series_id": s, "timestamp": T0 + pd.Timedelta(minutes=i), "flow_veh_h": float(i),
                         "avg_speed_kmh": 50.0 - i / 100.0, "rain_mm": 1.0 if i % 60 < 10 else 0.0,
                         "visibility_km": np.nan, "weather_condition": "rain" if i % 60 < 10 else "clear"})
    return pd.DataFrame(rows)


def spec(**kw) -> FeatureSpec:
    base = dict(dynamic=["flow_veh_h", "avg_speed_kmh"], targets=["flow_veh_h", "avg_speed_kmh"],
                weather=["rain_mm", "visibility_km", "weather_condition"], calendar=["hour", "day_of_week"],
                max_gap_fill_steps=2, lags=[1, 2, 5], rolling=[3])
    base.update(kw)
    return FeatureSpec(**base)


def pre_for(s: FeatureSpec | None = None) -> Preprocessor:
    return Preprocessor(s or spec(), resolution_s=60, tz="UTC", holiday_country=None)


# --------------------------------------------------------------------------- #
def test_regularize_fills_short_gaps_only() -> None:
    p = pre_for()
    reg = p.regularize(frame(drop=[10, 11] + list(range(100, 105))))
    assert len(reg) == 600                                           # full grid restored
    assert reg.loc[10, "flow_veh_h"] == pytest.approx(10.0)          # 2-step gap interpolated
    assert reg.loc[11, "observed"] == 0.0                            # ...but flagged as not observed
    assert reg.loc[100:104, "flow_veh_h"].isna().all()               # 5-step gap stays missing


def test_scaling_fitted_on_train_only_and_roundtrip() -> None:
    p = pre_for()
    reg = p.regularize(frame())
    train = reg[reg["timestamp"] < T0 + pd.Timedelta(minutes=300)]
    p.fit(train)
    assert p.mean["flow_veh_h"] == pytest.approx(149.5)              # mean of 0..299, not of 0..599
    assert "visibility_km" in p.dropped                              # all-NaN in train -> dropped
    tr = p.transform(reg)
    assert tr["flow_veh_h"].iloc[0] == pytest.approx((0 - 149.5) / p.std["flow_veh_h"])
    back = p.inverse_targets(tr[p.target_columns].to_numpy())
    np.testing.assert_allclose(back[:, 0], reg["flow_veh_h"], atol=1e-3)
    clone = Preprocessor.from_dict(p.to_dict())
    pd.testing.assert_frame_equal(clone.transform(reg), tr)
    assert clone.spec_hash() == p.spec_hash()


def test_weather_one_hot_uses_train_categories() -> None:
    p = pre_for()
    reg = p.regularize(frame())
    p.fit(reg)
    assert p.categories == ["clear", "rain"]
    reg2 = reg.copy()
    reg2.loc[:5, "weather_condition"] = "snow"                       # unseen at training time
    tr = p.transform(reg2)
    assert (tr.loc[:5, ["weather_clear", "weather_rain"]].to_numpy() == 0).all()


def test_calendar_features_local_time_holiday_and_rush_hour() -> None:
    times = pd.DatetimeIndex([pd.Timestamp("2016-07-04 15:00", tz="UTC"),    # 08:00 PDT, US holiday
                              pd.Timestamp("2016-07-05 15:00", tz="UTC"),    # 08:00 PDT Tuesday
                              pd.Timestamp("2016-07-09 15:00", tz="UTC")])   # Saturday
    cal = calendar_frame(times, tz="America/Los_Angeles", holiday_country="US", rush_hours=[(7, 10)])
    assert cal["is_holiday"].tolist() == [1.0, 0.0, 0.0]
    assert cal["is_rush_hour"].tolist() == [0.0, 1.0, 0.0]           # not on holidays or weekends
    assert cal["is_weekend"].tolist() == [0.0, 0.0, 1.0]
    assert cal["hour_sin"].iloc[1] == pytest.approx(np.sin(2 * np.pi * 8 / 24))
    assert wmo_condition(0) == "clear" and wmo_condition(63) == "rain" and wmo_condition(45) == "fog"


def test_chronological_split_is_ordered_and_disjoint() -> None:
    times = pd.date_range(T0, periods=1000, freq="60s")
    per = chronological_split(times, SplitConfig(train=0.6, val=0.2, test=0.2), resolution_s=60)
    assert per["train"].start == times[0]
    assert per["train"].end == per["val"].start and per["val"].end == per["test"].start
    assert per["train"].end == T0 + pd.Timedelta(minutes=600)        # snapped to the grid
    explicit = chronological_split(times, SplitConfig(val_start=str(times[500]), test_start=str(times[800])))
    assert explicit["test"].start == times[800]
    with pytest.raises(ValueError):
        chronological_split(times, SplitConfig(val_start=str(times[900]), test_start=str(times[100])))


def _setup(n: int = 600, drop: list[int] | None = None):  # noqa: ANN202
    p = pre_for()
    reg = p.regularize(frame(n, drop=drop))
    p.fit(reg)
    return p, build_blocks(p.transform(reg), p)


def test_windows_never_leak_across_periods_and_skip_gaps() -> None:
    p, blocks = _setup(drop=list(range(200, 210)))
    period = Period("train", T0, T0 + pd.Timedelta(minutes=400))
    ws = make_windows(blocks, period, lookback=10, horizon_steps=[1, 5], pre=p)
    t = ws.origins[:, 1]
    assert t.min() == 9                                               # first full window
    assert (t + 5 < 400).all()                                        # furthest target inside the period
    touching_gap = (t >= 200) & (t - 9 <= 209)
    target_in_gap = ((t + 1 >= 200) & (t + 1 <= 209)) | ((t + 5 >= 200) & (t + 5 <= 209))
    assert not (touching_gap | target_in_gap).any()
    assert len(make_windows(blocks, period, lookback=10, horizon_steps=[1, 5], pre=p, stride=3)) < len(ws) / 2.5


def test_window_alignment_inputs_targets_and_baselines() -> None:
    p, blocks = _setup()
    ws = make_windows(blocks, Period("all", T0, T0 + pd.Timedelta(days=1)), lookback=6, horizon_steps=[1, 3],
                      pre=p, season_steps=60)
    flow = lambda arr: p.inverse_targets(arr)[..., 0]                # noqa: E731
    t = ws.origins[:, 1].astype(float)
    np.testing.assert_allclose(flow(ws.y())[:, 0], t + 1, atol=1e-3)   # target = value at t+1
    np.testing.assert_allclose(flow(ws.y())[:, 1], t + 3, atol=1e-3)
    np.testing.assert_allclose(flow(ws.last()), t, atol=1e-3)
    seq = ws.seq()
    assert seq.shape == (len(ws), 6, len(p.input_columns))
    np.testing.assert_allclose(seq[:, -1, 0] * p.std["flow_veh_h"] + p.mean["flow_veh_h"], t, atol=1e-3)
    later = t >= 60
    np.testing.assert_allclose(flow(ws.seasonal())[later, 0], t[later] + 1 - 60, atol=1e-3)


def test_tabular_features_are_past_only() -> None:
    p, blocks = _setup()
    mat, names = tabular_matrix(blocks[0], p)
    raw = lambda v: v * p.std["flow_veh_h"] + p.mean["flow_veh_h"]  # noqa: E731
    assert raw(mat[100, names.index("flow_veh_h_t")]) == pytest.approx(100, abs=1e-3)
    assert raw(mat[100, names.index("flow_veh_h_lag5")]) == pytest.approx(95, abs=1e-3)
    assert raw(mat[100, names.index("flow_veh_h_mean3")]) == pytest.approx(99, abs=1e-3)   # mean(98, 99, 100)
    assert np.isnan(mat[2, names.index("flow_veh_h_lag5")])


def test_spec_validation() -> None:
    with pytest.raises(ValueError):
        pre_for(spec(targets=["queue_length_m"]))
    with pytest.raises(ValueError):
        pre_for(spec(weather=["snow_depth"]))
