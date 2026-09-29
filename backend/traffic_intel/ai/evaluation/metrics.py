"""Forecast accuracy metrics, computed in original units (after inverse scaling).

    MAE   mean |y - y_hat|                   same unit as the target; robust to outliers
    RMSE  sqrt(mean (y - y_hat)^2)           penalises large errors (e.g. missed congestion onsets)
    R2    1 - SS_res / SS_tot                share of variance explained (1 = perfect, 0 = predicts the mean,
                                             < 0 = worse than the mean of the test period)
    MAPE  mean |y - y_hat| / |y|  (%)        only where |y| >= 1 unit (km/h, veh/h, %): near-zero actuals (night
                                             flow, outage artefacts) would make it explode - the DCRNN convention
    skill 1 - RMSE_model / RMSE_persistence  > 0 = better than "no change"; the key question for this project
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def mae(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.abs(y - p)))


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - p) ** 2)))


MAPE_FLOOR = 1.0


def mape(y: np.ndarray, p: np.ndarray, floor: float = MAPE_FLOOR) -> float:
    m = np.abs(y) >= floor
    if not m.any():
        return float("nan")
    return float(100.0 * np.mean(np.abs(y[m] - p[m]) / np.abs(y[m])))


def r2(y: np.ndarray, p: np.ndarray) -> float:
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    return 1.0 - float(np.sum((y - p) ** 2)) / ss_tot


def metric_table(y: np.ndarray, preds: dict[str, np.ndarray], horizons_min: list[int],
                 targets: list[str]) -> pd.DataFrame:
    """Long table: model, target, horizon_min, MAE, RMSE, R2, MAPE, skill_vs_persistence.

    ``y`` and every prediction are [N, H, K] in original units.
    """
    rows = []
    pers = preds.get("persistence")
    for model, p in preds.items():
        for k, tgt in enumerate(targets):
            for j, h in enumerate(horizons_min):
                yt, pt = y[:, j, k], p[:, j, k]
                row: dict[str, Any] = {"model": model, "target": tgt, "horizon_min": h,
                       "MAE": mae(yt, pt), "RMSE": rmse(yt, pt), "R2": r2(yt, pt), "MAPE": mape(yt, pt)}
                if pers is not None:
                    rp = rmse(yt, pers[:, j, k])
                    row["skill_vs_persistence"] = 1.0 - row["RMSE"] / rp if rp > 0 else float("nan")
                rows.append(row)
    return pd.DataFrame(rows)


def summary_table(long: pd.DataFrame) -> pd.DataFrame:
    """Model | target | MAE | RMSE | R2 | MAPE | skill, averaged over horizons."""
    cols = [c for c in ("MAE", "RMSE", "R2", "MAPE", "skill_vs_persistence") if c in long]
    return long.groupby(["target", "model"], sort=False)[cols].mean().reset_index()
