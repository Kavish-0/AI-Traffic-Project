"""Naive baselines - the reference every learned model must beat.

persistence     value in h minutes = value now. Hard to beat at short horizons.
seasonal_naive  value in h minutes = value at the same time one season earlier
                (default one day). Captures the daily cycle, ignores today's deviations.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from traffic_intel.forecasting.data.windows import WindowSet
from traffic_intel.forecasting.interfaces import FORECASTERS, Forecaster


@FORECASTERS.register("persistence")
class PersistenceForecaster(Forecaster):
    name = "persistence"
    family = "baseline"
    framework = "numpy"

    def fit(self, train: WindowSet, valid: WindowSet | None = None) -> dict[str, Any]:
        return {"trained": False}

    def predict(self, data: WindowSet) -> np.ndarray:
        return np.repeat(data.last()[:, None, :], len(data.horizon_steps), axis=1)


@FORECASTERS.register("seasonal_naive")
class SeasonalNaiveForecaster(Forecaster):
    name = "seasonal_naive"
    family = "baseline"
    framework = "numpy"

    def fit(self, train: WindowSet, valid: WindowSet | None = None) -> dict[str, Any]:
        return {"trained": False, "season_steps": train.season_steps}

    def predict(self, data: WindowSet) -> np.ndarray:
        return data.seasonal()
