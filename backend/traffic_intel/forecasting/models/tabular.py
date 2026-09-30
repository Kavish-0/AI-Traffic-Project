"""Tabular forecasters on engineered lag features (windows.tabular_matrix).

linear         ordinary least squares (``alpha: 0``) or ridge - the learned BASELINE.
               Fast, interpretable coefficients, but linear in lags: cannot model
               regime changes (free flow <-> queueing) or interactions.
random_forest  bagged regression trees. Non-linear, robust, little tuning; cannot
               extrapolate beyond training range; large on disk.
xgboost        gradient-boosted trees with early stopping on the validation period.
               Usually the strongest tabular model; more hyper-parameters.

All three predict every horizon and target jointly (multi-output), from the
same features, so differences come from the model class, not the inputs.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from traffic_intel.forecasting.data.windows import WindowSet
from traffic_intel.forecasting.interfaces import FORECASTERS, LearnedForecaster

log = logging.getLogger(__name__)


class _TabularForecaster(LearnedForecaster):
    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self.model: Any = None
        self.feature_names: list[str] = []

    def _make(self) -> Any:  # pragma: no cover - abstract
        raise NotImplementedError

    def fit(self, train: WindowSet, valid: WindowSet | None = None) -> dict[str, Any]:
        X, y = train.tabular(), self._targets(train)
        self.feature_names = train.tabular_names
        t0 = time.perf_counter()
        self.model = self._make()
        self._fit(X, y, valid)
        return {"fit_s": round(time.perf_counter() - t0, 2), "n_train": len(X), "n_features": X.shape[1]}

    def _fit(self, X: np.ndarray, y: np.ndarray, valid: WindowSet | None) -> None:
        self.model.fit(X, y)

    def predict(self, data: WindowSet) -> np.ndarray:
        return self._levels(np.asarray(self.model.predict(data.tabular())), data)

    def _save_artifact(self, directory: Path) -> None:
        joblib.dump({"model": self.model, "feature_names": self.feature_names}, directory / "model.joblib",
                    compress=3)

    def _load_artifact(self, directory: Path) -> None:
        d = joblib.load(directory / "model.joblib")
        self.model, self.feature_names = d["model"], d["feature_names"]


@FORECASTERS.register("linear")
class LinearForecaster(_TabularForecaster):
    name = "linear"
    family = "linear"
    framework = "sklearn"

    def __init__(self, alpha: float = 0.0, **params: Any) -> None:
        super().__init__(alpha=alpha, **params)

    def _make(self) -> Any:
        from sklearn.linear_model import LinearRegression, Ridge

        a = float(self.params["alpha"])
        return LinearRegression() if a == 0 else Ridge(alpha=a)


@FORECASTERS.register("random_forest")
class RandomForestForecaster(_TabularForecaster):
    name = "random_forest"
    family = "tree"
    framework = "sklearn"

    def __init__(self, n_estimators: int = 200, max_depth: int | None = 16, min_samples_leaf: int = 5,
                 max_features: float | str = 0.5, max_samples: float | None = 0.5, n_jobs: int = -1,
                 random_state: int = 0, **params: Any) -> None:
        super().__init__(n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                         max_features=max_features, max_samples=max_samples, n_jobs=n_jobs,
                         random_state=random_state, **params)

    def _make(self) -> Any:
        from sklearn.ensemble import RandomForestRegressor

        p = self.params
        return RandomForestRegressor(n_estimators=p["n_estimators"], max_depth=p["max_depth"],
                                     min_samples_leaf=p["min_samples_leaf"], max_features=p["max_features"],
                                     max_samples=p["max_samples"], n_jobs=p["n_jobs"],
                                     random_state=p["random_state"])


@FORECASTERS.register("xgboost")
class XGBoostForecaster(_TabularForecaster):
    name = "xgboost"
    family = "tree"
    framework = "xgboost"

    def __init__(self, n_estimators: int = 800, learning_rate: float = 0.05, max_depth: int = 6,
                 subsample: float = 0.8, colsample_bytree: float = 0.8, min_child_weight: float = 5.0,
                 early_stopping_rounds: int = 50, device: str = "auto", random_state: int = 0,
                 **params: Any) -> None:
        super().__init__(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth,
                         subsample=subsample, colsample_bytree=colsample_bytree, min_child_weight=min_child_weight,
                         early_stopping_rounds=early_stopping_rounds, device=device, random_state=random_state,
                         **params)

    def _device(self) -> str:
        d = str(self.params["device"])
        if d != "auto":
            return d
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _make(self) -> Any:
        from xgboost import XGBRegressor

        p = self.params
        return XGBRegressor(
            n_estimators=p["n_estimators"], learning_rate=p["learning_rate"], max_depth=p["max_depth"],
            subsample=p["subsample"], colsample_bytree=p["colsample_bytree"],
            min_child_weight=p["min_child_weight"], tree_method="hist", device=self._device(),
            early_stopping_rounds=p["early_stopping_rounds"], random_state=p["random_state"],
            multi_strategy="one_output_per_tree",
        )

    def _fit(self, X: np.ndarray, y: np.ndarray, valid: WindowSet | None) -> None:
        if valid is None or len(valid) == 0:
            self.model.set_params(early_stopping_rounds=None)
            self.model.fit(X, y, verbose=False)
            return
        self.model.fit(X, y, eval_set=[(valid.tabular(), self._targets(valid))], verbose=False)
        log.info("xgboost best iteration %s", getattr(self.model, "best_iteration", None))

    def predict(self, data: WindowSet) -> np.ndarray:
        self.model.set_params(device="cpu")      # avoid host<->device copies for numpy input at inference
        return super().predict(data)

    def _save_artifact(self, directory: Path) -> None:
        self.model.save_model(directory / "model.json")
        joblib.dump({"feature_names": self.feature_names}, directory / "features.joblib")

    def _load_artifact(self, directory: Path) -> None:
        from xgboost import XGBRegressor

        self.model = XGBRegressor()
        self.model.load_model(directory / "model.json")
        self.feature_names = joblib.load(directory / "features.joblib")["feature_names"]
