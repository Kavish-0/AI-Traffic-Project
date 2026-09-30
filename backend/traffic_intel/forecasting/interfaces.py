"""Forecasting interface (Module 4).

Every forecaster - naive baselines, linear regression, random forest, XGBoost,
LSTM/GRU, temporal Transformer - implements the same contract, so all of them
are trained, evaluated and served by the same code and compared on identical
samples.

    fit(train: WindowSet, valid: WindowSet | None) -> training info   (offline only)
    predict(data: WindowSet) -> ndarray [N, H, K]                     (scaled target levels)

``WindowSet`` (forecasting/data/windows.py) gives each model family what it
needs from the same samples: engineered tabular features, [N, L, F] input
sequences, or the last observed value. Multi-horizon forecasting is *direct*:
one model predicts all horizons (5 / 15 / 30 min) at once; the horizon axis
is explicit in every output.

Learned models may forecast the change from the current value
(``target_transform: delta``, default) instead of the absolute level; they
always return levels.
"""

from __future__ import annotations

import json
from abc import abstractmethod
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from traffic_intel.ai.model_store import TrainableModel
from traffic_intel.core.registry import Registry
from traffic_intel.forecasting.data.windows import WindowSet


class Forecaster(TrainableModel):
    name: ClassVar[str] = ""
    family: ClassVar[str] = ""           # baseline | linear | tree | deep
    framework: ClassVar[str] = ""        # numpy | sklearn | xgboost | torch

    def __init__(self, **params: Any) -> None:
        self.params: dict[str, Any] = dict(params)

    @property
    def lookback_steps(self) -> int:
        """Input steps this model uses (tabular models: derived from lags)."""
        return int(self.params.get("lookback_steps", 1))

    @abstractmethod
    def fit(self, train: WindowSet, valid: WindowSet | None = None) -> dict[str, Any]: ...

    @abstractmethod
    def predict(self, data: WindowSet) -> np.ndarray: ...

    # persistence: params.json + model-specific artifact files ---------------
    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "params.json").write_text(json.dumps({"name": self.name, "params": self.params}, indent=2),
                                               encoding="utf-8")
        self._save_artifact(directory)

    @classmethod
    def load(cls, directory: Path) -> Forecaster:
        info = json.loads((directory / "params.json").read_text(encoding="utf-8"))
        model = cls(**info["params"])
        model._load_artifact(directory)
        return model

    def _save_artifact(self, directory: Path) -> None:  # noqa: B027 - baselines have none
        pass

    def _load_artifact(self, directory: Path) -> None:  # noqa: B027
        pass


class LearnedForecaster(Forecaster):
    """Shared target handling for trained models."""

    def __init__(self, target_transform: str = "delta", **params: Any) -> None:
        if target_transform not in ("delta", "level"):
            raise ValueError("target_transform must be 'delta' or 'level'")
        super().__init__(target_transform=target_transform, **params)

    def _targets(self, ws: WindowSet) -> np.ndarray:
        y = ws.y()
        if self.params["target_transform"] == "delta":
            y = y - ws.last()[:, None, :]
        return y.reshape(len(ws), -1)

    def _levels(self, flat: np.ndarray, ws: WindowSet) -> np.ndarray:
        out = flat.reshape(len(ws), len(ws.horizon_steps), ws.n_targets)
        if self.params["target_transform"] == "delta":
            out = out + ws.last()[:, None, :]
        return out.astype(np.float32)


FORECASTERS: Registry[Forecaster] = Registry("forecaster")
for _name, _module in {
    "persistence": "traffic_intel.forecasting.models.baselines",
    "seasonal_naive": "traffic_intel.forecasting.models.baselines",
    "linear": "traffic_intel.forecasting.models.tabular",
    "random_forest": "traffic_intel.forecasting.models.tabular",
    "xgboost": "traffic_intel.forecasting.models.tabular",
    "lstm": "traffic_intel.forecasting.models.recurrent",
    "gru": "traffic_intel.forecasting.models.recurrent",
    "transformer": "traffic_intel.forecasting.models.transformer",
}.items():
    FORECASTERS.register_lazy(_name, _module)


def forecaster_class(name: str) -> type[Forecaster]:
    """The registered class for ``name`` (for Forecaster.load)."""
    factory = FORECASTERS.get(name)
    return factory if isinstance(factory, type) else type(factory())
