"""Learned series detectors: Isolation Forest and a small autoencoder.

Both are fitted OFFLINE on normal traffic windows (``python -m traffic_intel.ai.training
train anomaly``) and see the site-independent deviation features of
``anomaly/features.py``. Raw outputs are mapped to [0, 1] with the TRAINING score
distribution:

    score = ramp(raw; q_normal, q_extreme)   q_normal = 99th percentile of normal windows
                                             q_extreme = 99.9th percentile x 2

so 0 means "within the normal 99%" and 1 means "far beyond anything seen in
normal training data". It is an outlier score, not an incident probability.

isolation_forest  anomaly = few random splits isolate the point (sklearn IsolationForest)
autoencoder       anomaly = high reconstruction error of a bottleneck MLP (sklearn
                  MLPRegressor trained to reproduce its input: 8 -> 16 -> 4 -> 16 -> 8)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from traffic_intel.ai.model_store import TrainableModel
from traffic_intel.anomaly.detectors.statistical import ramp
from traffic_intel.anomaly.features import FEATURE_NAMES, deviation_features
from traffic_intel.anomaly.interfaces import SERIES_ANOMALY_DETECTORS, LearnedSeriesAnomalyDetector
from traffic_intel.core.enums import AnomalyScope, AnomalyType, ModelTask
from traffic_intel.core.schemas import AnomalySignal

log = logging.getLogger(__name__)


class _Learned(LearnedSeriesAnomalyDetector, TrainableModel):
    kind = ""

    def __init__(self, window: int = 15, min_periods: int = 5, **params: Any) -> None:
        self.window, self.min_periods = window, min_periods
        self.params = {"window": window, "min_periods": min_periods, **params}
        self.model: Any = None
        self.q_normal = self.q_extreme = 0.0

    def _raw(self, X: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def _make(self) -> Any:  # pragma: no cover - abstract
        raise NotImplementedError

    def features(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Per-series deviation features; frame may contain several series (series_id)."""
        if "series_id" not in frame:
            return deviation_features(frame, self.window, self.min_periods)
        parts = [deviation_features(g.sort_values("timestamp"), self.window, self.min_periods)
                 for _, g in frame.groupby("series_id", sort=False)]
        return pd.concat(parts).loc[frame.index]

    def fit(self, train: pd.DataFrame) -> None:
        X = self.features(train).dropna().to_numpy()
        if len(X) < 50:
            raise ValueError(f"not enough normal windows to fit {self.kind} ({len(X)})")
        self.model = self._make()
        self._fit(X)
        raw = self._raw(X)
        self.q_normal = float(np.quantile(raw, 0.99))
        self.q_extreme = float(np.quantile(raw, 0.999)) * 2.0
        self.params["n_train"] = len(X)

    def _fit(self, X: np.ndarray) -> None:
        self.model.fit(X)

    def score_frame(self, frame: pd.DataFrame) -> np.ndarray:
        """Score every window (NaN where features are unavailable) - used for evaluation."""
        F = self.features(frame)
        ok = F.notna().all(axis=1).to_numpy()
        out = np.full(len(F), np.nan)
        if ok.any():
            raw = self._raw(F[ok].to_numpy())
            out[ok] = [ramp(r, self.q_normal, max(self.q_extreme, self.q_normal + 1e-9)) for r in raw]
        return out

    def score(self, approach_id: str, history: pd.DataFrame) -> list[AnomalySignal]:
        if self.model is None or len(history) < self.min_periods + 1:
            return []
        F = deviation_features(history, self.window, self.min_periods).iloc[[-1]]
        if F.isna().any(axis=None):
            return []
        raw = float(self._raw(F.to_numpy())[0])
        s = ramp(raw, self.q_normal, max(self.q_extreme, self.q_normal + 1e-9))
        if s <= 0:
            return []
        top = F.iloc[0].abs().sort_values(ascending=False).head(3)
        return [AnomalySignal(
            approach_id=approach_id, camera_id=None,
            timestamp=pd.Timestamp(history["timestamp"].iloc[-1]).to_pydatetime(),
            type=AnomalyType.STATISTICAL_OUTLIER, scope=AnomalyScope.SERIES, detector=self.kind, score=round(s, 3),
            details={"raw_score": round(raw, 4), "normal_q99": round(self.q_normal, 4),
                     "top_deviations": {k: round(float(F.iloc[0][k]), 2) for k in top.index}})]

    # persistence --------------------------------------------------------
    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "q_normal": self.q_normal, "q_extreme": self.q_extreme,
                     "params": self.params, "features": FEATURE_NAMES}, directory / "model.joblib", compress=3)

    @classmethod
    def load(cls, directory: Path) -> _Learned:
        d = joblib.load(directory / "model.joblib")
        obj = cls(**{k: v for k, v in d["params"].items() if k != "n_train"})
        obj.model, obj.q_normal, obj.q_extreme = d["model"], d["q_normal"], d["q_extreme"]
        obj.params = d["params"]
        return obj


class IsolationForestDetector(_Learned):
    name = kind = "isolation_forest"

    def __init__(self, n_estimators: int = 200, max_samples: int | str = 1024, random_state: int = 0,
                 **params: Any) -> None:
        super().__init__(n_estimators=n_estimators, max_samples=max_samples, random_state=random_state, **params)

    def _make(self) -> Any:
        from sklearn.ensemble import IsolationForest

        p = self.params
        return IsolationForest(n_estimators=p["n_estimators"], max_samples=p["max_samples"],
                               random_state=p["random_state"], n_jobs=-1)

    def _raw(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(-self.model.score_samples(X))            # higher = more anomalous


class AutoencoderDetector(_Learned):
    name = kind = "autoencoder"

    def __init__(self, hidden: tuple[int, ...] = (16, 4, 16), max_iter: int = 200, random_state: int = 0,
                 **params: Any) -> None:
        super().__init__(hidden=list(hidden), max_iter=max_iter, random_state=random_state, **params)

    def _make(self) -> Any:
        from sklearn.neural_network import MLPRegressor

        p = self.params
        return MLPRegressor(hidden_layer_sizes=tuple(p["hidden"]), max_iter=p["max_iter"], early_stopping=True,
                            random_state=p["random_state"])

    def _fit(self, X: np.ndarray) -> None:
        self.model.fit(X, X)

    def _raw(self, X: np.ndarray) -> np.ndarray:
        return np.mean((self.model.predict(X) - X) ** 2, axis=1)


def _from_registry(cls: type[_Learned], version: str = "production", **_: Any) -> _Learned:
    """Load the fitted detector from models/anomaly/<name>/<version> (train it first)."""
    from traffic_intel.ai.model_store import FileModelStore
    from traffic_intel.core.config import get_settings

    store = FileModelStore(get_settings().models_dir)
    v = store.resolve_version(ModelTask.ANOMALY, cls.kind, version)
    return cls.load(store._dir(ModelTask.ANOMALY, cls.kind, v))


@SERIES_ANOMALY_DETECTORS.register("isolation_forest")
def _iforest(**kw: Any) -> _Learned:
    return _from_registry(IsolationForestDetector, **kw)


@SERIES_ANOMALY_DETECTORS.register("autoencoder")
def _autoencoder(**kw: Any) -> _Learned:
    return _from_registry(AutoencoderDetector, **kw)
