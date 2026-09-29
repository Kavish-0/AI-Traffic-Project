"""Feature specification and the (fit-on-train-only) preprocessor.

Pipeline per series:
  1. regularise   reindex to the dataset's time grid; missing steps become NaN rows
  2. gap handling interpolate traffic gaps of <= ``max_gap_fill_steps``; longer gaps stay
                  NaN and every window touching them is excluded later (never invented)
  3. context      calendar features from timestamps (local time); weather numeric +
                  one-hot condition
  4. scaling      z-score for traffic and numeric weather columns with statistics from
                  the TRAIN period only; calendar features are already in [-1, 1] / {0, 1}
  5. imputation   remaining NaN in *context* columns -> 0 after scaling (= train mean);
                  traffic NaN are kept (they mark invalid windows)

The fitted state is plain JSON (``preprocessor.json``) stored next to every
model version, so inference reproduces training exactly.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from traffic_intel.core.feature_schema import CALENDAR_GROUPS, TRAFFIC_COLUMNS, WEATHER_COLUMNS
from traffic_intel.forecasting.data.context import calendar_frame

log = logging.getLogger(__name__)


class FeatureSpec(BaseModel):
    """``features:`` section of a training config."""

    model_config = ConfigDict(extra="forbid")

    dynamic: list[str]                                   # observed traffic series used as inputs
    targets: list[str]                                   # subset of dynamic to forecast
    weather: list[str] = []                              # subset of WEATHER_COLUMNS
    calendar: list[str] = ["hour", "day_of_week", "holiday", "rush_hour"]
    rush_hours: list[tuple[float, float]] = [(7.0, 10.0), (16.0, 19.0)]
    max_gap_fill_steps: int = Field(3, ge=0)
    # tabular models: lagged copies and rolling means of dynamic columns (in steps)
    lags: list[int] = [1, 2, 3, 6, 12]
    rolling: list[int] = [3, 12]

    def check(self) -> None:
        off_schema = set(self.dynamic) - set(TRAFFIC_COLUMNS)
        if off_schema:
            raise ValueError(f"dynamic features not in feature_schema.TRAFFIC_COLUMNS: {sorted(off_schema)}")
        missing = set(self.targets) - set(self.dynamic)
        if missing:
            raise ValueError(f"targets must be dynamic features too: {sorted(missing)}")
        bad = set(self.weather) - set(WEATHER_COLUMNS)
        if bad:
            raise ValueError(f"unknown weather columns {sorted(bad)}; allowed {WEATHER_COLUMNS}")
        bad = set(self.calendar) - set(CALENDAR_GROUPS)
        if bad:
            raise ValueError(f"unknown calendar features {sorted(bad)}; allowed {list(CALENDAR_GROUPS)}")


def _fill_short_gaps(df: pd.DataFrame, max_steps: int) -> pd.DataFrame:
    """Linearly interpolate interior NaN runs of length <= max_steps; longer runs stay NaN
    entirely (pandas' ``limit`` would fill the first steps of long gaps - never wanted here)."""
    filled = df.interpolate(limit_area="inside")
    for c in df.columns:
        na = df[c].isna().to_numpy()
        if not na.any():
            continue
        run_id = np.cumsum(~na)                       # constant within a NaN run
        lengths = pd.Series(na).groupby(run_id).transform("sum").to_numpy()
        too_long = na & (lengths > max_steps)
        filled.loc[too_long, c] = np.nan
    return filled


class Preprocessor:
    def __init__(self, spec: FeatureSpec, *, resolution_s: int, tz: str, holiday_country: str | None) -> None:
        spec.check()
        self.spec = spec
        self.resolution_s = resolution_s
        self.tz = tz
        self.holiday_country = holiday_country
        self.mean: dict[str, float] = {}
        self.std: dict[str, float] = {}
        self.categories: list[str] = []
        self.dropped: list[str] = []
        self.fitted = False
        self.fit_period: tuple[str, str] | None = None

    # ------------------------------------------------------------------ #
    def regularize(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Per series: full time grid, short traffic gaps interpolated, context filled."""
        out = []
        freq = f"{self.resolution_s}s"
        dyn = [c for c in self.spec.dynamic if c in frame.columns]
        missing = set(self.spec.dynamic) - set(dyn)
        if missing:
            raise ValueError(f"dataset lacks dynamic features {sorted(missing)}")
        for sid, g in frame.groupby("series_id", sort=True):
            g = g.set_index("timestamp").sort_index()
            g = g[~g.index.duplicated(keep="last")]
            grid = pd.date_range(g.index.min(), g.index.max(), freq=freq)
            g = g.reindex(grid)
            g["observed"] = g[dyn].notna().all(axis=1).astype(np.float32)
            if self.spec.max_gap_fill_steps > 0:
                g[dyn] = _fill_short_gaps(g[dyn], self.spec.max_gap_fill_steps)
            wcols = [c for c in self.spec.weather if c in g.columns]
            if wcols:
                g[wcols] = g[wcols].ffill().bfill()
            g["series_id"] = sid
            out.append(g.rename_axis("timestamp").reset_index())
        return pd.concat(out, ignore_index=True)

    def _context(self, df: pd.DataFrame) -> pd.DataFrame:
        times = pd.DatetimeIndex(df["timestamp"])
        cal = calendar_frame(times, tz=self.tz, holiday_country=self.holiday_country,
                             rush_hours=[tuple(r) for r in self.spec.rush_hours])  # type: ignore[misc]
        cols = [c for g in self.spec.calendar for c in CALENDAR_GROUPS[g]]
        ctx = cal[cols].reset_index(drop=True)
        for c in self.spec.weather:
            if c == "weather_condition":
                cond = df.get("weather_condition", pd.Series([None] * len(df))).reset_index(drop=True)
                for cat in (self.categories if self.fitted else sorted(set(cond.dropna()))):
                    ctx[f"weather_{cat}"] = (cond == cat).astype(np.float32).to_numpy()
            else:
                ctx[c] = pd.to_numeric(df[c], errors="coerce").to_numpy() if c in df else np.nan
        return ctx

    # ------------------------------------------------------------------ #
    def fit(self, train: pd.DataFrame) -> Preprocessor:
        """Statistics from the TRAIN period only (already regularised)."""
        self.categories = sorted(set(train["weather_condition"].dropna())) if (
            "weather_condition" in self.spec.weather and "weather_condition" in train) else []
        numeric = list(self.spec.dynamic) + [c for c in self.spec.weather if c != "weather_condition"]
        self.dropped = []
        for c in numeric:
            vals = pd.to_numeric(train[c], errors="coerce") if c in train else pd.Series(dtype=float)
            if vals.notna().sum() == 0:
                if c in self.spec.dynamic:
                    raise ValueError(f"dynamic feature '{c}' has no data in the training period")
                self.dropped.append(c)
                log.warning("feature '%s' has no data in the training period - dropped", c)
                continue
            self.mean[c] = float(vals.mean())
            self.std[c] = float(vals.std()) or 1.0
        self.fitted = True
        self.fit_period = (str(train["timestamp"].min()), str(train["timestamp"].max()))
        return self

    @property
    def weather_numeric(self) -> list[str]:
        return [c for c in self.spec.weather if c != "weather_condition" and c not in self.dropped]

    @property
    def context_columns(self) -> list[str]:
        cols = [c for g in self.spec.calendar for c in CALENDAR_GROUPS[g]]
        for c in self.spec.weather:
            if c == "weather_condition":
                cols += [f"weather_{cat}" for cat in self.categories]
            elif c not in self.dropped:
                cols.append(c)
        return cols

    @property
    def input_columns(self) -> list[str]:
        """Model input columns, in order: dynamic traffic features then context."""
        return list(self.spec.dynamic) + self.context_columns

    @property
    def target_columns(self) -> list[str]:
        return list(self.spec.targets)

    def transform(self, reg: pd.DataFrame) -> pd.DataFrame:
        """Regularised frame -> scaled model-input frame (series_id, timestamp, observed + inputs)."""
        if not self.fitted:
            raise RuntimeError("Preprocessor.transform before fit")
        ctx = self._context(reg)
        out = pd.DataFrame({"series_id": reg["series_id"].to_numpy(), "timestamp": reg["timestamp"].to_numpy(),
                            "observed": reg.get("observed", pd.Series(1.0, index=reg.index)).to_numpy()})
        for c in self.spec.dynamic:
            out[c] = ((pd.to_numeric(reg[c], errors="coerce") - self.mean[c]) / self.std[c]).to_numpy(np.float32)
        for c in self.context_columns:
            v = ctx[c].to_numpy(dtype=np.float64) if c in ctx else np.zeros(len(reg))
            if c in self.mean:
                v = (v - self.mean[c]) / self.std[c]
            out[c] = np.nan_to_num(v, nan=0.0).astype(np.float32)
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
        return out

    def scale_targets(self, values: np.ndarray) -> np.ndarray:
        m = np.array([self.mean[c] for c in self.target_columns])
        s = np.array([self.std[c] for c in self.target_columns])
        return np.asarray((values - m) / s)

    def inverse_targets(self, values: np.ndarray) -> np.ndarray:
        """[..., n_targets] scaled -> original units."""
        m = np.array([self.mean[c] for c in self.target_columns])
        s = np.array([self.std[c] for c in self.target_columns])
        return np.asarray(values * s + m)

    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        return {"spec": self.spec.model_dump(), "resolution_s": self.resolution_s, "tz": self.tz,
                "holiday_country": self.holiday_country, "mean": self.mean, "std": self.std,
                "categories": self.categories, "dropped": self.dropped, "fit_period": self.fit_period,
                "input_columns": self.input_columns, "target_columns": self.target_columns,
                "spec_hash": self.spec_hash()}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Preprocessor:
        p = cls(FeatureSpec(**d["spec"]), resolution_s=d["resolution_s"], tz=d["tz"],
                holiday_country=d["holiday_country"])
        p.mean, p.std, p.categories, p.dropped = d["mean"], d["std"], d["categories"], d["dropped"]
        p.fit_period = tuple(d["fit_period"]) if d.get("fit_period") else None
        p.fitted = True
        return p

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Preprocessor:
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def spec_hash(self) -> str:
        payload = json.dumps({"inputs": self.input_columns, "targets": self.target_columns,
                              "resolution_s": self.resolution_s}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]
