"""Chronological splits and supervised windows.

Split rule (no shuffling, no leakage)
    The global time range is cut into consecutive periods train | val | test.
    A sample belongs to a period if its *forecast origin* t lies in the period AND
    its furthest target t + max_horizon also lies in it (targets never cross into
    the next period). Input windows may reach back into the previous period - that
    is past data and legitimately available at forecast time.

Windows
    origin t, lookback L:  inputs X[t-L+1 .. t]  (all dynamic features observed)
    targets              :  Y[t + h] for every horizon h (in steps), all observed
    Every model in an experiment is evaluated on the SAME origins (built with the
    largest lookback any model needs); models with a shorter lookback use the tail.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from traffic_intel.forecasting.data.preprocess import Preprocessor


class SplitConfig(BaseModel):
    train: float = Field(0.6, gt=0)
    val: float = Field(0.2, gt=0)
    test: float = Field(0.2, gt=0)
    # Optional explicit boundaries (ISO timestamps); override the fractions.
    val_start: str | None = None
    test_start: str | None = None


@dataclass(frozen=True)
class Period:
    name: str
    start: pd.Timestamp       # inclusive
    end: pd.Timestamp         # exclusive

    def as_dict(self) -> dict[str, str]:
        return {"start": self.start.isoformat(), "end": self.end.isoformat()}


def chronological_split(times: pd.Series | pd.DatetimeIndex, cfg: SplitConfig,
                        resolution_s: int | None = None) -> dict[str, Period]:
    """Consecutive train | val | test periods. Boundaries snap to the data's time grid."""
    t = pd.DatetimeIndex(times)
    # periods are half-open; the data end is one step after the last timestamp
    t0, t1 = t.min(), t.max() + pd.Timedelta(seconds=resolution_s or 1)
    if cfg.val_start and cfg.test_start:
        v, s = pd.Timestamp(cfg.val_start), pd.Timestamp(cfg.test_start)
        v = v.tz_localize("UTC") if v.tzinfo is None else v.tz_convert("UTC")
        s = s.tz_localize("UTC") if s.tzinfo is None else s.tz_convert("UTC")
    else:
        total = cfg.train + cfg.val + cfg.test
        span = t1 - t0
        v = t0 + span * (cfg.train / total)
        s = t0 + span * ((cfg.train + cfg.val) / total)
        if resolution_s:
            v, s = v.floor(f"{resolution_s}s"), s.floor(f"{resolution_s}s")
    if not (t0 < v < s < t1):
        raise ValueError(f"invalid split boundaries: {t0} < {v} < {s} < {t1} violated")
    return {"train": Period("train", t0, v), "val": Period("val", v, s), "test": Period("test", s, t1)}


@dataclass
class SeriesBlock:
    series_id: str
    times: pd.DatetimeIndex
    X: np.ndarray            # [T, F] scaled model inputs (dynamic may be NaN)
    Y: np.ndarray            # [T, K] scaled targets (NaN = unobserved)
    ok: np.ndarray           # [T] all dynamic inputs finite


def build_blocks(transformed: pd.DataFrame, pre: Preprocessor) -> list[SeriesBlock]:
    blocks = []
    for sid, g in transformed.groupby("series_id", sort=True):
        g = g.sort_values("timestamp")
        X = g[pre.input_columns].to_numpy(np.float32)
        Y = g[pre.target_columns].to_numpy(np.float32)
        ok = np.isfinite(g[pre.spec.dynamic].to_numpy()).all(axis=1)
        blocks.append(SeriesBlock(str(sid), pd.DatetimeIndex(g["timestamp"]), X, Y, ok))
    return blocks


def tabular_matrix(block: SeriesBlock, pre: Preprocessor) -> tuple[np.ndarray, list[str]]:
    """Per-step engineered features for tabular models (computed on the past only):
    dynamic values at t, at t-lag, rolling means ending at t, first differences,
    plus context columns at t."""
    spec = pre.spec
    nd = len(spec.dynamic)
    D = block.X[:, :nd].astype(np.float64)
    C = block.X[:, nd:]
    cols: list[np.ndarray] = [D]
    names = [f"{c}_t" for c in spec.dynamic]
    T = len(D)
    for lag in spec.lags:
        sh = np.full_like(D, np.nan)
        if lag < T:
            sh[lag:] = D[:-lag]
        cols.append(sh)
        names += [f"{c}_lag{lag}" for c in spec.dynamic]
    cs = np.vstack([np.zeros((1, nd)), np.cumsum(np.nan_to_num(D), axis=0)])
    for w in spec.rolling:
        rm = np.full_like(D, np.nan)
        if w <= T:
            rm[w - 1:] = (cs[w:] - cs[:-w]) / w
        cols.append(rm)
        names += [f"{c}_mean{w}" for c in spec.dynamic]
    diff = np.full_like(D, np.nan)
    diff[1:] = D[1:] - D[:-1]
    cols.append(diff)
    names += [f"{c}_diff1" for c in spec.dynamic]
    cols.append(C.astype(np.float64))
    names += pre.context_columns
    return np.hstack(cols).astype(np.float32), names


class WindowSet:
    """Samples of one period: origins (block index, time index) + helpers to
    materialise inputs/targets for each model family."""

    def __init__(self, blocks: list[SeriesBlock], origins: np.ndarray, lookback: int, horizon_steps: list[int],
                 pre: Preprocessor, season_steps: int | None = None) -> None:
        self.blocks = blocks
        self.origins = origins.astype(np.int64).reshape(-1, 2)
        self.lookback = lookback
        self.horizon_steps = list(horizon_steps)
        self.pre = pre
        self.season_steps = season_steps
        self._tab_cache: dict[int, np.ndarray] = {}
        self.tabular_names: list[str] = []

    def __len__(self) -> int:
        return len(self.origins)

    @property
    def n_features(self) -> int:
        return int(self.blocks[0].X.shape[1])

    @property
    def n_targets(self) -> int:
        return int(self.blocks[0].Y.shape[1])

    def subset(self, mask: np.ndarray) -> WindowSet:
        return WindowSet(self.blocks, self.origins[mask], self.lookback, self.horizon_steps, self.pre,
                         self.season_steps)

    # targets ------------------------------------------------------------
    def y(self) -> np.ndarray:
        """[N, H, K] scaled target levels at t + h."""
        out = np.empty((len(self), len(self.horizon_steps), self.n_targets), np.float32)
        for b, blk in enumerate(self.blocks):
            m = self.origins[:, 0] == b
            t = self.origins[m, 1]
            for j, h in enumerate(self.horizon_steps):
                out[m, j] = blk.Y[t + h]
        return out

    def last(self) -> np.ndarray:
        """[N, K] scaled target values at the origin (what persistence predicts)."""
        out = np.empty((len(self), self.n_targets), np.float32)
        for b, blk in enumerate(self.blocks):
            m = self.origins[:, 0] == b
            out[m] = blk.Y[self.origins[m, 1]]
        return out

    def seasonal(self) -> np.ndarray:
        """[N, H, K] value one season before each target time; persistence where unavailable."""
        last = self.last()
        out = np.repeat(last[:, None, :], len(self.horizon_steps), axis=1)
        if not self.season_steps:
            return out
        for b, blk in enumerate(self.blocks):
            m = np.flatnonzero(self.origins[:, 0] == b)
            t = self.origins[m, 1]
            for j, h in enumerate(self.horizon_steps):
                idx = t + h - self.season_steps
                good = idx >= 0
                vals = np.full((len(m), self.n_targets), np.nan, np.float32)
                vals[good] = blk.Y[idx[good]]
                fine = np.isfinite(vals).all(axis=1)
                out[m[fine], j] = vals[fine]
        return out

    # inputs -------------------------------------------------------------
    def seq(self, lookback: int | None = None) -> np.ndarray:
        """[N, L, F] input windows (materialised)."""
        L = lookback or self.lookback
        out = np.empty((len(self), L, self.n_features), np.float32)
        ar = np.arange(-L + 1, 1)
        for b, blk in enumerate(self.blocks):
            m = self.origins[:, 0] == b
            idx = self.origins[m, 1][:, None] + ar[None, :]
            out[m] = blk.X[idx]
        return out

    def tabular(self) -> np.ndarray:
        """[N, F_tab] engineered features at each origin."""
        out = None
        for b, blk in enumerate(self.blocks):
            if b not in self._tab_cache:
                self._tab_cache[b], self.tabular_names = tabular_matrix(blk, self.pre)
            mat = self._tab_cache[b]
            if out is None:
                out = np.empty((len(self), mat.shape[1]), np.float32)
            m = self.origins[:, 0] == b
            out[m] = mat[self.origins[m, 1]]
        assert out is not None
        return out

    def concat(self) -> tuple[np.ndarray, np.ndarray]:
        """(X_all [sum T, F], global origin index [N]) for on-device window gathering."""
        offsets = np.cumsum([0] + [len(b.X) for b in self.blocks[:-1]])
        X_all = np.concatenate([b.X for b in self.blocks]).astype(np.float32)
        return X_all, offsets[self.origins[:, 0]] + self.origins[:, 1]

    # metadata -----------------------------------------------------------
    def origin_times(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex([self.blocks[b].times[t] for b, t in self.origins])

    def series_ids(self) -> np.ndarray:
        return np.array([self.blocks[b].series_id for b in self.origins[:, 0]])


def make_windows(
    blocks: list[SeriesBlock],
    period: Period,
    *,
    lookback: int,
    horizon_steps: list[int],
    pre: Preprocessor,
    stride: int = 1,
    season_steps: int | None = None,
) -> WindowSet:
    """Origins inside ``period`` whose input window is fully observed and whose
    targets (all horizons) are observed and inside the period."""
    max_h = max(horizon_steps)
    res = pd.Timedelta(seconds=pre.resolution_s)
    origins = []
    for b, blk in enumerate(blocks):
        T = len(blk.X)
        cs = np.concatenate([[0], np.cumsum(blk.ok.astype(np.int64))])
        t = np.arange(lookback - 1, T - max_h)
        if len(t) == 0:
            continue
        window_ok = (cs[t + 1] - cs[t + 1 - lookback]) == lookback
        tgt_ok = np.ones(len(t), bool)
        for h in horizon_steps:
            tgt_ok &= np.isfinite(blk.Y[t + h]).all(axis=1)
        times = blk.times[t]
        in_period = (times >= period.start) & (times + res * max_h < period.end)
        sel = t[window_ok & tgt_ok & np.asarray(in_period)]
        sel = sel[:: max(1, stride)]
        origins.append(np.stack([np.full(len(sel), b), sel], axis=1))
    o = np.concatenate(origins) if origins else np.zeros((0, 2), np.int64)
    return WindowSet(blocks, o, lookback, horizon_steps, pre, season_steps)
