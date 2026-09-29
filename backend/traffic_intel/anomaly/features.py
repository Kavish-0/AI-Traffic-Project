"""Site-independent feature vectors for learned anomaly detectors (Isolation Forest, autoencoder).

Raw traffic values differ by site (lanes, ROI length, demand), so a model trained
on one approach would flag another's normal state. Every feature is therefore a
*deviation from the approach's own recent past*: z-scores against the trailing
``window`` windows, one-window changes scaled the same way, and the inflow/outflow
imbalance ratio. The same function is used for training (whole series) and scoring
(the newest window), so there is no train/serve skew.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

COLUMNS = ["avg_speed_kmh", "density_ratio", "occupancy_pct", "queue_length_m", "flow_veh_h"]
FEATURE_NAMES = [f"z_{c}" for c in COLUMNS] + ["dz_avg_speed_kmh", "dz_queue_length_m", "flow_imbalance"]


def deviation_features(history: pd.DataFrame, window: int = 15, min_periods: int = 5) -> pd.DataFrame:
    """One row per window (NaN where the trailing baseline is too short)."""
    out = {}
    for c in COLUMNS:
        x = pd.to_numeric(history[c], errors="coerce") if c in history else pd.Series(np.nan, index=history.index)
        prev = x.shift(1)
        mean = prev.rolling(window, min_periods=min_periods).mean()
        std = prev.rolling(window, min_periods=min_periods).std()
        floor = 0.05 * mean.abs() + 1e-3
        z = (x - mean) / np.maximum(std.fillna(0), floor)
        out[f"z_{c}"] = z.clip(-10, 10)
        if c in ("avg_speed_kmh", "queue_length_m"):
            out[f"dz_{c}"] = ((x - prev) / np.maximum(std.fillna(0), floor)).clip(-10, 10)
    inflow = pd.to_numeric(history.get("inflow", pd.Series(np.nan, index=history.index)), errors="coerce")
    outflow = pd.to_numeric(history.get("outflow", pd.Series(np.nan, index=history.index)), errors="coerce")
    i3, o3 = inflow.rolling(3, min_periods=1).sum(), outflow.rolling(3, min_periods=1).sum()
    out["flow_imbalance"] = ((i3 - o3) / np.maximum(i3, 1.0)).clip(-1, 1)
    return pd.DataFrame(out, index=history.index)[FEATURE_NAMES]
