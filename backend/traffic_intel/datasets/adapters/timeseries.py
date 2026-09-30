"""Time-series adapters -> the forecasting pipeline's canonical ``TimeSeriesDataset``.

    metr_la, pems_bay   DCRNN HDF5 speed matrices (index = 5-min local timestamps, one column per
                        sensor, mph, 0 = missing) -> series_id, timestamp (UTC), avg_speed_kmh
    pems08              ASTGCN npz (flow / occupancy / speed)       (wraps forecasting.data.sources)
    synthetic           project generator, already canonical         (wraps forecasting.data.sources)
    own_recordings      the camera pipeline's 1-min TrafficState windows (wraps load_observations)

Only dataset-specific parsing lives here. Gap handling, calendar/weather features, train-only
scaling, chronological splits and window/sequence creation are the forecasting preprocessor's
job (forecasting/data/preprocess.py, windows.py) and identical for every dataset. Every frame is
checked against core/feature_schema.py before it leaves the adapter.
"""

from __future__ import annotations

import glob
import logging
from typing import Any

import numpy as np
import pandas as pd

from traffic_intel.core.feature_schema import validate_frame
from traffic_intel.datasets.adapters.base import DATASET_ADAPTERS, TimeSeriesAdapter

log = logging.getLogger(__name__)

MPH_TO_KMH = 1.609344


def _checked(ds: Any) -> Any:
    validate_frame(ds.frame.columns, source=ds.name)
    return ds


def _decode(v: Any) -> Any:  # noqa: ANN401 - HDF5 attribute / array element
    return v.decode("utf-8") if isinstance(v, bytes | np.bytes_) else v


def _read_legacy_fixed_frame(path: str, key: str) -> pd.DataFrame:
    """A pandas 'fixed'-format DataFrame (one float block, datetime64[ns] row index) read with PyTables
    directly. Only the arrays are read; pickled attributes (e.g. the index ``freq``) are never unpickled."""
    import tables

    with tables.open_file(path, mode="r") as h:
        g = h.get_node(f"/{key}")
        if _decode(g._v_attrs.pandas_type) != "frame" or int(g._v_attrs.nblocks) != 1:
            raise ValueError(f"{path}[{key}]: unsupported legacy HDF5 layout (need a single-block frame)")
        columns = [_decode(c) for c in g.axis0.read()]
        items = [_decode(c) for c in g.block0_items.read()]
        if items != columns:
            raise ValueError(f"{path}[{key}]: block items do not match the column axis")
        if _decode(g.axis1._v_attrs.kind) != "datetime64":
            raise ValueError(f"{path}[{key}]: row index is not datetime64")
        index = pd.DatetimeIndex(pd.to_datetime(g.axis1.read(), unit="ns"))
        values = g.block0_values.read()
    return pd.DataFrame(values, index=index, columns=[str(c) for c in columns])


class _SpeedMatrixAdapter(TimeSeriesAdapter):
    """METR-LA / PEMS-BAY: wide HDF5 speed matrix."""

    prefix = ""
    h5_keys: tuple[str, ...] = ()

    def h5_file(self) -> str:
        self.check()
        for req in self.entry.required:
            for pat in req.pattern:
                hits = sorted(glob.glob(str(self.root / pat)))
                if hits:
                    return hits[0]
        raise FileNotFoundError(self.root)  # pragma: no cover - check() already raised

    def raw_matrix(self) -> pd.DataFrame:
        path = self.h5_file()
        try:
            import tables  # noqa: F401 - pandas' HDF5 backend
        except ImportError as exc:
            raise ImportError("reading METR-LA / PEMS-BAY needs PyTables: pip install -e \".[datasets]\"") from exc
        with pd.HDFStore(path, mode="r") as store:
            keys = [k.strip("/") for k in store]
        key = next((k for k in self.h5_keys if k in keys), keys[0] if keys else None)
        if key is None:
            raise ValueError(f"{path}: empty HDF5 file")
        try:
            df = pd.read_hdf(path, key=key)
        except TypeError:
            # DCRNN files were written by pandas 0.15 under Python 2 (bytes attributes) - pandas 3 cannot
            # parse their metadata, so read the single-block 'fixed' frame's arrays directly.
            df = _read_legacy_fixed_frame(path, key)
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(f"{path}[{key}]: expected a DatetimeIndex, got {type(df.index).__name__}")
        return df

    def load(self, sensors: list[str] | None = None, every_k: int = 1, max_sensors: int | None = None,
             start: str | None = None, end: str | None = None, weather: dict[str, float] | bool | None = None,
             holiday_country: str | None = "default", zero_is_missing: bool = True, **_: Any) -> Any:
        """Canonical frame. ``weather=True`` uses the manifest's ``weather_point``; a dict gives lat/lon;
        None/False -> traffic-only (weather is never mandatory)."""
        from traffic_intel.forecasting.data.sources import TimeSeriesDataset, _attach_weather

        tz = getattr(self.entry, "timezone", "UTC")
        wide = self.raw_matrix()
        cols = [str(c) for c in wide.columns]
        wide.columns = cols
        chosen = [str(s) for s in sensors] if sensors else cols[::max(1, every_k)]
        if max_sensors:
            chosen = chosen[:max_sensors]
        unknown = set(chosen) - set(cols)
        if unknown:
            raise ValueError(f"{self.dataset_id}: unknown sensor ids {sorted(unknown)[:5]}")
        wide = wide[chosen]
        idx = wide.index
        if idx.tz is None:
            # naive local wall-clock stamps; the DST jump / repeat hour cannot be placed -> dropped
            idx = idx.tz_localize(tz, nonexistent="NaT", ambiguous="NaT")
        utc = idx.tz_convert("UTC")
        keep = ~utc.isna()
        dropped_dst = int((~keep).sum())
        wide = wide.loc[keep]
        wide.index = utc[keep]
        wide = wide[~wide.index.duplicated()]
        if start:
            wide = wide[wide.index >= pd.Timestamp(start, tz=tz)]
        if end:
            wide = wide[wide.index < pd.Timestamp(end, tz=tz)]
        values = wide.to_numpy(dtype=np.float64) * MPH_TO_KMH
        missing_zero = int((values == 0).sum()) if zero_is_missing else 0
        if zero_is_missing:
            values[values == 0] = np.nan
        frame = pd.DataFrame({
            "series_id": np.repeat([f"{self.prefix}_{c}" for c in chosen], len(wide)),
            "timestamp": np.tile(wide.index.to_numpy(), len(chosen)),
            "avg_speed_kmh": values.T.reshape(-1),
        })
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        point = getattr(self.entry, "weather_point", None) if weather is True else (weather or None)
        frame = _attach_weather(frame, point, pd.DatetimeIndex(wide.index))
        steps = wide.index.to_series().diff().dropna()
        resolution_s = int(steps.mode().iloc[0].total_seconds()) if len(steps) else int(
            getattr(self.entry, "resolution_s", 300))
        hol = getattr(self.entry, "holiday_country", None) if holiday_country == "default" else holiday_country
        return _checked(TimeSeriesDataset(
            name=self.dataset_id, frame=frame, resolution_s=resolution_s, tz=tz, holiday_country=hol,
            info={"file": self.h5_file(), "sensors": chosen, "n_sensors_total": len(cols), "steps": len(wide),
                  "period_utc": [str(wide.index.min()), str(wide.index.max())],
                  "missing_values": int(np.isnan(values).sum()), "zeros_treated_as_missing": missing_zero,
                  "dropped_dst_timestamps": dropped_dst, "weather": bool(point),
                  "units": "speed mph -> km/h", "provenance": self.entry.source}))


@DATASET_ADAPTERS.register("metr_la")
class METRLAAdapter(_SpeedMatrixAdapter):
    prefix = "metr_la"
    h5_keys = ("df",)


@DATASET_ADAPTERS.register("pems_bay")
class PEMSBAYAdapter(_SpeedMatrixAdapter):
    prefix = "pems_bay"
    h5_keys = ("speed", "df")


@DATASET_ADAPTERS.register("pems08")
class PEMS08Adapter(TimeSeriesAdapter):
    def load(self, **params: Any) -> Any:
        from traffic_intel.forecasting.data.sources import load_pems08

        self.check()
        return _checked(load_pems08(path=str(self.root / "pems08.npz"), **params))


@DATASET_ADAPTERS.register("synthetic")
class SyntheticAdapter(TimeSeriesAdapter):
    def load(self, **params: Any) -> Any:
        from traffic_intel.forecasting.data.sources import load_synthetic

        return _checked(load_synthetic(**params))


@DATASET_ADAPTERS.register("own_recordings")
class OwnRecordingsAdapter(TimeSeriesAdapter):
    """video -> OpenCV -> YOLO -> ByteTrack -> features -> 1-min windows -> this dataset.
    Optional: no public-data experiment depends on it."""

    def load(self, paths: str | list[str] | None = None, **params: Any) -> Any:
        from traffic_intel.forecasting.data.sources import load_observations

        paths = paths or self.params.get("states_glob")
        if not paths:
            raise ValueError("own_recordings: pass paths=... or set params.states_glob in the dataset config")
        return _checked(load_observations(paths=paths, **params))
