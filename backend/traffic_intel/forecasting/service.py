"""Forecasting service: recent traffic history -> +5 / +15 / +30 min predictions.

    svc = ForecastingService.from_registry()          # PRODUCTION model (any type) from models/forecasting
    res = svc.predict(history)                        # history: TrafficState / TrafficFeatures (dicts or models)
    res.model_dump()  ->
      {"approach_id": "junction_a_north", "current_state": "HIGH",
       "forecast": {"5_min": "HIGH", "15_min": "SEVERE", "30_min": "SEVERE"},
       "predictions": {"5_min": {"avg_speed_kmh": 14.2, "density_ratio": 0.41, ...}, ...}, ...}

How levels are produced: the model forecasts numeric traffic features; each
horizon's predicted features are classified by the SAME explainable
rule-based congestion classifier the live pipeline uses, so current and
forecast levels are comparable. Levels need the classifier's inputs among the
model's targets (at least speed); otherwise only numbers are returned.

Inputs the service needs: at least ``required_history_steps`` consecutive
windows at the model's resolution (short gaps are interpolated like in
training). Calendar features come from timestamps; weather is optional - if
the model uses weather and none is given, the training mean is assumed
(documented in the result).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel

from traffic_intel.ai.model_store import FileModelStore, ModelMetadata
from traffic_intel.core.config import get_settings
from traffic_intel.core.enums import CongestionLevel, ModelStatus, ModelTask
from traffic_intel.core.feature_schema import FORECAST_TARGETS
from traffic_intel.core.schemas import Forecast, ForecastPoint, TrafficFeatures
from traffic_intel.forecasting.data.preprocess import Preprocessor
from traffic_intel.forecasting.data.sources import observations_frame
from traffic_intel.forecasting.data.windows import WindowSet, build_blocks
from traffic_intel.forecasting.interfaces import Forecaster

log = logging.getLogger(__name__)

# physical bounds applied to predictions
_BOUNDS = {"avg_speed_kmh": (0.0, 200.0), "density_ratio": (0.0, 1.5), "occupancy_pct": (0.0, 100.0),
           "queue_length_m": (0.0, None), "flow_veh_h": (0.0, None), "vehicle_count": (0.0, None),
           "inflow": (0.0, None), "outflow": (0.0, None), "stopped_vehicles": (0.0, None)}


class InsufficientHistoryError(ValueError):
    pass


class ForecastResult(BaseModel):
    approach_id: str
    issued_at: datetime                       # time of the last observed window
    model: dict[str, str]                     # name, type, version
    horizons_min: list[int]
    current_state: CongestionLevel | None
    current: dict[str, float]                 # last observed values of the forecast targets
    forecast: dict[str, CongestionLevel | None]            # "5_min" -> level
    predictions: dict[str, dict[str, float]]               # "5_min" -> {target: value}
    explanations: dict[str, list[str]] = {}
    notes: list[str] = []

    def to_forecast(self) -> Forecast:
        """The system-wide ``Forecast`` schema (bus / API / database)."""
        points = [
            ForecastPoint(horizon_min=h, target_time=self.issued_at + pd.Timedelta(minutes=h),
                          target=t, value=v)
            for h in self.horizons_min for t, v in self.predictions[f"{h}_min"].items()
            if t in FORECAST_TARGETS
        ]
        return Forecast(approach_id=self.approach_id, issued_at=self.issued_at, model_name=self.model["name"],
                        model_version=self.model["version"], points=points,
                        predicted_level={h: lvl for h, lvl in
                                         zip(self.horizons_min, self.forecast.values(), strict=True)
                                         if lvl is not None})


class ForecastingService:
    def __init__(self, model: Forecaster, pre: Preprocessor, meta: ModelMetadata,
                 free_flow_speeds: dict[str, float] | None = None) -> None:
        self.model = model
        self.pre = pre
        self.meta = meta
        td = meta.train_data
        self.horizons_min: list[int] = list(td["horizons_min"])
        self.horizon_steps: list[int] = list(td["horizon_steps"])
        self.required_history_steps: int = int(td["window_lookback_steps"])
        self.free_flow = free_flow_speeds or {}
        from traffic_intel.traffic.congestion.rules import RuleBasedCongestionClassifier

        self.classifier = RuleBasedCongestionClassifier(rules_file="congestion_rules.yaml",
                                                        free_flow_speeds=self.free_flow, stateful=False)

    # ------------------------------------------------------------------ #
    @classmethod
    def from_registry(cls, model: str = "auto", version: str = "production", store: FileModelStore | None = None,
                      free_flow_speeds: dict[str, float] | None = None) -> ForecastingService:
        """``model="auto"``: the most recently trained PRODUCTION forecasting model of any type."""
        store = store or FileModelStore(get_settings().models_dir)
        if model == "auto":
            prod = [m for m in store.list(ModelTask.FORECASTING) if m.status is ModelStatus.PRODUCTION]
            if not prod:
                raise FileNotFoundError(
                    "no PRODUCTION forecasting model - train with --promote best-val or run: "
                    "python -m traffic_intel.ai.training promote forecasting <model> latest")
            model, version = sorted(prod, key=lambda m: m.trained_at)[-1].name, "production"
        version = store.resolve_version(ModelTask.FORECASTING, model, version)
        fc = store.load(ModelTask.FORECASTING, model, version)
        d = store._dir(ModelTask.FORECASTING, model, version)
        pre = Preprocessor.load(d / "preprocessor.json")
        meta = store.metadata(ModelTask.FORECASTING, model, version)
        if pre.spec_hash() != meta.feature_schema_hash:
            raise ValueError(f"preprocessor/model feature schema mismatch for {model}@{version}")
        if free_flow_speeds is None:
            free_flow_speeds = _network_free_flow()
        return cls(fc, pre, meta, free_flow_speeds)  # type: ignore[arg-type]

    @property
    def model_info(self) -> dict[str, str]:
        return {"name": self.meta.name, "type": str(self.meta.params.get("type", self.meta.name)),
                "version": self.meta.version}

    # ------------------------------------------------------------------ #
    def predict(
        self,
        history: Sequence[Any] | pd.DataFrame,
        *,
        approach_id: str | None = None,
        context: pd.DataFrame | None = None,
    ) -> ForecastResult:
        """``history``: TrafficState / TrafficFeatures objects or dicts, or a canonical DataFrame
        (series_id, timestamp, feature columns). ``context``: optional weather rows
        (timestamp + weather columns) covering the history."""
        frame, last_rows = self._history_frame(history, approach_id)
        sid = str(frame["series_id"].iloc[0])
        notes: list[str] = []
        wcols = [c for c in self.pre.spec.weather if c not in self.pre.dropped]
        if wcols:
            if context is not None:
                ctx = context.copy()
                ctx["timestamp"] = pd.to_datetime(ctx["timestamp"], utc=True)
                frame = frame.merge(ctx[["timestamp", *[c for c in wcols if c in ctx]]], on="timestamp", how="left")
            else:
                notes.append("no weather context supplied - model assumed training-average weather")
            for c in wcols:
                if c not in frame:
                    frame[c] = np.nan

        reg = self.pre.regularize(frame)
        L = self.required_history_steps
        if len(reg) < L:
            raise InsufficientHistoryError(
                f"need >= {L} consecutive {self.pre.resolution_s}s windows, got {len(reg)}")
        blocks = build_blocks(self.pre.transform(reg.tail(L).reset_index(drop=True)), self.pre)
        if not blocks[0].ok.all():
            raise InsufficientHistoryError(
                f"the last {L} windows contain missing values that cannot be filled (a gap longer than "
                f"{self.pre.spec.max_gap_fill_steps} steps, or missing at the start of the available history)")
        ws = WindowSet(blocks, np.array([[0, L - 1]]), L, self.horizon_steps, self.pre)
        pred = self.pre.inverse_targets(self.model.predict(ws))[0]         # [H, K]
        targets = self.pre.target_columns
        preds: dict[str, dict[str, float]] = {}
        for j, h in enumerate(self.horizons_min):
            row = {}
            for k, t in enumerate(targets):
                lo, hi = _BOUNDS.get(t, (None, None))
                row[t] = round(float(np.clip(pred[j, k], lo, hi)), 3)
            preds[f"{h}_min"] = row

        last = last_rows[-1]
        current = {t: round(float(last[t]), 3) for t in targets if last.get(t) is not None and pd.notna(last[t])}
        issued = pd.Timestamp(reg["timestamp"].iloc[-1]).to_pydatetime()
        levels_ok = "avg_speed_kmh" in targets
        current_state: CongestionLevel | None = None
        forecast: dict[str, CongestionLevel | None] = {}
        expl: dict[str, list[str]] = {}
        if levels_ok:
            current_state = _as_level(last.get("traffic_state")) or self._classify(sid, last, {})[0]
            for key, vals in preds.items():
                lvl, why = self._classify(sid, last, vals)
                forecast[key], expl[key] = lvl, why
            missing = {"density_ratio", "occupancy_pct", "queue_length_m"} - set(targets)
            if missing and not set(self.pre.spec.dynamic) <= {"avg_speed_kmh"}:
                notes.append(f"levels use last observed values for {sorted(missing)} (not forecast by this model)")
        else:
            forecast = {f"{h}_min": None for h in self.horizons_min}
            notes.append("model does not forecast speed - congestion levels not available, numbers only")
        return ForecastResult(approach_id=sid, issued_at=issued, model=self.model_info, horizons_min=self.horizons_min,
                              current_state=current_state, current=current, forecast=forecast, predictions=preds,
                              explanations=expl, notes=notes)

    # ------------------------------------------------------------------ #
    def _history_frame(self, history: Sequence[Any] | pd.DataFrame,
                       approach_id: str | None) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
        if isinstance(history, pd.DataFrame):
            frame = history.copy()
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
            rows = frame.to_dict("records")
        else:
            dicts = [h.model_dump(mode="json") if hasattr(h, "model_dump") else dict(h) for h in history]
            frame = observations_frame(dicts)
            frame = frame.drop(columns=[c for c in ("window_s",) if c in frame])
            rows = [{**d, **r} for d, r in zip([d for d in dicts if not d.get("is_live")],
                                               frame.to_dict("records"), strict=True)]
        if approach_id is not None:
            frame = frame[frame["series_id"] == approach_id]
            rows = [r for r in rows if r.get("series_id", r.get("approach_id")) == approach_id]
        if frame.empty:
            raise InsufficientHistoryError("empty history")
        if frame["series_id"].nunique() != 1:
            raise ValueError("history must contain exactly one approach (pass approach_id=...)")
        order = np.argsort(pd.to_datetime(frame["timestamp"], utc=True).to_numpy())
        frame = frame.iloc[order].reset_index(drop=True)
        rows = [rows[i] for i in order]
        return frame, rows

    def _classify(self, sid: str, last: dict[str, Any],
                  overrides: dict[str, float]) -> tuple[CongestionLevel, list[str]]:
        base = {k: last.get(k) for k in TrafficFeatures.model_fields if k in last and last.get(k) is not None}
        speed = overrides.get("avg_speed_kmh", last.get("avg_speed_kmh", last.get("average_speed_kmh")))
        feats = TrafficFeatures(
            approach_id=sid, camera_id=str(base.get("camera_id", "forecast")),
            window_start=pd.Timestamp(last.get("timestamp") or last.get("window_start")).to_pydatetime(),
            window_s=float(base.get("window_s") or self.pre.resolution_s),
            vehicle_count=int(overrides.get("vehicle_count", base.get("vehicle_count", 0)) or 0),
            count_by_class={}, pcu_count=0.0,
            avg_speed_kmh=None if speed is None or pd.isna(speed) else float(speed),
            speed_p85_kmh=None,
            density_veh_km=float(base.get("density_veh_km", 0.0) or 0.0),
            density_ratio=float(overrides.get("density_ratio", _num(last, "density_ratio", "density_pct", 0.01))),
            occupancy_pct=float(overrides.get("occupancy_pct", _num(last, "occupancy_pct"))),
            flow_veh_h=float(overrides.get("flow_veh_h", _num(last, "flow_veh_h"))),
            inflow=0, outflow=0,
            queue_length_m=float(overrides.get("queue_length_m", _num(last, "queue_length_m"))),
            stopped_vehicles=int(overrides.get("stopped_vehicles", _num(last, "stopped_vehicles",
                                                                         "stopped_vehicle_count"))),
            avg_gap_m=None, pedestrian_count=0, quality=1.0,
            # a model that only ever saw speed (PEMS-BAY) gets speed-only levels: the other indicators
            # of its history are "not measured", not zero
            extra={"speed_only": 1.0} if set(self.pre.spec.dynamic) <= {"avg_speed_kmh"} else {},
        )
        a = self.classifier.classify(feats)
        return a.level, a.explanation


def _num(d: dict[str, Any], key: str, alt: str | None = None, alt_scale: float = 1.0) -> float:
    v = d.get(key)
    if (v is None or pd.isna(v)) and alt is not None and d.get(alt) is not None:
        v = float(d[alt]) * alt_scale
    return 0.0 if v is None or pd.isna(v) else float(v)


def _as_level(v: Any) -> CongestionLevel | None:
    try:
        return CongestionLevel(v) if v else None
    except ValueError:
        return None


def _network_free_flow() -> dict[str, float]:
    try:
        from traffic_intel.core.config import load_network, load_pipeline_config

        st = get_settings()
        net = load_network(st.config_dir / load_pipeline_config(st).network_file)
        seg = {s.id: s for s in net.segments}
        return {a.id: seg[a.road_segment_id].free_flow_speed_kmh or seg[a.road_segment_id].speed_limit_kmh
                for a in net.approaches}
    except Exception:  # noqa: BLE001 - classifier falls back to its default free-flow speed
        return {}
