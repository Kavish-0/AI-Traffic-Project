"""Forecasting experiment: dataset -> chronological split -> preprocessing (fit on train)
-> identical windows for every model -> train -> evaluate -> registry + experiment record.

Outputs (never overwritten - every run gets a new experiment id / model version):

    experiments/forecasting/<experiment_id>/
        config.yaml          the resolved experiment configuration
        dataset.json         source, series, periods (train/val/test), sample counts, features, targets
        models.json          per model: type, hyper-parameters, version, fit time, epochs, parameters
        metrics_test.csv     long table: model x target x horizon -> MAE, RMSE, R2, skill vs persistence
        metrics_val.csv      same on the validation period (used for model selection)
        comparison.md        summary tables + automatically derived observations
        plots/*.png          RMSE-by-horizon per target; prediction-vs-actual per model
    experiments/forecasting/index.csv   one appended row per (experiment, model)
    models/forecasting/<model>/<experiment_id>/   model + preprocessor.json + metadata.json
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field

from traffic_intel.ai.evaluation import plots
from traffic_intel.ai.evaluation.metrics import metric_table, summary_table
from traffic_intel.ai.model_store import FileModelStore, ModelMetadata
from traffic_intel.core.config import PROJECT_ROOT, get_settings, load_yaml
from traffic_intel.core.enums import ModelStatus, ModelTask
from traffic_intel.core.registry import ComponentSpec
from traffic_intel.forecasting.data.preprocess import FeatureSpec, Preprocessor
from traffic_intel.forecasting.data.sources import load_dataset
from traffic_intel.forecasting.data.windows import (
    SplitConfig,
    build_blocks,
    chronological_split,
    make_windows,
)
from traffic_intel.forecasting.interfaces import FORECASTERS, Forecaster

log = logging.getLogger(__name__)

UNITS = {"flow_veh_h": "veh/h", "avg_speed_kmh": "km/h", "occupancy_pct": "%", "density_ratio": "ratio",
         "queue_length_m": "m", "vehicle_count": "veh", "stopped_vehicles": "veh"}


class ModelEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    type: str
    params: dict[str, Any] = {}


class DatasetEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: str
    params: dict[str, Any] = {}


class PlotEntry(BaseModel):
    series: str | None = None       # default: first series
    days: float = 1.0               # length of the prediction-vs-actual window
    start_offset_days: float = 1.0  # offset from the start of the test period


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    description: str = ""
    seed: int = 0
    dataset: DatasetEntry
    features: FeatureSpec
    horizons_min: list[int] = [5, 15, 30]
    season: str = "1D"                          # seasonal-naive period (pandas offset)
    split: SplitConfig = SplitConfig()
    train_stride: int = Field(1, ge=1)          # subsample training origins (memory/time)
    eval_stride: int = Field(1, ge=1)
    primary_target: str
    # Deep models are trained once per seed; the registered model is the seed with the best
    # VALIDATION RMSE, and the report shows mean +- std of test RMSE over seeds.
    deep_seeds: list[int] = [0]
    models: list[ModelEntry]
    plots: PlotEntry = PlotEntry()


def load_experiment_config(path: Path) -> ExperimentConfig:
    return ExperimentConfig.model_validate(load_yaml(path))


class ExperimentResult(BaseModel):
    experiment_id: str
    directory: str
    test: list[dict[str, Any]]
    val: list[dict[str, Any]]
    best_by_validation: dict[str, str]


@dataclass
class PreparedData:
    """Dataset -> regular grid -> chronological periods -> preprocessor fitted on TRAIN only -> blocks.
    Deterministic, so a finished experiment's windows can be rebuilt exactly (``windows``)."""

    cfg: ExperimentConfig
    ds: Any
    res: int
    steps: list[int]
    season_steps: int
    pre: Preprocessor
    periods: dict[str, Any]
    blocks: Any

    def windows(self, part: str, lookback: int) -> Any:  # noqa: ANN401
        stride = self.cfg.train_stride if part == "train" else self.cfg.eval_stride
        return make_windows(self.blocks, self.periods[part], lookback=lookback, horizon_steps=self.steps,
                            pre=self.pre, stride=stride, season_steps=self.season_steps)


def prepare_data(cfg: ExperimentConfig) -> PreparedData:
    ds = load_dataset(cfg.dataset.source, cfg.dataset.params)
    res = ds.resolution_s
    steps = []
    for h in cfg.horizons_min:
        if (h * 60) % res:
            raise ValueError(f"horizon {h} min is not a multiple of the {res} s resolution")
        steps.append(h * 60 // res)
    season_steps = int(pd.Timedelta(cfg.season).total_seconds() // res)
    pre = Preprocessor(cfg.features, resolution_s=res, tz=ds.tz, holiday_country=ds.holiday_country)
    reg = pre.regularize(ds.frame)
    periods = chronological_split(reg["timestamp"], cfg.split, res)
    pre.fit(reg[reg["timestamp"] < periods["train"].end])
    blocks = build_blocks(pre.transform(reg), pre)
    return PreparedData(cfg, ds, res, steps, season_steps, pre, periods, blocks)


def run_experiment(cfg: ExperimentConfig, *, experiments_root: Path | None = None, store: FileModelStore | None = None,
                   only: list[str] | None = None, smoke: bool = False) -> ExperimentResult:
    """``smoke``: pipeline check - deep models 1 epoch / 1 seed, nothing registered, not indexed."""
    if smoke:
        cfg = cfg.model_copy(deep=True)
        cfg.name += "_smoke"
        cfg.deep_seeds = cfg.deep_seeds[:1]
        for m in cfg.models:
            if "max_epochs" in m.params:
                m.params = {**m.params, "max_epochs": 1}
    np.random.seed(cfg.seed)
    settings = get_settings()
    experiments_root = experiments_root or PROJECT_ROOT / "experiments" / "forecasting"
    store = store or FileModelStore(settings.models_dir)
    exp_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{cfg.name}"
    out = experiments_root / exp_id
    out.mkdir(parents=True, exist_ok=False)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False), encoding="utf-8")
    t_all = time.perf_counter()

    # ---- data -------------------------------------------------------------
    data = prepare_data(cfg)
    ds, res, steps, season_steps, pre, periods = (data.ds, data.res, data.steps, data.season_steps, data.pre,
                                                   data.periods)

    models: list[tuple[ModelEntry, Forecaster]] = []
    for m in cfg.models:
        if only and m.name not in only:
            continue
        params = dict(m.params)
        if m.type in ("lstm", "gru", "transformer"):
            params.setdefault("seed", cfg.seed)
        models.append((m, FORECASTERS.create(ComponentSpec(type=m.type, params=params))))
    tab_lookback = max(max(cfg.features.lags, default=0) + 1, max(cfg.features.rolling, default=1))
    lookback = max([tab_lookback] + [f.lookback_steps for _, f in models if f.family == "deep"])
    wins = {k: data.windows(k, lookback) for k in ("train", "val", "test")}
    dataset_info = {
        "source": cfg.dataset.source, "dataset": ds.name, "resolution_s": res, "timezone": ds.tz,
        "holiday_country": ds.holiday_country, "series": ds.series_ids, "info": ds.info,
        "periods": {k: v.as_dict() for k, v in periods.items()},
        "samples": {k: len(v) for k, v in wins.items()},
        "window_lookback_steps": lookback, "horizons_min": cfg.horizons_min, "horizon_steps": steps,
        "season_steps": season_steps,
        "features": {"inputs": pre.input_columns, "targets": pre.target_columns, "dropped": pre.dropped,
                     "tabular_lags": cfg.features.lags, "tabular_rolling": cfg.features.rolling},
        "missing_fraction": {c: round(float(ds.frame[c].isna().mean()), 5) for c in cfg.features.dynamic},
        "preprocessor_spec_hash": pre.spec_hash(),
    }
    (out / "dataset.json").write_text(json.dumps(dataset_info, indent=2, default=str), encoding="utf-8")
    log.info("experiment %s: samples %s, lookback %d, inputs %d", exp_id, dataset_info["samples"], lookback,
             len(pre.input_columns))

    # ---- train / predict ---------------------------------------------------
    y_true = {k: pre.inverse_targets(wins[k].y()) for k in ("val", "test")}
    preds: dict[str, dict[str, np.ndarray]] = {"val": {}, "test": {}}
    model_records = []
    seed_runs: list[dict[str, Any]] = []
    k_prim = pre.target_columns.index(cfg.primary_target)
    for i, (entry, model) in enumerate(models):
        seeds = cfg.deep_seeds if model.family == "deep" else [None]
        best: tuple[float, Forecaster, dict[str, Any], dict[str, np.ndarray], float, float] | None = None
        for seed in seeds:
            cand = model if seed is None else FORECASTERS.create(
                ComponentSpec(type=entry.type, params={**model.params, "seed": seed}))
            log.info("training %s (%s)%s", entry.name, entry.type, "" if seed is None else f" seed {seed}")
            t0 = time.perf_counter()
            info = cand.fit(wins["train"], wins["val"])
            fit_s = time.perf_counter() - t0
            t1 = time.perf_counter()
            p = {k: pre.inverse_targets(cand.predict(wins[k])) for k in ("val", "test")}
            predict_ms = (time.perf_counter() - t1) / max(1, len(wins["val"]) + len(wins["test"])) * 1000
            val_rmse = float(np.sqrt(np.mean((p["val"][..., k_prim] - y_true["val"][..., k_prim]) ** 2)))
            test_rmse = float(np.sqrt(np.mean((p["test"][..., k_prim] - y_true["test"][..., k_prim]) ** 2)))
            if seed is not None:
                seed_runs.append({"model": entry.name, "seed": seed, "val_rmse": val_rmse, "test_rmse": test_rmse,
                                  "epochs": info.get("epochs")})
            if best is None or val_rmse < best[0]:
                best = (val_rmse, cand, info, p, fit_s, predict_ms)
        assert best is not None
        _, model, info, p, fit_s, predict_ms = best
        models[i] = (entry, model)
        for k in ("val", "test"):
            preds[k][entry.name] = p[k]
        if model.family == "deep":
            info = {**info, "selected_seed": model.params.get("seed"), "seeds": seeds}
        rec = {"name": entry.name, "type": entry.type, "family": model.family, "framework": model.framework,
               "params": model.params, "version": exp_id, "fit_s": round(fit_s, 2),
               "predict_ms_per_sample": round(predict_ms, 4),
               **{k: v for k, v in info.items() if k != "history"}}
        if "history" in info:
            rec["history"] = info["history"]
        model_records.append(rec)

    if seed_runs:
        pd.DataFrame(seed_runs).to_csv(out / "seed_runs.csv", index=False)
    metrics = {k: metric_table(y_true[k], preds[k], cfg.horizons_min, pre.target_columns) for k in ("val", "test")}
    metrics["val"].to_csv(out / "metrics_val.csv", index=False)
    metrics["test"].to_csv(out / "metrics_test.csv", index=False)

    # model selection on VALIDATION (test is for reporting only)
    val_primary = metrics["val"][metrics["val"]["target"] == cfg.primary_target]
    val_rank = val_primary.groupby("model")["RMSE"].mean().sort_values()
    best_val = str(val_rank.index[0])

    # ---- registry -------------------------------------------------------------
    test_summary = summary_table(metrics["test"])
    for entry, model in models:
        m_long = metrics["test"][metrics["test"]["model"] == entry.name]
        meta = ModelMetadata(
            task=ModelTask.FORECASTING, name=entry.name, version=exp_id, framework=model.framework,
            feature_schema_hash=pre.spec_hash(), trained_at=datetime.now(UTC),
            train_data={"experiment_id": exp_id, "dataset": ds.name, "series": ds.series_ids,
                        "periods": dataset_info["periods"], "samples": dataset_info["samples"],
                        "horizons_min": cfg.horizons_min, "horizon_steps": steps, "resolution_s": res,
                        "lookback_steps": model.lookback_steps if model.family == "deep" else lookback,
                        "window_lookback_steps": lookback, "inputs": pre.input_columns,
                        "targets": pre.target_columns},
            params={"type": entry.type, **model.params},
            metrics={"test": m_long.drop(columns=["model"]).to_dict("records"),
                     "val_rmse_primary": float(val_rank.get(entry.name, np.nan))},
            status=ModelStatus.CANDIDATE,
        )
        if smoke:
            continue
        path = store.save(model, meta, preprocessor=pre)
        next(r for r in model_records if r["name"] == entry.name)["artifact"] = str(path)
    (out / "models.json").write_text(json.dumps(model_records, indent=2, default=str), encoding="utf-8")

    # ---- report ------------------------------------------------------------
    plot_files = _plots(cfg, pre, wins["test"], y_true["test"], preds["test"], metrics["test"], out / "plots", ds.tz)
    md = _comparison_markdown(cfg, exp_id, dataset_info, metrics, test_summary, model_records, best_val, plot_files,
                              seed_runs)
    (out / "comparison.md").write_text(md, encoding="utf-8")
    if not smoke:
        _append_index(experiments_root / "index.csv", exp_id, cfg, dataset_info, test_summary, model_records)
    log.info("experiment %s done in %.1f s -> %s", exp_id, time.perf_counter() - t_all, out)
    return ExperimentResult(experiment_id=exp_id, directory=str(out),
                            test=metrics["test"].to_dict("records"), val=metrics["val"].to_dict("records"),
                            best_by_validation={"target": cfg.primary_target, "model": best_val})


# --------------------------------------------------------------------------- #
def _plots(cfg: ExperimentConfig, pre: Preprocessor, test: Any, y: np.ndarray, preds: dict[str, np.ndarray],  # noqa: ANN401
           long: pd.DataFrame, plot_dir: Path, tz: str) -> list[str]:
    files = []
    for tgt in pre.target_columns:
        f = plot_dir / f"rmse_by_horizon_{tgt}.png"
        plots.rmse_by_horizon(long, tgt, UNITS.get(tgt, ""), f, f"{cfg.name}: test RMSE by horizon - {tgt}")
        files.append(f.name)
    # prediction vs actual: one series, a window of the test period, every learned model + persistence
    sids = test.series_ids()
    series = cfg.plots.series or sids[0]
    times = test.origin_times()
    start = times[sids == series].min() + pd.Timedelta(days=cfg.plots.start_offset_days)
    sel = (sids == series) & (times >= start) & (times < start + pd.Timedelta(days=cfg.plots.days))
    if sel.sum() == 0:
        return files
    k = pre.target_columns.index(cfg.primary_target)
    order = np.argsort(times[sel])
    t_sel = times[sel][order]
    act = y[sel][order][:, :, k]
    for name, p in preds.items():
        if name in ("persistence", "seasonal_naive"):
            continue
        f = plot_dir / f"pred_vs_actual_{name}_{cfg.primary_target}.png"
        plots.prediction_vs_actual(
            t_sel, act, {name: p[sel][order][:, :, k], "persistence": preds["persistence"][sel][order][:, :, k]}
            if "persistence" in preds else {name: p[sel][order][:, :, k]},
            cfg.horizons_min, cfg.primary_target, UNITS.get(cfg.primary_target, ""), tz, f,
            f"{name} vs actual - {cfg.primary_target}, series {series}")
        files.append(f.name)
    return files


def _fmt(x: float) -> str:
    return "-" if pd.isna(x) else (f"{x:.3f}" if abs(x) < 10 else f"{x:.1f}")


def _table(df: pd.DataFrame, cols: list[str]) -> str:
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) if isinstance(r[c], str) else _fmt(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def _comparison_markdown(cfg: ExperimentConfig, exp_id: str, info: dict[str, Any], metrics: dict[str, pd.DataFrame],
                         summary: pd.DataFrame, records: list[dict[str, Any]], best_val: str | None,
                         plot_files: list[str], seed_runs: Any = None) -> str:  # noqa: ANN401
    test = metrics["test"]
    p = info["periods"]
    rec = {r["name"]: r for r in records}
    out = [f"# Forecasting experiment `{exp_id}`", "", cfg.description.strip(), "",
           f"- **Dataset:** {info['dataset']} ({info['source']}), {len(info['series'])} series, resolution "
           f"{info['resolution_s']} s",
           f"- **Periods (chronological):** train {p['train']['start']} -> {p['train']['end']}; "
           f"val {p['val']['start']} -> {p['val']['end']}; test {p['test']['start']} -> {p['test']['end']}",
           f"- **Samples:** {info['samples']} (identical test origins for every model)",
           f"- **Inputs ({len(info['features']['inputs'])}):** {', '.join(info['features']['inputs'])}",
           f"- **Targets:** {', '.join(info['features']['targets'])}; horizons {cfg.horizons_min} min",
           f"- **Model selection:** by validation RMSE on `{cfg.primary_target}` -> **{best_val}**", ""]
    for tgt in summary["target"].unique():
        s = summary[summary["target"] == tgt].sort_values("RMSE")
        out += [f"## {tgt} ({UNITS.get(tgt, '')}) - test, averaged over horizons", "",
                _table(s.rename(columns={"skill_vs_persistence": "skill"}),
                       ["model", "MAE", "RMSE", "R2", "MAPE", "skill"]),
                ""]
    prim = test[test["target"] == cfg.primary_target]
    out += [f"## {cfg.primary_target} by horizon (test)", ""]
    for h in cfg.horizons_min:
        s = prim[prim["horizon_min"] == h].sort_values("RMSE")
        out += [f"**+{h} min**", "", _table(s.rename(columns={"skill_vs_persistence": "skill"}),
                                             ["model", "MAE", "RMSE", "R2", "MAPE", "skill"]), ""]
    # automatic, factual observations
    obs = []
    for h in cfg.horizons_min:
        s = prim[prim["horizon_min"] == h].set_index("model")["RMSE"]
        best = s.idxmin()
        pers = _fmt(s.get("persistence", np.nan))
        obs.append(f"+{h} min: lowest test RMSE = **{best}** ({_fmt(s[best])}); persistence {pers}"
                   + (f"; linear {_fmt(s['linear'])}" if "linear" in s else ""))
    fam = {r["name"]: r["family"] for r in records}
    ps = summary[summary["target"] == cfg.primary_target].set_index("model")["RMSE"]
    deep = [m for m in ps.index if fam.get(m) == "deep"]
    trees = [m for m in ps.index if fam.get(m) == "tree"]
    if deep and trees:
        bd, bt = ps[deep].idxmin(), ps[trees].idxmin()
        diff = (ps[bt] - ps[bd]) / ps[bt] * 100
        obs.append(f"Best deep model {bd} vs best tree model {bt}: RMSE {_fmt(ps[bd])} vs {_fmt(ps[bt])} "
                   f"({'deep better' if diff > 0 else 'tree better'} by {abs(diff):.1f}%).")
    if "linear" in ps and trees:
        bt = ps[trees].idxmin()
        obs.append(f"Best tree model {bt} vs linear baseline: {(ps['linear'] - ps[bt]) / ps['linear'] * 100:+.1f}% "
                   "RMSE reduction.")
    if "persistence" in ps:
        beat = [m for m in ps.index if m != "persistence" and ps[m] < ps["persistence"]]
        obs.append(f"Models beating persistence on average: {', '.join(beat) if beat else 'none'}.")
    if seed_runs:
        sr = pd.DataFrame(seed_runs).groupby("model")["test_rmse"].agg(["mean", "std", "min", "max", "count"])
        out += [f"## Seed variability of deep models ({cfg.primary_target}, test RMSE, mean over horizons)", "",
                "| model | mean | std | min | max | seeds |", "|---|---|---|---|---|---|"]
        for m, r in sr.iterrows():
            out.append(f"| {m} | {_fmt(r['mean'])} | {_fmt(r['std'])} | {_fmt(r['min'])} | {_fmt(r['max'])} | "
                       f"{int(r['count'])} |")
        out += ["", "The tables above use, for each deep model, the seed with the best *validation* RMSE.", ""]
        if trees:
            bt = ps[trees].idxmin()
            better = [m for m, r in sr.iterrows() if r["mean"] + r["std"] < ps[bt]]
            obs.append(f"Deep models whose mean-over-seeds RMSE (+1 std) is below the best tree model ({bt}): "
                       f"{', '.join(better) if better else 'none'}.")
    out += ["## Observations (derived automatically from the numbers above)", ""] + [f"- {o}" for o in obs] + [""]
    out += ["## Training cost", "", "| model | family | fit (s) | predict (ms/sample) | epochs | parameters |",
            "|---|---|---|---|---|---|"]
    for name, r in rec.items():
        out.append(f"| {name} | {r['family']} | {r['fit_s']} | {r['predict_ms_per_sample']} | {r.get('epochs', '-')} | "
                   f"{r.get('n_parameters', '-')} |")
    out += ["", "## Figures", ""] + [f"![{f}](plots/{f})" for f in plot_files] + [""]
    return "\n".join(out)


def _append_index(path: Path, exp_id: str, cfg: ExperimentConfig, info: dict[str, Any], summary: pd.DataFrame,
                  records: list[dict[str, Any]]) -> None:
    prim = summary[summary["target"] == cfg.primary_target].set_index("model")
    rows = []
    for r in records:
        m = prim.loc[r["name"]] if r["name"] in prim.index else None
        rows.append({
            "experiment_id": exp_id, "created": datetime.now(UTC).isoformat(timespec="seconds"),
            "config": cfg.name, "dataset": info["dataset"], "model": r["name"], "type": r["type"],
            "version": exp_id, "primary_target": cfg.primary_target,
            "test_MAE": None if m is None else round(float(m["MAE"]), 4),
            "test_RMSE": None if m is None else round(float(m["RMSE"]), 4),
            "test_R2": None if m is None else round(float(m["R2"]), 4),
            "test_MAPE": None if m is None or "MAPE" not in m else round(float(m["MAPE"]), 3),
            "train_period": f"{info['periods']['train']['start']}..{info['periods']['train']['end']}",
            "test_period": f"{info['periods']['test']['start']}..{info['periods']['test']['end']}",
            "fit_s": r["fit_s"], "predict_ms_per_sample": r.get("predict_ms_per_sample"),
        })
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    # rewrite (not append) so a new column never misaligns rows written under an older header
    old = pd.read_csv(path) if path.exists() else pd.DataFrame()
    pd.concat([old, df], ignore_index=True).to_csv(path, index=False)
