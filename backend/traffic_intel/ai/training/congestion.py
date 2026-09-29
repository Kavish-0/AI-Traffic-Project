"""Congestion-level FORECASTING as classification (offline; never imported by the pipeline or the API).

    python -m traffic_intel.ai.training train congestion --config configs/training/congestion_level_metr_la.yaml

Question: which level (FREE / MODERATE / HIGH / SEVERE) will an approach be at in +h minutes?

Why this and not "classify the current state": current levels are DEFINED by the explainable rules
(configs/congestion_rules.yaml); no dataset carries independent congestion labels, so a classifier
trained on rule labels of the same window would only learn to imitate the rules. The FUTURE level is
different: its label is the rule level of the OBSERVED value at t + h, which the inputs (history up to
t) do not determine.

Pipeline (shared with the forecasting experiments - one feature definition, one leakage rule):
    dataset adapter -> Preprocessor (fit on TRAIN) -> chronological split -> windows (targets inside
    the period) -> tabular features at the origin t
    labels  = RuleBasedCongestionClassifier.levels_frame(values at t + h)  (only measured indicators)
    free flow per series: TRAIN-period speed quantile (or the network's value for live approaches)
Models:
    persistence_level      the level now (the "no change" baseline)
    regression_to_rules    XGBoost regressor of the values at t + h, then the same rules
                           (= what the live forecasting service does)
    logistic_regression, random_forest, xgboost   classifiers; small grid, selected on VALIDATION macro-F1
Metrics (test): accuracy, balanced accuracy, macro / weighted precision, recall, F1, per class, confusion.
"""

from __future__ import annotations

import itertools
import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field

from traffic_intel.ai.model_store import FileModelStore, ModelMetadata, TrainableModel
from traffic_intel.ai.training.experiment import DatasetEntry
from traffic_intel.core.config import PROJECT_ROOT, get_settings, load_yaml
from traffic_intel.core.enums import ModelStatus, ModelTask
from traffic_intel.forecasting.data.preprocess import FeatureSpec, Preprocessor
from traffic_intel.forecasting.data.sources import load_dataset
from traffic_intel.forecasting.data.windows import SplitConfig, build_blocks, chronological_split, make_windows
from traffic_intel.traffic.congestion.rules import LEVEL_ORDER, RuleBasedCongestionClassifier

log = logging.getLogger(__name__)
LEVELS = [lvl.value for lvl in LEVEL_ORDER]


class FreeFlowCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    method: str = "train_quantile"       # train_quantile | network (live approaches; quantile fallback)
    quantile: float = Field(0.85, gt=0.5, lt=1.0)


class ModelCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    type: str                             # persistence_level | regression_to_rules | logistic_regression | ...
    params: dict[str, Any] = {}           # fixed parameters
    grid: dict[str, list[Any]] = {}       # tuned on validation macro-F1 (small matrix)


class CongestionExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    description: str = ""
    seed: int = 0
    dataset: DatasetEntry
    features: FeatureSpec
    horizons_min: list[int] = [5, 15, 30]
    split: SplitConfig = SplitConfig()
    train_stride: int = Field(1, ge=1)
    eval_stride: int = Field(1, ge=1)
    free_flow: FreeFlowCfg = Field(default_factory=FreeFlowCfg)
    rules_file: str = "congestion_rules.yaml"
    models: list[ModelCfg]
    register_best: bool = False           # register the best-on-validation classifier per horizon


def load_congestion_config(path: Path) -> CongestionExperimentConfig:
    return CongestionExperimentConfig.model_validate(load_yaml(path))


# --------------------------------------------------------------------------- estimators
def _estimator(kind: str, params: dict[str, Any], seed: int) -> Any:  # noqa: ANN401
    if kind == "logistic_regression":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(max_iter=3000, random_state=seed, **params)
    if kind == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(n_jobs=-1, random_state=seed, **params)
    if kind == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(tree_method="hist", random_state=seed, n_jobs=-1, eval_metric="mlogloss", **params)
    raise ValueError(f"unknown classifier type {kind}")


class LevelClassifier(TrainableModel):
    """A fitted classifier + the level labels it predicts (registry artifact)."""

    def __init__(self, estimator: Any, classes: list[int], horizon_min: int,  # noqa: ANN401
                 feature_names: list[str]) -> None:
        self.estimator = estimator
        self.classes = classes
        self.horizon_min = horizon_min
        self.feature_names = feature_names

    def predict_levels(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.classes)[np.asarray(self.estimator.predict(X), dtype=int)]

    def save(self, directory: Path) -> None:
        import joblib

        joblib.dump({"estimator": self.estimator, "classes": self.classes, "horizon_min": self.horizon_min,
                     "feature_names": self.feature_names, "levels": LEVELS}, directory / "model.joblib")

    @classmethod
    def load(cls, directory: Path) -> LevelClassifier:
        import joblib

        d = joblib.load(directory / "model.joblib")
        return cls(d["estimator"], d["classes"], d["horizon_min"], d["feature_names"])


# --------------------------------------------------------------------------- metrics
def classification_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        precision_recall_fscore_support,
    )

    labels = list(range(len(LEVELS)))
    pr, rc, f1, sup = precision_recall_fscore_support(y, p, labels=labels, zero_division=0)
    present = sup > 0
    macro = {k: float(np.mean(v[present])) for k, v in (("precision", pr), ("recall", rc), ("f1", f1))}
    weighted = precision_recall_fscore_support(y, p, labels=labels, average="weighted", zero_division=0)
    return {"accuracy": float(accuracy_score(y, p)), "balanced_accuracy": float(balanced_accuracy_score(y, p)),
            "macro_precision": macro["precision"], "macro_recall": macro["recall"], "macro_f1": macro["f1"],
            "weighted_precision": float(weighted[0]), "weighted_recall": float(weighted[1]),
            "weighted_f1": float(weighted[2]),
            "per_class": {LEVELS[i]: {"precision": float(pr[i]), "recall": float(rc[i]), "f1": float(f1[i]),
                                      "support": int(sup[i])} for i in labels},
            "confusion": confusion_matrix(y, p, labels=labels).tolist()}


# --------------------------------------------------------------------------- experiment
def run_congestion_experiment(cfg: CongestionExperimentConfig, smoke: bool = False) -> dict[str, Any]:
    if smoke:
        cfg = cfg.model_copy(deep=True)
        cfg.name += "_smoke"
        cfg.train_stride = max(cfg.train_stride, 20)
        cfg.register_best = False
        for m in cfg.models:
            m.grid = {k: v[:1] for k, v in m.grid.items()}
    np.random.seed(cfg.seed)
    exp_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{cfg.name}"
    out = PROJECT_ROOT / "experiments" / "congestion" / exp_id
    # a smoke run is a pipeline check, not a result: its figures stay in its experiment folder
    fig = out / "figures" if smoke else PROJECT_ROOT / "reports" / "figures" / "congestion" / exp_id
    out.mkdir(parents=True)
    fig.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False), encoding="utf-8")
    t_all = time.perf_counter()

    # ---- data (identical preparation to the forecasting experiments) ------------------------
    ds = load_dataset(cfg.dataset.source, cfg.dataset.params)
    res = ds.resolution_s
    steps = [h * 60 // res for h in cfg.horizons_min]
    pre = Preprocessor(cfg.features, resolution_s=res, tz=ds.tz, holiday_country=ds.holiday_country)
    reg = pre.regularize(ds.frame)
    periods = chronological_split(reg["timestamp"], cfg.split, res)
    train_rows = reg[reg["timestamp"] < periods["train"].end]
    pre.fit(train_rows)
    blocks = build_blocks(pre.transform(reg), pre)
    lookback = max(max(cfg.features.lags, default=0) + 1, max(cfg.features.rolling, default=1))
    wins = {k: make_windows(blocks, periods[k], lookback=lookback, horizon_steps=steps, pre=pre,
                            stride=cfg.train_stride if k == "train" else cfg.eval_stride) for k in periods}
    free_flow = _free_flow(cfg, train_rows)
    rules = RuleBasedCongestionClassifier(rules_file=cfg.rules_file, stateful=False)
    targets = pre.target_columns

    def levels(values: np.ndarray, series: np.ndarray) -> np.ndarray:
        frame = pd.DataFrame(values, columns=targets)
        ff = np.array([free_flow[s] for s in series])
        return np.asarray(rules.levels_frame(frame, ff)[1])

    data: dict[str, dict[str, Any]] = {}
    for k, w in wins.items():
        X = np.nan_to_num(w.tabular(), nan=0.0)
        sid = w.series_ids()
        yv = pre.inverse_targets(w.y())                    # [N, H, K] original units at t + h
        now = pre.inverse_targets(w.last())                # [N, K] at the origin
        data[k] = {"X": X, "series": sid, "values": yv, "now": levels(now, sid),
                   "y": {h: levels(yv[:, j], sid) for j, h in enumerate(cfg.horizons_min)}}
    feature_names = wins["train"].tabular_names
    dist = {k: {h: dict(zip(LEVELS, np.bincount(d["y"][h], minlength=4).tolist(), strict=True))
                for h in cfg.horizons_min} for k, d in data.items()}
    dataset_info = {"source": cfg.dataset.source, "dataset": ds.name, "resolution_s": res, "series": ds.series_ids,
                    "periods": {k: v.as_dict() for k, v in periods.items()},
                    "samples": {k: len(w) for k, w in wins.items()}, "features": feature_names,
                    "label_indicators": [t for t in targets if t in ("avg_speed_kmh", "density_ratio",
                                                                      "occupancy_pct", "queue_length_m")],
                    "free_flow_kmh": {k: round(v, 2) for k, v in free_flow.items()},
                    "free_flow_method": cfg.free_flow.model_dump(), "class_distribution": dist,
                    "preprocessor_spec_hash": pre.spec_hash()}
    (out / "dataset.json").write_text(json.dumps(dataset_info, indent=2, default=str), encoding="utf-8")
    log.info("congestion %s: samples %s; test labels %s", exp_id, dataset_info["samples"], dist["test"])

    # ---- models ------------------------------------------------------------------------------
    rows: list[dict[str, Any]] = []
    grid_rows: list[dict[str, Any]] = []
    registry: list[dict[str, Any]] = []
    tr, va, te = data["train"], data["val"], data["test"]
    for j, h in enumerate(cfg.horizons_min):
        best_clf: tuple[float, str, Any, list[int], dict[str, Any]] | None = None
        for m in cfg.models:
            t0 = time.perf_counter()
            chosen: dict[str, Any]
            if m.type == "persistence_level":
                p_val, p_test, chosen = va["now"], te["now"], {}
            elif m.type == "regression_to_rules":
                p_val, p_test, chosen = _regression_to_rules(m, tr, va, te, j, levels, cfg.seed)
            else:
                p_val, p_test, chosen, est, classes, grid = _tuned_classifier(m, tr, va, te, h, cfg.seed)
                grid_rows += [{"horizon_min": h, "model": m.name, **g} for g in grid]
                f1v = classification_metrics(va["y"][h], p_val)["macro_f1"]
                if best_clf is None or f1v > best_clf[0]:
                    best_clf = (f1v, m.name, est, classes, chosen)
            fit_s = time.perf_counter() - t0
            mv = classification_metrics(va["y"][h], p_val)
            mt = classification_metrics(te["y"][h], p_test)
            rows.append({"horizon_min": h, "model": m.name, "type": m.type, "params": json.dumps(chosen),
                         "val_macro_f1": round(mv["macro_f1"], 4), "fit_s": round(fit_s, 1),
                         **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in mt.items()}})
            log.info("h=%d %s: test macro-F1 %.3f acc %.3f (val macro-F1 %.3f)", h, m.name, mt["macro_f1"],
                     mt["accuracy"], mv["macro_f1"])
        if cfg.register_best and best_clf is not None:
            _, name, est, classes, chosen = best_clf
            registry.append(_register(cfg, exp_id, h, name, est, classes, feature_names, pre, dataset_info,
                                      [r for r in rows if r["horizon_min"] == h and r["model"] == name][0]))

    # ---- outputs -----------------------------------------------------------------------------
    long = pd.DataFrame(rows)
    long.drop(columns=["per_class", "confusion"]).to_csv(out / "metrics_test.csv", index=False)
    (out / "metrics_test_detail.json").write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    pd.DataFrame(grid_rows).to_csv(out / "grid_search_val.csv", index=False)
    (out / "models.json").write_text(json.dumps(registry, indent=2, default=str), encoding="utf-8")
    plots = _plots(long, fig, cfg)
    (out / "report.md").write_text(_report(cfg, exp_id, dataset_info, long, plots), encoding="utf-8")
    _index(out.parent / "index.csv", exp_id, cfg, long, time.perf_counter() - t_all)
    log.info("congestion experiment %s done in %.1f s -> %s", exp_id, time.perf_counter() - t_all, out)
    return {"experiment_id": exp_id, "directory": str(out), "metrics": rows, "registered": registry}


def _free_flow(cfg: CongestionExperimentConfig, train_rows: pd.DataFrame) -> dict[str, float]:
    q = train_rows.groupby("series_id")["avg_speed_kmh"].quantile(cfg.free_flow.quantile).to_dict()
    out = {str(k): float(v) for k, v in q.items()}
    if cfg.free_flow.method == "network":
        from traffic_intel.core.config import load_network, load_pipeline_config

        pipeline = load_pipeline_config()
        net = load_network(get_settings().config_dir / pipeline.network_file)
        seg = {s.id: s for s in net.segments}
        for a in net.approaches:
            s = seg.get(a.road_segment_id)
            if a.id in out and s is not None:
                out[a.id] = float(s.free_flow_speed_kmh or s.speed_limit_kmh)
    return out


def _tuned_classifier(m: ModelCfg, tr: dict[str, Any], va: dict[str, Any], te: dict[str, Any], h: int,
                      seed: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any], Any, list[int], list[dict[str, Any]]]:
    classes = sorted(set(tr["y"][h].tolist()))
    enc = {c: i for i, c in enumerate(classes)}
    y_tr = np.array([enc[c] for c in tr["y"][h]])
    keys = list(m.grid)
    best: tuple[float, Any, dict[str, Any]] | None = None
    grid = []
    for values in itertools.product(*[m.grid[k] for k in keys]) if keys else [()]:
        params = {**m.params, **dict(zip(keys, values, strict=True))}
        est = _estimator(m.type, params, seed)
        t0 = time.perf_counter()
        est.fit(tr["X"], y_tr)
        p_val = np.asarray(classes)[np.asarray(est.predict(va["X"]), dtype=int)]
        f1 = classification_metrics(va["y"][h], p_val)["macro_f1"]
        grid.append({"params": json.dumps(params), "val_macro_f1": round(f1, 4),
                     "fit_s": round(time.perf_counter() - t0, 1)})
        if best is None or f1 > best[0]:
            best = (f1, est, params)
    assert best is not None
    _, est, params = best
    p_val = np.asarray(classes)[np.asarray(est.predict(va["X"]), dtype=int)]
    p_test = np.asarray(classes)[np.asarray(est.predict(te["X"]), dtype=int)]
    return p_val, p_test, params, est, classes, grid


def _regression_to_rules(m: ModelCfg, tr: dict[str, Any], va: dict[str, Any], te: dict[str, Any], j: int,
                         levels: Any, seed: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:  # noqa: ANN401
    from xgboost import XGBRegressor

    params = {"n_estimators": 1500, "learning_rate": 0.05, "max_depth": 6, "early_stopping_rounds": 50, **m.params}
    preds_val, preds_test = [], []
    for k in range(tr["values"].shape[2]):
        reg = XGBRegressor(tree_method="hist", random_state=seed, n_jobs=-1, **params)
        reg.fit(tr["X"], tr["values"][:, j, k], eval_set=[(va["X"], va["values"][:, j, k])], verbose=False)
        preds_val.append(reg.predict(va["X"]))
        preds_test.append(reg.predict(te["X"]))
    return (levels(np.stack(preds_val, 1), va["series"]), levels(np.stack(preds_test, 1), te["series"]), params)


def _register(cfg: CongestionExperimentConfig, exp_id: str, h: int, name: str, est: Any, classes: list[int],  # noqa: ANN401
              feature_names: list[str], pre: Preprocessor, info: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    reg_name = f"level_h{h}"
    meta = ModelMetadata(
        task=ModelTask.CONGESTION, name=reg_name, version=exp_id, framework=type(est).__module__.split(".")[0],
        feature_schema_hash=pre.spec_hash(), trained_at=datetime.now(UTC),
        train_data={"experiment_id": exp_id, "dataset": info["dataset"], "periods": info["periods"],
                    "samples": info["samples"], "horizon_min": h, "label_indicators": info["label_indicators"],
                    "free_flow_method": info["free_flow_method"]},
        params={"type": name, "estimator": type(est).__name__, **json.loads(row["params"])},
        metrics={"test": {k: row[k] for k in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1")},
                 "val_macro_f1": row["val_macro_f1"]}, status=ModelStatus.CANDIDATE)
    path = FileModelStore(get_settings().models_dir).save(LevelClassifier(est, classes, h, feature_names), meta,
                                                          preprocessor=pre)
    return {"name": reg_name, "horizon_min": h, "model": name, "artifact": str(path)}


def _plots(long: pd.DataFrame, fig: Path, cfg: CongestionExperimentConfig) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = []
    piv = long.pivot(index="model", columns="horizon_min", values="macro_f1")
    ax = piv.plot.bar(figsize=(8, 4))
    ax.set_ylabel("test macro-F1")
    ax.set_ylim(0, 1)
    ax.set_title(f"{cfg.name}: future congestion level (test)")
    plt.tight_layout()
    f = fig / "macro_f1_by_model.png"
    plt.savefig(f, dpi=130)
    plt.close("all")
    files.append(str(f))
    for h in cfg.horizons_min:
        sub = long[long["horizon_min"] == h].sort_values("val_macro_f1", ascending=False)
        best = sub.iloc[0]
        cm = np.array(best["confusion"], float)
        norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        fig_, ax_ = plt.subplots(figsize=(4.8, 4.2))
        ax_.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for (r, c), v in np.ndenumerate(cm):
            ax_.text(c, r, f"{int(v)}\n{norm[r, c]:.0%}", ha="center", va="center", fontsize=7,
                     color="white" if norm[r, c] > 0.5 else "black")
        ax_.set_xticks(range(4), LEVELS)
        ax_.set_yticks(range(4), LEVELS)
        ax_.set_xlabel("predicted")
        ax_.set_ylabel("actual (rule level of the observed value)")
        ax_.set_title(f"+{h} min: {best['model']} (best on validation)")
        plt.tight_layout()
        f = fig / f"confusion_h{h}.png"
        plt.savefig(f, dpi=130)
        plt.close("all")
        files.append(str(f))
    return files


def _report(cfg: CongestionExperimentConfig, exp_id: str, info: dict[str, Any], long: pd.DataFrame,
            plots: list[str]) -> str:
    lines = [f"# Congestion-level forecast: {cfg.name}", "", f"Experiment `{exp_id}`.", "", cfg.description.strip(), "",
             f"- Dataset: {info['dataset']} ({len(info['series'])} series, {info['resolution_s']} s)",
             "- Periods: " + ", ".join(f"{k} {v['start'][:10]}..{v['end'][:10]}" for k, v in info["periods"].items()),
             f"- Samples: {info['samples']}",
             f"- Label = rule level of the observed value at t + h, indicators: {info['label_indicators']}",
             f"- Test class distribution: {info['class_distribution']['test']}", ""]
    cols = ["model", "accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1",
            "weighted_f1", "val_macro_f1"]
    for h in cfg.horizons_min:
        sub = long[long["horizon_min"] == h].sort_values("val_macro_f1", ascending=False)
        lines += [f"## +{h} min (test; sorted by validation macro-F1)", "", "| " + " | ".join(cols) + " |",
                  "|" + "---|" * len(cols)]
        for _, r in sub.iterrows():
            lines.append("| " + " | ".join(str(r[c]) if c == "model" else f"{r[c]:.3f}" for c in cols) + " |")
        lines.append("")
    lines += ["Figures: " + ", ".join(Path(p).name for p in plots), ""]
    return "\n".join(lines)


def _index(path: Path, exp_id: str, cfg: CongestionExperimentConfig, long: pd.DataFrame, seconds: float) -> None:
    df = long[["horizon_min", "model", "accuracy", "macro_f1", "weighted_f1", "balanced_accuracy",
               "val_macro_f1", "fit_s"]].assign(experiment_id=exp_id, dataset=cfg.dataset.source,
                                                created=datetime.now(UTC).isoformat(timespec="seconds"),
                                                experiment_seconds=round(seconds, 1))
    old = pd.read_csv(path) if path.exists() else pd.DataFrame()
    pd.concat([old, df], ignore_index=True).to_csv(path, index=False)
