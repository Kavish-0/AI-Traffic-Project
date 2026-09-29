"""Evaluation summary tables straight from the experiment records (no number is typed by hand).

    python -m traffic_intel.ai.training summary [--out reports/model_evaluation_tables.md]

For each experiment config the LATEST non-smoke run is used. Every table cites its experiment id.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from traffic_intel.core.config import PROJECT_ROOT

EXP = PROJECT_ROOT / "experiments"


DONE = {"forecasting": "comparison.md", "congestion": "report.md", "incident": "report.md",
        "detection": "metrics_test.csv", "tracking": "metrics_test.csv", "anomaly": "report.md"}


def _latest(task: str, suffix: str) -> Path | None:
    """Newest COMPLETED, non-smoke run of a config (a run still in progress has no report yet)."""
    runs = sorted(p for p in (EXP / task).glob(f"*_{suffix}")
                  if p.is_dir() and not p.name.endswith("_smoke") and (p / DONE[task]).exists())
    return runs[-1] if runs else None


FORECAST_CONFIGS = [("metr_la_traffic_only", "METR-LA - traffic only"),
                    ("metr_la_context", "METR-LA - traffic + context"),
                    ("pems_bay_traffic_only", "PEMS-BAY - traffic only"),
                    ("pems_bay_context", "PEMS-BAY - traffic + context"),
                    ("pems08_benchmark", "PEMS08 (flow, primary target)"),
                    ("synthetic_junction", "Synthetic junction (live feature set)")]
CONGESTION_CONFIGS = [("congestion_level_metr_la", "METR-LA"), ("congestion_level_pems_bay", "PEMS-BAY"),
                      ("congestion_level_synthetic", "Synthetic junction (live feature set)")]


def selected_model(d: Path) -> str:
    """The forecasting model an experiment selected on VALIDATION (as written in its comparison.md)."""
    cmp_md = (d / "comparison.md").read_text(encoding="utf-8")
    return cmp_md.split("-> **", 1)[1].split("**", 1)[0] if "-> **" in cmp_md else "?"


def _fmt(v: Any, nd: int = 3) -> str:  # noqa: ANN401
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}" if abs(v) < 1000 else f"{v:.0f}"
    return str(v)


def _table(df: pd.DataFrame, cols: list[str], nd: int = 3) -> list[str]:
    text_cols = ("model", "method", "tracker", "dataset", "task", "experiment")
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" if c in text_cols else "--:" for c in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(_fmt(r[c], nd) for c in cols) + " |")
    return out


def forecasting(configs: list[tuple[str, str]]) -> list[str]:
    lines: list[str] = []
    for suffix, title in configs:
        d = _latest("forecasting", suffix)
        if d is None:
            lines += [f"### {title}", "", "_not run_", ""]
            continue
        m = pd.read_csv(d / "metrics_test.csv")
        info = json.loads((d / "dataset.json").read_text(encoding="utf-8"))
        selected = selected_model(d)
        target = yaml.safe_load((d / "config.yaml").read_text(encoding="utf-8"))["primary_target"]  # = selection
        mt = m[m["target"] == target]
        avg = mt.groupby("model", sort=False)[["MAE", "RMSE", "R2", "MAPE", "skill_vs_persistence"]].mean() \
            .reset_index().sort_values("RMSE")
        avg = avg.rename(columns={"skill_vs_persistence": "skill"})
        lines += [f"### {title}", "",
                  f"Experiment `{d.name}` · {len(info['series'])} series · samples {info['samples']} · "
                  f"test {info['periods']['test']['start'][:10]} .. {info['periods']['test']['end'][:10]} · "
                  f"target `{target}` · **selected on validation: {selected}**", "",
                  "Test, averaged over +5 / +15 / +30 min:", ""]
        lines += _table(avg, ["model", "MAE", "RMSE", "R2", "MAPE", "skill"])
        lines += ["", "Test, per horizon (persistence, linear, best tree, best deep, selected):", ""]
        keep = {"persistence", "linear", selected}
        trees = avg[avg["model"].str.startswith(("random_forest", "xgboost"))]
        deep = avg[avg["model"].isin(["lstm", "gru", "transformer", "lstm_L24"])]
        if len(trees):
            keep.add(trees.iloc[0]["model"])
        if len(deep):
            keep.add(deep.iloc[0]["model"])
        ph = mt[mt["model"].isin(keep)].sort_values(["horizon_min", "RMSE"])
        lines += _table(ph.rename(columns={"horizon_min": "horizon"}),
                        ["horizon", "model", "MAE", "RMSE", "R2", "MAPE"])
        lines.append("")
    return lines


def congestion(configs: list[tuple[str, str]]) -> list[str]:
    lines: list[str] = []
    for suffix, title in configs:
        d = _latest("congestion", suffix)
        if d is None:
            continue
        m = pd.read_csv(d / "metrics_test.csv")
        info = json.loads((d / "dataset.json").read_text(encoding="utf-8"))
        lines += [f"### {title}", "", f"Experiment `{d.name}` · label indicators {info['label_indicators']} · "
                  f"test class counts (+15 min) {info['class_distribution']['test'].get('15')}", ""]
        for h, g in m.groupby("horizon_min"):
            g = g.sort_values("val_macro_f1", ascending=False)
            lines += [f"**+{h} min** (sorted by validation macro-F1; test metrics)", ""]
            lines += _table(g, ["model", "accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1",
                                "val_macro_f1"])
            lines.append("")
    return lines


def anomaly() -> list[str]:
    runs = sorted(p for p in (EXP / "anomaly").glob("*")
                  if p.is_dir() and not p.name.endswith("_smoke") and (p / "report.md").exists())
    if not runs:
        return []
    d = runs[-1]
    rep = (d / "report.md").read_text(encoding="utf-8")
    table = [ln for ln in rep.splitlines() if ln.startswith("|")]
    return [f"Experiment `{d.name}` (synthetic junction, injected capacity drops)", ""] + table + [""]


def incident() -> list[str]:
    d = _latest("incident", "aic21_track4")
    if d is None:
        return ["_not run_", ""]
    rec = json.loads((d / "experiment.json").read_text(encoding="utf-8"))
    m = pd.DataFrame(rec["metrics"])
    lines = [f"Experiment `{d.name}` · {rec['videos']} videos · windows {rec['windows']} · positive windows "
             f"{rec['positive_windows']} · test videos {len(rec['split']['test'])}", ""]
    lines += _table(m, ["method", "window_roc_auc", "window_pr_auc", "window_precision", "window_recall", "window_f1",
                        "event_precision", "event_recall", "event_f1", "event_median_delay_s"])
    lines += ["", f"AI City timing score (stationary rule, onset back-dated): `{rec['aic21_timing_rule']}`", "",
              f"Calibration (best score on validation windows -> test windows): `{rec['calibration']}`", ""]
    return lines


def detection_runs() -> list[Path]:
    """Completed (or partially completed) non-smoke detection runs, oldest first."""
    return sorted(p for p in (EXP / "detection").glob("*")
                  if p.is_dir() and not p.name.endswith("_smoke") and (p / "metrics_test.csv").exists())


def detection_rows() -> pd.DataFrame:
    """Every evaluated model x input size; the newest evaluation of a (model, imgsz) wins.
    All runs score the same test split through the same code, so rows are comparable."""
    frames = [pd.read_csv(d / "metrics_test.csv").assign(experiment=d.name) for d in detection_runs()]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates(["model", "imgsz"], keep="last")


def detection() -> list[str]:
    m = detection_rows()
    if m.empty:
        return ["_not run_", ""]
    m["model"] = m["model"] + " @" + m["imgsz"].astype(str)
    exps = ", ".join(f"`{e}`" for e in dict.fromkeys(m["experiment"]))
    return [f"Experiments {exps} · UA-DETRAC official test split (40 sequences, every 10th frame), "
            "scored through the production detector adapter", ""] + \
        _table(m, ["model", "precision", "recall", "mAP50", "mAP50-95", "car_mAP50-95", "bus_mAP50-95",
                   "truck_mAP50-95", "ms_per_image", "train_seconds", "experiment"]) + [""]


def tracking() -> list[str]:
    d = _latest("tracking", "tracker_ua_detrac")
    if d is None:
        return ["_not run_", ""]
    m = pd.read_csv(d / "metrics_test.csv")
    cols = ["tracker", "chosen_on_validation", "val_MOTA", "MOTA", "MOTP_iou", "precision", "recall", "id_switches",
            "ms_per_frame"]
    return [f"Experiment `{d.name}`", ""] + _table(m, cols, 4) + [""]


def build() -> str:
    fc, cg = FORECAST_CONFIGS, CONGESTION_CONFIGS
    parts = ["# Model evaluation tables (generated)", "",
             "Generated by `python -m traffic_intel.ai.training summary` from the experiment records.", "",
             "## Forecasting", ""] + forecasting(fc) + ["## Congestion level (+h forecast)", ""] + congestion(cg) + \
            ["## Anomaly detection (live feature set)", ""] + anomaly() + \
            ["## Incident detection on real video (AI City 2021 Track 4)", ""] + incident() + \
            ["## Vehicle detection (UA-DETRAC)", ""] + detection() + ["## Tracking (UA-DETRAC)", ""] + tracking()
    return "\n".join(parts) + "\n"


def write(out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build(), encoding="utf-8")
    return out
