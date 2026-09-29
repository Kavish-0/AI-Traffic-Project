"""Offline training / evaluation CLI (never imported by the online pipeline).

    python -m traffic_intel.ai.training train forecasting --config configs/training/forecasting_pems08.yaml
    python -m traffic_intel.ai.training train forecasting --config configs/training/forecasting.yaml --promote best-val
    python -m traffic_intel.ai.training train forecasting --config ... --only persistence,linear,xgboost
    python -m traffic_intel.ai.training evaluate forecasting --experiment <experiment_id>
    python -m traffic_intel.ai.training list forecasting
    python -m traffic_intel.ai.training promote forecasting <model> <version|latest>
    python -m traffic_intel.ai.training train anomaly --config configs/training/anomaly.yaml [--promote best-val]
    python -m traffic_intel.ai.training promote anomaly isolation_forest latest
    python -m traffic_intel.ai.training train detection --config configs/training/detection_ua_detrac.yaml [--smoke]
    python -m traffic_intel.ai.training train detection --config ... --only yolo11m_ua_detrac --baselines-only
    python -m traffic_intel.ai.training promote detection yolo11s_ua_detrac <version>
    python -m traffic_intel.ai.training train congestion --config configs/training/congestion_level_metr_la.yaml
    python -m traffic_intel.ai.training summary      # reports/model_evaluation_tables.md
    python -m traffic_intel.ai.training figures      # reports/figures/ (+ INDEX.md)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from traffic_intel.core.config import PROJECT_ROOT, get_settings
from traffic_intel.core.enums import ModelTask
from traffic_intel.core.logging import configure_logging


def _experiments_root() -> Path:
    return PROJECT_ROOT / "experiments" / "forecasting"


def cmd_train_anomaly(a: argparse.Namespace) -> None:
    from traffic_intel.ai.model_store import FileModelStore
    from traffic_intel.ai.training.anomaly import load_anomaly_config, run_anomaly_experiment

    cfg_path = Path(a.config)
    cfg = load_anomaly_config(cfg_path if cfg_path.is_absolute() else Path.cwd() / cfg_path)
    res = run_anomaly_experiment(cfg)
    print(f"\nexperiment {res.experiment_id} -> {res.directory}\n")
    print((Path(res.directory) / "report.md").read_text(encoding="utf-8"))
    if a.promote == "best-val":
        # learned anomaly detectors are promoted together (both are optional pipeline inputs)
        store = FileModelStore(get_settings().models_dir)
        for m in res.metrics:
            if m["detector"] in ("isolation_forest", "autoencoder"):
                store.promote(ModelTask.ANOMALY, m["detector"], res.experiment_id)
                print(f"promoted {m['detector']}@{res.experiment_id} to PRODUCTION")


def cmd_train_detection(a: argparse.Namespace) -> None:
    import json

    from traffic_intel.ai.training.detection import load_detection_config, run_detection_experiment

    cfg_path = Path(a.config)
    cfg = load_detection_config(cfg_path if cfg_path.is_absolute() else Path.cwd() / cfg_path)
    rec = run_detection_experiment(cfg, only=a.only.split(",") if a.only else None, smoke=a.smoke,
                                   baselines_only=a.baselines_only)
    for m in rec["models"]:
        print(json.dumps({k: m[k] for k in ("name", "artifact", "train_seconds", "best_epoch")}, default=str))
    print(f"experiment {rec['experiment_id']} -> experiments/detection/{rec['experiment_id']}/metrics_test.csv")


def cmd_train_congestion(a: argparse.Namespace) -> None:
    from traffic_intel.ai.training.congestion import load_congestion_config, run_congestion_experiment

    cfg_path = Path(a.config)
    cfg = load_congestion_config(cfg_path if cfg_path.is_absolute() else Path.cwd() / cfg_path)
    res = run_congestion_experiment(cfg, smoke=a.smoke)
    print(f"experiment {res['experiment_id']} -> {res['directory']}")
    print((Path(res["directory"]) / "report.md").read_text(encoding="utf-8"))


def cmd_train_incident(a: argparse.Namespace) -> None:
    from traffic_intel.ai.training.incident_video import load_incident_config, run_incident_experiment

    cfg_path = Path(a.config)
    cfg = load_incident_config(cfg_path if cfg_path.is_absolute() else Path.cwd() / cfg_path)
    if a.smoke:
        cfg.videos = ["2", "9", "11", "3", "4", "5"]      # 3 labelled + 3 normal videos
    rec = run_incident_experiment(cfg, smoke=a.smoke)
    print((Path(PROJECT_ROOT) / "experiments" / "incident" / rec["experiment_id"] / "report.md").read_text("utf-8"))


def cmd_train_tracking(a: argparse.Namespace) -> None:
    import json

    from traffic_intel.ai.training.tracking import load_tracking_config, run_tracking_experiment

    cfg_path = Path(a.config)
    cfg = load_tracking_config(cfg_path if cfg_path.is_absolute() else Path.cwd() / cfg_path)
    rec = run_tracking_experiment(cfg, smoke=a.smoke)
    print(json.dumps(rec["results"], indent=2, default=str))


def cmd_train(a: argparse.Namespace) -> None:
    if a.task == "tracking":
        cmd_train_tracking(a)
        return
    if a.task == "incident":
        cmd_train_incident(a)
        return
    if a.task == "congestion":
        cmd_train_congestion(a)
        return
    if a.task == "anomaly":
        cmd_train_anomaly(a)
        return
    if a.task == "detection":
        cmd_train_detection(a)
        return
    from traffic_intel.ai.model_store import FileModelStore
    from traffic_intel.ai.training.experiment import load_experiment_config, run_experiment

    cfg_path = Path(a.config)
    cfg = load_experiment_config(cfg_path if cfg_path.is_absolute() else Path.cwd() / cfg_path)
    only = a.only.split(",") if a.only else None
    res = run_experiment(cfg, only=only, smoke=a.smoke)
    print(f"\nexperiment {res.experiment_id} -> {res.directory}")
    print((Path(res.directory) / "comparison.md").read_text(encoding="utf-8").split("## Observations")[0][-3000:])
    if a.promote == "best-val":
        store = FileModelStore(get_settings().models_dir)
        best = res.best_by_validation["model"]
        store.promote(ModelTask.FORECASTING, best, res.experiment_id)
        print(f"promoted {best}@{res.experiment_id} to PRODUCTION (selected on validation, not test)")


def cmd_evaluate(a: argparse.Namespace) -> None:
    root = _experiments_root()
    exp = a.experiment or sorted(p.name for p in root.iterdir() if p.is_dir())[-1]
    d = root / exp
    m = pd.read_csv(d / "metrics_test.csv")
    pd.set_option("display.width", 160)
    print(f"experiment {exp}\n")
    for tgt, g in m.groupby("target", sort=False):
        s = g.groupby("model", sort=False)[["MAE", "RMSE", "R2", "skill_vs_persistence"]].mean().sort_values("RMSE")
        print(f"== {tgt} (test, mean over horizons)\n{s.round(3).to_string()}\n")
    print(f"full report: {d / 'comparison.md'}")


def cmd_list(a: argparse.Namespace) -> None:
    idx = _experiments_root() / "index.csv"
    if not idx.exists():
        raise SystemExit("no experiments yet")
    df = pd.read_csv(idx)
    print(df[["experiment_id", "model", "dataset", "primary_target", "test_MAE", "test_RMSE", "test_R2"]]
          .to_string(index=False))


def cmd_promote(a: argparse.Namespace) -> None:
    from traffic_intel.ai.model_store import FileModelStore

    store = FileModelStore(get_settings().models_dir)
    store.promote(ModelTask(a.task), a.model, a.version)
    print(f"promoted {a.model}@{store.resolve_version(ModelTask(a.task), a.model, 'production')}")


def cmd_summary(a: argparse.Namespace) -> None:
    from traffic_intel.ai.evaluation.summary import write

    out = Path(a.out) if a.out else PROJECT_ROOT / "reports" / "model_evaluation_tables.md"
    print(f"written {write(out)}")


def cmd_figures(a: argparse.Namespace) -> None:
    from traffic_intel.ai.evaluation.figures import FIG, build_all

    files = build_all(rescore=not a.no_rescore, samples=not a.no_samples)
    print(f"written {len(files)} figures under {FIG} (index: {FIG / 'INDEX.md'})")


def cmd_retire(a: argparse.Namespace) -> None:
    from traffic_intel.ai.model_store import FileModelStore

    v = FileModelStore(get_settings().models_dir).retire(ModelTask(a.task), a.model)
    print(f"retired {a.model}@{v}" if v else f"{a.model} has no PRODUCTION version")


def main() -> None:
    p = argparse.ArgumentParser(prog="traffic_intel.ai.training")
    sub = p.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train")
    t.add_argument("task", choices=["forecasting", "anomaly", "detection", "congestion", "incident", "tracking"])
    t.add_argument("--config", required=True)
    t.add_argument("--only", help="comma-separated model names from the config")
    t.add_argument("--promote", choices=["best-val"], help="promote the model with the best validation RMSE")
    t.add_argument("--smoke", action="store_true", help="pipeline check: 1 epoch, nothing registered")
    t.add_argument("--baselines-only", action="store_true", help="detection: zero-shot COCO evaluation, no training")
    t.set_defaults(func=cmd_train)
    e = sub.add_parser("evaluate")
    e.add_argument("task", choices=["forecasting"])
    e.add_argument("--experiment", help="experiment id (default: latest)")
    e.set_defaults(func=cmd_evaluate)
    ls = sub.add_parser("list")
    ls.add_argument("task", choices=["forecasting"])
    ls.set_defaults(func=cmd_list)
    pr = sub.add_parser("promote")
    pr.add_argument("task", choices=["forecasting", "anomaly", "detection", "congestion"])
    pr.add_argument("model")
    pr.add_argument("version")
    pr.set_defaults(func=cmd_promote)
    rt = sub.add_parser("retire", help="take a model out of production (its version becomes ARCHIVED)")
    rt.add_argument("task", choices=["forecasting", "anomaly", "detection", "congestion"])
    rt.add_argument("model")
    rt.set_defaults(func=cmd_retire)
    sm = sub.add_parser("summary", help="evaluation tables from the experiment records (markdown)")
    sm.add_argument("--out", default=None)
    sm.set_defaults(func=cmd_summary)
    fg = sub.add_parser("figures", help="evaluation figures under reports/figures from the experiment records")
    fg.add_argument("--no-rescore", action="store_true", help="forecasting: copy run plots only (no re-scoring)")
    fg.add_argument("--no-samples", action="store_true", help="detection: skip the sample-prediction images")
    fg.set_defaults(func=cmd_figures)
    a = p.parse_args()
    configure_logging(get_settings().log_level)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    a.func(a)


if __name__ == "__main__":
    main()
