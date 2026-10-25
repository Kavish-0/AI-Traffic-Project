"""Model registry (read-only, metadata files only - no model is loaded in the API process)."""

from __future__ import annotations

import csv
from collections.abc import Callable
from pathlib import Path
from typing import Any

from traffic_intel.ai.model_store import FileModelStore, ModelMetadata
from traffic_intel.api.schemas import Evidence, ModelInfo, ServingComponent
from traffic_intel.core.enums import ModelStatus, ModelTask
from traffic_intel.services.context import ServiceContext


def _store(ctx: ServiceContext) -> FileModelStore:
    return FileModelStore(ctx.settings.models_dir)


def to_info(md: ModelMetadata) -> ModelInfo:
    td = md.train_data or {}
    return ModelInfo(task=md.task.value, name=md.name, version=md.version, trained_on=td.get("dataset"),
                     resolution_s=td.get("resolution_s"), required_history=td.get("window_lookback_steps"),
                     trained_at=md.trained_at, metrics=md.metrics if md.task is ModelTask.ANOMALY else {})


def production_forecaster(ctx: ServiceContext) -> ModelInfo | None:
    """The forecaster the runtime serves: ``configs/pipeline.yaml`` forecaster.params (model / version),
    resolved exactly like ForecastingService.from_registry; ``auto`` = newest PRODUCTION forecaster."""
    from traffic_intel.core.config import load_pipeline_config

    store = _store(ctx)
    params = load_pipeline_config(ctx.settings).forecaster.params
    model, version = str(params.get("model", "auto")), str(params.get("version", "production"))
    if model != "auto":
        try:
            return to_info(store.metadata(ModelTask.FORECASTING, model,
                                          store.resolve_version(ModelTask.FORECASTING, model, version)))
        except (FileNotFoundError, OSError):
            return None
    prod = [m for m in store.list(ModelTask.FORECASTING) if m.status is ModelStatus.PRODUCTION]
    return to_info(sorted(prod, key=lambda m: m.trained_at)[-1]) if prod else None


def list_models(ctx: ServiceContext, task: ModelTask | None = None, production_only: bool = False) -> list[ModelInfo]:
    metas = _store(ctx).list(task)
    if production_only:
        metas = [m for m in metas if m.status is ModelStatus.PRODUCTION]
    return [to_info(m) for m in sorted(metas, key=lambda m: (m.task.value, m.name, m.version))]


# --------------------------------------------------------------------------- serving components (Models page)
def _latest_rows(task_dir: str, name: str,
                 match: Callable[[dict[str, str]], bool]) -> tuple[str, dict[str, str]] | None:
    """Newest non-smoke experiment under experiments/<task_dir>/ whose <name> CSV has a row ``match(row)``."""
    from traffic_intel.core.config import PROJECT_ROOT

    root = PROJECT_ROOT / "experiments" / task_dir
    if not root.exists():
        return None
    for run in sorted((p for p in root.iterdir() if p.is_dir() and "smoke" not in p.name), reverse=True):
        f = run / name
        if not f.exists():
            continue
        with f.open(encoding="utf-8") as fh:
            row = next((r for r in csv.DictReader(fh) if match(r)), None)
        if row is not None:
            return run.name, row
    return None


def _num(v: Any, nd: int = 3) -> float | None:  # noqa: ANN401 - CSV / JSON cell
    try:
        return round(float(v), nd)
    except (TypeError, ValueError):
        return None


def serving_components(ctx: ServiceContext) -> list[ServingComponent]:
    """What each pipeline stage runs NOW (the active pipeline file) and the measured evidence behind it, read
    from the experiment records and the model registry. Numbers are never invented: a stage without a
    measurement says why."""
    from traffic_intel.core.config import load_pipeline_config

    p = load_pipeline_config(ctx.settings)
    out: list[ServingComponent] = []

    weights = Path(str(p.detector.params.get("weights", ""))).name
    det = _latest_rows("detection", "metrics_val.csv", lambda r: Path(r.get("weights", "")).name == weights)
    out.append(ServingComponent(
        stage="Vehicle detection", implementation=f"{p.detector.type}: {weights or '-'}",
        detail=f"image size {p.detector.params.get('imgsz', '-')}",
        evidence=Evidence(dataset=f"UA-DETRAC validation (car / bus / truck) at {det[1].get('imgsz', '?')} px",
                          experiment=det[0],
                          metrics={"mAP50": _num(det[1].get("mAP50")), "mAP50-95": _num(det[1].get("mAP50-95")),
                                   "ms per image": _num(det[1].get("ms_per_image"), 1)}) if det else None,
        no_evidence_reason=None if det else "no evaluation record for these weights"))

    trk = _latest_rows("tracking", "metrics_test.csv", lambda r: r.get("tracker") == p.tracker.type)
    out.append(ServingComponent(
        stage="Tracking", implementation=p.tracker.type,
        detail=f"new_track_thresh {p.tracker.params.get('new_track_thresh', '-')}",
        evidence=Evidence(dataset="UA-DETRAC test", experiment=trk[0],
                          metrics={"MOTA": _num(trk[1].get("MOTA")), "precision": _num(trk[1].get("precision")),
                                   "recall": _num(trk[1].get("recall")),
                                   "ID switches": _num(trk[1].get("id_switches"), 0)}) if trk else None,
        no_evidence_reason=None if trk else "no evaluation record for this tracker"))

    fc = production_forecaster(ctx)
    fc_ev = None
    if fc is not None:
        try:
            md = _store(ctx).metadata(ModelTask.FORECASTING, fc.name, fc.version)
            test = [m for m in (md.metrics or {}).get("test", [])
                    if isinstance(m, dict) and m.get("target", "avg_speed_kmh") == "avg_speed_kmh"]
        except (FileNotFoundError, OSError):
            test = []
        if test:
            metrics: dict[str, float | None] = {
                f"RMSE +{m.get('horizon_min')} min": _num(m.get("RMSE"), 2)
                for m in sorted(test, key=lambda m: m.get("horizon_min", 0))}
            skill = [float(m["skill_vs_persistence"]) for m in test if m.get("skill_vs_persistence") is not None]
            if skill:
                metrics["% better than 'no change'"] = _num(sum(skill) / len(skill) * 100, 1)
            fc_ev = Evidence(dataset=f"{fc.trained_on} held-out test period", experiment=fc.version, metrics=metrics,
                             unit_note="RMSE in km/h")
    out.append(ServingComponent(
        stage="Speed forecasting (+5 / +15 / +30 min)", implementation=fc.name if fc else "none loaded",
        detail=f"trained on {fc.trained_on}" if fc else "", evidence=fc_ev,
        no_evidence_reason=None if fc_ev else "no test metrics stored with the model"))

    out.append(ServingComponent(stage="Congestion level", implementation=p.congestion_classifier.type,
                                detail="explainable rules (configs/congestion_rules.yaml)",
                                no_evidence_reason="no independent congestion labels exist"))

    inc = _latest_rows("incident", "metrics_test.csv", lambda r: r.get("method") == "stationary_rule")
    out.append(ServingComponent(
        stage="Anomaly / incident detection", implementation=f"{p.incident_fuser.type} (UNCALIBRATED)",
        detail="series detectors: " + ", ".join(d.type for d in p.series_anomaly_detectors)
               + "; stopped-vehicle and trajectory rules",
        evidence=Evidence(dataset="AI City 2021 Track 4 test videos, stopped-vehicle rule (evaluation only)",
                          experiment=inc[0],
                          metrics={"window ROC-AUC": _num(inc[1].get("window_roc_auc")),
                                   "share of incidents found": _num(inc[1].get("event_recall")),
                                   "share of alerts correct": _num(inc[1].get("event_precision")),
                                   "median delay (s)": _num(inc[1].get("event_median_delay_s"), 1)}) if inc else None,
        no_evidence_reason=None if inc else "no evaluation record"))

    out.append(ServingComponent(stage="Risk score", implementation=p.risk_engine.type,
                                detail="documented weighted formula (configs/risk.yaml)",
                                no_evidence_reason="no risk labels exist in any dataset"))
    out.append(ServingComponent(stage="What-if simulation", implementation=p.simulator.type,
                                detail="static macroscopic re-assignment (decision support)",
                                no_evidence_reason="no ground truth exists for counterfactual closures"))
    return out
