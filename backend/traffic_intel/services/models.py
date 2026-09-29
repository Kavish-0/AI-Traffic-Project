"""Model registry (read-only, metadata files only - no model is loaded in the API process)."""

from __future__ import annotations

from traffic_intel.ai.model_store import FileModelStore, ModelMetadata
from traffic_intel.api.schemas import ModelInfo
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
