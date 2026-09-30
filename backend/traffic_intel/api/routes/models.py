"""Model registry and evaluation reports, read-only (metadata files; no model is loaded here).
Training and promotion are offline CLI actions (python -m traffic_intel.ai.training)."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from traffic_intel.api.deps import get_ctx
from traffic_intel.api.schemas import ModelInfo
from traffic_intel.core.config import PROJECT_ROOT
from traffic_intel.core.enums import ModelTask
from traffic_intel.services import models as svc
from traffic_intel.services.context import ServiceContext

router = APIRouter(prefix="/models", tags=["models"])


@router.get("")
def list_models(task: ModelTask | None = None, production_only: bool = False,
                ctx: ServiceContext = Depends(get_ctx)) -> list[ModelInfo]:
    return svc.list_models(ctx, task, production_only)


@router.get("/{task}/evaluation")
def evaluation_report(task: ModelTask) -> dict[str, Any]:
    """Latest experiment's test metrics (forecasting: per model and horizon; anomaly: per detector)."""
    root = PROJECT_ROOT / "experiments" / task.value
    runs: list[Path] = sorted(p for p in root.iterdir() if p.is_dir()) if root.exists() else []
    name = "metrics_test.csv" if task is ModelTask.FORECASTING else "metrics.csv"
    runs = [r for r in runs if (r / name).exists()]
    if not runs:
        raise HTTPException(404, f"no {task.value} experiment with {name} under experiments/{task.value}/")
    latest: Path = runs[-1]
    with (latest / name).open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return {"experiment_id": latest.name, "rows": rows}
