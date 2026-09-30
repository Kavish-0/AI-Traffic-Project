"""Shared ML infrastructure: trainable-model contract and artifact store.

Training (offline) and inference (online) meet only here:

    ai.training  --fit()--> TrainableModel --save()--> models/<task>/<name>/<version>/
                                                           model.*        (weights)
                                                           metadata.json  (ModelMetadata)
    pipeline     <--load()-- ModelStore.load_production(task, name)

Online code never calls ``fit``. A model whose ``feature_schema_hash`` does
not match ``core.feature_schema.schema_hash()`` is refused at load time.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from traffic_intel.core.enums import ModelStatus, ModelTask


class ModelMetadata(BaseModel):
    task: ModelTask
    name: str                                   # e.g. "xgboost", "lstm"
    version: str                                # e.g. "2026-09-23T10-00_ab12cd"
    framework: str                              # "sklearn" | "xgboost" | "torch" | "rules"
    feature_schema_hash: str
    trained_at: datetime
    git_sha: str | None = None
    train_data: dict[str, Any] = Field(default_factory=dict)   # date range, approaches, rows
    params: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)      # e.g. {"h15": {"mae": ...}}
    status: ModelStatus = ModelStatus.CANDIDATE


class TrainableModel(ABC):
    """Base for every learned component (forecasters, anomaly models, ML
    congestion classifiers, incident fusion)."""

    metadata: ModelMetadata | None = None

    @abstractmethod
    def save(self, directory: Path) -> None:
        """Write weights + any scalers to ``directory`` (metadata is written by the store)."""

    @classmethod
    @abstractmethod
    def load(cls, directory: Path) -> TrainableModel: ...


class ModelStore(ABC):
    """Versioned artifact store. Phase 5 ships a filesystem implementation under
    ``models/``; the ``model_registry`` DB table mirrors it for the API/UI."""

    @abstractmethod
    def save(self, model: TrainableModel, metadata: ModelMetadata) -> Path: ...

    @abstractmethod
    def load(self, task: ModelTask, name: str, version: str) -> TrainableModel: ...

    @abstractmethod
    def load_production(self, task: ModelTask, name: str) -> TrainableModel: ...

    @abstractmethod
    def list(self, task: ModelTask | None = None) -> list[ModelMetadata]: ...

    @abstractmethod
    def promote(self, task: ModelTask, name: str, version: str) -> None:
        """Mark a version PRODUCTION (and archive the previous one)."""


class FileModelStore(ModelStore):
    """Filesystem registry.

        <root>/<task>/<name>/<version>/
            params.json, model.*        (written by the model)
            preprocessor.json           (fitted preprocessing, if any)
            metadata.json               (ModelMetadata: data periods, features, hyper-parameters, metrics)
        <root>/<task>/<name>/PRODUCTION (text: the promoted version)

    Versions are never overwritten; a new training run always creates a new version.
    """

    def __init__(self, root: Path) -> None:
        self.root = root

    def _dir(self, task: ModelTask, name: str, version: str | None = None) -> Path:
        d = self.root / ModelTask(task).value / name
        return d / version if version else d

    @staticmethod
    def _model_class(meta: ModelMetadata) -> type[TrainableModel]:
        if meta.task is ModelTask.FORECASTING:
            from traffic_intel.forecasting.interfaces import forecaster_class

            return forecaster_class(meta.params.get("type", meta.name))
        if meta.task is ModelTask.ANOMALY:
            from traffic_intel.anomaly.detectors.learned import AutoencoderDetector, IsolationForestDetector

            kinds: dict[str, type[TrainableModel]] = {"isolation_forest": IsolationForestDetector,
                                                      "autoencoder": AutoencoderDetector}
            return kinds[meta.params.get("type", meta.name)]
        raise NotImplementedError(f"no loader for task {meta.task}")

    def save(self, model: TrainableModel, metadata: ModelMetadata, **extras: object) -> Path:
        d = self._dir(metadata.task, metadata.name, metadata.version)
        if d.exists():
            raise FileExistsError(f"model version already exists: {d}")
        d.mkdir(parents=True)
        model.save(d)
        pre = extras.get("preprocessor")
        if pre is not None:
            pre.save(d / "preprocessor.json")  # type: ignore[attr-defined]
        (d / "metadata.json").write_text(metadata.model_dump_json(indent=2), encoding="utf-8")
        return d

    def metadata(self, task: ModelTask, name: str, version: str) -> ModelMetadata:
        return ModelMetadata.model_validate_json((self._dir(task, name, version) / "metadata.json").read_text("utf-8"))

    def resolve_version(self, task: ModelTask, name: str, version: str = "production") -> str:
        """'production' -> promoted version; 'latest' -> newest; else as given."""
        base = self._dir(task, name)
        if version == "production":
            f = base / "PRODUCTION"
            if not f.exists():
                raise FileNotFoundError(f"no PRODUCTION version for {task}/{name} - promote one first")
            return f.read_text(encoding="utf-8").strip()
        if version == "latest":
            versions = sorted(p.name for p in base.iterdir() if (p / "metadata.json").exists()) if base.exists() else []
            if not versions:
                raise FileNotFoundError(f"no versions of {task}/{name} under {base}")
            return versions[-1]
        return version

    def load(self, task: ModelTask, name: str, version: str) -> TrainableModel:
        version = self.resolve_version(task, name, version)
        meta = self.metadata(task, name, version)
        model = self._model_class(meta).load(self._dir(task, name, version))
        model.metadata = meta
        return model

    def load_production(self, task: ModelTask, name: str) -> TrainableModel:
        return self.load(task, name, "production")

    def list(self, task: ModelTask | None = None) -> list[ModelMetadata]:
        out = []
        for meta_file in sorted(self.root.glob(f"{ModelTask(task).value if task else '*'}/*/*/metadata.json")):
            out.append(ModelMetadata.model_validate_json(meta_file.read_text("utf-8")))
        return out

    def promote(self, task: ModelTask, name: str, version: str) -> None:
        version = self.resolve_version(task, name, version)
        d = self._dir(task, name, version)
        if not (d / "metadata.json").exists():
            raise FileNotFoundError(f"unknown version {d}")
        prev = self._dir(task, name) / "PRODUCTION"
        if prev.exists():
            old = prev.read_text(encoding="utf-8").strip()
            if old != version and (self._dir(task, name, old) / "metadata.json").exists():
                m = self.metadata(task, name, old)
                m.status = ModelStatus.ARCHIVED
                (self._dir(task, name, old) / "metadata.json").write_text(m.model_dump_json(indent=2), "utf-8")
        m = self.metadata(task, name, version)
        m.status = ModelStatus.PRODUCTION
        (d / "metadata.json").write_text(m.model_dump_json(indent=2), encoding="utf-8")
        prev.write_text(version, encoding="utf-8")

    def retire(self, task: ModelTask, name: str) -> str | None:
        """Take ``name`` out of production (its promoted version becomes ARCHIVED). Returns that version."""
        marker = self._dir(task, name) / "PRODUCTION"
        if not marker.exists():
            return None
        version = marker.read_text(encoding="utf-8").strip()
        if (self._dir(task, name, version) / "metadata.json").exists():
            m = self.metadata(task, name, version)
            m.status = ModelStatus.ARCHIVED
            (self._dir(task, name, version) / "metadata.json").write_text(m.model_dump_json(indent=2), "utf-8")
        marker.unlink()
        return version
