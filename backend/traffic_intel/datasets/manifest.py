"""Dataset manifest (``configs/datasets/manifest.yaml``): what each dataset is, where it lives,
under which licence, and how to check that it is present.

Nothing here downloads anything. ``validate`` checks the expected files under the dataset's
local path and, if they are missing, fails with a message that says what is missing, where to
put it, how to obtain it and under which terms.

Local paths are relative to the data directory (``TI_DATA_DIR``, default ``<project>/data``).
A git-ignored ``configs/datasets/local.yaml`` may override single datasets' locations:

    paths:
      ua_detrac: D:/datasets/UA-DETRAC        # any absolute or data-dir-relative path
"""

from __future__ import annotations

import glob
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from traffic_intel.core.config import get_settings, load_yaml

Task = Literal["detection", "classification", "tracking", "multi_camera", "calibration", "anomaly",
               "incident_evaluation", "forecasting", "context", "perception", "integrated_demo",
               "reid", "traffic_light"]


class LicenseInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    access: Literal["open", "registration", "request", "private"]
    commercial_use: bool | Literal["unknown"] = "unknown"
    redistribution: bool = False
    notes: str = ""


class RequiredFiles(BaseModel):
    """At least ``min_count`` files must match ANY of the glob ``pattern`` alternatives."""

    model_config = ConfigDict(extra="forbid")
    pattern: list[str]
    min_count: int = 1
    description: str = ""

    @field_validator("pattern", mode="before")
    @classmethod
    def _as_list(cls, v: Any) -> Any:
        return [v] if isinstance(v, str) else v


class AdapterSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: str
    params: dict[str, Any] = {}


class DatasetEntry(BaseModel):
    model_config = ConfigDict(extra="allow")          # task-specific extras: classes, features, fps, ...
    dataset_id: str = ""
    name: str
    tasks: list[Task]
    source: str
    version: str
    license: LicenseInfo
    size: str = ""
    expected_format: str
    local_path: str
    required: list[RequiredFiles] = []
    download: str
    preprocessing: str
    citation: str
    splits: str = ""
    adapter: AdapterSpec
    classes: dict[str, Any] | None = None
    features: dict[str, Any] | None = None
    labels: dict[str, Any] | None = None
    params: dict[str, Any] = {}

    @field_validator("local_path")
    @classmethod
    def _relative(cls, v: str) -> str:
        if Path(v).is_absolute() or ":" in v:
            raise ValueError(f"local_path must be relative to the data directory, got {v!r} "
                             "(put machine-specific locations in configs/datasets/local.yaml)")
        return v


class ExperimentEntry(BaseModel):
    model_config = ConfigDict(extra="allow")
    purpose: str
    datasets: list[str]
    metrics: list[str] = []


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    datasets: dict[str, DatasetEntry]
    experiments: dict[str, ExperimentEntry] = Field(default_factory=dict)

    def model_post_init(self, _: Any) -> None:
        for k, e in self.datasets.items():
            e.dataset_id = k
        for name, exp in self.experiments.items():
            unknown = set(exp.datasets) - set(self.datasets)
            if unknown:
                raise ValueError(f"experiment {name} references unknown datasets {sorted(unknown)}")

    def get(self, dataset_id: str) -> DatasetEntry:
        if dataset_id not in self.datasets:
            raise KeyError(f"unknown dataset '{dataset_id}' (known: {', '.join(self.datasets)})")
        return self.datasets[dataset_id]


class DatasetNotAvailableError(FileNotFoundError):
    """Raised when a dataset's files are not where the manifest expects them."""


@dataclass
class ValidationReport:
    dataset_id: str
    root: Path
    ok: bool
    found: dict[str, int] = field(default_factory=dict)       # requirement description -> matching files
    missing: list[str] = field(default_factory=list)

    def summary(self) -> str:
        state = "OK" if self.ok else "MISSING"
        detail = ", ".join(f"{k}: {v}" for k, v in self.found.items()) or "nothing required"
        return f"{self.dataset_id:20s} {state:8s} {self.root}  [{detail}]"


def manifest_path() -> Path:
    return get_settings().config_dir / "datasets" / "manifest.yaml"


@lru_cache(maxsize=4)
def _load(path: str) -> Manifest:
    return Manifest.model_validate(load_yaml(Path(path)))


def load_manifest(path: Path | None = None) -> Manifest:
    return _load(str(path or manifest_path()))


def _local_overrides() -> dict[str, str]:
    f = get_settings().config_dir / "datasets" / "local.yaml"
    return (load_yaml(f).get("paths") or {}) if f.exists() else {}


def dataset_root(entry: DatasetEntry, data_dir: Path | None = None) -> Path:
    override = _local_overrides().get(entry.dataset_id)
    base = Path(data_dir or get_settings().data_dir)
    if override:
        p = Path(override)
        return p if p.is_absolute() else base / p
    return base / entry.local_path


def validate(entry: DatasetEntry, root: Path | None = None) -> ValidationReport:
    root = root or dataset_root(entry)
    report = ValidationReport(entry.dataset_id, root, ok=True)
    for req in entry.required:
        n = max((len(glob.glob(str(root / pat), recursive=True)) for pat in req.pattern), default=0)
        label = req.description or " | ".join(req.pattern)
        report.found[label] = n
        if n < req.min_count:
            report.ok = False
            report.missing.append(f"{label} (expected {' or '.join(req.pattern)}; found {n}, need >= {req.min_count})")
    return report


def missing_message(entry: DatasetEntry, report: ValidationReport) -> str:
    lic = entry.license
    lines = [
        f"Dataset '{entry.dataset_id}' ({entry.name}) is not available at {report.root}",
        "Missing:", *[f"  - {m}" for m in report.missing],
        f"Source: {entry.source}",
        f"Licence/access: {lic.name} [access: {lic.access}; redistribution: {'yes' if lic.redistribution else 'no'}]",
    ]
    if lic.notes:
        lines.append(f"  {lic.notes}")
    lines += ["How to get it (nothing is downloaded automatically):",
              *["  " + ln for ln in entry.download.strip().splitlines()]]
    if "status" not in entry.download:
        lines.append(f"Then check: python -m traffic_intel.datasets status {entry.dataset_id}")
    return "\n".join(lines)


def require(entry: DatasetEntry, root: Path | None = None) -> Path:
    """Return the dataset root, or raise DatasetNotAvailableError with instructions."""
    report = validate(entry, root)
    if not report.ok:
        raise DatasetNotAvailableError(missing_message(entry, report))
    return report.root
