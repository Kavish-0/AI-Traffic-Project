"""Dataset manifest, presence validation and the single canonical feature schema."""

from __future__ import annotations

from pathlib import Path

import pytest

from traffic_intel.core.feature_schema import (
    CALENDAR_COLUMNS,
    TRAFFIC_COLUMNS,
    WEATHER_COLUMNS,
    SchemaError,
    validate_frame,
)
from traffic_intel.datasets.adapters.base import DATASET_ADAPTERS, adapter_for
from traffic_intel.datasets.manifest import (
    DatasetEntry,
    DatasetNotAvailableError,
    load_manifest,
    require,
    validate,
)
from traffic_intel.forecasting.data import context
from traffic_intel.forecasting.data.preprocess import FeatureSpec

REQUIRED_DATASETS = {"ua_detrac", "aicity_mtmc", "aicity21_anomaly", "metr_la", "pems_bay"}


def test_manifest_covers_the_dataset_strategy() -> None:
    m = load_manifest()
    assert set(m.datasets) >= REQUIRED_DATASETS
    for dsid, e in m.datasets.items():
        assert e.dataset_id == dsid
        for field in ("name", "source", "version", "expected_format", "download", "preprocessing", "citation"):
            assert getattr(e, field).strip(), f"{dsid}: empty {field}"
        assert e.license.name.strip() and e.license.access in {"open", "registration", "request", "private"}
        assert not Path(e.local_path).is_absolute() and ":" not in e.local_path
        if e.adapter.type != "none":
            assert e.adapter.type in DATASET_ADAPTERS.names()
    # licence-restricted datasets are never marked redistributable
    for dsid in REQUIRED_DATASETS:
        assert m.get(dsid).license.redistribution is False


def test_experiments_keep_tasks_apart() -> None:
    m = load_manifest()
    task_of = {k: set(e.tasks) for k, e in m.datasets.items()}
    expect = {"E1_perception_detection": {"detection"}, "E2_tracking": {"tracking"},
              "E4_incident_anomaly": {"anomaly"}, "E5_forecasting": {"forecasting"}}
    for exp, needed in expect.items():
        for ds in m.experiments[exp].datasets:
            assert task_of[ds] & needed, f"{exp} uses {ds}, which is not a {needed} dataset"
    assert m.experiments["E5_forecasting"].model_extra["variants"] == ["traffic_only", "traffic_plus_context"]
    assert "own_recordings" not in {d for x in list(m.experiments.values())[:-1] for d in x.datasets}


def test_absolute_local_path_is_rejected() -> None:
    base = load_manifest().get("metr_la").model_dump()
    base["local_path"] = "C:/data/metr_la"
    with pytest.raises(ValueError, match="relative"):
        DatasetEntry.model_validate(base)


def test_missing_dataset_fails_with_instructions(tmp_path: Path) -> None:
    e = load_manifest().get("metr_la")
    report = validate(e, tmp_path)
    assert not report.ok and report.found == {"speed matrix (HDF5)": 0}
    with pytest.raises(DatasetNotAvailableError) as exc:
        require(e, tmp_path)
    msg = str(exc.value)
    for part in (str(tmp_path), "metr-la.h5", "DCRNN", "nothing is downloaded automatically", "redistribution: no",
                 "status metr_la"):
        assert part in msg, part
    with pytest.raises(DatasetNotAvailableError):
        adapter_for("metr_la", tmp_path).load()                   # the adapter fails the same way


def test_present_dataset_validates(tmp_path: Path) -> None:
    (tmp_path / "metr-la.h5").write_bytes(b"")
    assert validate(load_manifest().get("metr_la"), tmp_path).ok


def test_one_feature_schema() -> None:
    """Forecasting context names ARE the core schema's (no second schema)."""
    assert context.WEATHER_COLUMNS is WEATHER_COLUMNS
    assert {c for cols in context.CALENDAR_GROUPS.values() for c in cols} == set(CALENDAR_COLUMNS)
    validate_frame(["series_id", "timestamp", "avg_speed_kmh", "hour_sin", "rain_mm", "gt_incident"])
    with pytest.raises(SchemaError, match="speed_mph"):
        validate_frame(["series_id", "timestamp", "speed_mph"])
    with pytest.raises(SchemaError, match="key"):
        validate_frame(["avg_speed_kmh"])
    with pytest.raises(ValueError, match="TRAFFIC_COLUMNS"):
        FeatureSpec(dynamic=["speed_mph"], targets=["speed_mph"]).check()
    assert "avg_speed_kmh" in TRAFFIC_COLUMNS


def test_traffic_only_feature_spec_is_valid() -> None:
    FeatureSpec(dynamic=["avg_speed_kmh"], targets=["avg_speed_kmh"], calendar=[], weather=[]).check()
