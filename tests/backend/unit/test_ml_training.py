"""ML-training support code: discovery safety, evaluation maths, leakage-free splits, registry wiring."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# --------------------------------------------------------------------------- discovery
class _Boom:
    """A pickle payload that would run a command when unpickled."""

    def __reduce__(self) -> tuple[object, tuple[str]]:
        import os

        return (os.system, ("echo pwned",))


def test_inventory_never_executes_pickles_and_groups_by_folder(tmp_path: Path) -> None:
    from traffic_intel.datasets.inventory import discover, inspect_unit

    data = tmp_path / "data"
    (data / "raw" / "mystery").mkdir(parents=True)
    (data / "raw" / "mystery" / "evil.pkl").write_bytes(pickle.dumps(_Boom()))
    (data / "raw" / "mystery" / "graph.pkl").write_bytes(pickle.dumps([np.eye(3, dtype=np.float32)], protocol=2))
    (data / "raw" / "mystery" / "labels.csv").write_text("time,level\n2020-01-01 00:00,free\n2020-01-01 00:05,high\n",
                                                          encoding="utf-8")
    (data / "raw" / "docs_only").mkdir()
    (data / "raw" / "docs_only" / "README.md").write_text("# nothing", encoding="utf-8")
    units = discover(data)
    assert [u.key for u in units] == ["raw/mystery"]                      # README-only folders are not data
    rec = inspect_unit(units[0])
    pk = {p["file"]: p for p in rec["findings"]["pkl"]["files"]}
    assert "posix.system" in pk["evil.pkl"]["globals"] or "nt.system" in pk["evil.pkl"]["globals"]
    assert str(pk["evil.pkl"]["structure"]).startswith("not loaded")      # inspected, never executed
    assert pk["graph.pkl"]["structure"]["items"][0]["ndarray"] == [3, 3]  # NumPy-only pickle is described
    csv = rec["findings"]["csv"]["schemas"][0]
    assert csv["categorical"]["level"] == {"free": 1, "high": 1}
    assert csv["time_range"]["time"][0].startswith("2020-01-01")


# --------------------------------------------------------------------------- detection evaluation
def test_detection_matching_follows_ultralytics() -> None:
    torch = pytest.importorskip("torch")
    from ultralytics.engine.validator import BaseValidator

    from traffic_intel.ai.training.detection import IOU_THRESHOLDS, _box_iou, _match

    rng = np.random.default_rng(1)
    gt = np.c_[rng.uniform(0, 500, (12, 2)), np.zeros((12, 2))]
    gt[:, 2:] = gt[:, :2] + rng.uniform(20, 80, (12, 2))
    pred = gt[rng.permutation(12)[:9]] + rng.normal(0, 6, (9, 4))
    pred = np.vstack([pred, gt[:3] + 40.0])
    gcls, pcls = rng.integers(0, 3, 12), rng.integers(0, 3, len(pred))
    ours = _match(pcls, gcls, _box_iou(gt, pred))

    class V(BaseValidator):
        def __init__(self) -> None:
            self.iouv = torch.tensor(IOU_THRESHOLDS)

    ref = V().match_predictions(torch.tensor(pcls), torch.tensor(gcls), torch.tensor(_box_iou(gt, pred)))
    assert np.array_equal(ours, ref.numpy())


def test_validation_sequences_are_whole_and_disjoint() -> None:
    from traffic_intel.ai.training.detection import _split_sequences

    seqs = [(f"MVI_{i:05d}", ["sunny", "night", "cloudy", "rainy"][i % 4]) for i in range(60)]
    train, val = _split_sequences(seqs, 10)
    assert len(val) == 10 and not set(train) & set(val) and len(train) + len(val) == 60
    assert {w for s, w in seqs if s in val} == {"sunny", "night", "cloudy", "rainy"}   # weather-stratified
    assert _split_sequences(seqs, 10) == (train, val)                                   # deterministic


def test_repeat_factor_sampling_repeats_only_rare_class_training_images(tmp_path: Path) -> None:
    import yaml

    from traffic_intel.ai.training.detection import repeat_factor_list

    (tmp_path / "images" / "train").mkdir(parents=True)
    (tmp_path / "labels" / "train").mkdir(parents=True)
    labels = {"a": "0 .5 .5 .1 .1\n0 .2 .2 .1 .1", "b": "0 .5 .5 .1 .1\n2 .3 .3 .2 .2", "c": "0 .5 .5 .1 .1",
              "d": "0 .5 .5 .1 .1"}
    for stem, text in labels.items():
        (tmp_path / "images" / "train" / f"{stem}.jpg").write_bytes(b"")
        (tmp_path / "labels" / "train" / f"{stem}.txt").write_text(text, encoding="utf-8")
    (tmp_path / "data.yaml").write_text(yaml.safe_dump({"train": "images/train", "val": "images/val"}), "utf-8")
    data_yaml, st = repeat_factor_list(tmp_path, t=1.0, seed=0, n_classes=3)
    # truck (class 2) is in 1 of 4 images -> r = sqrt(1 / 0.25) = 2; car images keep r = 1
    listed = (tmp_path / yaml.safe_load(data_yaml.read_text(encoding="utf-8"))["train"]).read_text().splitlines()
    assert [Path(p).stem for p in listed] == ["a", "b", "b", "c", "d"]
    assert st["boxes_per_class_before"] == [5, 0, 1] and st["boxes_per_class_after"] == [6, 0, 2]
    assert yaml.safe_load(data_yaml.read_text(encoding="utf-8"))["val"] == "images/val"    # val untouched


def test_registry_weights_reference(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from traffic_intel.computer_vision.detectors.yolo import resolve_weights
    from traffic_intel.core.config import get_settings

    d = tmp_path / "detection" / "yolo_x" / "v1"
    d.mkdir(parents=True)
    (d / "model.pt").write_bytes(b"weights")
    (d / "metadata.json").write_text("{}", encoding="utf-8")
    (tmp_path / "detection" / "yolo_x" / "PRODUCTION").write_text("v1", encoding="utf-8")
    monkeypatch.setenv("TI_MODELS_DIR", str(tmp_path))
    get_settings.cache_clear()
    try:
        assert resolve_weights("registry:detection/yolo_x", auto_download=False) == str(d / "model.pt")
        assert resolve_weights("registry:detection/yolo_x@v1", auto_download=False) == str(d / "model.pt")
        with pytest.raises(FileNotFoundError):
            resolve_weights("registry:detection/missing", auto_download=False)
    finally:
        monkeypatch.delenv("TI_MODELS_DIR")
        get_settings.cache_clear()


# --------------------------------------------------------------------------- incident video evaluation
def test_video_split_is_by_video_and_stratified() -> None:
    from traffic_intel.ai.training.incident_video import SplitCfg, split_videos

    vids = [str(i) for i in range(1, 101)]
    labelled = {str(i) for i in (2, 9, 11, 14, 33, 35, 49, 51, 63, 72, 73, 74, 83, 91, 93, 95, 97)}
    s = split_videos(vids, labelled, SplitCfg())
    assert sorted(s["train"] + s["val"] + s["test"], key=int) == vids
    assert not (set(s["train"]) & set(s["test"])) and not (set(s["val"]) & set(s["test"]))
    for part in ("train", "val", "test"):
        assert set(s[part]) & labelled                                      # every split has labelled videos


def test_event_metrics_first_alarm_per_video() -> None:
    from traffic_intel.ai.training.incident_video import event_metrics

    df = pd.DataFrame({"video": ["1"] * 4 + ["2"] * 4 + ["3"] * 4,
                       "t_end": [10, 20, 30, 40] * 3})
    s = np.array([0, 0, 1, 1,   1, 0, 0, 0,   0, 0, 0, 0], float)
    m = event_metrics(df, s, 0.5, {"1": [(25.0, 40.0)], "3": [(5.0, 40.0)]}, tol=10.0)
    # video 1: alarm at 30 inside [25, 40] -> TP (delay 5); video 2: normal with an alarm -> FP; video 3: missed -> FN
    assert (m["tp"], m["fp"], m["fn"]) == (1, 1, 1) and m["median_delay_s"] == 5.0


def test_persistence_survives_dropouts_and_ignores_movers() -> None:
    from traffic_intel.ai.training.incident_video import WindowCfg, persistence_series

    rows = []
    for t in np.arange(0, 60, 0.2):
        if not 20 <= t < 22:
            rows.append((t, 2, 0.9, 100, 100, 140, 130))                    # parked, 2 s detector dropout
        x = 10 + 30 * t
        rows.append((t, 2, 0.9, x, 200, x + 40, 230))                       # moving 30 px/s
    s = persistence_series(pd.DataFrame(rows, columns=["t_s", "class_id", "conf", "x1", "y1", "x2", "y2"]),
                           800 * 410, WindowCfg())
    assert s["max_persist_s"].iloc[-1] >= 55                                # kept growing through the dropout
    assert (s.set_index("t").loc[40.0, "n_persist"]) == 1                   # only the parked box persists


# --------------------------------------------------------------------------- congestion
def test_classification_metrics_and_level_classifier_roundtrip(tmp_path: Path) -> None:
    from sklearn.dummy import DummyClassifier

    from traffic_intel.ai.training.congestion import LevelClassifier, classification_metrics

    y = np.array([0, 0, 0, 1, 2, 3])
    p = np.array([0, 0, 1, 1, 2, 2])
    m = classification_metrics(y, p)
    assert m["accuracy"] == pytest.approx(4 / 6)
    assert m["per_class"]["severe"]["recall"] == 0.0 and m["confusion"][3][2] == 1
    est = DummyClassifier(strategy="constant", constant=1).fit(np.zeros((3, 2)), [0, 1, 1])
    lc = LevelClassifier(est, classes=[0, 2], horizon_min=15, feature_names=["a", "b"])
    lc.save(tmp_path)
    back = LevelClassifier.load(tmp_path)
    assert back.predict_levels(np.zeros((2, 2))).tolist() == [2, 2]         # encoded 1 -> level index 2


def test_retire_takes_a_model_out_of_production(tmp_path: Path) -> None:
    from datetime import UTC, datetime

    from traffic_intel.ai.model_store import FileModelStore, ModelMetadata
    from traffic_intel.core.enums import ModelStatus, ModelTask

    store = FileModelStore(tmp_path)
    d = tmp_path / "forecasting" / "old" / "v1"
    d.mkdir(parents=True)
    meta = ModelMetadata(task=ModelTask.FORECASTING, name="old", version="v1", framework="sklearn",
                         feature_schema_hash="x", trained_at=datetime.now(UTC))
    (d / "metadata.json").write_text(meta.model_dump_json(), encoding="utf-8")
    store.promote(ModelTask.FORECASTING, "old", "v1")
    assert store.metadata(ModelTask.FORECASTING, "old", "v1").status is ModelStatus.PRODUCTION
    assert store.retire(ModelTask.FORECASTING, "old") == "v1"
    assert store.metadata(ModelTask.FORECASTING, "old", "v1").status is ModelStatus.ARCHIVED
    assert not (tmp_path / "forecasting" / "old" / "PRODUCTION").exists()
    assert store.retire(ModelTask.FORECASTING, "old") is None                # idempotent
