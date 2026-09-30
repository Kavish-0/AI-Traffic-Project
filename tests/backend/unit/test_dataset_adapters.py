"""Dataset adapters on tiny fixtures written in each dataset's documented raw format, plus the
evaluation code that consumes their canonical output."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from traffic_intel.ai.evaluation.detection import DetectionEvaluator, MOTEvaluator
from traffic_intel.ai.evaluation.incidents import TimedPrediction, anomaly_timing_score, predictions_from_incidents
from traffic_intel.computer_vision.detectors.precomputed import load_detection_table
from traffic_intel.computer_vision.homography import HomographyProjector
from traffic_intel.core.enums import ObjectClass, class_id_of
from traffic_intel.core.feature_schema import validate_frame
from traffic_intel.core.types import Detections, Tracks
from traffic_intel.datasets.adapters.aicity import apply_h
from traffic_intel.datasets.adapters.base import adapter_for
from traffic_intel.datasets.manifest import load_manifest
from traffic_intel.datasets.prepare import (
    ground_truth_tracked_frames,
    prepare_detection,
    prepare_incidents,
    prepare_timeseries,
)

CAR, BUS, TRUCK = (class_id_of(c) for c in (ObjectClass.CAR, ObjectClass.BUS, ObjectClass.TRUCK))

DETRAC_XML = """<?xml version="1.0" encoding="utf-8"?>
<sequence name="MVI_00001">
  <sequence_attribute camera_state="unstable" sence_weather="sunny"/>
  <ignored_region><box left="0" top="0" width="100" height="50"/></ignored_region>
  <frame density="3" num="1"><target_list>
    <target id="1"><box left="100" top="200" width="50" height="40"/>
      <attribute orientation="0" speed="5" trajectory_length="1" truncation_ratio="0" vehicle_type="car"/></target>
    <target id="2"><box left="300" top="200" width="60" height="40"/>
      <attribute orientation="0" speed="5" trajectory_length="1" truncation_ratio="0" vehicle_type="van"/></target>
    <target id="3"><box left="500" top="150" width="120" height="90"/>
      <attribute orientation="0" speed="5" trajectory_length="1" truncation_ratio="0" vehicle_type="others"/></target>
  </target_list></frame>
  <frame density="2" num="2"><target_list>
    <target id="1"><box left="104" top="204" width="50" height="40"/>
      <attribute orientation="0" speed="5" trajectory_length="2" truncation_ratio="0" vehicle_type="car"/></target>
    <target id="4"><box left="700" top="100" width="150" height="100"/>
      <attribute orientation="0" speed="5" trajectory_length="1" truncation_ratio="0" vehicle_type="bus"/></target>
    <target id="5"><box left="10" top="10" width="10" height="10"/>
      <attribute orientation="0" speed="5" trajectory_length="1" truncation_ratio="0" vehicle_type="tram"/></target>
  </target_list></frame>
</sequence>
"""


@pytest.fixture
def detrac_root(tmp_path: Path) -> Path:
    root = tmp_path / "uadetrac"
    (root / "DETRAC-Train-Annotations-XML").mkdir(parents=True)
    (root / "DETRAC-Train-Annotations-XML" / "MVI_00001.xml").write_text(DETRAC_XML, encoding="utf-8")
    (root / "Insight-MVT_Annotation_Train" / "MVI_00001").mkdir(parents=True)
    (root / "Insight-MVT_Annotation_Train" / "MVI_00001" / "img00001.jpg").write_bytes(b"")
    return root


def test_uadetrac_maps_classes_through_the_manifest(detrac_root: Path, tmp_path: Path) -> None:
    ad = adapter_for("ua_detrac", detrac_root)
    ad.check()
    (seq,) = ad.sequences()
    assert seq.sequence_id == "MVI_00001" and seq.fps == 25 and seq.n_frames == 2
    assert seq.metadata["weather"] == "sunny" and seq.media.name == "MVI_00001"
    f1, f2 = list(ad.frames("MVI_00001"))
    assert f1.frame_index == 0 and f2.time_s == pytest.approx(1 / 25)
    assert f1.tracks.class_id.tolist() == [CAR, CAR, TRUCK]            # car, van -> car, others -> truck
    assert f1.tracks.xyxy[0].tolist() == [100, 200, 150, 240]          # left/top/width/height -> xyxy
    assert f2.tracks.track_id.tolist() == [1, 4] and f2.unmapped == 1  # 'tram' is not guessed
    assert f1.ignore_regions.tolist() == [[0, 0, 100, 50]]

    out = prepare_detection(ad, tmp_path / "data")
    assert out == {"output": str(tmp_path / "data" / "annotations" / "ua_detrac"), "sequences": 1, "boxes": 5,
                   "unmapped_boxes": 1}
    gt = pd.read_csv(tmp_path / "data" / "annotations" / "ua_detrac" / "MVI_00001.csv")
    assert set(gt["class"]) == {"car", "truck", "bus"}
    table = load_detection_table(tmp_path / "data" / "detections" / "ua_detrac" / "MVI_00001_gt.csv")
    assert sorted(table) == [0, 1] and len(table[0]) == 3               # replayable by the precomputed detector
    tfs = list(ground_truth_tracked_frames(ad, "MVI_00001", "detrac_cam", (540, 960)))
    assert tfs[1].tracks.track_id.tolist() == [1, 4] and tfs[0].image_shape == (540, 960)


# --------------------------------------------------------------------------- AI City MTMC
LAT0, LON0 = 42.49, -90.72                     # somewhere in Iowa
M_PER_DEG_LAT = 111_194.9


def _gps_homography() -> np.ndarray:
    """A GPS->image mapping built from an image->metres affine (0.05 m/px, v grows southwards)."""
    m_per_deg_lon = M_PER_DEG_LAT * np.cos(np.radians(LAT0))
    img_to_gps = np.array([[0.0, -0.05 / M_PER_DEG_LAT, LAT0 + 20 / M_PER_DEG_LAT],
                           [0.05 / m_per_deg_lon, 0.0, LON0 - 30 / m_per_deg_lon],
                           [0.0, 0.0, 1.0]])
    return np.linalg.inv(img_to_gps)


@pytest.fixture
def aicity_root(tmp_path: Path) -> Path:
    cam = tmp_path / "aicity" / "train" / "S01" / "c001"
    (cam / "gt").mkdir(parents=True)
    (cam / "gt" / "gt.txt").write_text("1,7,100,200,50,40,1,-1,-1,-1\n1,8,300,210,60,45,1,-1,-1,-1\n"
                                       "2,7,110,205,50,40,1,-1,-1,-1\n", encoding="utf-8")
    H = _gps_homography()
    rows = ";".join(" ".join(f"{v:.12e}" for v in r) for r in H)
    (cam / "calibration.txt").write_text(f"Homography matrix: {rows}\nReprojection error: 1.0\n", encoding="utf-8")
    (cam / "vdo.avi").write_bytes(b"")
    (tmp_path / "aicity" / "cam_timestamp").mkdir()
    (tmp_path / "aicity" / "cam_timestamp" / "S01.txt").write_text("c001 0\nc002 1.64\n", encoding="utf-8")
    return tmp_path / "aicity"


def test_aicity_tracks_and_calibration_to_project_config(aicity_root: Path) -> None:
    ad = adapter_for("aicity_mtmc", aicity_root)
    ad.check()
    (seq,) = ad.sequences()
    assert seq.sequence_id == "S01_c001" and seq.metadata["scenario"] == "S01" and seq.metadata["has_calibration"]
    frames = list(ad.frames("S01_c001"))
    assert [f.frame_index for f in frames] == [0, 1] and frames[0].tracks.track_id.tolist() == [7, 8]
    assert set(frames[0].tracks.class_id.tolist()) == {CAR}             # 'vehicle' -> configurable canonical class

    cal = ad.calibration_config("S01_c001", (1280, 720))
    proj = HomographyProjector(cal, (1280, 720))
    px = np.array([[640.0, 600.0], [200.0, 400.0]])
    lonlat = proj.ground_to_lonlat(proj.image_to_ground(px))
    gps = apply_h(np.linalg.inv(ad.homography("S01_c001")), px)          # (lat, lon)
    np.testing.assert_allclose(lonlat[:, 1], gps[:, 0], atol=2e-6)       # < 0.25 m
    np.testing.assert_allclose(lonlat[:, 0], gps[:, 1], atol=3e-6)
    g = proj.image_to_ground(np.array([[100.0, 500.0], [300.0, 500.0]]))
    assert np.linalg.norm(g[1] - g[0]) == pytest.approx(200 * 0.05, rel=0.01)   # metric scale preserved


# --------------------------------------------------------------------------- AI City anomaly
def test_anomaly_labels_stay_intervals_with_their_own_label(tmp_path: Path) -> None:
    root = tmp_path / "aic21"
    (root / "aic21-track4-train-data").mkdir(parents=True)
    for v in ("1", "2", "3"):
        (root / "aic21-track4-train-data" / f"{v}.mp4").write_bytes(b"")
    (root / "train-anomaly-results.txt").write_text("video_id start end\n2 587 894\n3 180.5 450\n", encoding="utf-8")
    ad = adapter_for("aicity21_anomaly", root)
    ad.check()
    ann = ad.annotations()
    assert [(a.video_id, a.start_s, a.end_s) for a in ann] == [("2", 587.0, 894.0), ("3", 180.5, 450.0)]
    assert {a.source_label for a in ann} == {"anomaly"}                   # never renamed "accident"
    assert ann[0].frame_interval(30) == (17610, 26820)
    out = prepare_incidents(ad, tmp_path / "data")
    assert out["intervals"] == 2 and out["videos_found"] == 3 and out["labelled_videos_without_file"] == 0

    positive = load_manifest().get("aicity21_anomaly").labels["positive_system_types"]
    preds = predictions_from_incidents([
        {"video_id": "2", "time_s": 592, "incident_type": "stopped_vehicle", "probability": 0.4},
        {"video_id": "3", "time_s": 400, "incident_type": "possible_incident", "probability": 0.9},   # too late
        {"video_id": "1", "time_s": 30, "incident_type": "normal", "probability": 0.1},              # not positive
    ], positive)
    assert len(preds) == 2
    s = anomaly_timing_score(ann, preds)
    assert (s["tp"], s["fp"], s["fn"]) == (1, 1, 1) and s["f1"] == 0.5
    assert s["rmse_s"] == 5.0 and s["s4"] == pytest.approx(0.5 * (1 - 5 / 300), abs=1e-4)
    assert anomaly_timing_score(ann, [TimedPrediction("2", 587.0)])["nrmse"] == 0.0


def test_anomaly_videos_exclude_unlabelled_test_split(tmp_path: Path) -> None:
    """The real release ships train (1..100, labelled) and test (1..150, unlabelled) folders with the SAME
    file names; only training videos may be paired with the training labels."""
    root = tmp_path / "aic21"
    for split, n in (("aic21-track4-train-data", 3), ("aic21-track4-test-data", 5)):
        (root / split).mkdir(parents=True)
        for v in range(1, n + 1):
            (root / split / f"{v}.mp4").write_bytes(b"")
    (root / "train-anomaly-results.txt").write_text("2 587 894\n", encoding="utf-8")
    vids = adapter_for("aicity21_anomaly", root).videos()
    assert sorted(vids, key=int) == ["1", "2", "3"]
    assert all(p.parent.name == "aic21-track4-train-data" for p in vids.values())


# --------------------------------------------------------------------------- METR-LA / PEMS-BAY
def _speed_h5(path: Path, key: str, start: str = "2012-03-10 00:00", periods: int = 2 * 288) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    idx = pd.date_range(start, periods=periods, freq="5min")                   # naive local time, like DCRNN
    df = pd.DataFrame(rng.uniform(30, 70, (periods, 3)), index=idx, columns=["773869", "767541", "767542"])
    df.iloc[10:13, 1] = 0.0                                                    # zeros = missing
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_hdf(path, key=key)
    return df


def test_metr_la_to_canonical_series(tmp_path: Path) -> None:
    pytest.importorskip("tables")
    raw = _speed_h5(tmp_path / "metr_la" / "metr-la.h5", "df")               # spans the 2012-03-11 DST jump
    ds = adapter_for("metr_la", tmp_path / "metr_la").load(every_k=1)
    validate_frame(ds.frame.columns)
    assert list(ds.frame.columns) == ["series_id", "timestamp", "avg_speed_kmh"]
    assert ds.series_ids == ["metr_la_767541", "metr_la_767542", "metr_la_773869"]
    assert ds.resolution_s == 300 and ds.tz == "America/Los_Angeles" and ds.holiday_country == "US"
    assert ds.info["dropped_dst_timestamps"] == 12                            # 02:00-02:55 does not exist locally
    s = ds.frame[ds.frame["series_id"] == "metr_la_773869"].set_index("timestamp")["avg_speed_kmh"]
    first = pd.Timestamp("2012-03-10 00:00", tz="America/Los_Angeles").tz_convert("UTC")
    assert s.loc[first] == pytest.approx(raw.iloc[0]["773869"] * 1.609344)
    assert ds.frame["avg_speed_kmh"].isna().sum() == 3 and ds.info["zeros_treated_as_missing"] == 3
    assert str(ds.frame["timestamp"].dt.tz) == "UTC"
    sub = adapter_for("metr_la", tmp_path / "metr_la").load(sensors=["767542"])
    assert sub.series_ids == ["metr_la_767542"]
    with pytest.raises(ValueError, match="unknown sensor"):
        adapter_for("metr_la", tmp_path / "metr_la").load(sensors=["nope"])

    out = prepare_timeseries(adapter_for("metr_la", tmp_path / "metr_la"), tmp_path / "data", max_sensors=2)
    assert out["series"] == 2 and (tmp_path / "data" / "processed" / "metr_la" / "meta.json").exists()


def test_legacy_dcrnn_hdf5_layout_is_read(tmp_path: Path) -> None:
    """The DCRNN files (pandas 0.15 / Python 2: bytes attributes, pickled index freq) that pandas 3
    cannot parse are read from their arrays; the pickled attribute is never loaded."""
    tables = pytest.importorskip("tables")
    path = tmp_path / "metr_la" / "METR-LA.h5"
    path.parent.mkdir(parents=True)
    idx = pd.date_range("2012-03-01", periods=4, freq="5min")
    cols = np.array([b"773869", b"767541"])
    vals = np.array([[60.0, 0.0], [61.0, 55.0], [62.0, 56.0], [63.0, 57.0]])
    with tables.open_file(path, mode="w") as h:
        g = h.create_group("/", "df")
        for k, v in {"pandas_type": b"frame", "pandas_version": b"0.15.2", "nblocks": np.int64(1),
                     "ndim": np.int64(2)}.items():
            g._v_attrs[k] = v
        h.create_array(g, "axis0", cols)
        ax1 = h.create_array(g, "axis1", idx.as_unit("ns").asi8)      # legacy files store ns
        ax1._v_attrs["kind"] = b"datetime64"
        ax1._v_attrs["freq"] = b"cnot.a.module\nnever_unpickled\n."    # would fail if unpickled
        h.create_array(g, "block0_items", cols)
        h.create_array(g, "block0_values", vals)
    raw = adapter_for("metr_la", path.parent).raw_matrix()
    assert list(raw.columns) == ["773869", "767541"] and raw.shape == (4, 2)
    assert raw.index[0] == pd.Timestamp("2012-03-01 00:00") and raw.iloc[1, 1] == 55.0
    ds = adapter_for("metr_la", path.parent).load()
    assert ds.frame["avg_speed_kmh"].isna().sum() == 1                          # the 0 mph reading


def test_pems_bay_uses_its_own_key(tmp_path: Path) -> None:
    pytest.importorskip("tables")
    _speed_h5(tmp_path / "pems_bay" / "pems-bay.h5", "speed", start="2017-01-01 00:00", periods=288)
    ds = adapter_for("pems_bay", tmp_path / "pems_bay").load(max_sensors=2)
    assert ds.series_ids == ["pems_bay_767541", "pems_bay_773869"] and ds.info["dropped_dst_timestamps"] == 0


# --------------------------------------------------------------------------- evaluation
def _det(boxes: list[list[float]], cls: list[int], conf: list[float] | None = None) -> Detections:
    n = len(boxes)
    return Detections(np.asarray(boxes, np.float32).reshape(n, 4), np.asarray(conf or [0.9] * n, np.float32),
                      np.asarray(cls, np.int64))


def test_detection_ap_and_ignored_regions() -> None:
    gt = _det([[0, 0, 10, 10], [20, 20, 30, 30]], [CAR, BUS])
    ev = DetectionEvaluator()
    ev.add(_det([[0, 0, 10, 10], [20, 20, 30, 30]], [CAR, BUS]), gt)
    r = ev.results()
    assert r["car"]["ap50"] == 1.0 and r["bus"]["ap50"] == 1.0 and r["mean"]["ap50"] == 1.0
    ev = DetectionEvaluator()
    ev.add(_det([[0, 0, 10, 10], [100, 100, 110, 110], [200, 0, 210, 5]], [CAR, CAR, CAR], [0.9, 0.8, 0.95]),
           gt, ignore_regions=np.array([[190, 0, 220, 10]], np.float32))
    r = ev.results()["car"]
    assert r["n_pred"] == 2 and r["precision"] == 0.5 and r["recall"] == 1.0   # the ignored box is not an FP


def test_mot_counts_identity_switches() -> None:
    def tr(ids: list[int], boxes: list[list[float]]) -> Tracks:
        return Tracks(track_id=np.array(ids), xyxy=np.array(boxes, np.float32),
                      confidence=np.ones(len(ids), np.float32), class_id=np.zeros(len(ids), np.int64))

    ev = MOTEvaluator()
    gt_box = [[0, 0, 10, 10]]
    ev.add(tr([100], gt_box), tr([1], gt_box))
    ev.add(tr([100], gt_box), tr([1], gt_box))
    ev.add(tr([200], gt_box), tr([1], gt_box))                  # tracker changed the ID
    ev.add(tr([200, 300], [[0, 0, 10, 10], [50, 50, 60, 60]]), tr([1], gt_box))
    r = ev.results()
    assert (r["id_switches"], r["fp"], r["fn"], r["gt"]) == (1, 1, 0, 4)
    assert r["MOTA"] == pytest.approx(1 - 2 / 4) and r["MOTP_iou"] == 1.0
