"""Data-quality checks on every dataset found on disk, run through the adapters (canonical form).

    python -m traffic_intel.datasets quality     -> reports/data_quality_report.{json,md}

Checks are chosen by the KIND of data (adapter family or manifest task), never by dataset name:
    time series        missing values, gaps, duplicates, irregular sampling, impossible values,
                       stuck sensors, DST handling, and the actual chronological split of every
                       training config that uses the dataset (ordered, disjoint periods)
    object annotations per official split: annotated vs image frames, invalid / out-of-image / tiny
                       boxes, duplicate ids per frame, unmapped labels, boxes in ignored regions,
                       adjacent-frame redundancy (why frames are subsampled and splits are by sequence)
    incident intervals label validity against video durations, overlaps, labelled vs normal
                       videos, frame-rate differences, unlabelled test videos
    identity labels    images per identity / camera, file references
    box annotations    invalid boxes, tag balance, referenced frames that exist
    context tables     duplicates, impossible values, constant columns, time coverage
Each check reports ok / warn / fail / info with the measured numbers.
"""

from __future__ import annotations

import glob
import json
import logging
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class Check:
    name: str
    status: str                      # ok | warn | fail | info
    detail: str
    value: Any = None


@dataclass
class DatasetQuality:
    dataset: str
    kind: str
    path: str
    checks: list[Check] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def add(self, name: str, status: str, detail: str, value: Any = None) -> None:  # noqa: ANN401
        self.checks.append(Check(name, status, detail, value))


# --------------------------------------------------------------------------- time series
def check_timeseries(dsid: str, root: Path) -> DatasetQuality:
    from traffic_intel.datasets.adapters.base import TimeSeriesAdapter, adapter_for

    ds = cast(TimeSeriesAdapter, adapter_for(dsid, root)).load()
    df = ds.frame
    q = DatasetQuality(dsid, "time series", str(root))
    values = [c for c in df.columns if c not in ("series_id", "timestamp")]
    q.stats.update(rows=len(df), series=int(df["series_id"].nunique()), columns=values,
                   start=str(df["timestamp"].min()), end=str(df["timestamp"].max()), resolution_s=ds.resolution_s,
                   adapter_info={k: v for k, v in ds.info.items() if k not in ("sensors",)})
    dup = int(df.duplicated(["series_id", "timestamp"]).sum())
    q.add("duplicate (series, timestamp)", "ok" if dup == 0 else "fail", f"{dup} duplicates", dup)

    steps = df.sort_values(["series_id", "timestamp"]).groupby("series_id")["timestamp"].diff().dropna()
    irregular = int((steps != pd.Timedelta(seconds=ds.resolution_s)).sum())
    q.add("regular sampling", "ok" if irregular == 0 else "warn",
          f"{irregular} steps differ from {ds.resolution_s} s "
          f"(largest {steps.max()}); DST: {ds.info.get('dropped_dst_timestamps', 0)} non-existent local "
          f"timestamps dropped", irregular)

    for c in values:
        miss = float(df[c].isna().mean())
        per_series = df.groupby("series_id")[c].apply(lambda s: s.isna().mean())
        q.add(f"missing {c}", "ok" if miss < 0.02 else "warn" if miss < 0.2 else "fail",
              f"{miss:.2%} missing overall; worst series {per_series.max():.1%}, "
              f"{int((per_series > 0.2).sum())} series > 20% missing", round(miss, 5))
    if "zeros_treated_as_missing" in ds.info:
        z = ds.info["zeros_treated_as_missing"]
        q.add("zeros = sensor outage", "info",
              f"{z} zero readings converted to missing (documented source convention)", z)

    target = "avg_speed_kmh" if "avg_speed_kmh" in values else values[0]
    gaps = _longest_nan_runs(df, target)
    q.stats["longest_gap_steps"] = {"max": int(gaps.max()), "median": float(gaps.median())}
    q.add(f"longest gap ({target})", "info", f"longest run of missing steps per series: max {int(gaps.max())}, "
          f"median {gaps.median():.0f} (gaps > max_gap_fill_steps are never interpolated; windows touching them "
          f"are excluded)")

    bounds = {"avg_speed_kmh": (0, 200), "occupancy_pct": (0, 100), "flow_veh_h": (0, 20000),
              "vehicle_count": (0, 10000), "density_ratio": (0, 1.5)}
    for c, (lo, hi) in bounds.items():
        if c in df:
            bad = int(((df[c] < lo) | (df[c] > hi)).sum())
            q.add(f"physical range {c}", "ok" if bad == 0 else "fail", f"{bad} values outside [{lo}, {hi}]", bad)

    stuck = _stuck_fraction(df, target, min_run=max(2, int(3600 / ds.resolution_s)))
    q.add(f"stuck sensor ({target})", "ok" if stuck < 0.01 else "warn",
          f"{stuck:.2%} of readings sit in runs of identical non-missing values lasting >= 1 h", round(stuck, 5))

    q.stats["splits"] = _config_splits(dsid, df, ds.resolution_s)
    for name, sp in q.stats["splits"].items():
        ordered = sp.get("ordered_disjoint")
        q.add(f"chronological split ({name})", "ok" if ordered else "fail",
              "train < val < test, disjoint periods; targets never cross a boundary (windows.py)"
              if ordered else sp.get("error", "invalid split"), sp)
    return q


def _longest_nan_runs(df: pd.DataFrame, col: str) -> pd.Series:
    def longest(s: pd.Series) -> int:
        isna = s.isna().to_numpy()
        if not isna.any():
            return 0
        runs = np.diff(np.flatnonzero(np.diff(np.r_[0, isna.astype(int), 0])))[::2]
        return int(runs.max()) if len(runs) else 0

    return df.sort_values(["series_id", "timestamp"]).groupby("series_id")[col].apply(longest)


def _stuck_fraction(df: pd.DataFrame, col: str, min_run: int) -> float:
    s = df.sort_values(["series_id", "timestamp"])
    v = s[col].to_numpy()
    sid = s["series_id"].to_numpy()
    new = np.r_[True, (v[1:] != v[:-1]) | (sid[1:] != sid[:-1])]
    run_id = np.cumsum(new)
    counts = np.bincount(run_id)
    run_len = counts[run_id]
    stuck = (run_len >= min_run) & ~np.isnan(v)
    return float(stuck.mean()) if len(v) else 0.0


def _config_splits(dsid: str, df: pd.DataFrame, resolution_s: int) -> dict[str, Any]:
    from traffic_intel.core.config import get_settings, load_yaml
    from traffic_intel.forecasting.data.windows import SplitConfig, chronological_split

    out: dict[str, Any] = {}
    for f in sorted((get_settings().config_dir / "training").glob("*.yaml")):
        cfg = load_yaml(f)
        if (cfg.get("dataset") or {}).get("source") != dsid or "split" not in cfg:
            continue
        try:
            p = chronological_split(df["timestamp"], SplitConfig(**cfg["split"]), resolution_s)
            ok = p["train"].end <= p["val"].start and p["val"].end <= p["test"].start
            out[f.name] = {k: v.as_dict() for k, v in p.items()} | {"ordered_disjoint": ok}
        except ValueError as exc:
            out[f.name] = {"ordered_disjoint": False, "error": str(exc)}
    return out


# --------------------------------------------------------------------------- object annotations
def check_object_annotations(dsid: str, root: Path) -> DatasetQuality:
    from traffic_intel.core.enums import class_from_id
    from traffic_intel.datasets.adapters.base import DetectionTrackingAdapter, adapter_for

    q = DatasetQuality(dsid, "object annotations (detection / tracking)", str(root))
    names: dict[str, set[str]] = {}
    splits: tuple[str, ...] = getattr(type(adapter_for(dsid, root)), "splits", ("train",))
    for split in splits:
        params = {"split": split} if len(splits) > 1 else {}
        a = cast(DetectionTrackingAdapter, adapter_for(dsid, root, **params))
        seqs = a.sequences()
        names[split] = {s.sequence_id for s in seqs}
        st: Counter[str] = Counter()
        classes: Counter[str] = Counter()
        redundancy: list[float] = []
        for s in seqs:
            imgs = {int(Path(p).stem.replace("img", "")) - 1 for p in glob.glob(str(s.media / "img*.jpg"))} \
                if s.media else set()
            frames = list(a.frames(s.sequence_id))
            ann = {f.frame_index for f in frames}
            st["sequences"] += 1
            st["annotated_frames"] += len(ann)
            st["image_frames"] += len(imgs)
            st["images_without_annotation"] += len(imgs - ann)
            st["annotations_without_image"] += len(ann - imgs)
            w, h = s.frame_size or (10**6, 10**6)
            prev: dict[int, np.ndarray] = {}
            for fr in frames:
                t = fr.tracks
                b = t.xyxy
                st["boxes"] += len(b)
                st["unmapped_labels"] += fr.unmapped
                if len(b):
                    bw, bh = b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]
                    st["invalid_boxes"] += int(((bw <= 0) | (bh <= 0)).sum())
                    st["out_of_image_boxes"] += int(((b[:, 0] < -1) | (b[:, 1] < -1) | (b[:, 2] > w + 1)
                                                     | (b[:, 3] > h + 1)).sum())
                    st["tiny_boxes_lt8px"] += int((np.minimum(bw, bh) < 8).sum())
                    st["duplicate_ids_in_frame"] += len(t.track_id) - len(set(t.track_id.tolist()))
                    if len(fr.ignore_regions):
                        cx, cy = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2
                        r = fr.ignore_regions
                        inside = ((cx[:, None] >= r[None, :, 0]) & (cx[:, None] <= r[None, :, 2])
                                  & (cy[:, None] >= r[None, :, 1]) & (cy[:, None] <= r[None, :, 3])).any(1)
                        st["boxes_centred_in_ignored_regions"] += int(inside.sum())
                    classes.update(class_from_id(int(c)).value for c in t.class_id)
                    cur = {int(i): bb for i, bb in zip(t.track_id, b, strict=True)}
                    common = set(cur) & set(prev)
                    if common and fr.frame_index % 25 == 0:
                        redundancy.append(float(np.mean([_iou(cur[i], prev[i]) for i in common])))
                    prev = cur
        q.stats[split] = dict(st) | {"classes": dict(classes),
                                     "median_consecutive_frame_iou": round(float(np.median(redundancy)), 3)
                                     if redundancy else None}
        q.add(f"{split}: images vs annotations", "ok" if st["annotations_without_image"] == 0 else "fail",
              f"{st['annotated_frames']} annotated frames, {st['image_frames']} images; "
              f"{st['images_without_annotation']} images have no annotation entry (excluded - never assumed empty); "
              f"{st['annotations_without_image']} annotations without an image")
        bad = st["invalid_boxes"] + st["out_of_image_boxes"] + st["duplicate_ids_in_frame"]
        q.add(f"{split}: box validity", "ok" if bad == 0 else "warn",
              f"{st['invalid_boxes']} zero/negative-size, {st['out_of_image_boxes']} outside the image, "
              f"{st['duplicate_ids_in_frame']} duplicate ids in a frame, {st['tiny_boxes_lt8px']} smaller than 8 px")
        q.add(f"{split}: ignored regions", "info",
              f"{st['boxes_centred_in_ignored_regions']} annotated boxes centred inside ignored regions "
              "(ignored regions are masked for training and excluded from evaluation)")
        if st["unmapped_labels"]:
            q.add(f"{split}: unmapped labels", "warn", f"{st['unmapped_labels']} boxes without a project class")
        total = sum(classes.values()) or 1
        q.add(f"{split}: class balance", "warn" if min(classes.values(), default=0) / total < 0.05 else "ok",
              ", ".join(f"{k} {v} ({v / total:.1%})" for k, v in classes.most_common()))
        if redundancy:
            q.add(f"{split}: adjacent-frame redundancy", "info",
                  f"median IoU of the same object in consecutive frames {np.median(redundancy):.2f} - adjacent "
                  "frames are near-duplicates, so splits are by sequence and training frames are subsampled")
    if len(names) > 1:
        s1, s2 = list(names)[:2]
        overlap = names[s1] & names[s2]
        q.add(f"{s1}/{s2} sequence overlap", "ok" if not overlap else "fail",
              f"{len(names[s1])} {s1} and {len(names[s2])} {s2} sequences; overlap: {sorted(overlap) or 'none'}")
    return q


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


# --------------------------------------------------------------------------- incident intervals
def check_incidents(dsid: str, root: Path) -> DatasetQuality:
    import cv2

    from traffic_intel.datasets.adapters.base import IncidentAdapter, adapter_for

    a = cast(IncidentAdapter, adapter_for(dsid, root))
    ann = a.annotations()
    vids = a.videos()
    q = DatasetQuality(dsid, "incident intervals (video level)", str(root))
    dur: dict[str, float] = {}
    fps: dict[str, float] = {}
    for vid, p in vids.items():
        cap = cv2.VideoCapture(str(p))
        f, n = cap.get(5), cap.get(7)
        cap.release()
        fps[vid], dur[vid] = f, (n / f if f else 0.0)
    labelled = {x.video_id for x in ann}
    invalid = [x for x in ann if x.end_s is not None and not (0 <= x.start_s < x.end_s <= dur.get(x.video_id, 0) + 1)]
    missing = sorted(labelled - set(vids), key=int)
    q.add("labels reference existing training videos", "ok" if not missing else "fail",
          f"{len(labelled)} labelled videos; without a file: {missing or 'none'}")
    q.add("interval validity", "ok" if not invalid else "fail",
          f"{len(ann)} intervals; {len(invalid)} with start >= end or end beyond the video "
          f"({[(x.video_id, x.start_s, x.end_s) for x in invalid][:5]})")
    by_vid: dict[str, list[tuple[float, float]]] = {}
    for x in ann:
        by_vid.setdefault(x.video_id, []).append((x.start_s, x.end_s or dur.get(x.video_id, 0)))
    overlaps = sum(1 for iv in by_vid.values() for i, j in zip(sorted(iv), sorted(iv)[1:], strict=False) if j[0] < i[1])
    q.add("overlapping intervals", "ok" if overlaps == 0 else "warn", f"{overlaps} overlapping pairs; "
          f"{sum(len(v) > 1 for v in by_vid.values())} videos have more than one interval")
    anomalous_s = sum(e - s for iv in by_vid.values() for s, e in iv)
    total_s = sum(dur.values())
    at_start = sum(1 for x in ann if x.start_s <= 1)
    q.add("class balance", "warn",
          f"{len(labelled)}/{len(vids)} training videos contain an anomaly; anomalous time "
          f"{anomalous_s / 3600:.1f} h of {total_s / 3600:.1f} h ({anomalous_s / max(total_s, 1):.1%}); "
          f"{at_start} intervals start at 0 s (anomaly already present when the video starts)")
    odd = {v: round(f, 2) for v, f in fps.items() if abs(f - 30) > 0.5}
    q.add("frame rates", "warn" if odd else "ok",
          f"{len(odd)} training videos not at 30 fps: {odd} - labels are in seconds, so times are converted with "
          "each video's own fps")
    test = glob.glob(str(root / "*test*" / "*.mp4"))
    q.add("test split", "info", f"{len(test)} test videos without public labels - not usable for supervised "
          "evaluation; excluded")
    q.stats.update(videos=len(vids), labelled_videos=len(labelled), intervals=len(ann),
                   hours=round(total_s / 3600, 2), anomalous_hours=round(anomalous_s / 3600, 2), test_videos=len(test))
    return q


# --------------------------------------------------------------------------- identity labels (re-id)
def check_identities(dsid: str, root: Path) -> DatasetQuality:
    from traffic_intel.datasets.inventory import _parse_xml

    q = DatasetQuality(dsid, "identity labels (re-identification)", str(root))
    items = list(_parse_xml(root / "train_label.xml").iter("Item"))
    per_id = Counter(it.get("vehicleID") for it in items)
    cams_per_id = Counter({v: len({it.get("cameraID") for it in items if it.get("vehicleID") == v}) for v in per_id})
    names = set((root / "name_train.txt").read_text().split())
    on_disk = {Path(p).name for p in glob.glob(str(root / "image_train" / "*.jpg"))}
    labelled = {it.get("imageName") for it in items}
    q.add("label file vs images", "ok" if labelled == names == on_disk else "fail",
          f"{len(labelled)} labelled, {len(names)} listed, {len(on_disk)} on disk")
    counts = np.array(list(per_id.values()))
    q.add("images per identity", "info", f"{len(per_id)} identities; images per identity min {counts.min()}, "
          f"median {np.median(counts):.0f}, max {counts.max()}; cameras per identity median "
          f"{np.median(list(cams_per_id.values())):.0f}")
    test_xml = root / "test_label.xml"
    has_ids = test_xml.exists() and _parse_xml(test_xml).find(".//Item[@vehicleID]") is not None
    q.add("test identities", "info" if not has_ids else "ok",
          "test/query labels carry camera ids only - identities are withheld (challenge evaluation), so a held-out "
          "evaluation must split the TRAINING identities" if not has_ids else "test identities present")
    q.stats.update(identities=len(per_id), train_images=len(items))
    return q


# --------------------------------------------------------------------------- box annotations (CSV)
def check_box_csv(dsid: str, root: Path) -> DatasetQuality:
    q = DatasetQuality(dsid, "box annotations (CSV)", str(root))
    files = sorted(root.rglob("frameAnnotationsBOX.csv")) or sorted(root.rglob("*.csv"))
    df = pd.concat([pd.read_csv(f, sep=";") for f in files], ignore_index=True)
    x1, y1, x2, y2 = (df[c] for c in ("Upper left corner X", "Upper left corner Y",
                                      "Lower right corner X", "Lower right corner Y"))
    invalid = int(((x2 <= x1) | (y2 <= y1)).sum())
    import cv2

    sample = next(iter(root.rglob("*.jpg")), None)
    img = cv2.imread(str(sample)) if sample else None
    h, w = img.shape[:2] if img is not None else (0, 0)
    oob = int(((x1 < 0) | (y1 < 0) | (x2 > w) | (y2 > h)).sum()) if w else 0
    q.add("box validity", "ok" if invalid + oob == 0 else "warn",
          f"{len(df)} boxes in {len(files)} files; {invalid} zero/negative-size, {oob} outside the {w}x{h} frame")
    tags = df["Annotation tag"].value_counts()
    q.add("tag balance", "warn" if tags.min() / tags.sum() < 0.02 else "ok",
          ", ".join(f"{k} {v}" for k, v in tags.items()))
    jpgs = {p.name for p in root.rglob("*.jpg")}
    referenced = df["Filename"].map(lambda s: Path(str(s)).name)
    found = float(referenced.isin(jpgs).mean())
    q.add("referenced frames exist", "ok" if found > 0.99 else "warn",
          f"{found:.1%} of annotation rows point to an image on disk "
          f"({referenced.nunique()} distinct frames referenced, {len(jpgs)} images)")
    dups = int(df.duplicated().sum())
    q.add("duplicate rows", "ok" if dups == 0 else "warn", f"{dups} duplicate annotation rows")
    q.stats.update(boxes=len(df), tags=tags.to_dict(), frame_size=[w, h])
    return q


# --------------------------------------------------------------------------- context tables
def check_context_table(dsid: str, root: Path, traffic_ranges: dict[str, tuple[str, str]]) -> DatasetQuality:
    q = DatasetQuality(dsid, "context table (weather)", str(root))
    files = sorted(root.rglob("*.csv"))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    tcol = next(c for c in df.columns if "date" in c.lower() or "time" in c.lower())
    t = pd.to_datetime(df[tcol], utc=True, errors="coerce")
    q.add("timestamps", "ok" if t.notna().all() else "fail",
          f"{int(t.isna().sum())} unparseable; {df[tcol].str[-5:].nunique()} UTC offsets present "
          f"({sorted(df[tcol].str[-5:].unique())}); range {t.min()} .. {t.max()}")
    q.add("duplicates", "ok" if not df.duplicated().any() else "warn",
          f"{int(df.duplicated().sum())} duplicate rows, {int(t.duplicated().sum())} duplicate timestamps")
    impossible = {}
    for c in df.select_dtypes("number").columns:
        low = c.lower()
        if "pressure" in low:
            impossible[c] = int(((df[c] < 850) | (df[c] > 1090)).sum())
        if "humidity" in low:
            impossible[c] = int((df[c] <= 0).sum())
    q.add("impossible values", "warn" if any(impossible.values()) else "ok",
          ", ".join(f"{k}: {v}" for k, v in impossible.items()) + " (pressure outside 850-1090 mbar, humidity 0)")
    const = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    q.add("constant columns", "warn" if const else "ok", f"{const or 'none'} carry no information")
    miss = {c: int(v) for c, v in df.isna().sum().items() if v}
    q.add("missing values", "warn" if miss else "ok", str(miss or "none"))
    overlap = {k: bool(t.min() <= pd.Timestamp(e, tz="UTC") and t.max() >= pd.Timestamp(s, tz="UTC"))
               for k, (s, e) in traffic_ranges.items()}
    q.add("overlap with traffic datasets", "warn",
          "time overlap: " + ", ".join(f"{k}: {'yes' if v else 'no'}" for k, v in overlap.items())
          + " - and a single site; weather must match the traffic's location AND time to be a feature")
    q.stats.update(rows=len(df), start=str(t.min()), end=str(t.max()))
    return q


def check_parquet_cache(key: str, root: Path) -> DatasetQuality:
    q = DatasetQuality(key, "context cache (Parquet)", str(root))
    for f in sorted(root.glob("*.parquet")):
        df = pd.read_parquet(f)
        miss = {c: f"{v / len(df):.0%}" for c, v in df.isna().sum().items() if v}
        q.add(f.name, "warn" if miss else "ok",
              f"{len(df)} rows {df.index.min()} .. {df.index.max()}; missing: {miss or 'none'}")
    return q


# --------------------------------------------------------------------------- driver
def run_checks() -> dict[str, Any]:
    from traffic_intel.datasets.adapters.base import (
        DATASET_ADAPTERS,
        DetectionTrackingAdapter,
        IncidentAdapter,
        TimeSeriesAdapter,
    )
    from traffic_intel.datasets.manifest import dataset_root, load_manifest, validate

    results: list[DatasetQuality] = []
    ranges: dict[str, tuple[str, str]] = {}
    m = load_manifest()
    # context tables last: their check compares coverage with the traffic datasets' time ranges
    order = sorted(m.datasets.items(), key=lambda kv: kv[1].adapter.type == "none" and "context" in kv[1].tasks)
    for dsid, e in order:
        root = dataset_root(e)
        if e.adapter.type in ("synthetic", "own_recordings") or not validate(e).ok or not e.required:
            continue                                    # generated / nothing on disk / cache-only
        log.info("quality: %s", dsid)
        try:
            if e.adapter.type == "none":
                q = (check_identities(dsid, root) if "reid" in e.tasks
                     else check_box_csv(dsid, root) if "traffic_light" in e.tasks
                     else check_context_table(dsid, root, ranges) if "context" in e.tasks else None)
                if q is None:
                    continue
            else:
                cls = DATASET_ADAPTERS.get(e.adapter.type)
                if isinstance(cls, type) and issubclass(cls, TimeSeriesAdapter):
                    q = check_timeseries(dsid, root)
                    ranges[dsid] = (q.stats["start"], q.stats["end"])
                elif isinstance(cls, type) and issubclass(cls, DetectionTrackingAdapter):
                    q = check_object_annotations(dsid, root)
                elif isinstance(cls, type) and issubclass(cls, IncidentAdapter):
                    q = check_incidents(dsid, root)
                else:
                    continue
            if q is not None:
                results.append(q)
        except Exception as exc:  # noqa: BLE001 - a dataset that cannot be checked is reported, not hidden
            log.exception("quality check failed for %s", dsid)
            results.append(DatasetQuality(dsid, "error", str(root),
                                          [Check("load", "fail", f"{type(exc).__name__}: {exc}")]))
    from traffic_intel.core.config import get_settings

    cache = get_settings().data_dir / "external" / "weather"
    if cache.is_dir():
        results.append(check_parquet_cache("external/weather (Open-Meteo cache)", cache))
    return {"generated": datetime.now(UTC).isoformat(timespec="seconds"),
            "datasets": [asdict(r) for r in results]}


def render_markdown(rep: dict[str, Any]) -> str:
    lines = ["# Data quality report", "",
             f"Generated {rep['generated']} by `python -m traffic_intel.datasets quality`.",
             "Every number below was measured on the files on disk.", "",
             "| Dataset | Kind | ok | warn | fail | info |", "|---|---|--:|--:|--:|--:|"]
    for d in rep["datasets"]:
        c = Counter(ch["status"] for ch in d["checks"])
        lines.append(f"| {d['dataset']} | {d['kind']} | {c['ok']} | {c['warn']} | {c['fail']} | {c['info']} |")
    for d in rep["datasets"]:
        lines += ["", f"## {d['dataset']} ({d['kind']})", "", f"`{d['path']}`", "", "| Check | Status | Result |",
                  "|---|---|---|"]
        for ch in d["checks"]:
            detail = str(ch["detail"]).replace("|", "/").replace("\n", " ")
            lines.append(f"| {ch['name']} | **{ch['status']}** | {detail} |")
    return "\n".join(lines) + "\n"


def write_report(out: Path) -> str:
    rep = run_checks()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    md = out.with_suffix(".md")
    md.write_text(render_markdown(rep), encoding="utf-8")
    counts = Counter(ch["status"] for d in rep["datasets"] for ch in d["checks"])
    return f"{len(rep['datasets'])} datasets checked: {dict(counts)}\n  {out}\n  {md}"
