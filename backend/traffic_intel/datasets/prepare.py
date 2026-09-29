"""Raw dataset -> canonical files on disk (``python -m traffic_intel.datasets prepare <id>``).

    time series   data/processed/<id>/series.parquet (+ meta.json)     -> forecasting pipeline
    detection /   data/annotations/<id>/<sequence>.csv                   canonical ground-truth tracks
    tracking      data/detections/<id>/<sequence>_gt.csv                 precomputed-detector table: replay
                                                                          GT boxes through tracker/features
                  data/annotations/<id>/sequences.json                   fps, size, frames, camera metadata
    incidents     data/annotations/<id>/incidents.csv                    video_id, start_s, end_s, source_label

Plus ``ground_truth_tracked_frames``: canonical ground truth as ``TrackedFrame`` objects, the exact
input of the trajectory store / traffic feature engine, so features can be computed from annotated
tracks (isolating feature-engine errors from detector and tracker errors).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from traffic_intel.computer_vision.detectors.precomputed import save_detection_table
from traffic_intel.core.config import get_settings
from traffic_intel.core.enums import OBJECT_CLASSES
from traffic_intel.core.feature_schema import validate_frame
from traffic_intel.core.types import TrackedFrame
from traffic_intel.datasets.adapters.base import (
    DetectionTrackingAdapter,
    IncidentAdapter,
    TimeSeriesAdapter,
    adapter_for,
)


def _out(kind: str, dataset_id: str, data_dir: Path | None = None) -> Path:
    d = Path(data_dir or get_settings().data_dir) / kind / dataset_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def prepare(dataset_id: str, *, root: Path | None = None, data_dir: Path | None = None,
            max_sequences: int | None = None, **params: Any) -> dict[str, Any]:
    ad = adapter_for(dataset_id, root)
    ad.check()
    if isinstance(ad, TimeSeriesAdapter):
        return prepare_timeseries(ad, data_dir, **params)
    if isinstance(ad, DetectionTrackingAdapter):
        return prepare_detection(ad, data_dir, max_sequences)
    if isinstance(ad, IncidentAdapter):
        return prepare_incidents(ad, data_dir)
    raise TypeError(f"no prepare step for {type(ad).__name__}")


def prepare_timeseries(ad: TimeSeriesAdapter, data_dir: Path | None = None, **params: Any) -> dict[str, Any]:
    ds = ad.load(**params)
    validate_frame(ds.frame.columns, source=ad.dataset_id)
    out = _out("processed", ad.dataset_id, data_dir)
    ds.frame.to_parquet(out / "series.parquet", index=False)
    meta = {"dataset_id": ad.dataset_id, "created": datetime.now(UTC).isoformat(timespec="seconds"),
            "resolution_s": ds.resolution_s, "tz": ds.tz, "holiday_country": ds.holiday_country,
            "series": len(ds.series_ids), "rows": len(ds.frame), "columns": list(ds.frame.columns),
            "schema": "core/feature_schema.py", "params": params, "info": ds.info}
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return {"output": str(out), "series": len(ds.series_ids), "rows": len(ds.frame)}


def prepare_detection(ad: DetectionTrackingAdapter, data_dir: Path | None = None,
                      max_sequences: int | None = None) -> dict[str, Any]:
    ann, det = _out("annotations", ad.dataset_id, data_dir), _out("detections", ad.dataset_id, data_dir)
    seqs = ad.sequences()[:max_sequences] if max_sequences else ad.sequences()
    boxes = unmapped = 0
    for s in seqs:
        rows, table = [], {}
        for fr in ad.frames(s.sequence_id):
            t = fr.tracks
            unmapped += fr.unmapped
            table[fr.frame_index] = fr.as_detections()
            for i in range(len(t)):
                x1, y1, x2, y2 = (round(float(v), 2) for v in t.xyxy[i])
                rows.append((fr.frame_index, round(fr.time_s, 3), int(t.track_id[i]),
                             OBJECT_CLASSES[int(t.class_id[i])].value, x1, y1, x2, y2))
        boxes += len(rows)
        pd.DataFrame(rows, columns=["frame", "time_s", "track_id", "class", "x1", "y1", "x2", "y2"]).to_csv(
            ann / f"{s.sequence_id}.csv", index=False)
        save_detection_table(det / f"{s.sequence_id}_gt.csv", table)
    (ann / "sequences.json").write_text(json.dumps(
        [{"sequence_id": s.sequence_id, "fps": s.fps, "frame_size": s.frame_size, "n_frames": s.n_frames,
          "media": str(s.media) if s.media else None, **s.metadata} for s in seqs], indent=2), encoding="utf-8")
    return {"output": str(ann), "sequences": len(seqs), "boxes": boxes, "unmapped_boxes": unmapped}


def prepare_incidents(ad: IncidentAdapter, data_dir: Path | None = None) -> dict[str, Any]:
    out = _out("annotations", ad.dataset_id, data_dir)
    rows = [a.model_dump() for a in ad.annotations()]
    videos = ad.videos()
    df = pd.DataFrame(rows, columns=["dataset_id", "video_id", "start_s", "end_s", "source_label", "notes"])
    df["video_file"] = df["video_id"].map(lambda v: str(videos[v]) if v in videos else None)
    df.to_csv(out / "incidents.csv", index=False)
    return {"output": str(out / "incidents.csv"), "intervals": len(df), "videos_found": len(videos),
            "labelled_videos_without_file": int(df["video_file"].isna().sum())}


def ground_truth_tracked_frames(ad: DetectionTrackingAdapter, sequence_id: str, camera_id: str,
                                image_shape: tuple[int, int], t0: datetime | None = None) -> Iterator[TrackedFrame]:
    """Canonical GT as TrackedFrame (h, w image shape) - input of TrajectoryStore / feature engine."""
    from traffic_intel.core.types import Detections

    t0 = t0 or datetime(2000, 1, 1, tzinfo=UTC)
    for fr in ad.frames(sequence_id):
        yield TrackedFrame(camera_id=camera_id, frame_index=fr.frame_index,
                           timestamp=t0 + timedelta(seconds=fr.time_s), image_shape=image_shape,
                           detections=Detections(), tracks=fr.tracks)
