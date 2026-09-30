"""Replay detections from a CSV file (``detector.type: precomputed``).

Lets the whole downstream pipeline (tracking, trajectories, features, state)
run without a GPU or a detector - e.g. on the synthetic dataset, on
detections exported from another system, or on ground-truth boxes from an
annotated dataset (to isolate tracking/feature errors from detection errors).

CSV columns (header required):
    frame, class, confidence, x1, y1, x2, y2
``class`` is an ObjectClass name (car, motorcycle, ..., pedestrian).
Frames are matched on ``Frame.index`` (source frame index).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from traffic_intel.computer_vision.interfaces import DETECTORS, Detector
from traffic_intel.core.config import resolve_path
from traffic_intel.core.enums import OBJECT_CLASSES, ObjectClass, class_id_of
from traffic_intel.core.types import Detections, Frame

COLUMNS = ["frame", "class", "confidence", "x1", "y1", "x2", "y2"]


def load_detection_table(path: str | Path) -> dict[int, Detections]:
    df = pd.read_csv(resolve_path(path))
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    df["class_id"] = df["class"].map(lambda c: class_id_of(ObjectClass(c)))
    table: dict[int, Detections] = {}
    for frame, g in df.groupby("frame", sort=True):
        table[int(frame)] = Detections(
            g[["x1", "y1", "x2", "y2"]].to_numpy(np.float32),
            g["confidence"].to_numpy(np.float32),
            g["class_id"].to_numpy(np.int64),
        )
    return table


def save_detection_table(path: str | Path, detections: dict[int, Detections]) -> None:
    rows = []
    for frame in sorted(detections):
        d = detections[frame]
        for i in range(len(d)):
            x1, y1, x2, y2 = (round(float(v), 2) for v in d.xyxy[i])
            rows.append((frame, OBJECT_CLASSES[int(d.class_id[i])].value,
                         round(float(d.confidence[i]), 3), x1, y1, x2, y2))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=COLUMNS).to_csv(path, index=False)


class PrecomputedDetector(Detector):
    def __init__(self, path: str | None = None, table: dict[int, Detections] | None = None,
                 conf_threshold: float = 0.0, **_: object) -> None:
        if table is None:
            if path is None:
                raise ValueError("precomputed detector needs `path` or `table`")
            table = load_detection_table(path)
        self.table = table
        self.conf_threshold = conf_threshold

    @property
    def classes(self) -> Sequence[ObjectClass]:
        return OBJECT_CLASSES

    def detect(self, frames: Sequence[Frame]) -> list[Detections]:
        out = []
        for f in frames:
            d = self.table.get(f.index, Detections())
            out.append(d.select(d.confidence >= self.conf_threshold) if len(d) else d)
        return out


DETECTORS.register("precomputed")(PrecomputedDetector)
