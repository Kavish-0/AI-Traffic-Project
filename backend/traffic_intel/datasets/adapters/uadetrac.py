"""UA-DETRAC adapter: per-sequence XML annotations -> canonical ground-truth frames.

XML (one file per sequence, frame numbers 1-based):

    <sequence name="MVI_20011">
      <sequence_attribute camera_state="unstable" sence_weather="sunny"/>
      <ignored_region><box left=".." top=".." width=".." height=".."/>...</ignored_region>
      <frame density="7" num="1">
        <target_list>
          <target id="1">
            <box left="592.75" top="378.8" width="160.05" height="162.2"/>
            <attribute orientation=".." speed=".." trajectory_length=".." truncation_ratio="0.1"
                       vehicle_type="car"/>
          </target> ...

Source classes (car, van, bus, others) are mapped to project ObjectClass via the manifest's
``classes.class_map``; unmapped labels are counted, not guessed. Frames are
``<images>/<sequence>/img00001.jpg``.
"""

from __future__ import annotations

import glob
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path

import numpy as np

from traffic_intel.datasets.adapters.base import (
    DATASET_ADAPTERS,
    AnnotatedFrame,
    DetectionTrackingAdapter,
    SequenceInfo,
    make_tracks,
)

ANNOTATION_GLOBS = ("DETRAC-Train-Annotations-XML/*.xml", "DETRAC-Train-Annotations-XML-v3/*.xml",
                    "*Annotations*/*.xml")
# official test split (its annotations are public); selected with params split="test"
TEST_ANNOTATION_GLOBS = ("DETRAC-Test-Annotations-XML/*.xml", "DETRAC-Test-Annotations-XML-v3/*.xml")


@DATASET_ADAPTERS.register("ua_detrac")
class UADETRACAdapter(DetectionTrackingAdapter):
    """Official splits are kept apart: ``split="train"`` (default) or ``split="test"``; frames of one
    sequence never move between splits."""

    splits = ("train", "test")

    def _xml_files(self) -> dict[str, Path]:
        split = str(self.params.get("split", "train"))
        if split not in ("train", "test"):
            raise ValueError(f"ua_detrac: split must be 'train' or 'test', got {split!r}")
        for pat in ANNOTATION_GLOBS if split == "train" else TEST_ANNOTATION_GLOBS:
            files = sorted(glob.glob(str(self.root / pat)))
            if files:
                return {Path(f).stem.replace("_v3", ""): Path(f) for f in files}
        return {}

    def _image_dir(self, seq: str) -> Path | None:
        hits = glob.glob(str(self.root / "*" / seq))
        return Path(hits[0]) if hits else None

    @lru_cache(maxsize=8)  # noqa: B019 - small, per-adapter-instance cache of parsed sequences
    def _parse(self, seq: str) -> tuple[dict[str, str], np.ndarray, dict[int, list[_Target]]]:
        path = self._xml_files()[seq]
        root = ET.parse(path).getroot()
        seq_attr = root.find("sequence_attribute")
        attrs = dict(seq_attr.attrib) if seq_attr is not None else {}
        ign = [_xyxy(b) for region in root.iter("ignored_region") for b in region.iter("box")]
        frames: dict[int, list[_Target]] = {}
        for fr in root.iter("frame"):
            num = int(_attr(fr, "num"))
            rows = []
            for t in fr.iter("target"):
                box, attr = t.find("box"), t.find("attribute")
                if box is None:
                    continue
                rows.append((int(_attr(t, "id")), _xyxy(box), attr.get("vehicle_type", "") if attr is not None else ""))
            frames[num] = rows
        return attrs, np.asarray(ign, np.float32).reshape(-1, 4), frames

    def sequences(self) -> list[SequenceInfo]:
        out = []
        fps = float(self.params.get("fps") or getattr(self.entry, "fps", 25) or 25)
        size = getattr(self.entry, "frame_size", None)
        for seq in self._xml_files():
            attrs, _, frames = self._parse(seq)
            out.append(SequenceInfo(seq, fps, tuple(size) if size else None, max(frames, default=0),
                                    self._image_dir(seq), {"weather": attrs.get("sence_weather", attrs.get("weather")),
                                                           "camera_state": attrs.get("camera_state")}))
        return out

    def frames(self, sequence_id: str) -> Iterator[AnnotatedFrame]:
        _, ignore, frames = self._parse(sequence_id)
        fps = float(self.params.get("fps") or getattr(self.entry, "fps", 25) or 25)
        for num in sorted(frames):
            ids, boxes, cls, unmapped = [], [], [], 0
            for tid, box, label in frames[num]:
                cid = self.classes(label)
                if cid is None:
                    unmapped += 1
                    continue
                ids.append(tid)
                boxes.append(box)
                cls.append(cid)
            k = num - 1
            yield AnnotatedFrame(sequence_id, k, k / fps, make_tracks(ids, boxes, cls), ignore, unmapped)


_Target = tuple[int, tuple[float, float, float, float], str]


def _attr(el: ET.Element, name: str) -> str:
    v = el.get(name)
    if v is None:
        raise ValueError(f"UA-DETRAC XML: <{el.tag}> without '{name}'")
    return v


def _xyxy(box: ET.Element) -> tuple[float, float, float, float]:
    x, y = float(_attr(box, "left")), float(_attr(box, "top"))
    return (x, y, x + float(_attr(box, "width")), y + float(_attr(box, "height")))
