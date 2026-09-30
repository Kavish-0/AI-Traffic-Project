"""Detection layer: data types, configurable class mapping, YOLO adapter, precomputed detections."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import FIXTURES

from traffic_intel.computer_vision.class_mapping import ClassMapper
from traffic_intel.computer_vision.detectors.precomputed import (
    PrecomputedDetector,
    load_detection_table,
    save_detection_table,
)
from traffic_intel.computer_vision.detectors.yolo import YOLODetector
from traffic_intel.computer_vision.interfaces import DETECTORS
from traffic_intel.core.config import PROJECT_ROOT
from traffic_intel.core.enums import ObjectClass, class_id_of
from traffic_intel.core.registry import ComponentSpec
from traffic_intel.core.types import Detections, Frame

COCO_NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck", 9: "traffic light"}
NOW = datetime(2026, 9, 1, tzinfo=UTC)


def test_detection_view_has_class_name_bbox_timestamp() -> None:
    d = Detections(np.array([[1, 2, 3, 4]], np.float32), np.array([0.8], np.float32),
                   np.array([class_id_of(ObjectClass.BUS)]))
    (det,) = d.to_list(NOW)
    assert (det.class_name, det.class_id, det.bbox, det.timestamp) == ("bus", 2, (1.0, 2.0, 3.0, 4.0), NOW)
    assert det.confidence == pytest.approx(0.8)


def test_coco_mapping_by_name_drops_other_classes() -> None:
    m = ClassMapper(COCO_NAMES)
    ours, keep = m.map(np.array([2, 0, 9, 7, 3]), np.array([0.9, 0.9, 0.9, 0.9, 0.9], np.float32))
    assert keep.tolist() == [True, True, False, True, True]
    assert ours[keep].tolist() == [class_id_of(c) for c in ("car", "pedestrian", "truck", "motorcycle")]
    assert m.model_class_ids == [0, 1, 2, 3, 5, 7]


def test_custom_model_with_different_ids_and_int_keys() -> None:
    """A fine-tuned model numbering classes its own way."""
    names = {0: "Vehicle-Car", 1: "two_wheeler", 2: "HeavyBus", 3: "lorry"}
    m = ClassMapper(names, {"vehicle-car": "car", "two_wheeler": "motorcycle", 2: "bus", "3": "truck"})
    ours, keep = m.map(np.array([3, 2, 1, 0]), np.full(4, 0.9, np.float32))
    assert keep.all()
    assert ours.tolist() == [class_id_of(c) for c in ("truck", "bus", "motorcycle", "car")]


def test_per_class_confidence() -> None:
    m = ClassMapper(COCO_NAMES, class_confidence={"pedestrian": 0.5}, default_confidence=0.25)
    _, keep = m.map(np.array([0, 0, 2]), np.array([0.4, 0.6, 0.3], np.float32))
    assert keep.tolist() == [False, True, True]


def test_mapping_errors() -> None:
    with pytest.raises(ValueError):
        ClassMapper(COCO_NAMES, {"spaceship": "car"})          # nothing resolves
    with pytest.raises(ValueError):
        ClassMapper(COCO_NAMES, {42: "car"})                   # unknown id
    with pytest.raises(ValueError):
        ClassMapper(COCO_NAMES, {"car": "tram"})               # unknown target class


class FakeYOLO:
    """Duck-typed stand-in for ultralytics.YOLO (names + predict)."""

    names = COCO_NAMES

    def __init__(self) -> None:
        self.kwargs: dict[str, object] = {}

    def predict(self, images: list[np.ndarray], **kwargs: object) -> list[SimpleNamespace]:
        self.kwargs = kwargs
        boxes = SimpleNamespace(
            xyxy=np.array([[10, 10, 50, 40], [60, 10, 70, 40], [0, 0, 5, 5]], np.float32),
            conf=np.array([0.9, 0.3, 0.8], np.float32),
            cls=np.array([2, 0, 9], np.float32),
        )
        return [SimpleNamespace(boxes=boxes) for _ in images]


def test_yolo_adapter_maps_filters_and_forwards_config() -> None:
    fake = FakeYOLO()
    det = YOLODetector(weights="unused.pt", conf_threshold=0.25, iou_threshold=0.45, imgsz=320,
                       device="cpu", class_confidence={"pedestrian": 0.35}, model=fake)
    frames = [Frame("c", i, NOW, np.zeros((40, 80, 3), np.uint8)) for i in range(2)]
    out = det.detect(frames)
    assert len(out) == 2
    assert out[0].class_id.tolist() == [class_id_of(ObjectClass.CAR)]   # person 0.3 < 0.35, light dropped
    assert fake.kwargs["iou"] == 0.45 and fake.kwargs["imgsz"] == 320 and fake.kwargs["conf"] == 0.25
    assert fake.kwargs["classes"] == [0, 1, 2, 3, 5, 7]
    assert fake.kwargs.get("half", False) is False and fake.kwargs.get("quantize") is None   # fp16 only on CUDA


def test_precomputed_roundtrip_and_fixture(tmp_path: Path) -> None:
    table = {0: Detections(np.array([[1, 2, 3, 4]], np.float32), np.array([0.9], np.float32), np.array([0])),
             5: Detections()}
    save_detection_table(tmp_path / "d.csv", table)
    loaded = load_detection_table(tmp_path / "d.csv")
    assert list(loaded) == [0] and loaded[0].xyxy.tolist() == [[1, 2, 3, 4]]
    spec = ComponentSpec(type="precomputed", params={"path": str(FIXTURES / "synthetic_small" / "detections.csv")})
    det = DETECTORS.create(spec)
    assert isinstance(det, PrecomputedDetector)
    frames = [Frame("c", i, NOW, np.zeros((1, 1, 3), np.uint8)) for i in (0, 10**6)]
    a, b = det.detect(frames)
    assert len(a) > 0 and len(b) == 0


WEIGHTS = PROJECT_ROOT / "models" / "detection" / "yolo11s.pt"


@pytest.mark.cv
@pytest.mark.skipif(not WEIGHTS.exists(), reason="YOLO weights not downloaded")
def test_real_yolo_on_sample_image() -> None:
    pytest.importorskip("ultralytics")
    import cv2
    from ultralytics.utils import ASSETS

    det = YOLODetector(weights=str(WEIGHTS), imgsz=640, device="cpu", auto_download=False)
    img = cv2.imread(str(ASSETS / "bus.jpg"))
    (d,) = det.detect([Frame("c", 0, NOW, img)])
    names = {x.class_name for x in d.to_list(NOW)}
    assert {"bus", "pedestrian"} <= names
