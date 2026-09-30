"""Ultralytics YOLO adapter (``detector.type: yolo``).

Supports any Ultralytics detection checkpoint (.pt) or export (.onnx,
.engine, ...). Class ids are translated through ``ClassMapper`` using the
model's own ``names`` table, so COCO and custom-trained models both work.
Requires the ``cv`` extra: ``pip install -e ".[cv]"``.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from traffic_intel.computer_vision.class_mapping import ClassMapper
from traffic_intel.computer_vision.device import DeviceInfo, resolve_device
from traffic_intel.computer_vision.interfaces import DETECTORS, Detector
from traffic_intel.core.config import resolve_path
from traffic_intel.core.enums import ObjectClass
from traffic_intel.core.types import Detections, Frame

log = logging.getLogger(__name__)


def resolve_weights(weights: str, auto_download: bool) -> str:
    """A file path (downloaded if missing and allowed) or a model-registry reference
    ``registry:detection/<name>[@version]`` (default version: production) -> weights file."""
    if weights.startswith("registry:"):
        from traffic_intel.ai.model_store import FileModelStore
        from traffic_intel.core.config import get_settings
        from traffic_intel.core.enums import ModelTask

        ref, _, version = weights.removeprefix("registry:").partition("@")
        task, _, name = ref.partition("/")
        store = FileModelStore(get_settings().models_dir)
        v = store.resolve_version(ModelTask(task), name, version or "production")
        f = get_settings().models_dir / task / name / v / "model.pt"
        if not f.exists():
            raise FileNotFoundError(f"registry model {weights} -> {f} does not exist")
        return str(f)
    path = resolve_path(weights)
    if path.exists():
        return str(path)
    if not auto_download:
        raise FileNotFoundError(f"YOLO weights not found: {path}")
    from ultralytics.utils.downloads import attempt_download_asset

    path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading %s -> %s", path.name, path)
    out = Path(attempt_download_asset(str(path)))
    if not out.exists():
        raise FileNotFoundError(f"YOLO weights not found and not downloadable: {path}")
    return str(out)


class YOLODetector(Detector):
    def __init__(
        self,
        weights: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.5,
        imgsz: int = 960,
        device: str = "auto",
        fallback_to_cpu: bool = True,
        half: bool | str = "auto",
        max_det: int = 300,
        agnostic_nms: bool = False,
        class_map: Mapping[str | int, str] | None = None,
        class_confidence: Mapping[str, float] | None = None,
        auto_download: bool = True,
        model: Any = None,  # injected model (tests); must expose .names and .predict()
    ) -> None:
        self.device_info = _resolve(device, fallback_to_cpu, half)
        self.device = self.device_info.device
        self.half = self.device_info.half
        if model is None:
            from ultralytics import YOLO  # type: ignore[attr-defined]  # lazy re-export

            model = YOLO(resolve_weights(weights, auto_download))
            model.to(self.device)
        self.model = model
        self.weights = weights
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self.max_det = max_det
        self.agnostic_nms = agnostic_nms
        names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
        self.mapper = ClassMapper(names, class_map, class_confidence, default_confidence=conf_threshold)
        # The model filters at the lowest threshold in use; the mapper then applies
        # each class's own threshold (class_confidence may go below conf_threshold).
        self._predict_conf = min([conf_threshold, *(class_confidence or {}).values()])
        log.info("Detector: YOLO %s | device %s | imgsz %d | classes %s", Path(weights).name,
                 self.device_info.describe(), imgsz, {names[k]: v.value for k, v in self.mapper.mapping.items()})

    @property
    def classes(self) -> Sequence[ObjectClass]:
        return self.mapper.classes

    def detect(self, frames: Sequence[Frame]) -> list[Detections]:
        if not frames:
            return []
        results = self.model.predict(
            [f.image for f in frames],
            conf=self._predict_conf,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            device=self.device,
            **_precision_kwargs(self.half),
            max_det=self.max_det,
            agnostic_nms=self.agnostic_nms,
            classes=self.mapper.model_class_ids,
            verbose=False,
        )
        out: list[Detections] = []
        for r in results:
            boxes = r.boxes
            xyxy = _np(boxes.xyxy).reshape(-1, 4).astype(np.float32)
            conf = _np(boxes.conf).reshape(-1).astype(np.float32)
            cls = _np(boxes.cls).reshape(-1).astype(np.int64)
            ours, keep = self.mapper.map(cls, conf)
            out.append(Detections(xyxy[keep], conf[keep], ours[keep]))
        return out

    def warmup(self) -> None:
        dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        from datetime import UTC, datetime

        self.detect([Frame("warmup", 0, datetime.now(UTC), dummy)])


def _precision_kwargs(half: bool) -> dict[str, Any]:
    """Ultralytics >= 8.4 takes ``quantize="fp16"``; older versions take ``half=True``."""
    try:
        from ultralytics.cfg import DEFAULT_CFG_DICT  # type: ignore[attr-defined]

        if "quantize" in DEFAULT_CFG_DICT:
            return {"quantize": "fp16" if half else None}
    except ImportError:
        pass
    return {"half": half}


def _resolve(device: str, fallback_to_cpu: bool, half: bool | str) -> DeviceInfo:
    try:
        return resolve_device(device, fallback_to_cpu=fallback_to_cpu, half=half)
    except ImportError:  # torch not installed (only possible with an injected test model)
        return DeviceInfo("cpu", "cpu", "CPU", False, "n/a", None, device, device not in ("cpu", "auto"))


def _np(x: Any) -> np.ndarray:
    return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)


DETECTORS.register("yolo")(YOLODetector)
