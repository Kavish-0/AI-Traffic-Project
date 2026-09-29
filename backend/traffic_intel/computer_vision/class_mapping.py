"""Configurable mapping from a model's class ids to ``ObjectClass``.

Different detectors number classes differently (COCO: person=0, car=2; a
fine-tuned traffic model might use car=0, bus=1 ...). The mapping is given in
config by *label name* (robust across models that share names) and/or by
*integer id* (for models with unusual names), and resolved against the
model's own ``names`` table at load time.

    class_map:            # model label or id  ->  ObjectClass
      car: car
      person: pedestrian
      7: truck            # int key = model class id
    class_confidence:     # optional per-class minimum confidence
      pedestrian: 0.35
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

import numpy as np

from traffic_intel.core.enums import OBJECT_CLASSES, ObjectClass, class_id_of
from traffic_intel.core.types import FloatArray, IntArray

log = logging.getLogger(__name__)

COCO_DEFAULT_MAP: dict[str | int, str] = {
    "car": "car",
    "motorcycle": "motorcycle",
    "bus": "bus",
    "truck": "truck",
    "bicycle": "bicycle",
    "person": "pedestrian",
}


class ClassMapper:
    def __init__(
        self,
        model_names: Mapping[int, str],
        class_map: Mapping[str | int, str] | None = None,
        class_confidence: Mapping[str, float] | None = None,
        default_confidence: float = 0.0,
    ) -> None:
        class_map = class_map if class_map is not None else COCO_DEFAULT_MAP
        name_to_id = {str(n).lower(): int(i) for i, n in model_names.items()}
        size = max(model_names) + 1 if model_names else 0
        self._lookup = np.full(size, -1, dtype=np.int64)
        self.mapping: dict[int, ObjectClass] = {}

        for key, target in class_map.items():
            target_cls = ObjectClass(target)
            if isinstance(key, int) or (isinstance(key, str) and key.isdigit()):
                model_id = int(key)
                if model_id not in model_names:
                    raise ValueError(f"class_map id {model_id} not in model classes")
            else:
                if key.lower() not in name_to_id:
                    log.warning("class_map label '%s' not in model classes; ignored", key)
                    continue
                model_id = name_to_id[key.lower()]
            self._lookup[model_id] = class_id_of(target_cls)
            self.mapping[model_id] = target_cls

        if not self.mapping:
            raise ValueError("class_map resolved to no classes for this model")

        self._min_conf = np.full(len(OBJECT_CLASSES), default_confidence, dtype=np.float32)
        for name, conf in (class_confidence or {}).items():
            self._min_conf[class_id_of(ObjectClass(name))] = conf

    @property
    def model_class_ids(self) -> list[int]:
        """Model ids to request from the detector (skips NMS work on others)."""
        return sorted(self.mapping)

    @property
    def classes(self) -> list[ObjectClass]:
        return sorted(set(self.mapping.values()), key=OBJECT_CLASSES.index)

    def map(self, model_cls: IntArray, confidence: FloatArray) -> tuple[IntArray, np.ndarray]:
        """Returns (our class ids, keep-mask). Unmapped or under-confident rows are dropped."""
        model_cls = np.asarray(model_cls, dtype=np.int64)
        in_range = (model_cls >= 0) & (model_cls < len(self._lookup))
        ours = np.full(len(model_cls), -1, dtype=np.int64)
        ours[in_range] = self._lookup[model_cls[in_range]]
        keep = ours >= 0
        keep[keep] &= confidence[keep] >= self._min_conf[ours[keep]]
        return ours, keep
