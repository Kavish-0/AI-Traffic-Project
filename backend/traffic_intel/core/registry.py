"""Component registry: how replaceable models are selected by configuration.

Every pluggable interface (Detector, Tracker, Forecaster, ...) owns one
``Registry``. Implementations register themselves under a short name and the
pipeline builds them from a ``ComponentSpec`` in ``configs/pipeline.yaml``::

    detector:
      type: yolo
      params: {weights: models/detection/yolo11s.pt, conf: 0.25}

Swapping YOLO for another detector is then a new module plus a one-line
config change; nothing downstream imports the concrete class.

Implementation modules are imported lazily (``module`` argument to
``register_lazy``) so that e.g. the API process never imports torch.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class ComponentSpec(BaseModel):
    type: str
    params: dict[str, Any] = Field(default_factory=dict)


class RegistryError(KeyError):
    pass


class Registry(Generic[T]):
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._factories: dict[str, Callable[..., T]] = {}
        self._lazy: dict[str, str] = {}

    def register(self, name: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """Decorator: ``@DETECTORS.register("yolo")``."""

        def decorator(factory: Callable[..., T]) -> Callable[..., T]:
            if name in self._factories:
                raise RegistryError(f"{self.kind} '{name}' registered twice")
            self._factories[name] = factory
            return factory

        return decorator

    def register_lazy(self, name: str, module: str) -> None:
        """Declare that importing ``module`` will register ``name``."""
        self._lazy[name] = module

    def names(self) -> list[str]:
        return sorted(set(self._factories) | set(self._lazy))

    def get(self, name: str) -> Callable[..., T]:
        if name not in self._factories and name in self._lazy:
            importlib.import_module(self._lazy[name])
        try:
            return self._factories[name]
        except KeyError:
            raise RegistryError(
                f"Unknown {self.kind} '{name}'. Available: {self.names()}"
            ) from None

    def create(self, spec: ComponentSpec, **extra: Any) -> T:
        return self.get(spec.type)(**spec.params, **extra)
