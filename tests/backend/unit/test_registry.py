from __future__ import annotations

import pytest

from traffic_intel.core.registry import ComponentSpec, Registry, RegistryError


class Base:
    def __init__(self, x: int = 0) -> None:
        self.x = x


def test_register_and_create() -> None:
    reg: Registry[Base] = Registry("thing")
    reg.register("a")(Base)
    obj = reg.create(ComponentSpec(type="a", params={"x": 3}))
    assert isinstance(obj, Base) and obj.x == 3


def test_unknown_and_duplicate() -> None:
    reg: Registry[Base] = Registry("thing")
    reg.register("a")(Base)
    with pytest.raises(RegistryError):
        reg.register("a")(Base)
    with pytest.raises(RegistryError, match="Available"):
        reg.get("missing")


def test_planned_implementations_are_declared() -> None:
    from traffic_intel.computer_vision.interfaces import DETECTORS
    from traffic_intel.forecasting.interfaces import FORECASTERS

    assert "yolo" in DETECTORS.names()
    assert {"persistence", "linear", "random_forest", "xgboost", "lstm", "transformer"} <= set(
        FORECASTERS.names()
    )
