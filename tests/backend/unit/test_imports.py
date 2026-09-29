"""Every module in the package must import without optional (cv/ml) extras."""

from __future__ import annotations

import importlib
import pkgutil

import traffic_intel


def test_all_modules_import() -> None:
    skip = {"traffic_intel.database.migrations.env"}  # runs alembic context on import
    for mod in pkgutil.walk_packages(traffic_intel.__path__, "traffic_intel."):
        if mod.name not in skip:
            importlib.import_module(mod.name)
