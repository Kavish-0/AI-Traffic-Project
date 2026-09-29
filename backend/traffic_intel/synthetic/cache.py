"""Disk cache + parallel generation of synthetic scenes (long scenes take tens of seconds to simulate).

Cache key = SHA-256 of the scene config; files under ``<data_dir>/synthetic/cache/``. Scenes are
deterministic for a config, so a cached scene is exactly what generation would produce.
"""

from __future__ import annotations

import hashlib
import logging
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from traffic_intel.core.config import get_settings
from traffic_intel.synthetic.scene import SceneConfig, SyntheticScene, generate_scene

log = logging.getLogger(__name__)


def _path(cfg: SceneConfig) -> Path:
    key = hashlib.sha256(cfg.model_dump_json().encode()).hexdigest()[:16]
    return Path(get_settings().data_dir) / "synthetic" / "cache" / f"{cfg.name}-{key}.pkl"


def _generate(cfg: SceneConfig) -> SyntheticScene:
    return generate_scene(cfg)


def load_or_generate(cfgs: list[SceneConfig]) -> dict[str, SyntheticScene]:
    """name -> scene; missing scenes are generated in parallel processes and cached."""
    out: dict[str, SyntheticScene] = {}
    todo = []
    for c in cfgs:
        f = _path(c)
        if f.exists():
            with f.open("rb") as fh:
                out[c.name] = pickle.load(fh)  # noqa: S301 - our own cache file
        else:
            todo.append(c)
    if todo:
        log.info("generating %d synthetic scene(s) in parallel (first run only; cached afterwards)", len(todo))
        with ProcessPoolExecutor(max_workers=min(len(todo), 8)) as ex:
            for c, scene in zip(todo, ex.map(_generate, todo), strict=True):
                f = _path(c)
                f.parent.mkdir(parents=True, exist_ok=True)
                with f.open("wb") as fh:
                    pickle.dump(scene, fh, protocol=pickle.HIGHEST_PROTOCOL)
                out[c.name] = scene
    return out
