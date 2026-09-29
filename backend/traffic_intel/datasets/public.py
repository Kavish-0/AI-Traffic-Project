"""Catalog-driven download of public development footage (``configs/datasets/public_videos.yaml``).

Nothing here is referenced by the perception pipeline: a downloaded file is
used only by pointing a camera's ``source`` at it (or ``replay --source``).
"""

from __future__ import annotations

import hashlib
import logging
from datetime import UTC, datetime
from pathlib import Path

import httpx
from pydantic import BaseModel

from traffic_intel.core.config import PROJECT_ROOT, get_settings, load_yaml, resolve_path

log = logging.getLogger(__name__)


class PublicVideo(BaseModel):
    url: str
    file: str
    source: str
    license: str
    description: str = ""
    sha256: str | None = None


CATALOGS = {"videos": "public_videos.yaml", "timeseries": "public_timeseries.yaml"}


def load_catalog(path: Path | None = None, catalog: str = "videos") -> dict[str, PublicVideo]:
    """Entries of a catalog file (``videos:`` or ``files:`` mapping)."""
    path = path or get_settings().config_dir / "datasets" / CATALOGS[catalog]
    raw = load_yaml(path)
    entries = raw.get("videos") or raw.get("files") or {}
    return {k: PublicVideo.model_validate(v) for k, v in entries.items()}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(name: str, video: PublicVideo, *, force: bool = False) -> Path:
    dest = resolve_path(video.file)
    if dest.exists() and not force:
        log.info("%s already present at %s", name, dest)
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        with httpx.stream("GET", video.url, follow_redirects=True, timeout=60.0) as r:
            r.raise_for_status()
            with tmp.open("wb") as fh:
                for chunk in r.iter_bytes(1 << 20):
                    fh.write(chunk)
        tmp.replace(dest)
        _record_source(name, video, dest)
    if video.sha256 and sha256_of(dest) != video.sha256:
        raise ValueError(f"{name}: sha256 mismatch for {dest} - file changed upstream or is corrupt")
    return dest


def _record_source(name: str, video: PublicVideo, dest: Path) -> None:
    sources = PROJECT_ROOT / "data" / "external" / "SOURCES.md"
    sources.parent.mkdir(parents=True, exist_ok=True)
    if not sources.exists():
        sources.write_text("# External data sources\n\n", encoding="utf-8")
    with sources.open("a", encoding="utf-8") as fh:
        fh.write(
            f"- **{name}** -> `{dest.relative_to(PROJECT_ROOT).as_posix()}`  \n"
            f"  source: {video.source}; url: {video.url}  \n"
            f"  license: {video.license}; sha256: {sha256_of(dest)}; "
            f"downloaded {datetime.now(UTC).date().isoformat()}\n"
        )
