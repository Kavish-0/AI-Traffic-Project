"""Database engine (synchronous SQLAlchemy 2.0).

URL from ``TI_DATABASE_URL`` (``.env``). Without it the project uses a local SQLite file
(``<data_dir>/db/traffic.db``) so that everything runs with zero setup; point it at
PostgreSQL (``postgresql+psycopg://user:pass@host:5432/db``) for the full deployment - PostGIS
geometry columns are used automatically when the extension is available.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine

from traffic_intel.core.config import AppSettings, get_settings
from traffic_intel.database.types import mark_postgis

log = logging.getLogger(__name__)


def database_url(settings: AppSettings | None = None) -> str:
    s = settings or get_settings()
    if s.database_url:
        return s.database_url
    path = Path(s.data_dir) / "db" / s.database_file
    path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{path.as_posix()}"


def create_db_engine(url: str | None = None) -> Engine:
    url = url or database_url()
    if url.startswith("sqlite"):
        engine = create_engine(url, connect_args={"check_same_thread": False, "timeout": 30})

        @event.listens_for(engine, "connect")
        def _pragmas(dbapi_conn: Any, _: Any) -> None:     # WAL: the API reads while the pipeline writes
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA foreign_keys=ON")
            cur.close()
    else:
        engine = create_engine(url, pool_pre_ping=True)
        postgis = mark_postgis(engine)
        log.info("database: PostgreSQL (%s)", "PostGIS geometry" if postgis else "no PostGIS - geometry stored as WKT")
    return engine

