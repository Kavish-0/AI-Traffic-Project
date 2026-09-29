"""Portable column types: one ORM schema for PostgreSQL + PostGIS (production), plain PostgreSQL
and SQLite (zero-setup local development and tests).

    JSONType    JSONB on PostgreSQL, JSON elsewhere
    GeoType     PostGIS ``geometry(<type>, 4326)`` when the connected PostgreSQL has PostGIS,
                otherwise WKT text. Python side is ALWAYS a WKT string ("POINT(lon lat)").
    enum_type   native ENUM on PostgreSQL, VARCHAR + CHECK elsewhere (values = enum values)

PostGIS availability is detected once per engine (``mark_postgis``) and stored on the dialect.
"""

from __future__ import annotations

import struct
from enum import StrEnum
from typing import Any

from sqlalchemy import JSON, BigInteger, Integer, Text
from sqlalchemy import Enum as SAEnum
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import Dialect
from sqlalchemy.types import TypeDecorator

JSONType = JSON().with_variant(JSONB(), "postgresql")
# 64-bit auto-increment key; SQLite only auto-increments "INTEGER PRIMARY KEY"
BigIntPK = BigInteger().with_variant(Integer(), "sqlite")


def enum_type(enum_cls: type[StrEnum], name: str) -> SAEnum:
    return SAEnum(enum_cls, name=name, values_callable=lambda e: [m.value for m in e])


def _postgis(dialect: Dialect) -> bool:
    return dialect.name == "postgresql" and bool(getattr(dialect, "ti_postgis", False))


class GeoType(TypeDecorator[str]):
    """Geometry as WKT in Python; PostGIS geometry or TEXT in the database."""

    impl = Text
    cache_ok = True

    def __init__(self, geometry_type: str = "GEOMETRY", srid: int = 4326) -> None:
        super().__init__()
        self.geometry_type, self.srid = geometry_type, srid

    def load_dialect_impl(self, dialect: Dialect) -> Any:
        if _postgis(dialect):
            from geoalchemy2 import Geometry

            return dialect.type_descriptor(Geometry(self.geometry_type, srid=self.srid))
        return dialect.type_descriptor(Text())

    def process_bind_param(self, value: Any, dialect: Dialect) -> Any:
        if value is None or not _postgis(dialect):
            return value
        from geoalchemy2 import WKTElement

        return WKTElement(value, srid=self.srid)

    def process_result_value(self, value: Any, dialect: Dialect) -> str | None:
        if value is None or isinstance(value, str):
            return value
        data = getattr(value, "data", value)                   # geoalchemy2 WKBElement
        return wkb_to_wkt(bytes.fromhex(data) if isinstance(data, str) else bytes(data))


def mark_postgis(engine: Any) -> bool:
    """Detect PostGIS on a PostgreSQL engine (creating the extension if allowed)."""
    if engine.dialect.name != "postgresql":
        return False
    from sqlalchemy import text

    ok = False
    with engine.begin() as c:
        if c.execute(text("SELECT 1 FROM pg_available_extensions WHERE name = 'postgis'")).first():
            try:
                c.execute(text("CREATE EXTENSION IF NOT EXISTS postgis"))
                ok = True
            except Exception:  # noqa: BLE001 - no privilege: fall back to WKT text
                ok = False
    engine.dialect.ti_postgis = ok
    return ok


# --------------------------------------------------------------------------- WKT helpers
def point_wkt(lon: float, lat: float) -> str:
    return f"POINT({lon:.7f} {lat:.7f})"


def line_wkt(coords: list[tuple[float, float]] | list[list[float]]) -> str:
    return "LINESTRING(" + ", ".join(f"{x:.7f} {y:.7f}" for x, y in coords) + ")"


def wkt_coords(wkt: str | None) -> list[tuple[float, float]]:
    """Coordinates of a POINT / LINESTRING / POLYGON (outer ring) WKT."""
    if not wkt:
        return []
    inner = wkt[wkt.index("(") :].strip("() ")
    inner = inner.split("),")[0].strip("() ")
    out = []
    for p in inner.split(","):
        if p.strip():
            x, y = p.split()[:2]
            out.append((float(x), float(y)))
    return out


def point_coords(wkt: str | None) -> tuple[float | None, float | None]:
    """(lon, lat) of a POINT WKT, (None, None) when there is no geometry."""
    coords = wkt_coords(wkt)
    return coords[0] if coords else (None, None)


def wkb_to_wkt(b: bytes) -> str:
    """Minimal (E)WKB -> WKT for Point, LineString and Polygon (what the schema stores)."""
    endian = "<" if b[0] == 1 else ">"
    gtype = struct.unpack(endian + "I", b[1:5])[0]
    off = 5
    if gtype & 0x20000000:                                   # EWKB with SRID
        off += 4
    kind = gtype & 0xFF

    def pts(n: int, o: int) -> tuple[list[tuple[float, float]], int]:
        vals = struct.unpack(endian + "d" * 2 * n, b[o:o + 16 * n])
        return [(vals[2 * i], vals[2 * i + 1]) for i in range(n)], o + 16 * n

    if kind == 1:
        (x, y), _ = struct.unpack(endian + "dd", b[off:off + 16]), None
        return f"POINT({x} {y})"
    if kind == 2:
        n = struct.unpack(endian + "I", b[off:off + 4])[0]
        p, _ = pts(n, off + 4)
        return "LINESTRING(" + ", ".join(f"{x} {y}" for x, y in p) + ")"
    if kind == 3:
        rings = struct.unpack(endian + "I", b[off:off + 4])[0]
        o, parts = off + 4, []
        for _ in range(rings):
            n = struct.unpack(endian + "I", b[o:o + 4])[0]
            p, o = pts(n, o + 4)
            parts.append("(" + ", ".join(f"{x} {y}" for x, y in p) + ")")
        return "POLYGON(" + ", ".join(parts) + ")"
    raise ValueError(f"unsupported WKB geometry type {kind}")
