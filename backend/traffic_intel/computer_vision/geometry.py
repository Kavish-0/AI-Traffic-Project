"""Small, dependency-light 2-D geometry helpers (vectorised NumPy)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from traffic_intel.core.types import BoolArray, FloatArray


def as_points(points: Sequence[Sequence[float]] | FloatArray) -> FloatArray:
    arr = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    return arr.astype(np.float32)


def scale_points(
    points: FloatArray, from_size: tuple[int, int] | None, to_size: tuple[int, int]
) -> FloatArray:
    """Rescale pixel coordinates between resolutions given as (width, height)."""
    if from_size is None or tuple(from_size) == tuple(to_size):
        return points.astype(np.float32)
    sx = to_size[0] / from_size[0]
    sy = to_size[1] / from_size[1]
    return (points * np.array([sx, sy], dtype=np.float32)).astype(np.float32)


def points_in_polygon(points: FloatArray, polygon: FloatArray) -> BoolArray:
    """Even-odd ray casting for many points against one polygon. Points exactly
    on an edge may fall either way (irrelevant at pixel precision)."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    poly = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
    if len(pts) == 0:
        return np.zeros(0, dtype=bool)
    x = pts[:, 0][:, None]
    y = pts[:, 1][:, None]
    x1, y1 = poly[:, 0][None, :], poly[:, 1][None, :]
    x2, y2 = np.roll(poly[:, 0], -1)[None, :], np.roll(poly[:, 1], -1)[None, :]
    straddles = (y1 > y) != (y2 > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        x_cross = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
    hits = straddles & (x < x_cross)
    return np.asarray(np.count_nonzero(hits, axis=1) % 2 == 1, dtype=np.bool_)


def _cross(ax: np.ndarray, ay: np.ndarray, bx: np.ndarray, by: np.ndarray) -> np.ndarray:
    return np.asarray(ax * by - ay * bx)


def segments_intersect(p: FloatArray, q: FloatArray, a: FloatArray, b: FloatArray) -> BoolArray:
    """For N movement segments p[i]->q[i], does each properly intersect the
    single line segment a->b? Touching at an endpoint counts once on the side
    where the mover ends (q), so a point resting on the line is not
    double-counted over consecutive frames."""
    p64 = np.asarray(p, dtype=np.float64).reshape(-1, 2)
    q64 = np.asarray(q, dtype=np.float64).reshape(-1, 2)
    a64 = np.asarray(a, dtype=np.float64).reshape(2)
    b64 = np.asarray(b, dtype=np.float64).reshape(2)
    ab = b64 - a64
    d1 = _cross(ab[0], ab[1], p64[:, 0] - a64[0], p64[:, 1] - a64[1])   # side of p
    d2 = _cross(ab[0], ab[1], q64[:, 0] - a64[0], q64[:, 1] - a64[1])   # side of q
    pq = q64 - p64
    d3 = _cross(pq[:, 0], pq[:, 1], a64[0] - p64[:, 0], a64[1] - p64[:, 1])
    d4 = _cross(pq[:, 0], pq[:, 1], b64[0] - p64[:, 0], b64[1] - p64[:, 1])
    # p strictly on one side, q on the other side or on the line.
    crosses_line = ((d1 > 0) & (d2 <= 0)) | ((d1 < 0) & (d2 >= 0))
    within_segment = (d3 * d4) <= 0
    return np.asarray(crosses_line & within_segment, dtype=np.bool_)


def expand_polygon(polygon: FloatArray, factor: float) -> FloatArray:
    """Scale a polygon about its centroid (factor 0.2 = 20% larger)."""
    c = polygon.mean(axis=0)
    return ((polygon - c) * (1.0 + factor) + c).astype(np.float32)


def convex_hull(points: FloatArray) -> FloatArray:
    """Monotone-chain convex hull, counter-clockwise, no repeated first point."""
    pts = sorted({(float(x), float(y)) for x, y in np.asarray(points).reshape(-1, 2)})
    if len(pts) <= 2:
        return np.asarray(pts, dtype=np.float32)

    def half(seq: list[tuple[float, float]]) -> list[tuple[float, float]]:
        out: list[tuple[float, float]] = []
        for pt in seq:
            while len(out) >= 2 and (
                (out[-1][0] - out[-2][0]) * (pt[1] - out[-2][1])
                - (out[-1][1] - out[-2][1]) * (pt[0] - out[-2][0])
            ) <= 0:
                out.pop()
            out.append(pt)
        return out

    lower, upper = half(pts), half(pts[::-1])
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float32)


def unit(v: Sequence[float] | FloatArray) -> FloatArray:
    arr = np.asarray(v, dtype=np.float64).reshape(2)
    n = float(np.linalg.norm(arr))
    if n == 0.0:
        raise ValueError("zero-length direction vector")
    return (arr / n).astype(np.float32)
