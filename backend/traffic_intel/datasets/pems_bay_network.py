"""PEMS-BAY sensors -> road network config for the real-data runtime mode.

    python -m traffic_intel.datasets pems-bay-network          # -> configs/network/pems_bay.yaml

Every PEMS-BAY loop-detector station (325, all in the South Bay / San Jose) becomes one monitored
approach at its REAL location. Stations of one freeway and direction (e.g. US-101 northbound) are
chained in travel order (post mile), so each approach's road segment runs from the previous station to
this one; opposite carriageways are drawn a few metres apart. Each freeway direction is one "junction"
(a corridor, not a signalised intersection) so the dashboard can group its stations.

Measured from the data: free-flow speed per station = 90th percentile of its speeds in the TRAINING
part of the recording (first 70%, the same split the forecaster uses), clipped to 60..120 km/h.

Assumed (not in the data, labelled in the names): interchange connectors between freeways where two
corridors pass within ``connector_max_m`` of each other, and a one-lane local-street detour beside every
freeway segment (ids ``detour_*``), so what-if closures have alternative routes. The dashboard hides the
detours on the map unless a what-if run puts traffic on them.

The station metadata file stores its text columns (direction, name) as a pickled NumPy object array;
it is read with the inventory's restricted unpickler (NumPy arrays only), never with plain pickle.
"""

from __future__ import annotations

import io
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

FREEWAYS = {17: "SR-17", 85: "SR-85", 87: "SR-87", 101: "US-101", 237: "SR-237", 280: "I-280", 680: "I-680",
            880: "I-880"}
DIRECTIONS = {"N": "northbound", "S": "southbound", "E": "eastbound", "W": "westbound"}
SPEED_LIMIT_KMH = 105.0            # 65 mph, the usual South Bay freeway limit
_M_PER_DEG_LAT = 111_320.0


def _decode(v: Any) -> Any:  # noqa: ANN401 - HDF5 array element
    return v.decode("utf-8") if isinstance(v, bytes | np.bytes_) else v


def read_sensor_meta(path: Path | str) -> pd.DataFrame:
    """pems-bay-meta.h5 (pandas 0.x 'fixed' frame) -> one row per station, index = station id (str).

    Columns: fwy, dir, name, lat, lon, abs_pm, lanes, type."""
    import tables

    from traffic_intel.datasets.inventory import _RestrictedUnpickler

    with tables.open_file(str(path), mode="r") as h:
        g = h.get_node("/meta")
        ids = [str(_decode(i)) for i in g.axis1.read()]
        blocks = {}
        for b in range(3):
            items = [_decode(c) for c in getattr(g, f"block{b}_items").read()]
            node = getattr(g, f"block{b}_values")
            if isinstance(node.atom, tables.ObjectAtom):
                node.atom.fromarray = lambda a: a.tobytes()        # raw pickle bytes, no unpickling here
                values = _RestrictedUnpickler(io.BytesIO(node.read(0, 1)[0]), encoding="latin1").load()
            else:
                values = node.read()
            blocks.update({c: values[:, k] for k, c in enumerate(items)})
    meta = pd.DataFrame({
        "fwy": pd.Series(blocks["Fwy"]).astype(int).to_numpy(),
        "dir": [str(d) for d in blocks["Dir"]],
        "name": [str(n).strip() for n in blocks["Name"]],
        "lat": pd.Series(blocks["Latitude"]).astype(float).to_numpy(),
        "lon": pd.Series(blocks["Longitude"]).astype(float).to_numpy(),
        "abs_pm": pd.Series(blocks["Abs_PM"]).astype(float).to_numpy(),
        "lanes": pd.Series(blocks["Lanes"]).astype(int).to_numpy(),
        "type": [str(t) for t in blocks["Type"]],
    }, index=pd.Index(ids, name="sensor"))
    return meta


def free_flow_speeds(speeds_kmh: pd.DataFrame, train_fraction: float = 0.7, q: float = 0.9) -> dict[str, float]:
    """``speeds_kmh``: wide matrix (rows = time, columns = station id, NaN = missing)."""
    train = speeds_kmh.iloc[: int(len(speeds_kmh) * train_fraction)]
    ff = train.quantile(q).clip(60.0, 120.0).fillna(SPEED_LIMIT_KMH)
    return {str(k): round(float(v), 1) for k, v in ff.items()}


def _metres(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    """Local east/north metres from lon/lat a to b."""
    lat0 = math.radians((a[1] + b[1]) / 2)
    return (b[0] - a[0]) * _M_PER_DEG_LAT * math.cos(lat0), (b[1] - a[1]) * _M_PER_DEG_LAT


def _shift(p: tuple[float, float], east_m: float, north_m: float) -> tuple[float, float]:
    return (p[0] + east_m / (_M_PER_DEG_LAT * math.cos(math.radians(p[1]))), p[1] + north_m / _M_PER_DEG_LAT)


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(*_metres(a, b))


_KEEP_UPPER = {"NB", "SB", "EB", "WB", "SR", "US", "I", "HOV", "CA"}


def _title(name: str) -> str:
    return " ".join(w.upper() if w.upper() in _KEEP_UPPER or any(c.isdigit() for c in w) else w.capitalize()
                    for w in name.split())


def corridor_id(fwy: int, d: str) -> str:
    return f"pems_bay_{FREEWAYS.get(fwy, str(fwy)).lower().replace('-', '')}_{d.lower()}"


def build_network(meta: pd.DataFrame, free_flow: dict[str, float], *, offset_m: float = 12.0,
                  connector_max_m: float = 800.0, detour_offset_m: float = 120.0) -> dict[str, Any]:
    junctions: list[dict[str, Any]] = []
    nodes: list[dict[str, Any]] = []
    segments: list[dict[str, Any]] = []
    approaches: list[dict[str, Any]] = []
    corridor_nodes: dict[str, list[tuple[str, tuple[float, float]]]] = {}

    for (fwy, d), grp in meta.groupby(["fwy", "dir"], sort=True):
        fwy, d = int(fwy), str(d)
        grp = grp.sort_values("abs_pm", ascending=d in ("N", "E"))           # travel order
        pts = [(float(r.lon), float(r.lat)) for r in grp.itertuples()]
        # upstream end of the first segment: half the first spacing (>= 300 m) back along the road
        if len(pts) > 1:
            e, n = _metres(pts[1], pts[0])
            scale = max(0.5, 300.0 / max(math.hypot(e, n), 1.0))
            first = _shift(pts[0], e * scale, n * scale)
        else:
            first = _shift(pts[0], 0.0, -300.0 if d == "N" else 300.0)
        chain = [first, *pts]
        # draw each carriageway offset_m to the right of its travel direction
        shifted = []
        for i, p in enumerate(chain):
            a, b = chain[max(0, i - 1)], chain[min(len(chain) - 1, i + 1)]
            e, n = _metres(a, b)
            norm = math.hypot(e, n) or 1.0
            shifted.append(_shift(p, n / norm * offset_m, -e / norm * offset_m))
        cid = corridor_id(fwy, d)
        road = f"{FREEWAYS.get(fwy, fwy)} {d}"
        mid = shifted[len(shifted) // 2]
        junctions.append({"id": cid, "name": f"{FREEWAYS.get(fwy, fwy)} {DIRECTIONS.get(d, d)}",
                          "location": [round(mid[0], 6), round(mid[1], 6)], "signalized": False})
        start_id = f"{cid}_start"
        nodes.append({"id": start_id, "location": [round(shifted[0][0], 6), round(shifted[0][1], 6)]})
        prev = start_id
        corridor_nodes[cid] = []
        for (sid, row), loc in zip(grp.iterrows(), shifted[1:], strict=True):
            nid, seg_id, aid = f"n_pems_bay_{sid}", f"seg_pems_bay_{sid}", f"pems_bay_{sid}"
            prev_loc = next(n["location"] for n in nodes if n["id"] == prev)
            nodes.append({"id": nid, "junction_id": cid, "location": [round(loc[0], 6), round(loc[1], 6)]})
            label = _title(row["name"]) or f"station {sid}"
            segments.append({
                "id": seg_id, "name": f"{road} · {label}", "from_node": prev, "to_node": nid,
                "geometry": [prev_loc, [round(loc[0], 6), round(loc[1], 6)]],
                "lanes": max(1, int(row["lanes"])), "speed_limit_kmh": SPEED_LIMIT_KMH,
                "free_flow_speed_kmh": free_flow.get(str(sid), SPEED_LIMIT_KMH), "importance": 0.9})
            approaches.append({"id": aid, "junction_id": cid, "road_segment_id": seg_id,
                               "name": f"{road} · {label} ({sid})", "direction": DIRECTIONS.get(d, d)})
            # assumed local-street detour around this freeway segment (exit before it, re-enter after it), so
            # a what-if closure has somewhere to go; via its own mid node because the road graph is a DiGraph
            e, n = _metres(tuple(prev_loc), loc)
            norm = math.hypot(e, n) or 1.0
            mid_loc = _shift(((prev_loc[0] + loc[0]) / 2, (prev_loc[1] + loc[1]) / 2),
                             n / norm * detour_offset_m, -e / norm * detour_offset_m)
            mid_pt = [round(mid_loc[0], 6), round(mid_loc[1], 6)]
            did = f"n_detour_pems_bay_{sid}"
            nodes.append({"id": did, "location": mid_pt})
            for part, (a_id, a_loc, b_id, b_loc) in (("a", (prev, prev_loc, did, mid_pt)),
                                                       ("b", (did, mid_pt, nid, [round(loc[0], 6), round(loc[1], 6)]))):
                segments.append({
                    "id": f"detour_pems_bay_{sid}_{part}", "name": f"Local streets around {road} · {label} "
                    "(assumed detour)", "from_node": a_id, "to_node": b_id, "geometry": [a_loc, b_loc],
                    "lanes": 1, "speed_limit_kmh": 40.0})
            corridor_nodes[cid].append((nid, loc))
            prev = nid

    # assumed interchange connectors (no approach, no measurements) - one per corridor pair
    fwy_of = {corridor_id(int(f), str(d)): int(f) for f, d in meta[["fwy", "dir"]].drop_duplicates().to_numpy()}
    for ca, na in corridor_nodes.items():
        for cb, nb in corridor_nodes.items():
            if fwy_of[ca] == fwy_of[cb]:
                continue
            best = min(((_dist(pa, pb), ia, ib, pa, pb) for ia, pa in na for ib, pb in nb), default=None)
            if best is None or best[0] > connector_max_m:
                continue
            _, ia, ib, pa, pb = best
            segments.append({
                "id": f"conn_{ca}_{cb}", "name": f"Interchange {ca.split('_')[2].upper()} to "
                                                f"{cb.split('_')[2].upper()} (assumed connector)",
                "from_node": ia, "to_node": ib, "geometry": [[round(pa[0], 6), round(pa[1], 6)],
                                                             [round(pb[0], 6), round(pb[1], 6)]],
                "lanes": 1, "speed_limit_kmh": 60.0})

    return {"name": "pems_bay", "junctions": junctions, "nodes": nodes, "segments": segments,
            "approaches": approaches}


HEADER = """\
# GENERATED by: python -m traffic_intel.datasets pems-bay-network   (do not edit by hand - regenerate)
# PEMS-BAY real-data network: {n} loop-detector stations at their real locations (South Bay, CA),
# chained per freeway direction in post-mile order. Free-flow speed per station = 90th percentile of
# its speeds in the training period (first 70% of the recording). Segments named "(assumed connector)"
# are NOT in the data: they join freeways that pass close to each other; "(assumed detour)" segments are
# one-lane local streets beside each freeway segment. Both exist only so what-if closures can reroute.
"""


def write_network(out: Path, *, adapter: Any = None) -> dict[str, Any]:
    from traffic_intel.core.config import NetworkConfig
    from traffic_intel.datasets.adapters import adapter_for

    adapter = adapter or adapter_for("pems_bay")
    meta_path = Path(adapter.h5_file()).with_name("pems-bay-meta.h5")
    meta = read_sensor_meta(meta_path)
    ds = adapter.load(weather=False, holiday_country=None)
    wide = ds.frame.pivot(index="timestamp", columns="series_id", values="avg_speed_kmh")
    wide.columns = [c.removeprefix("pems_bay_") for c in wide.columns]
    meta = meta.loc[[s for s in meta.index if s in wide.columns]]
    net = build_network(meta, free_flow_speeds(wide))
    NetworkConfig.model_validate(net)                    # fail here, not at runtime start
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(HEADER.format(n=len(net["approaches"])) + yaml.safe_dump(net, sort_keys=False, width=140),
                   encoding="utf-8")
    return net
