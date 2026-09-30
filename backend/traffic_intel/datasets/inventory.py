"""Dataset discovery: find every dataset under the data directory and inspect its CONTENT.

    python -m traffic_intel.datasets inventory [--out reports/dataset_inventory.json]

Nothing here knows dataset names. Files are grouped into dataset units: every manifest
``local_path`` that exists is a unit (longest prefix wins), and any other folder with data files
becomes an "unregistered" unit (``<root>/<child>``, e.g. ``raw/public``). Each unit is then
inspected by FORMAT: video headers, image sizes, XML annotation structure, CSV columns / labels /
time ranges, HDF5 tables, NumPy archives, pickles (restricted, never executing code), Parquet
metadata, SQLite tables, JSON(L). The result is written as JSON; docs/DATASET_INVENTORY.md is
written from it.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import pickle
import pickletools
import sqlite3
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

VIDEO = {".mp4", ".avi", ".mkv", ".mov", ".m4v"}
IMAGE = {".jpg", ".jpeg", ".png", ".bmp"}
DOCS = {".md", ".pdf", ".rst"}
CODE = {".py", ".m", ".mexa64", ".mexmaci", ".mexmaci64", ".mexw32", ".mexw64", ".c", ".cpp", ".h", ".sh"}
IGNORED = {".gitkeep", ".ds_store", ".db-shm", ".db-wal"}
MAX_XML_BYTES = 600_000_000          # parse every XML file of a unit up to this total size
MAX_CSV_FILES = 80
MAX_TXT_FILES = 25
IMAGE_SAMPLE = 16

# Globals a pickle may reference to be loaded for inspection: NumPy array reconstruction only
# (``_codecs.encode`` turns a str into bytes - protocol-2 pickles use it for array buffers).
_SAFE_PICKLE = {("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
                ("numpy", "ndarray"), ("numpy", "dtype"), ("numpy.core.multiarray", "scalar"),
                ("numpy._core.multiarray", "scalar"), ("_codecs", "encode")}


@dataclass
class Unit:
    key: str                                   # path relative to the scanned root, "/"-separated
    root: Path
    manifest_ids: list[str] = field(default_factory=list)
    files: list[Path] = field(default_factory=list)

    @property
    def path(self) -> Path:
        return self.root / self.key


# --------------------------------------------------------------------------- grouping
def _manifest_paths() -> dict[str, list[str]]:
    from traffic_intel.datasets.manifest import load_manifest

    out: dict[str, list[str]] = defaultdict(list)
    for dsid, e in load_manifest().datasets.items():
        out[Path(e.local_path).as_posix().strip("/")].append(dsid)
    return out


def discover(data_dir: Path, extra_roots: dict[str, Path] | None = None) -> list[Unit]:
    """Group every file under ``data_dir`` (and ``extra_roots``) into dataset units."""
    manifest = _manifest_paths()
    units: dict[str, Unit] = {}
    roots = {"": data_dir, **(extra_roots or {})}
    for prefix, root in roots.items():
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in {"__pycache__", ".git"}]
            rel_dir = Path(dirpath).relative_to(root).as_posix()
            rel_dir = "" if rel_dir == "." else rel_dir
            for f in filenames:
                if f.lower() in IGNORED or Path(f).suffix.lower() in IGNORED:
                    continue
                rel = f"{rel_dir}/{f}" if rel_dir else f
                if prefix:                                                   # outside data/: one unit per child
                    key = f"{prefix}/{rel.split('/', 1)[0]}" if "/" in rel else prefix
                    u = units.setdefault(key, Unit(key, root.parent))
                else:
                    key = _unit_key(rel, manifest)
                    u = units.setdefault(key, Unit(key, data_dir, list(manifest.get(key, []))))
                u.files.append(Path(dirpath) / f)
    return [u for u in sorted(units.values(), key=lambda u: u.key)
            if any(p.suffix.lower() not in DOCS for p in u.files)]            # README-only folders are not data


def _unit_key(rel: str, manifest: dict[str, list[str]]) -> str:
    parts = rel.split("/")
    for n in range(len(parts) - 1, 0, -1):                                  # longest manifest prefix
        cand = "/".join(parts[:n])
        if cand in manifest:
            return cand
    return "/".join(parts[:2]) if len(parts) > 2 else parts[0] if len(parts) == 2 else "(root)"


# --------------------------------------------------------------------------- inspectors
def inspect_unit(u: Unit) -> dict[str, Any]:
    by_ext: dict[str, list[Path]] = defaultdict(list)
    for p in u.files:
        by_ext[p.suffix.lower() or "(none)"].append(p)
    total = sum(_size(p) for p in u.files)
    rec: dict[str, Any] = {
        "unit": u.key, "path": str(u.path), "manifest_ids": u.manifest_ids, "registered": bool(u.manifest_ids),
        "files": len(u.files), "bytes": total,
        "extensions": dict(Counter({e: len(v) for e, v in by_ext.items()}).most_common()),
        "docs": sorted(p.name for p in u.files if p.suffix.lower() in DOCS or p.name.lower().startswith("readme")),
        "code_files": sum(len(v) for e, v in by_ext.items() if e in CODE),
    }
    findings: dict[str, Any] = {}
    videos = [p for e in VIDEO for p in by_ext.get(e, [])]
    images = [p for e in IMAGE for p in by_ext.get(e, [])]
    if videos:
        findings["video"] = _videos(videos, u.path)
    if images:
        findings["images"] = _images(images, u.path)
    for ext, fn in ((".xml", _xmls), (".csv", _csvs), (".txt", _txts), (".h5", _h5s), (".hdf5", _h5s),
                    (".npz", _npzs), (".pkl", _pickles), (".parquet", _parquets), (".jsonl", _jsonls),
                    (".json", _jsons), (".db", _sqlite), (".sqlite", _sqlite)):
        if by_ext.get(ext):
            try:
                findings[ext.lstrip(".")] = fn(sorted(by_ext[ext]), u.path)
            except Exception as exc:  # noqa: BLE001 - a broken file is itself a finding
                findings[ext.lstrip(".")] = {"error": f"{type(exc).__name__}: {exc}"}
    rec["findings"] = findings
    return rec


def _size(p: Path) -> int:
    try:
        return p.stat().st_size
    except OSError:
        return 0


def _rel(p: Path, base: Path) -> str:
    try:
        return p.relative_to(base).as_posix()
    except ValueError:
        return p.as_posix()


def _videos(files: list[Path], base: Path) -> dict[str, Any]:
    import cv2

    res: Counter[str] = Counter()
    fps_c: Counter[float] = Counter()
    per_dir: Counter[str] = Counter()
    durations: list[float] = []
    bad: list[str] = []
    for p in files:
        cap = cv2.VideoCapture(str(p))
        if not cap.isOpened():
            bad.append(_rel(p, base))
            continue
        w, h, fps, n = int(cap.get(3)), int(cap.get(4)), cap.get(5), cap.get(7)
        cap.release()
        res[f"{w}x{h}"] += 1
        fps_c[round(fps, 2)] += 1
        durations.append(n / fps if fps else 0.0)
        per_dir[_rel(p.parent, base)] += 1
    return {"count": len(files), "unreadable": bad, "resolutions": dict(res),
            "fps": {str(k): v for k, v in fps_c.items()},
            "duration_s": {"min": round(min(durations), 1), "max": round(max(durations), 1),
                           "total_h": round(sum(durations) / 3600, 2)} if durations else None,
            "per_folder": dict(per_dir)}


def _images(files: list[Path], base: Path) -> dict[str, Any]:
    import cv2

    step = max(1, len(files) // IMAGE_SAMPLE)
    sizes: Counter[str] = Counter()
    bad: list[str] = []
    for p in sorted(files)[::step][:IMAGE_SAMPLE]:
        im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if im is None:
            bad.append(_rel(p, base))
        else:
            sizes[f"{im.shape[1]}x{im.shape[0]}"] += 1
    parents = Counter(_rel(p.parent, base) for p in files)
    return {"count": len(files), "sampled_sizes": dict(sizes), "unreadable_in_sample": bad,
            "folders": len(parents), "largest_folders": dict(parents.most_common(5))}


def _xmls(files: list[Path], base: Path) -> dict[str, Any]:
    if sum(_size(p) for p in files) > MAX_XML_BYTES:
        files = files[:20]
    kinds: Counter[str] = Counter()
    track = {"files": 0, "frames": 0, "objects": 0, "track_ids": 0, "ignored_regions": 0}
    attr_values: dict[str, Counter[str]] = defaultdict(Counter)
    seq_attrs: dict[str, Counter[str]] = defaultdict(Counter)
    items: dict[str, dict[str, Any]] = {}
    per_folder: Counter[str] = Counter()
    for p in files:
        per_folder[_rel(p.parent, base)] += 1
        root = _parse_xml(p)
        if root.find(".//frame") is not None and root.find(".//target") is not None:
            kinds["per-frame object annotations (frame/target)"] += 1
            track["files"] += 1
            ids: set[str] = set()
            for fr in root.iter("frame"):
                track["frames"] += 1
                for t in fr.iter("target"):
                    track["objects"] += 1
                    ids.add(t.get("id", ""))
                    a = t.find("attribute")
                    if a is not None and a.get("vehicle_type") is not None:
                        attr_values["vehicle_type"][a.get("vehicle_type", "")] += 1
            track["track_ids"] += len(ids)
            track["ignored_regions"] += sum(1 for r in root.iter("ignored_region") for _ in r.iter("box"))
            sa = root.find("sequence_attribute")
            if sa is not None:
                for k, v in sa.attrib.items():
                    seq_attrs[k][v] += 1
        elif root.find(".//Item") is not None:
            kinds["item list (Item attributes)"] += 1
            its = list(root.iter("Item"))
            keys = sorted({k for it in its for k in it.attrib})
            items[p.name] = {"items": len(its), "attributes": keys,
                             "unique": {k: len({it.get(k) for it in its}) for k in keys if k != "imageName"}}
        else:
            kinds[f"other (root <{root.tag}>)"] += 1
    out: dict[str, Any] = {"count": len(files), "kinds": dict(kinds), "per_folder": dict(per_folder)}
    if track["files"]:
        out["object_annotations"] = {**track, "class_counts": {k: dict(v) for k, v in attr_values.items()},
                                     "sequence_attributes": {k: dict(v) for k, v in seq_attrs.items()}}
    if items:
        out["item_lists"] = items
    return out


def _parse_xml(p: Path) -> ET.Element:
    raw = p.read_bytes()
    try:
        return ET.fromstring(raw)
    except (ET.ParseError, ValueError):         # e.g. encoding="gb2312", which expat cannot decode
        head = raw[:200].decode("ascii", errors="ignore")
        enc = head.split('encoding="', 1)[1].split('"', 1)[0] if 'encoding="' in head else "utf-8"
        text = raw.decode(enc, errors="replace")
        return ET.fromstring(text.split("?>", 1)[1] if text.lstrip().startswith("<?xml") else text)


def _sniff(p: Path) -> str:
    head = p.read_text(encoding="utf-8", errors="replace").splitlines()[0] if _size(p) else ""
    return max((";", ",", "\t"), key=head.count)


def _csvs(files: list[Path], base: Path) -> dict[str, Any]:
    import pandas as pd

    schemas: dict[tuple[str, ...], dict[str, Any]] = {}
    for p in files[:MAX_CSV_FILES]:
        df = pd.read_csv(p, sep=_sniff(p), low_memory=False)
        key = tuple(df.columns)
        s = schemas.setdefault(key, {"files": 0, "rows": 0, "missing": Counter(), "categorical": defaultdict(Counter),
                                     "numeric": {}, "time": {}, "duplicate_rows": 0, "examples": [],
                                     "rows_by_file_name": Counter()})
        s["files"] += 1
        s["rows"] += len(df)
        s["rows_by_file_name"][p.name] += len(df)
        s["duplicate_rows"] += int(df.duplicated().sum())
        if len(s["examples"]) < 3:
            s["examples"].append(_rel(p, base))
        for c in df.columns:
            s["missing"][c] += int(df[c].isna().sum())
            col = df[c]
            if col.dtype == object or pd.api.types.is_string_dtype(col):
                t = _as_time(col)
                if t is not None:
                    lo, hi = s["time"].get(c, (t.min(), t.max()))
                    s["time"][c] = (min(lo, t.min()), max(hi, t.max()))
                elif col.nunique() <= 50:
                    s["categorical"][c].update(col.dropna().astype(str).value_counts().to_dict())
            elif pd.api.types.is_numeric_dtype(col) and len(col.dropna()):
                lo, hi = s["numeric"].get(c, (float(col.min()), float(col.max())))
                s["numeric"][c] = (min(lo, float(col.min())), max(hi, float(col.max())))
    out = []
    for cols, s in schemas.items():
        out.append({"columns": list(cols), "files": s["files"], "rows": s["rows"],
                    "duplicate_rows": s["duplicate_rows"], "rows_by_file_name": dict(s["rows_by_file_name"]),
                    "missing": {k: v for k, v in s["missing"].items() if v},
                    "categorical": {k: dict(Counter(v).most_common(12)) for k, v in s["categorical"].items()},
                    "numeric_range": {k: [round(a, 4), round(b, 4)] for k, (a, b) in s["numeric"].items()},
                    "time_range": {k: [str(a), str(b)] for k, (a, b) in s["time"].items()},
                    "examples": s["examples"]})
    return {"count": len(files), "inspected": min(len(files), MAX_CSV_FILES), "schemas": out}


def _as_time(col: Any) -> Any:
    import pandas as pd

    sample = col.dropna().astype(str).head(20)
    if sample.empty or not sample.str.match(r"^\d{4}-\d{2}-\d{2}").all():
        return None
    return pd.to_datetime(col, errors="coerce", utc=True).dropna()


def _txts(files: list[Path], base: Path) -> dict[str, Any]:
    out = []
    for p in files[:MAX_TXT_FILES]:
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        rows = [ln.split() for ln in lines if ln.strip()]
        numeric = bool(rows) and all(_is_num(x) for r in rows[:50] for x in r)
        widths = Counter(len(r) for r in rows)
        out.append({"file": _rel(p, base), "lines": len(lines), "numeric_table": numeric,
                    "columns": dict(widths.most_common(3)), "first_line": lines[0][:120] if lines else ""})
    return {"count": len(files), "inspected": out}


def _is_num(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def _h5s(files: list[Path], base: Path) -> dict[str, Any]:
    import pandas as pd

    from traffic_intel.datasets.adapters.timeseries import _read_legacy_fixed_frame

    out = []
    for p in files:
        with pd.HDFStore(p, mode="r") as st:
            keys = [k.strip("/") for k in st]
        for k in keys:
            try:
                df = pd.read_hdf(p, key=k)
            except TypeError:
                df = _read_legacy_fixed_frame(str(p), k)
            rec: dict[str, Any] = {"file": _rel(p, base), "key": k, "shape": list(df.shape),
                                   "columns_sample": [str(c) for c in list(df.columns)[:5]],
                                   "index": type(df.index).__name__}
            if isinstance(df.index, pd.DatetimeIndex) and len(df.index) > 1:
                steps = Counter(df.index.to_series().diff().dropna().astype("timedelta64[s]").astype(int))
                rec.update(start=str(df.index.min()), end=str(df.index.max()),
                           step_s=dict(steps.most_common(3)), duplicate_timestamps=int(df.index.duplicated().sum()))
            num = df.select_dtypes("number")
            if not num.empty:
                rec.update(value_min=float(num.min().min()), value_max=float(num.max().max()),
                           nan=int(num.isna().sum().sum()), zeros=int((num == 0).sum().sum()), cells=int(num.size))
            else:
                rec["dtypes"] = {str(c): str(t) for c, t in list(df.dtypes.items())[:17]}
            out.append(rec)
    return {"count": len(files), "tables": out}


def _npzs(files: list[Path], base: Path) -> dict[str, Any]:
    import numpy as np

    out = []
    for p in files:
        with np.load(p, allow_pickle=False) as z:
            for k in z.files:
                a = z[k]
                rec: dict[str, Any] = {"file": _rel(p, base), "array": k, "shape": list(a.shape), "dtype": str(a.dtype)}
                if a.dtype.kind in "fiu" and a.size:
                    rec.update(min=float(np.nanmin(a)), max=float(np.nanmax(a)),
                               nan=int(np.isnan(a).sum()) if a.dtype.kind == "f" else 0)
                    if a.ndim == 3:
                        rec["per_channel_range"] = [[float(np.nanmin(a[..., i])), float(np.nanmax(a[..., i]))]
                                                    for i in range(a.shape[-1])]
                out.append(rec)
    return {"count": len(files), "arrays": out}


class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:  # noqa: ANN401
        if (module, name) not in _SAFE_PICKLE:
            raise pickle.UnpicklingError(f"blocked {module}.{name}")
        return super().find_class(module.replace("numpy.core.", "numpy._core."), name)


def _pickles(files: list[Path], base: Path) -> dict[str, Any]:
    out = []
    for p in files:
        refs = _pickle_globals(p)
        rec: dict[str, Any] = {"file": _rel(p, base), "globals": sorted(".".join(r) for r in refs)}
        if refs <= _SAFE_PICKLE:
            with p.open("rb") as f:
                obj = _RestrictedUnpickler(f, encoding="latin1").load()
            rec["structure"] = _describe(obj)
        else:
            rec["structure"] = "not loaded - references code beyond NumPy arrays (inspected with pickletools only)"
        out.append(rec)
    return {"count": len(files), "files": out}


def _pickle_globals(p: Path) -> set[tuple[str, str]]:
    """Every class/function a pickle would import, read from its opcodes without executing it.
    Protocol 0-3 name them in GLOBAL; protocol 4+ push module and name as strings before
    STACK_GLOBAL (possibly through the memo)."""
    refs: set[tuple[str, str]] = set()
    strings: list[str] = []
    memo: dict[int, str] = {}
    with p.open("rb") as f:
        for op, arg, _ in pickletools.genops(f):
            if op.name == "GLOBAL" and isinstance(arg, str):
                mod, _, name = arg.partition(" ")
                refs.add((mod, name))
            elif isinstance(arg, str) and "UNICODE" in op.name:
                strings.append(arg)
            elif op.name == "MEMOIZE" and strings:
                memo[len(memo)] = strings[-1]
            elif op.name in ("BINGET", "LONG_BINGET", "GET") and isinstance(arg, int) and arg in memo:
                strings.append(memo[arg])
            elif op.name == "STACK_GLOBAL" and len(strings) >= 2:
                refs.add((strings[-2], strings[-1]))
    return refs


def _describe(obj: Any, depth: int = 0) -> Any:  # noqa: ANN401
    import numpy as np

    if isinstance(obj, np.ndarray):
        return {"ndarray": list(obj.shape), "dtype": str(obj.dtype), "nonzero": int(np.count_nonzero(obj))}
    if isinstance(obj, (list, tuple)):
        return {type(obj).__name__: len(obj),
                "items": [_describe(x, depth + 1) for x in obj[:3]] if depth < 2 else "..."}
    if isinstance(obj, dict):
        return {"dict": len(obj), "sample_keys": [str(k) for k in list(obj)[:3]]}
    return type(obj).__name__


def _parquets(files: list[Path], base: Path) -> dict[str, Any]:
    import pyarrow.parquet as pq

    schemas: dict[tuple[str, ...], dict[str, Any]] = {}
    for p in files:
        md = pq.read_metadata(p)
        cols = tuple(md.schema.names)
        s = schemas.setdefault(cols, {"files": 0, "rows": 0, "examples": []})
        s["files"] += 1
        s["rows"] += md.num_rows
        if len(s["examples"]) < 2:
            s["examples"].append(_rel(p, base))
    return {"count": len(files), "schemas": [{"columns": list(k), **v} for k, v in schemas.items()]}


def _jsonls(files: list[Path], base: Path) -> dict[str, Any]:
    lines = 0
    keys: Counter[str] = Counter()
    per_name = Counter(p.name for p in files)
    for p in files:
        for ln in p.read_text(encoding="utf-8", errors="replace").splitlines():
            if ln.strip():
                lines += 1
                if lines <= 50:
                    with contextlib.suppress(json.JSONDecodeError, AttributeError):
                        keys.update(json.loads(ln).keys())
    return {"count": len(files), "records": lines, "by_file_name": dict(per_name), "keys_sample": sorted(keys)[:25]}


def _jsons(files: list[Path], base: Path) -> dict[str, Any]:
    names = Counter(p.name for p in files)
    sample = {}
    for p in files[:3]:
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            sample[_rel(p, base)] = sorted(obj)[:15] if isinstance(obj, dict) else type(obj).__name__
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            sample[_rel(p, base)] = f"unreadable: {exc}"
    return {"count": len(files), "by_file_name": dict(names), "top_level_keys": sample}


def _sqlite(files: list[Path], base: Path) -> dict[str, Any]:
    out = {}
    for p in files:
        con = sqlite3.connect(f"file:{p.as_posix()}?mode=ro", uri=True)
        try:
            tables = [r[0] for r in con.execute("select name from sqlite_master where type='table' order by name")]
            out[_rel(p, base)] = {t: con.execute(f'select count(*) from "{t}"').fetchone()[0] for t in tables}
        finally:
            con.close()
    return {"count": len(files), "tables": out}


# --------------------------------------------------------------------------- entry point
def build_inventory(data_dir: Path | None = None, project_root: Path | None = None) -> dict[str, Any]:
    from datetime import UTC, datetime

    from traffic_intel.core.config import PROJECT_ROOT, get_settings

    data_dir = Path(data_dir or get_settings().data_dir)
    project_root = Path(project_root or PROJECT_ROOT)
    extra = {"runs": project_root / "runs"} if (project_root / "runs").is_dir() else {}
    units = discover(data_dir, extra)
    records = []
    for u in units:
        log.info("inspecting %s (%d files)", u.key, len(u.files))
        records.append(inspect_unit(u))
    from traffic_intel.datasets.manifest import load_manifest

    present = {i for r in records for i in r["manifest_ids"]}
    return {"generated": datetime.now(UTC).isoformat(timespec="seconds"), "data_dir": str(data_dir),
            "units": records,
            "manifest_entries_without_files": sorted(set(load_manifest().datasets) - present)}


def summary_lines(inv: dict[str, Any]) -> list[str]:
    lines = [f"{'unit':<40} {'manifest id':<24} {'files':>7} {'GB':>7}  formats"]
    for r in inv["units"]:
        ids = ",".join(r["manifest_ids"]) or "(unregistered)"
        fmts = ", ".join(f"{e}:{n}" for e, n in list(r["extensions"].items())[:4])
        lines.append(f"{r['unit']:<40} {ids:<24} {r['files']:>7} {r['bytes'] / 1e9:>7.3f}  {fmts}")
    if inv["manifest_entries_without_files"]:
        lines.append(f"manifest entries with no files on disk: {', '.join(inv['manifest_entries_without_files'])}")
    return lines


def write_inventory(out: Path, **kw: Any) -> dict[str, Any]:  # noqa: ANN401
    inv = build_inventory(**kw)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(inv, indent=2, default=str), encoding="utf-8")
    return inv


__all__ = ["build_inventory", "discover", "inspect_unit", "summary_lines", "write_inventory"]
