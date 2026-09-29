"""Dataset CLI.

Manifest (configs/datasets/manifest.yaml) - nothing is downloaded automatically:
    python -m traffic_intel.datasets status [dataset_id ...]     present / missing, with instructions
    python -m traffic_intel.datasets info <dataset_id>           source, licence, layout, citation
    python -m traffic_intel.datasets experiments                 which dataset feeds which experiment
    python -m traffic_intel.datasets layout                      create the data/ folder structure
    python -m traffic_intel.datasets prepare <dataset_id> [--max-sequences N] [--every-k K] [--max-sensors N]

Discovery and quality (every dataset actually on disk, inspected by content):
    python -m traffic_intel.datasets inventory [--out reports/dataset_inventory.json]
    python -m traffic_intel.datasets quality   [--out reports/data_quality_report.json]

Small public files with a verified checksum (catalogs in configs/datasets/public_*.yaml):
    python -m traffic_intel.datasets [--catalog videos|timeseries] list | download <name>... | download --all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from traffic_intel.core.config import get_settings
from traffic_intel.core.logging import configure_logging

DATA_LAYOUT = ["raw/uadetrac", "raw/aicity", "raw/anomaly", "raw/metr_la", "raw/pems_bay", "raw/weather",
               "raw/public", "interim", "processed", "annotations", "features", "predictions", "detections",
               "tracks", "recordings", "external", "labels"]


def cmd_status(ids: list[str]) -> int:
    from traffic_intel.datasets.manifest import load_manifest, missing_message, validate

    m = load_manifest()
    missing = 0
    for dsid in ids or list(m.datasets):
        e = m.get(dsid)
        r = validate(e)
        print(r.summary())
        if not r.ok:
            missing += 1
            if ids:
                print("\n" + missing_message(e, r) + "\n")
    if missing and not ids:
        print(f"\n{missing} dataset(s) missing - `status <id>` shows how to obtain each (licence first).")
    return 1 if (ids and missing) else 0


def cmd_info(dsid: str) -> None:
    from traffic_intel.datasets.manifest import dataset_root, load_manifest, validate

    e = load_manifest().get(dsid)
    print(f"{e.name}  [{dsid}]\n  tasks: {', '.join(e.tasks)}\n  source: {e.source}\n  version: {e.version}")
    print(f"  licence: {e.license.name} (access: {e.license.access}, redistribution: {e.license.redistribution})")
    print(f"  size: {e.size}\n  local path: {dataset_root(e)}  -> {'present' if validate(e).ok else 'MISSING'}")
    print(f"  format: {e.expected_format}\n  preprocessing: {e.preprocessing}")
    for k in ("classes", "features", "labels"):
        v = getattr(e, k)
        if v:
            print(f"  {k}: {json.dumps(v)}")
    print(f"  splits: {e.splits}\n  download:\n    " + e.download.strip().replace("\n", "\n    "))
    print(f"  citation: {e.citation}")


def cmd_experiments() -> None:
    from traffic_intel.datasets.manifest import load_manifest

    for name, x in load_manifest().experiments.items():
        print(f"{name}\n  purpose:  {x.purpose}\n  datasets: {', '.join(x.datasets)}")
        print(f"  metrics:  {', '.join(x.metrics)}")


def cmd_layout() -> None:
    root = get_settings().data_dir
    for d in DATA_LAYOUT:
        (root / d).mkdir(parents=True, exist_ok=True)
        (root / d / ".gitkeep").touch()
    print(f"data layout ready under {root}")


def main() -> None:
    p = argparse.ArgumentParser(prog="traffic_intel.datasets")
    p.add_argument("--catalog", default="videos", choices=["videos", "timeseries"])
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    d = sub.add_parser("download")
    d.add_argument("names", nargs="*")
    d.add_argument("--all", action="store_true")
    d.add_argument("--force", action="store_true")
    s = sub.add_parser("status")
    s.add_argument("ids", nargs="*")
    i = sub.add_parser("info")
    i.add_argument("dataset_id")
    sub.add_parser("experiments")
    sub.add_parser("layout")
    inv = sub.add_parser("inventory")
    inv.add_argument("--out", default=None, help="JSON output (default: reports/dataset_inventory.json)")
    q = sub.add_parser("quality")
    q.add_argument("--out", default=None, help="JSON output (default: reports/data_quality_report.json)")
    pr = sub.add_parser("prepare")
    pr.add_argument("dataset_id")
    pr.add_argument("--max-sequences", type=int)
    pr.add_argument("--every-k", type=int)
    pr.add_argument("--max-sensors", type=int)
    a = p.parse_args()
    configure_logging()

    if a.cmd == "status":
        sys.exit(cmd_status(a.ids))
    if a.cmd == "info":
        cmd_info(a.dataset_id)
        return
    if a.cmd == "experiments":
        cmd_experiments()
        return
    if a.cmd == "layout":
        cmd_layout()
        return
    if a.cmd == "inventory":
        from traffic_intel.core.config import PROJECT_ROOT
        from traffic_intel.datasets.inventory import summary_lines, write_inventory

        out = Path(a.out) if a.out else PROJECT_ROOT / "reports" / "dataset_inventory.json"
        for line in summary_lines(write_inventory(out)):
            print(line)
        print(f"full inventory: {out}")
        return
    if a.cmd == "quality":
        from traffic_intel.core.config import PROJECT_ROOT
        from traffic_intel.datasets.quality import write_report

        out = Path(a.out) if a.out else PROJECT_ROOT / "reports" / "data_quality_report.json"
        print(write_report(out))
        return
    if a.cmd == "prepare":
        from traffic_intel.datasets.manifest import DatasetNotAvailableError
        from traffic_intel.datasets.prepare import prepare

        params = {k: v for k, v in (("every_k", a.every_k), ("max_sensors", a.max_sensors)) if v}
        try:
            print(json.dumps(prepare(a.dataset_id, max_sequences=a.max_sequences, **params), indent=2))
        except DatasetNotAvailableError as exc:
            raise SystemExit(str(exc)) from None
        return

    from traffic_intel.datasets.public import download, load_catalog

    catalog = load_catalog(catalog=a.catalog)
    if a.cmd == "list":
        for name, v in catalog.items():
            print(f"{name:26s} {v.license:60s} {v.description}")
        return
    names = list(catalog) if a.all else a.names
    unknown = set(names) - set(catalog)
    if not names or unknown:
        raise SystemExit(f"choose from: {', '.join(catalog)}" + (f" (unknown: {unknown})" if unknown else ""))
    for name in names:
        print(name, "->", download(name, catalog[name], force=a.force))


if __name__ == "__main__":
    main()
