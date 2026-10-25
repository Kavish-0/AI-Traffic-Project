"""UVH-26 (IISc, Bengaluru traffic cameras; CC BY 4.0) -> YOLO detection dataset with 8 merged classes.

    python -m traffic_intel.datasets.uvh26 --src ../data/raw/uvh26 --dst ../data/processed/uvh26_yolo

Uses the majority-vote (MV) COCO labels, the ones IISc's released models were trained on. Images listed in
the JSON but not downloaded (only part of the train split is on disk) are skipped. Images are hard-linked,
so the YOLO copy costs no extra disk space on the same drive. Writes ``uvh26.yaml`` for Ultralytics.

The 14 UVH-26 classes are merged into 8 that keep what matters for Indian traffic (autos and two-wheelers
are separate classes); the live pipeline maps these names onto its own classes in configs/pipeline.yaml.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path
from typing import Any

MERGE = {
    "hatchback": "car", "sedan": "car", "suv": "car", "muv": "car",
    "bus": "bus", "mini-bus": "bus",
    "truck": "truck", "lcv": "truck",
    "van": "van", "tempo-traveller": "van",
    "three-wheeler": "auto_rickshaw",
    "two-wheeler": "two_wheeler",
    "bicycle": "bicycle",
    "other": "other", "others": "other",
}
NAMES = ["car", "bus", "truck", "van", "auto_rickshaw", "two_wheeler", "bicycle", "other"]
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


def _norm(s: str) -> str:
    return s.strip().lower().replace("_", "-").replace(" ", "-")


def _labels_json(split_dir: Path, split: str) -> Path:
    for pat in (f"*MV*{split.capitalize()}*.json", "*MV*.json"):
        hits = sorted(split_dir.glob(pat))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"no majority-vote (MV) COCO json in {split_dir}")


def _place(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def convert_split(split_dir: Path, split: str, out: Path, max_images: int = 0, seed: int = 0) -> dict[str, int]:
    coco = json.loads(_labels_json(split_dir, split).read_text(encoding="utf-8"))
    cat: dict[int, int] = {}
    for c in coco["categories"]:
        key = _norm(c["name"])
        if key not in MERGE:
            raise KeyError(f"unmapped UVH-26 class {c['name']!r}: add it to MERGE")
        cat[c["id"]] = NAMES.index(MERGE[key])
    on_disk = {p.name: p for p in split_dir.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES}
    anns: dict[int, list[dict[str, Any]]] = {}
    for a in coco["annotations"]:
        anns.setdefault(a["image_id"], []).append(a)
    images = list(coco["images"])
    if max_images:
        random.Random(seed).shuffle(images)
        images = images[:max_images]
    kept = missing = boxes = 0
    for im in images:
        src = on_disk.get(Path(im["file_name"]).name)
        if src is None:
            missing += 1
            continue
        w, h = float(im["width"]), float(im["height"])
        lines = []
        for a in anns.get(im["id"], []):
            x, y, bw, bh = a["bbox"]
            if bw <= 1 or bh <= 1:
                continue
            lines.append(f"{cat[a['category_id']]} {(x + bw / 2) / w:.6f} {(y + bh / 2) / h:.6f} "
                         f"{bw / w:.6f} {bh / h:.6f}")
        stem = f"{src.parent.name}_{src.stem}"
        _place(src, out / "images" / split / f"{stem}{src.suffix.lower()}")
        label = out / "labels" / split / f"{stem}.txt"
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text("\n".join(lines), encoding="utf-8")
        kept += 1
        boxes += len(lines)
    return {"images": kept, "boxes": boxes, "not_downloaded": missing}


def main() -> None:
    ap = argparse.ArgumentParser(prog="traffic_intel.datasets.uvh26")
    ap.add_argument("--src", default="../data/raw/uvh26")
    ap.add_argument("--dst", default="../data/processed/uvh26_yolo")
    ap.add_argument("--max-train", type=int, default=0, help="0 = every downloaded train image")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.dst)
    for split, sub, mx in (("train", "UVH-26-Train", a.max_train), ("val", "UVH-26-Val", 0)):
        print(split, convert_split(src / sub, split, out, mx, a.seed))
    yaml = [f"path: {out.resolve().as_posix()}", "train: images/train", "val: images/val", "names:"]
    yaml += [f"  {i}: {n}" for i, n in enumerate(NAMES)]
    (out / "uvh26.yaml").write_text("\n".join(yaml) + "\n", encoding="utf-8")
    print("wrote", out / "uvh26.yaml")


if __name__ == "__main__":
    main()
