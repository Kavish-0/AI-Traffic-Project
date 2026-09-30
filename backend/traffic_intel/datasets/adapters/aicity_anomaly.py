"""AI City Challenge 2021 Track 4 (traffic anomaly detection) adapter.

Canonical output: ``IncidentAnnotation`` intervals (video_id, start_s, end_s, source label).
They stay at interval level - nothing is expanded to frames until an experiment asks for it
(``IncidentAnnotation.frame_interval(fps)``).

Label file: one anomaly per line, ``video_id start_s end_s [label]``, whitespace- or
comma-separated, optional header. The file name is a parameter (``label_file``); by default the
first file matching the manifest's requirement patterns is used. Every labelled interval keeps
the dataset's own label (``anomaly``) - it covers stalled vehicles AND crashes and is never
re-labelled "accident".
"""

from __future__ import annotations

import glob
import re
from pathlib import Path

from traffic_intel.datasets.adapters.base import DATASET_ADAPTERS, IncidentAdapter, IncidentAnnotation

# first pattern with matches wins; the root-level fallback covers a flat layout without a train folder
VIDEO_GLOBS = ("*train*/**/*.mp4", "*.mp4")


@DATASET_ADAPTERS.register("aicity_anomaly")
class AICityAnomalyAdapter(IncidentAdapter):
    def label_file(self) -> Path:
        if self.params.get("label_file"):
            p = Path(self.params["label_file"])
            return p if p.is_absolute() else self.root / p
        patterns = self.entry.required[0].pattern if self.entry.required else ["*anomaly*.txt"]
        for pat in patterns:
            hits = sorted(glob.glob(str(self.root / pat)))
            if hits:
                return Path(hits[0])
        self.check()                                   # raises with download instructions
        raise FileNotFoundError(self.root)             # pragma: no cover

    def annotations(self) -> list[IncidentAnnotation]:
        source_label = (self.entry.labels or {}).get("source", ["anomaly"])[0]
        out = []
        for ln in self.label_file().read_text(encoding="utf-8").splitlines():
            parts = [p for p in re.split(r"[,\s]+", ln.strip()) if p]
            if len(parts) < 2 or not _is_number(parts[1]):
                continue                               # blank line or header
            end = float(parts[2]) if len(parts) > 2 and _is_number(parts[2]) else None
            label = parts[3] if len(parts) > 3 and not _is_number(parts[3]) else source_label
            out.append(IncidentAnnotation(dataset_id=self.dataset_id, video_id=_vid(parts[0]),
                                          start_s=float(parts[1]), end_s=end, source_label=label))
        return out

    def videos(self) -> dict[str, Path]:
        """Labelled TRAINING videos only. The release also ships aic21-track4-test-data/ whose files reuse
        the ids 1..150 without public labels - mixing them in would pair training labels with test videos."""
        pats = self.params.get("video_glob") or VIDEO_GLOBS
        for pat in [pats] if isinstance(pats, str) else pats:
            hits = sorted(glob.glob(str(self.root / pat), recursive=True))
            if hits:
                return {_vid(Path(p).stem): Path(p) for p in hits}
        return {}


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def _vid(s: str) -> str:
    """'1', '001', '1.mp4' -> '1' (AIC video ids are integers)."""
    stem = Path(s).stem
    return str(int(stem)) if stem.isdigit() else stem
