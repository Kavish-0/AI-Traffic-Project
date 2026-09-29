"""Calibration check tool writes its images and reports sane numbers."""

from __future__ import annotations

from pathlib import Path

from traffic_intel.computer_vision.calibration_tool import check
from traffic_intel.synthetic.sources import synthetic_camera


def test_calibration_check_on_synthetic_camera(tmp_path: Path) -> None:
    report = check(synthetic_camera("small"), frame_index=5, out=tmp_path)
    assert {p.name for p in tmp_path.iterdir()} == {"frame.png", "overlay.png", "birdseye.png"}
    assert report["reprojection_error_m"] < 0.3          # 1 px simulated picking noise
    assert report["anchor_reference"] == "front" and report["lanes"] == 2
    assert 55 < report["roi_length_m"] < 70  # type: ignore[operator]
