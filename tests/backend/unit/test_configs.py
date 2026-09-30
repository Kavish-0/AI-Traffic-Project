"""All shipped YAML configs validate and reference each other consistently."""

from __future__ import annotations

from pathlib import Path

from traffic_intel.core.config import (
    AppSettings,
    load_camera,
    load_network,
    load_pipeline_config,
)
from traffic_intel.core.feature_schema import BASE_RESOLUTION_S


def test_pipeline_network_and_cameras(configs_dir: Path) -> None:
    pipeline = load_pipeline_config(AppSettings(config_dir=configs_dir))
    network = load_network(configs_dir / pipeline.network_file)
    approach_ids = {a.id for a in network.approaches}
    for rel in pipeline.camera_files:
        camera = load_camera(configs_dir / rel)
        assert camera.approach_id in approach_ids
    assert pipeline.feature_engine.params["window_s"] == BASE_RESOLUTION_S


def test_engine_rules_and_projector_configs_validate(configs_dir: Path) -> None:
    from traffic_intel.computer_vision.homography import HomographyProjector
    from traffic_intel.computer_vision.zones import CameraGeometry
    from traffic_intel.traffic.congestion.rules import RuleBasedCongestionClassifier
    from traffic_intel.traffic.features.samples import FeatureEngineConfig

    pipeline = load_pipeline_config(AppSettings(config_dir=configs_dir))
    params = {k: v for k, v in pipeline.feature_engine.params.items() if k != "live_tick_s"}
    FeatureEngineConfig(**params)
    RuleBasedCongestionClassifier(rules_file=str(configs_dir / "congestion_rules.yaml"))
    cam = load_camera(configs_dir / "cameras" / "cam_01.yaml")
    assert cam.calibration is not None
    size = cam.resolution or cam.calibration.image_size
    proj = HomographyProjector(cam.calibration, size, **pipeline.ground_projector.params)
    geo = CameraGeometry(cam, size, proj)
    assert proj.reprojection_error_m < 0.5 and geo.anchor_reference == "front" and 40 < geo.roi_length_m < 80
