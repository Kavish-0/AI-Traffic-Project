"""Explainable rule-based congestion classification."""

from __future__ import annotations

from typing import Any

import pytest
from conftest import T0

from traffic_intel.core.config import get_settings, load_yaml
from traffic_intel.core.enums import CongestionLevel
from traffic_intel.core.schemas import TrafficFeatures
from traffic_intel.traffic.congestion.rules import RuleBasedCongestionClassifier, safe_eval

RULES = load_yaml(get_settings().config_dir / "congestion_rules.yaml")


def feats(**kw: Any) -> TrafficFeatures:
    base: dict[str, Any] = dict(
        approach_id="a", camera_id="c", window_start=T0, window_s=60, vehicle_count=10, count_by_class={},
        pcu_count=10, avg_speed_kmh=45.0, speed_p85_kmh=50.0, density_veh_km=10, density_ratio=0.07,
        occupancy_pct=5.0, flow_veh_h=600, inflow=10, outflow=10, queue_length_m=0.0, stopped_vehicles=0,
        avg_gap_m=30.0, pedestrian_count=0, quality=1.0, extra={"roi_length_m": 60.0},
    )
    base.update(kw)
    return TrafficFeatures(**base)


def clf(**kw: Any) -> RuleBasedCongestionClassifier:
    return RuleBasedCongestionClassifier(rules=RULES, free_flow_speeds={"a": 50.0}, **kw)


def test_free_flow_is_free_and_explained() -> None:
    a = clf().classify(feats())
    assert a.level is CongestionLevel.FREE and a.score < 0.3
    assert any("speed 45.0 km/h = 90% of free-flow 50" in e for e in a.explanation)
    assert a.classifier == "rule_based"


@pytest.mark.parametrize(
    "kw, level",
    [
        (dict(avg_speed_kmh=30.0, density_ratio=0.3, occupancy_pct=20.0, queue_length_m=10.0),
         CongestionLevel.MODERATE),
        (dict(avg_speed_kmh=18.0, density_ratio=0.5, occupancy_pct=40.0, queue_length_m=30.0), CongestionLevel.HIGH),
        (dict(avg_speed_kmh=6.0, density_ratio=0.75, occupancy_pct=70.0, queue_length_m=60.0), CongestionLevel.SEVERE),
    ],
)
def test_levels_increase_with_congestion(kw: dict[str, Any], level: CongestionLevel) -> None:
    assert clf().classify(feats(**kw)).level is level


def test_override_raises_level_with_reason() -> None:
    a = clf().classify(feats(stopped_vehicles=6, avg_speed_kmh=3.0, density_ratio=0.2, occupancy_pct=10.0))
    assert a.level is CongestionLevel.SEVERE
    assert any("override: Standstill" in e for e in a.explanation)


def test_missing_speed_renormalises_weights() -> None:
    a = clf().classify(feats(avg_speed_kmh=None, density_ratio=0.0, occupancy_pct=0.0))
    assert a.level is CongestionLevel.FREE and a.score == 0.0
    assert any("speed unavailable" in e for e in a.explanation)


def test_hysteresis_delays_downgrade_but_not_upgrade() -> None:
    c = clf()
    severe = feats(avg_speed_kmh=5.0, density_ratio=0.85, occupancy_pct=80.0, queue_length_m=60.0)
    assert c.classify(severe).level is CongestionLevel.SEVERE
    held = c.classify(feats())
    assert held.level is CongestionLevel.SEVERE and any("hysteresis" in e for e in held.explanation)
    assert c.classify(feats()).level is CongestionLevel.FREE                 # 2nd lower window
    assert c.classify(severe).level is CongestionLevel.SEVERE                # upgrade immediate
    stateless = clf(stateful=False)
    stateless.classify(severe)
    assert stateless.classify(feats()).level is CongestionLevel.FREE


def test_low_quality_is_flagged() -> None:
    a = clf().classify(feats(quality=0.1))
    assert any("low data quality" in e for e in a.explanation)


def test_safe_eval_only_allows_simple_expressions() -> None:
    ns = {"a": 3.0, "b": None}
    assert safe_eval("a >= 3 and not a > 5", ns)
    assert not safe_eval("b > 1", ns)                    # missing values never fire a rule
    for bad in ("__import__('os')", "a.real > 1", "[a][0] > 1", "(lambda: 1)() > 0"):
        with pytest.raises(ValueError):
            safe_eval(bad, ns)


def test_config_typos_rejected() -> None:
    with pytest.raises(Exception, match="hysterisis"):
        RuleBasedCongestionClassifier(rules={**RULES, "hysterisis": {}})


def test_vectorised_levels_match_classify_exactly() -> None:
    """levels_frame (training labels, evaluation) must equal the live classify() row by row -
    including overrides - so labels are produced by the SAME rules the pipeline applies."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    rows = []
    for _ in range(400):
        rows.append(dict(avg_speed_kmh=float(rng.uniform(0, 60)) if rng.random() > 0.1 else None,
                         density_ratio=float(rng.uniform(0, 1)), occupancy_pct=float(rng.uniform(0, 80)),
                         queue_length_m=float(rng.uniform(0, 90)), stopped_vehicles=int(rng.integers(0, 8))))
    c = clf(stateful=False)
    expected = [c.classify(feats(**r)) for r in rows]
    frame = pd.DataFrame(rows).astype(float).assign(roi_length_m=60.0)
    score, level = c.levels_frame(frame, 50.0)
    order = [CongestionLevel.FREE, CongestionLevel.MODERATE, CongestionLevel.HIGH, CongestionLevel.SEVERE]
    assert [order[i] for i in level] == [e.level for e in expected]
    np.testing.assert_allclose(score, [e.score for e in expected], atol=1e-4)


def test_vectorised_levels_use_only_measured_indicators() -> None:
    """A speed-only dataset is scored on speed alone - an unmeasured density is not an implied 0."""
    import pandas as pd

    c = clf(stateful=False)
    speed_only = pd.DataFrame({"avg_speed_kmh": [48.0, 33.0, 24.0, 10.0]})
    _, level = c.levels_frame(speed_only, 50.0)
    assert list(level) == [0, 1, 2, 3]            # 96% / 66% / 48% / 20% of free flow (bands 76 / 56 / 40%)
    with_zero_density = speed_only.assign(density_ratio=0.0)
    _, diluted = c.levels_frame(with_zero_density, 50.0)
    assert diluted[3] < level[3]                   # what an implied 0 density would have done
