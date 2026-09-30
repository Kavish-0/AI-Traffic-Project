"""Explainable rule-based congestion classifier (``congestion_classifier.type: rule_based``).

    indicator_i in [0, 1]  (1 = worst)
      speed      = 1 - avg_speed / free_flow_speed
      density    = density_ratio / density saturation
      occupancy  = occupancy_pct / occupancy saturation
      queue      = queue_length_m / queue saturation (capped by ROI length)
    score = sum(w_i * indicator_i) / sum(w_i over available indicators)
    level = first level whose upper bound >= score
    overrides (boolean expressions over features) may only RAISE the level
    hysteresis: a lower level is accepted only after N consecutive windows

Every step is written to ``CongestionAssessment.explanation``. All numbers
come from ``configs/congestion_rules.yaml``. An ML classifier replaces this by
implementing the same ``CongestionClassifier`` interface.
"""

from __future__ import annotations

import ast
import operator
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from traffic_intel.core.config import get_settings, load_yaml
from traffic_intel.core.enums import CongestionLevel
from traffic_intel.core.schemas import CongestionAssessment, ContextFeatures, TrafficFeatures
from traffic_intel.traffic.interfaces import CONGESTION_CLASSIFIERS, CongestionClassifier

LEVEL_ORDER = [CongestionLevel.FREE, CongestionLevel.MODERATE, CongestionLevel.HIGH, CongestionLevel.SEVERE]


class Indicator(BaseModel):
    weight: float = Field(ge=0)
    saturation: float = Field(1.0, gt=0)     # value at which the indicator reaches 1


class Override(BaseModel):
    if_: str = Field(alias="if")
    level: CongestionLevel
    reason: str


class Hysteresis(BaseModel):
    min_windows_before_downgrade: int = Field(2, ge=1)


class RulesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    free_flow_speed_kmh: float = Field(50.0, gt=0)
    indicators: dict[str, Indicator]
    levels: dict[CongestionLevel, float]
    overrides: list[Override] = []
    hysteresis: Hysteresis = Hysteresis()
    min_quality: float = Field(0.3, ge=0, le=1)


# --------------------------------------------------------------------------- #
# Safe expression evaluation for override rules
# --------------------------------------------------------------------------- #
_CMP = {ast.Lt: operator.lt, ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge,
        ast.Eq: operator.eq, ast.NotEq: operator.ne}
_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


def safe_eval(expr: str, names: dict[str, Any]) -> bool:
    """Evaluate a boolean rule like ``"stopped_vehicles >= 5 and avg_speed_kmh < 5"``.
    Only comparisons, and/or/not, + - * /, numbers and known names are allowed.
    Any comparison involving a missing/None value is False."""

    def ev(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.BoolOp):
            vals = [ev(v) for v in node.values]
            return all(vals) if isinstance(node.op, ast.And) else any(vals)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return not ev(node.operand)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            v = ev(node.operand)
            return None if v is None else -v
        if isinstance(node, ast.Compare):
            left = ev(node.left)
            for op, comp in zip(node.ops, node.comparators, strict=True):
                right = ev(comp)
                if left is None or right is None or type(op) not in _CMP:
                    return False
                if not _CMP[type(op)](left, right):
                    return False
                left = right
            return True
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
            a, b = ev(node.left), ev(node.right)
            return None if a is None or b is None else _BIN[type(node.op)](a, b)
        if isinstance(node, ast.Name):
            if node.id not in names:
                raise ValueError(f"unknown name '{node.id}' in rule: {expr}")
            return names[node.id]
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float | bool):
            return node.value
        raise ValueError(f"disallowed syntax in rule: {expr}")

    return bool(ev(ast.parse(expr, mode="eval")))


def _eval_vector(node: ast.AST, cols: dict[str, Any]) -> Any:  # noqa: ANN401
    """Element-wise twin of ``safe_eval`` over arrays (same allowed syntax; NaN comparisons are False)."""
    import numpy as np

    if isinstance(node, ast.BoolOp):
        vals = [np.asarray(_eval_vector(v, cols), dtype=bool) for v in node.values]
        return np.logical_and.reduce(vals) if isinstance(node.op, ast.And) else np.logical_or.reduce(vals)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return ~np.asarray(_eval_vector(node.operand, cols), dtype=bool)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -np.asarray(_eval_vector(node.operand, cols), dtype=float)
    if isinstance(node, ast.Compare):
        left = np.asarray(_eval_vector(node.left, cols), dtype=float)
        out = np.ones(np.broadcast(left, left).shape, dtype=bool)
        for op, comp in zip(node.ops, node.comparators, strict=True):
            right = np.asarray(_eval_vector(comp, cols), dtype=float)
            if type(op) not in _CMP:
                raise ValueError("disallowed comparison")
            with np.errstate(invalid="ignore"):
                out = out & _CMP[type(op)](left, right) & np.isfinite(left) & np.isfinite(right)
            left = right
        return out
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        return _BIN[type(node.op)](np.asarray(_eval_vector(node.left, cols), dtype=float),
                                   np.asarray(_eval_vector(node.right, cols), dtype=float))
    if isinstance(node, ast.Name):
        return np.asarray(cols[node.id], dtype=float)
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float | bool):
        return float(node.value)
    raise ValueError("disallowed syntax in rule")


def feature_namespace(f: TrafficFeatures, free_flow_kmh: float) -> dict[str, Any]:
    ns: dict[str, Any] = {k: v for k, v in f.model_dump().items() if isinstance(v, int | float | type(None))}
    ns.update(f.extra)
    ns["density_pct"] = f.density_ratio * 100.0
    ns["speed_ratio"] = None if f.avg_speed_kmh is None else f.avg_speed_kmh / free_flow_kmh
    ns["free_flow_speed_kmh"] = free_flow_kmh
    return ns


class RuleBasedCongestionClassifier(CongestionClassifier):
    name = "rule_based"

    def __init__(
        self,
        rules_file: str | None = None,
        rules: dict[str, Any] | None = None,
        free_flow_speeds: dict[str, float] | None = None,
        stateful: bool = True,
    ) -> None:
        if rules is None:
            path = Path(rules_file or "congestion_rules.yaml")
            if not path.is_absolute():
                path = get_settings().config_dir / path
            rules = load_yaml(path)
        self.cfg = RulesConfig.model_validate(rules)
        if set(self.cfg.levels) != set(LEVEL_ORDER):
            raise ValueError("levels must define free, moderate, high, severe")
        self.free_flow = free_flow_speeds or {}
        self.stateful = stateful
        self._state: dict[str, tuple[CongestionLevel, int]] = {}   # approach -> (level, lower-count)
        # validate override syntax once
        probe = {k: 0.0 for k in (*TrafficFeatures.model_fields, "density_pct", "speed_ratio", "free_flow_speed_kmh")}
        for o in self.cfg.overrides:
            try:
                safe_eval(o.if_, probe)
            except ValueError as e:
                if "unknown name" not in str(e):   # extras are only known at runtime
                    raise

    # ------------------------------------------------------------------ #
    def _indicators(self, f: TrafficFeatures, ff: float) -> list[tuple[str, float, float, str]]:
        """(name, value 0..1, weight, explanation) for each available indicator.

        ``extra["speed_only"]``: the source measures speed and nothing else (e.g. PEMS-BAY loop-detector
        speeds) - its zero density / occupancy / queue are "not measured", not "empty road", so only the
        speed indicator is used."""
        out = []
        ind = self.cfg.indicators
        if "speed" in ind and f.avg_speed_kmh is not None:
            ratio = f.avg_speed_kmh / ff
            v = min(1.0, max(0.0, (1.0 - ratio) / ind["speed"].saturation))
            out.append(("speed", v, ind["speed"].weight,
                        f"speed {f.avg_speed_kmh:.1f} km/h = {ratio:.0%} of free-flow {ff:.0f} km/h"))
        if f.extra.get("speed_only"):
            return out
        if "density" in ind:
            v = min(1.0, f.density_ratio / ind["density"].saturation)
            out.append(("density", v, ind["density"].weight,
                        f"density {f.density_veh_km:.1f} veh/km/lane = {f.density_ratio:.0%} of jam density"))
        if "occupancy" in ind:
            v = min(1.0, f.occupancy_pct / ind["occupancy"].saturation)
            out.append(("occupancy", v, ind["occupancy"].weight, f"occupancy {f.occupancy_pct:.1f}%"))
        if "queue" in ind:
            sat = ind["queue"].saturation
            roi = f.extra.get("roi_length_m")
            if roi:
                sat = min(sat, roi)
            v = min(1.0, f.queue_length_m / sat)
            out.append(("queue", v, ind["queue"].weight, f"queue {f.queue_length_m:.0f} m"))
        return out

    def _level_for(self, score: float) -> CongestionLevel:
        for lvl in LEVEL_ORDER:
            if score <= self.cfg.levels[lvl] + 1e-12:
                return lvl
        return CongestionLevel.SEVERE

    def classify(self, features: TrafficFeatures, context: ContextFeatures | None = None) -> CongestionAssessment:
        ff = self.free_flow.get(features.approach_id, self.cfg.free_flow_speed_kmh)
        parts = self._indicators(features, ff)
        wsum = sum(w for _, _, w, _ in parts)
        score = sum(v * w for _, v, w, _ in parts) / wsum if wsum > 0 else 0.0
        explanation = [f"{text} -> {v:.2f} x w{w:.2f}" for _, v, w, text in parts]
        if features.avg_speed_kmh is None and "speed" in self.cfg.indicators:
            explanation.append("speed unavailable (no reliable samples); weights renormalised")
        level = self._level_for(score)
        explanation.append(f"score {score:.2f} -> {level.value.upper()}")

        ns = feature_namespace(features, ff)
        for o in self.cfg.overrides:
            try:
                fired = safe_eval(o.if_, ns)
            except ValueError:
                fired = False
            if fired and LEVEL_ORDER.index(o.level) > LEVEL_ORDER.index(level):
                level = o.level
                score = max(score, self.cfg.levels[LEVEL_ORDER[LEVEL_ORDER.index(o.level) - 1]] + 1e-3)
                explanation.append(f"override: {o.reason} -> {level.value.upper()}")

        if features.quality < self.cfg.min_quality:
            explanation.append(f"low data quality ({features.quality:.0%} of expected frames) - treat with caution")

        if self.stateful:
            level = self._apply_hysteresis(features.approach_id, level, explanation)

        return CongestionAssessment(
            approach_id=features.approach_id,
            timestamp=features.window_start,
            level=level,
            score=round(min(1.0, max(0.0, score)), 4),
            classifier=self.name,
            model_version=None,
            explanation=explanation,
        )

    def levels_frame(self, frame: Any, free_flow_kmh: Any = None) -> tuple[Any, Any]:  # noqa: ANN401
        """Vectorised, STATELESS version of ``classify`` for many windows (training labels, evaluation).

        Same indicator formulas, weights, level bounds and overrides as ``classify``; no hysteresis.
        An indicator is available only where its column exists and the value is not NaN - a dataset
        that does not measure density (e.g. loop-detector speeds) is scored on what it does measure,
        never on an implied 0. ``free_flow_kmh``: scalar or array per row (default: the config value).
        Returns (score [N], level index [N] into LEVEL_ORDER).
        """
        import numpy as np

        n = len(frame)
        ff = np.broadcast_to(np.asarray(self.cfg.free_flow_speed_kmh if free_flow_kmh is None else free_flow_kmh,
                                        dtype=float), (n,))
        ind = self.cfg.indicators
        num = np.zeros(n)
        den = np.zeros(n)

        def add(name: str, col: str, value: Any) -> None:  # noqa: ANN401
            if name in ind and col in frame:
                v = np.asarray(value, dtype=float)
                ok = np.isfinite(v)
                num[ok] += np.minimum(1.0, np.maximum(0.0, v[ok])) * ind[name].weight
                den[ok] += ind[name].weight

        if "avg_speed_kmh" in frame and "speed" in ind:
            add("speed", "avg_speed_kmh", (1.0 - frame["avg_speed_kmh"].to_numpy(float) / ff) / ind["speed"].saturation)
        if "density_ratio" in frame and "density" in ind:
            add("density", "density_ratio", frame["density_ratio"].to_numpy(float) / ind["density"].saturation)
        if "occupancy_pct" in frame and "occupancy" in ind:
            add("occupancy", "occupancy_pct", frame["occupancy_pct"].to_numpy(float) / ind["occupancy"].saturation)
        if "queue_length_m" in frame and "queue" in ind:
            sat = np.full(n, ind["queue"].saturation)
            if "roi_length_m" in frame:
                roi = frame["roi_length_m"].to_numpy(float)
                sat = np.where(np.isfinite(roi) & (roi > 0), np.minimum(sat, roi), sat)
            add("queue", "queue_length_m", frame["queue_length_m"].to_numpy(float) / sat)
        score = np.where(den > 0, num / np.maximum(den, 1e-12), 0.0)
        bounds = np.array([self.cfg.levels[lvl] for lvl in LEVEL_ORDER]) + 1e-12
        level = np.minimum(np.searchsorted(bounds, score, side="left"), len(LEVEL_ORDER) - 1)
        # overrides: only those whose names the frame provides (a missing name = rule cannot fire)
        cols = {**{c: frame[c] for c in frame.columns}, "free_flow_speed_kmh": ff}
        if "avg_speed_kmh" in frame:
            cols["speed_ratio"] = frame["avg_speed_kmh"].to_numpy(float) / ff
        if "density_ratio" in frame:
            cols["density_pct"] = frame["density_ratio"].to_numpy(float) * 100.0
        for o in self.cfg.overrides:
            names = {n_.id for n_ in ast.walk(ast.parse(o.if_, mode="eval")) if isinstance(n_, ast.Name)}
            if not names <= set(cols):
                continue
            fired = np.asarray(_eval_vector(ast.parse(o.if_, mode="eval").body, cols), dtype=bool)
            target = LEVEL_ORDER.index(o.level)
            raise_it = fired & (level < target)
            level = np.where(raise_it, target, level)
            floor = self.cfg.levels[LEVEL_ORDER[target - 1]] + 1e-3
            score = np.where(raise_it, np.maximum(score, floor), score)
        return np.clip(score, 0.0, 1.0), level

    def _apply_hysteresis(self, approach: str, level: CongestionLevel, explanation: list[str]) -> CongestionLevel:
        prev = self._state.get(approach)
        need = self.cfg.hysteresis.min_windows_before_downgrade
        if prev is None or LEVEL_ORDER.index(level) >= LEVEL_ORDER.index(prev[0]):
            self._state[approach] = (level, 0)
            return level
        count = prev[1] + 1
        if count >= need:
            self._state[approach] = (level, 0)
            return level
        self._state[approach] = (prev[0], count)
        explanation.append(f"held at {prev[0].value.upper()} (hysteresis {count}/{need} windows lower)")
        return prev[0]


CONGESTION_CLASSIFIERS.register("rule_based")(RuleBasedCongestionClassifier)
