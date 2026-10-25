/**
 * Plain-language wording for everything an operator reads. The API keeps its technical codes
 * (free / moderate / high / severe, approach, ...); the UI translates them in this one place.
 */
import type { CongestionLevel, RiskLevel } from "../api/types";

export const LEVEL_LABEL: Record<CongestionLevel, string> = {
  free: "Flowing",
  moderate: "Slow",
  high: "Heavy",
  severe: "Jammed",
};

export const RISK_LABEL: Record<RiskLevel, string> = {
  low: "Low risk",
  medium: "Medium risk",
  high: "High risk",
  critical: "Critical risk",
};

export const NO_DATA_LABEL = "No data";

/** Severity order used to sort "needs attention" lists (worst first). */
export const LEVEL_RANK: Record<CongestionLevel, number> = { severe: 3, high: 2, moderate: 1, free: 0 };

export function levelLabel(level: CongestionLevel | null | undefined): string {
  return level ? LEVEL_LABEL[level] : NO_DATA_LABEL;
}

/** Uncalibrated incident scores are shown as an estimate, never as a verified probability. */
export const ESTIMATE_NOTE = "estimate, not yet verified";
