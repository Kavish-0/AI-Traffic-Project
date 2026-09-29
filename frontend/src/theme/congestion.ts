import type { CongestionLevel, RiskLevel } from "../api/types";

// Map road colours (Module 7), also used by legends and charts. The level itself always comes from
// the backend - these are only the colours.
export const CONGESTION_COLORS: Record<CongestionLevel, string> = {
  free: "#22c55e",
  moderate: "#eab308",
  high: "#f97316",
  severe: "#dc2626",
};

export const UNKNOWN_COLOR = "#64748b"; // no recent data / camera offline / road not monitored

export const RISK_COLORS: Record<RiskLevel, string> = {
  low: "#22c55e",
  medium: "#eab308",
  high: "#f97316",
  critical: "#dc2626",
};
