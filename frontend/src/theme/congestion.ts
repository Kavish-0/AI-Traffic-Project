import type { CongestionLevel, RiskLevel } from "../api/types";

// Aurora palette. Road levels are shown by colour AND width AND dash pattern, so the map stays readable
// without colour vision. The level itself always comes from the backend - these are only the styles.
export const CONGESTION_COLORS: Record<CongestionLevel, string> = {
  free: "#2DD4BF", // free flow
  moderate: "#FDE047", // slow
  high: "#FB923C", // heavy
  severe: "#F43F5E", // jammed
};

export const UNKNOWN_COLOR = "#6E6790"; // no recent data / camera offline / road not monitored

/** Line width and dash pattern (both px; null = solid) per level; "none" = no fresh data. */
export const ROAD_STYLE: Record<CongestionLevel | "none", { width: number; dash: [number, number] | null }> = {
  none: { width: 3, dash: [2, 5] },
  free: { width: 4, dash: null },
  moderate: { width: 5, dash: null },
  high: { width: 6, dash: [12, 4] },
  severe: { width: 7, dash: [5, 3] },
};

export const ROAD_CASING = "#07060F"; // drawn under every road, 4 px wider

export const RISK_COLORS: Record<RiskLevel, string> = {
  low: "#2DD4BF",
  medium: "#FDE047",
  high: "#FB923C",
  critical: "#F43F5E",
};

/** Aurora accents: magenta = data / AI output, violet = actions. */
export const AURORA = {
  data: "#E879F9",
  action: "#7C3AED",
  actionHover: "#8B5CF6",
  mapGround: "#120E22",
  mapLabel: "#6A6390",
} as const;
