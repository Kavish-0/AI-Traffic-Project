import type { CongestionLevel, RiskLevel } from "../api/types";
import { ACCENT, LEVEL, MAP, SURFACE } from "./tokens";

// Road levels are shown by colour AND width AND dash pattern, so the map stays readable without colour
// vision. The level itself always comes from the backend - these are only the styles (values: tokens.ts).
export const CONGESTION_COLORS: Record<CongestionLevel, string> = {
  free: LEVEL.free,
  moderate: LEVEL.moderate,
  high: LEVEL.high,
  severe: LEVEL.severe,
};

export const UNKNOWN_COLOR = LEVEL.none; // no recent data / camera offline / road not monitored

/** Line width and dash pattern (both px; null = solid) per level; "none" = no fresh data. */
export const ROAD_STYLE: Record<CongestionLevel | "none", { width: number; dash: [number, number] | null }> = {
  none: { width: 3, dash: [2, 5] },
  free: { width: 4, dash: null },
  moderate: { width: 5, dash: null },
  high: { width: 6, dash: [12, 4] },
  severe: { width: 7, dash: [5, 3] },
};

export const ROAD_CASING = SURFACE.casing; // drawn under every road, 4 px wider

export const RISK_COLORS: Record<RiskLevel, string> = {
  low: LEVEL.free,
  medium: LEVEL.moderate,
  high: LEVEL.high,
  critical: LEVEL.severe,
};

/** Accents used by the map layers. */
export const AURORA = {
  data: ACCENT.data,
  action: ACCENT.action,
  actionHover: ACCENT.actionHover,
  mapGround: MAP.ground,
} as const;
