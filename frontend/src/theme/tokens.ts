/**
 * Aurora design tokens: the ONLY place colour values are defined.
 *
 * - Components use Tailwind classes for layout and surfaces; index.css maps Tailwind's slate / sky / cyan
 *   scales onto these values, so class names and tokens stay in step.
 * - Anything drawn outside CSS (MapLibre layers, ECharts series, inline SVG, badge tints) imports from here.
 * - Meaning, not decoration: magenta = data / AI output, violet = actions, teal-yellow-orange-red = traffic
 *   level, red = incident / error. A colour is never reused for a different meaning.
 */

export const SURFACE = {
  page: "#0C0A18",
  card: "#151226",
  raised: "#1D1933",
  border: "#2F2A4D",
  casing: "#07060F",
} as const;

export const TEXT = {
  primary: "#F3F0FF",
  secondary: "#C7C0E6",
  muted: "#A49DC4",
  faint: "#8A83AD",
} as const;

export const ACCENT = {
  data: "#E879F9",        // measurements and AI output
  action: "#7C3AED",      // buttons, selection
  actionHover: "#8B5CF6",
  synthetic: "#A855F7",   // synthetic test data badge
} as const;

export const STATUS = {
  ok: "#2DD4BF",
  warn: "#FDE047",
  alert: "#FB923C",
  error: "#F43F5E",
  unknown: "#6E6790",
} as const;

/** Traffic level colours (map roads, legends, badges). The level itself always comes from the backend. */
export const LEVEL = {
  free: "#2DD4BF",
  moderate: "#FDE047",
  high: "#FB923C",
  severe: "#F43F5E",
  none: "#6E6790",
} as const;

/** One colour per measured quantity, so a chart series means the same thing everywhere. */
export const SERIES = {
  speed: ACCENT.data,
  density: LEVEL.high,
  count: "#A3E635",
  queue: STATUS.error,
  flow: "#C084FC",
  forecast: LEVEL.moderate,
  incident: STATUS.error,
} as const;

/** Change direction in what-if results: closed / more traffic / less traffic. */
export const CHANGE = {
  closed: STATUS.error,
  up: STATUS.alert,
  down: STATUS.ok,
} as const;

/** Basemap recolouring (lib/basemap.ts). Labels are bright enough to read on the dark ground. */
export const MAP = {
  ground: "#120E22",
  landuse: "#161129",
  park: "#132526",
  water: "#15163A",
  building: "#1A1530",
  street: "#1F1A36",
  streetMajor: "#2A2447",
  freeway: "#352D58",
  rail: "#2A2447",
  boundary: "#3A3460",
  labelCity: TEXT.primary,
  labelDistrict: "#D5CEF2",
  labelNeighbourhood: "#B8B0DC",
  labelRoad: TEXT.muted,
  labelWater: "#7E86D6",
  labelPoi: "#9A92C0",
  halo: SURFACE.page,
} as const;

/** Hex colour + alpha (0..1) as #RRGGBBAA, for tinted badge backgrounds and borders. */
export function alpha(hex: string, a: number): string {
  return `${hex}${Math.round(Math.max(0, Math.min(1, a)) * 255).toString(16).padStart(2, "0")}`;
}
