/** Map label helpers (pure, so they are unit-tested without a WebGL map). */
import { CONGESTION_COLORS, UNKNOWN_COLOR } from "../theme/congestion";
import { LEVEL_LABEL, NO_DATA_LABEL } from "./labels";

export interface MapLabel {
  id: string;
  kind: "road" | "junction";
  lngLat: [number, number];
  text: string;
  color: string | null;          // level colour for roads; null for junctions
  approachId: string | null;
  roadId: string;
}

/** Point halfway along a polyline (planar lon/lat distance is fine at street scale). */
export function lineMidpoint(coords: number[][]): [number, number] {
  if (coords.length === 1) return [coords[0][0], coords[0][1]];
  const seg = coords.slice(1).map((c, i) => Math.hypot(c[0] - coords[i][0], c[1] - coords[i][1]));
  let half = seg.reduce((a, b) => a + b, 0) / 2;
  for (let i = 0; i < seg.length; i++) {
    if (half <= seg[i] || i === seg.length - 1) {
      const r = seg[i] > 0 ? Math.min(half / seg[i], 1) : 0;
      return [coords[i][0] + r * (coords[i + 1][0] - coords[i][0]), coords[i][1] + r * (coords[i + 1][1] - coords[i][1])];
    }
    half -= seg[i];
  }
  return [coords[0][0], coords[0][1]];
}

/**
 * One label per MONITORED road ("Road A · Severe", in the level colour; "no data" when the backend
 * has no fresh level) and per junction. Unmonitored context roads stay unlabelled.
 */
export function mapLabels(features: { geometry: unknown; properties: unknown }[]): MapLabel[] {
  const out: MapLabel[] = [];
  for (const f of features) {
    const p = (f.properties ?? {}) as Record<string, unknown>;
    const g = f.geometry as { type?: string; coordinates?: unknown };
    if (p.kind === "road" && p.monitored && g.type === "LineString") {
      const level = !p.stale && typeof p.level === "string" ? (p.level as keyof typeof CONGESTION_COLORS) : null;
      const name = String(p.name ?? p.id).split(" (")[0];
      out.push({
        id: String(p.id), kind: "road", lngLat: lineMidpoint(g.coordinates as number[][]),
        text: `${name} · ${level ? LEVEL_LABEL[level] : NO_DATA_LABEL}`,
        color: level ? CONGESTION_COLORS[level] ?? UNKNOWN_COLOR : UNKNOWN_COLOR,
        approachId: (p.approach_id as string | null) ?? null, roadId: String(p.id),
      });
    } else if (p.kind === "junction" && g.type === "Point") {
      const c = g.coordinates as number[];
      out.push({ id: String(p.id), kind: "junction", lngLat: [c[0], c[1]], text: String(p.name ?? p.id), color: null,
                 approachId: null, roadId: "" });
    }
  }
  return out;
}
