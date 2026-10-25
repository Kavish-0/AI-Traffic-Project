/**
 * Detailed basemap (city, district and neighbourhood names, streets, water, parks) recoloured to the
 * Aurora palette. Default: CARTO Dark Matter vector tiles (free, no API key; attribution kept).
 *   VITE_MAP_STYLE_URL=<any MapLibre style URL>   use another style (recoloured the same way)
 *   VITE_MAP_STYLE_URL=offline                     no basemap (plain background, works without internet)
 * If the style cannot be loaded, the map falls back to the plain background - roads still draw.
 */
import type { LayerSpecification, StyleSpecification } from "maplibre-gl";
import { MAP } from "../theme/tokens";

export const DEFAULT_STYLE_URL = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json";

const GROUND = MAP.ground;
export const OFFLINE_STYLE: StyleSpecification = {
  version: 8,
  sources: {},
  layers: [{ id: "bg", type: "background", paint: { "background-color": GROUND } }],
};

// Aurora map colours (theme/tokens.ts); labels are bright enough to read on the dark ground
const C = {
  ground: GROUND,
  landuse: MAP.landuse,
  park: MAP.park,
  water: MAP.water,
  building: MAP.building,
  street: MAP.street,
  streetMajor: MAP.streetMajor,
  freeway: MAP.freeway,
  casing: MAP.halo,
  rail: MAP.rail,
  boundary: MAP.boundary,
  city: MAP.labelCity,
  district: MAP.labelDistrict,
  neighbourhood: MAP.labelNeighbourhood,
  roadLabel: MAP.labelRoad,
  waterLabel: MAP.labelWater,
  poi: MAP.labelPoi,
  halo: MAP.halo,
};

function lineColor(id: string): string | null {
  if (/_case/.test(id)) return C.casing;
  if (/rail/.test(id)) return C.rail;
  if (/boundary/.test(id)) return C.boundary;
  if (/waterway/.test(id)) return C.water;
  if (/_mot_|_trunk_/.test(id)) return C.freeway;
  if (/_pri_|_sec_/.test(id)) return C.streetMajor;
  if (/road|tunnel|bridge|aeroway/.test(id)) return C.street;
  return null;
}

function fillColor(id: string): string | null {
  if (/water/.test(id)) return C.water;
  if (/park|landcover|nature/.test(id)) return C.park;
  if (/building/.test(id)) return C.building;
  if (/landuse/.test(id)) return C.landuse;
  return null;
}

function labelColor(id: string): { color: string; size?: number } {
  if (/city|capital|state|country|continent/.test(id)) return { color: C.city };
  if (/town|suburb/.test(id)) return { color: C.district };
  if (/hamlet|village/.test(id)) return { color: C.neighbourhood };
  if (/water/.test(id)) return { color: C.waterLabel };
  if (/roadname|housenumber/.test(id)) return { color: C.roadLabel };
  return { color: C.poi };
}

export function auroraStyle(style: StyleSpecification): StyleSpecification {
  const layers = style.layers.map((layer): LayerSpecification => {
    const l = { ...layer, paint: { ...(layer.paint ?? {}) } } as LayerSpecification & { paint: Record<string, unknown> };
    if (l.type === "background") l.paint["background-color"] = C.ground;
    else if (l.type === "fill") {
      const c = fillColor(l.id);
      if (c) l.paint["fill-color"] = c;
      delete l.paint["fill-pattern"];
    } else if (l.type === "line") {
      const c = lineColor(l.id);
      if (c) l.paint["line-color"] = c;
    } else if (l.type === "symbol") {
      const { color } = labelColor(l.id);
      l.paint["text-color"] = color;
      l.paint["text-halo-color"] = C.halo;
      l.paint["text-halo-width"] = 1.4;
      l.paint["icon-opacity"] = 0.7;
    }
    return l;
  });
  return { ...style, layers };
}

export async function loadBasemap(url: string | undefined): Promise<{ style: StyleSpecification; online: boolean }> {
  if (url === "offline") return { style: OFFLINE_STYLE, online: false };
  try {
    const res = await fetch(url || DEFAULT_STYLE_URL);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return { style: auroraStyle((await res.json()) as StyleSpecification), online: true };
  } catch {
    return { style: OFFLINE_STYLE, online: false };
  }
}
