import maplibregl, { type GeoJSONSource, type StyleSpecification } from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import { useEffect, useMemo, useRef } from "react";
import type { NetworkGeoJSON } from "../api/types";
import { mapLabels } from "../lib/mapLabels";
import { AURORA, CONGESTION_COLORS, ROAD_CASING, ROAD_STYLE, UNKNOWN_COLOR } from "../theme/congestion";

const MAX_ROAD_LABELS_ALWAYS = 40;
const ROAD_LABEL_MIN_ZOOM = 13;

// Basemap: VITE_MAP_STYLE_URL (any MapLibre style URL) or, by default, an offline schematic background.
const STYLE_URL = import.meta.env.VITE_MAP_STYLE_URL as string | undefined;
const BLANK_STYLE: StyleSpecification = {
  version: 8,
  sources: {},
  layers: [{ id: "bg", type: "background", paint: { "background-color": AURORA.mapGround } }],
};

// effective level of a road: its backend level, or "none" when there is no fresh level
const LEVEL: maplibregl.ExpressionSpecification = [
  "case",
  ["any", ["==", ["get", "stale"], true], ["!", ["to-boolean", ["get", "level"]]]],
  "none",
  ["to-string", ["get", "level"]],
];
const LEVEL_COLOR: maplibregl.ExpressionSpecification = [
  "match", LEVEL,
  "free", CONGESTION_COLORS.free,
  "moderate", CONGESTION_COLORS.moderate,
  "high", CONGESTION_COLORS.high,
  "severe", CONGESTION_COLORS.severe,
  UNKNOWN_COLOR,
];
const LEVEL_WIDTH: maplibregl.ExpressionSpecification = [
  "+",
  ["case", ["==", ["get", "selected"], true], 3, 0],
  ["match", LEVEL,
    "free", ROAD_STYLE.free.width,
    "moderate", ROAD_STYLE.moderate.width,
    "high", ROAD_STYLE.high.width,
    "severe", ROAD_STYLE.severe.width,
    ROAD_STYLE.none.width],
];
const OPEN_ROAD: maplibregl.ExpressionSpecification = ["all", ["==", ["get", "kind"], "road"], ["!=", ["get", "closed"], true]];
/** MapLibre dash lengths are in line widths; the Aurora spec gives them in px. */
const dashFor = (level: "high" | "severe" | "none"): number[] => {
  const s = ROAD_STYLE[level];
  return (s.dash ?? [1, 0]).map((px) => px / s.width);
};

export interface MapProps {
  network: NetworkGeoJSON | undefined;
  selectedApproach: string | null;
  selectedJunction: string | null;
  whatIfChanges: Record<string, number> | null;
  closedRoads: string[];
  onRoad: (approachId: string | null, roadId: string) => void;
  onJunction: (id: string) => void;
  onCamera: (id: string, approachId: string | null) => void;
}

/**
 * Adds UI state (selection, what-if overlay) to the REST GeoJSON. Road levels are the backend's
 * (GET /network, refetched when the live poll detects a new window) - never computed here.
 */
function decorate(net: NetworkGeoJSON, props: MapProps): NetworkGeoJSON {
  return {
    ...net,
    features: net.features.map((f) => {
      const p = { ...(f.properties ?? {}) } as Record<string, unknown>;
      if (p.kind === "road") {
        p.selected = p.approach_id !== null && p.approach_id === props.selectedApproach;
        const ch = props.whatIfChanges?.[p.id as string];
        p.whatif = ch === undefined ? 0 : Math.min(Math.abs(ch), 200);
        p.closed = props.closedRoads.includes(p.id as string);
      }
      if (p.kind === "junction") p.selected = p.id === props.selectedJunction;
      return { ...f, properties: p };
    }),
  };
}

export function MapView(props: MapProps) {
  const container = useRef<HTMLDivElement>(null);
  const mapRef = useRef<maplibregl.Map | null>(null);
  const loaded = useRef(false);
  const cb = useRef(props);
  useEffect(() => {
    cb.current = props;
  });

  const data = useMemo(
    () => (props.network ? decorate(props.network, props) : undefined),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [props.network, props.selectedApproach, props.selectedJunction, props.whatIfChanges, props.closedRoads],
  );

  useEffect(() => {
    if (!container.current) return;
    const map = new maplibregl.Map({
      container: container.current,
      style: STYLE_URL || BLANK_STYLE,
      center: [0, 0],
      zoom: 1,
      attributionControl: STYLE_URL ? undefined : false,
    });
    map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-right");
    mapRef.current = map;
    map.on("load", () => {
      map.addSource("net", { type: "geojson", data: { type: "FeatureCollection", features: [] }, promoteId: "id" });
      map.addLayer({
        id: "roads-casing", type: "line", source: "net", filter: ["==", ["get", "kind"], "road"],
        paint: { "line-color": ROAD_CASING, "line-width": ["+", LEVEL_WIDTH, 4] },
        layout: { "line-cap": "round" },
      });
      map.addLayer({
        id: "roads-whatif", type: "line", source: "net",
        filter: ["all", ["==", ["get", "kind"], "road"], [">", ["get", "whatif"], 0]],
        paint: { "line-color": AURORA.data, "line-opacity": 0.35, "line-width": ["+", 10, ["/", ["get", "whatif"], 6]] },
      });
      // solid levels share one layer; each dashed level has its own (line-dasharray cannot be data-driven)
      map.addLayer({
        id: "roads", type: "line", source: "net",
        filter: ["all", OPEN_ROAD, ["match", LEVEL, ["free", "moderate"], true, false]],
        paint: { "line-color": LEVEL_COLOR, "line-width": LEVEL_WIDTH },
        layout: { "line-cap": "round" },
      });
      for (const lvl of ["high", "severe", "none"] as const) {
        map.addLayer({
          id: `roads-${lvl}`, type: "line", source: "net", filter: ["all", OPEN_ROAD, ["==", LEVEL, lvl]],
          paint: { "line-color": LEVEL_COLOR, "line-width": LEVEL_WIDTH, "line-dasharray": dashFor(lvl) },
        });
      }
      map.addLayer({
        id: "roads-closed", type: "line", source: "net",
        filter: ["all", ["==", ["get", "kind"], "road"], ["==", ["get", "closed"], true]],
        paint: { "line-color": CONGESTION_COLORS.severe, "line-width": 3, "line-dasharray": [1, 1.5] },
      });
      map.addLayer({
        id: "incidents", type: "circle", source: "net", filter: ["==", ["get", "kind"], "incident"],
        paint: {
          "circle-radius": 9,
          "circle-color": CONGESTION_COLORS.severe,
          "circle-opacity": 0.9,
          "circle-stroke-color": "#F3F0FF",
          "circle-stroke-width": 2,
        },
      });
      map.addLayer({
        id: "junctions", type: "circle", source: "net", filter: ["==", ["get", "kind"], "junction"],
        paint: {
          "circle-radius": ["case", ["get", "selected"], 11, 8],
          "circle-color": "#F3F0FF",
          "circle-stroke-color": ["case", ["get", "selected"], AURORA.data, ROAD_CASING],
          "circle-stroke-width": 3,
        },
      });
      map.addLayer({
        id: "cameras", type: "circle", source: "net", filter: ["==", ["get", "kind"], "camera"],
        paint: {
          "circle-radius": 5,
          "circle-color": [
            "match", ["get", "status"],
            "online", AURORA.data, "error", CONGESTION_COLORS.severe, "offline", CONGESTION_COLORS.high, UNKNOWN_COLOR,
          ],
          "circle-stroke-color": ROAD_CASING,
          "circle-stroke-width": 1.5,
        },
      });
      // the casing is under every road and wider than it, so it is the click / hover target for all levels
      for (const layer of ["roads-casing", "junctions", "cameras", "incidents"]) {
        map.on("mouseenter", layer, () => (map.getCanvas().style.cursor = "pointer"));
        map.on("mouseleave", layer, () => (map.getCanvas().style.cursor = ""));
      }
      map.on("click", "roads-casing", (e) => {
        const p = e.features?.[0]?.properties;
        if (p) cb.current.onRoad((p.approach_id as string) || null, p.id as string);
      });
      map.on("click", "junctions", (e) => {
        const p = e.features?.[0]?.properties;
        if (p) cb.current.onJunction(p.id as string);
      });
      map.on("click", "incidents", (e) => {
        const p = e.features?.[0]?.properties;
        if (p?.approach_id) cb.current.onRoad(p.approach_id as string, "");
      });
      map.on("click", "cameras", (e) => {
        const p = e.features?.[0]?.properties;
        if (p) cb.current.onCamera(p.id as string, (p.approach_id as string) || null);
      });
      loaded.current = true;
      if (cb.current.network) apply(map, cb.current.network);
    });
    return () => {
      loaded.current = false;
      map.remove();
    };
  }, []);

  useEffect(() => {
    const map = mapRef.current;
    if (map && loaded.current && data) apply(map, data);
  }, [data]);

  // road / junction name labels as HTML markers (no glyph server needed; clickable)
  const markers = useRef(new Map<string, { marker: maplibregl.Marker; el: HTMLDivElement }>());
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !props.network) return;
    const want = mapLabels(props.network.features);
    const seen = new Set<string>();
    for (const l of want) {
      seen.add(l.id);
      let m = markers.current.get(l.id);
      if (!m) {
        const el = document.createElement("div");
        el.className = l.kind === "road"
          ? "cursor-pointer rounded bg-slate-950/85 px-1.5 py-0.5 text-[11px] font-medium whitespace-nowrap text-slate-100 shadow"
          : "pointer-events-none rounded px-1 text-[12px] font-semibold tracking-wide whitespace-nowrap text-slate-200";
        if (l.kind === "road") el.addEventListener("click", () => cb.current.onRoad(l.approachId, l.roadId));
        const marker = new maplibregl.Marker({ element: el, anchor: l.kind === "road" ? "left" : "bottom",
                                               offset: l.kind === "road" ? [12, 0] : [0, -14] });
        marker.setLngLat(l.lngLat).addTo(map);
        m = { marker, el };
        markers.current.set(l.id, m);
      }
      m.marker.setLngLat(l.lngLat);
      m.el.innerHTML = "";
      if (l.color) {
        const dot = document.createElement("span");
        dot.className = "mr-1 inline-block h-2 w-2 rounded-full align-middle";
        dot.style.background = l.color;
        m.el.appendChild(dot);
      }
      m.el.appendChild(document.createTextNode(l.text));
    }
    for (const [id, m] of markers.current) {
      if (!seen.has(id)) {
        m.marker.remove();
        markers.current.delete(id);
      }
    }
    roadLabelIds.current = new Set(want.filter((l) => l.kind === "road").map((l) => l.id));
    labelVisibility();
  }, [props.network]);

  // a large network (e.g. 325 PEMS-BAY stations) shows road labels only when zoomed in; junction names stay
  const roadLabelIds = useRef(new Set<string>());
  const labelVisibility = (): void => {
    const map = mapRef.current;
    if (!map) return;
    const show = roadLabelIds.current.size <= MAX_ROAD_LABELS_ALWAYS || map.getZoom() >= ROAD_LABEL_MIN_ZOOM;
    for (const id of roadLabelIds.current) {
      const m = markers.current.get(id);
      if (m) m.el.style.display = show ? "" : "none";
    }
  };
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    map.on("zoomend", labelVisibility);
    return () => {
      map.off("zoomend", labelVisibility);
    };
  });

  // zoom to the selected junction (once per selection, not on every data refresh)
  const fittedJunction = useRef<string | null>(null);
  useEffect(() => {
    const map = mapRef.current;
    const jid = props.selectedJunction;
    if (!map || !props.network || !jid || fittedJunction.current === jid) return;
    const b = junctionBounds(props.network, jid);
    if (!b) return;
    const fit = () => map.fitBounds(b, { padding: 70, maxZoom: 17.5, duration: fittedJunction.current ? 600 : 0 });
    if (loaded.current) fit();
    else map.once("load", fit);
    fittedJunction.current = jid;
  }, [props.selectedJunction, props.network]);

  // the map card can change height (it follows the row on wide screens)
  useEffect(() => {
    const el = container.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(() => mapRef.current?.resize());
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  return <div ref={container} className="h-full w-full" data-testid="map" />;
}

/** Bounds of a junction: its point plus every road / camera / incident that belongs to it. */
function junctionBounds(net: NetworkGeoJSON, jid: string): maplibregl.LngLatBoundsLike | null {
  const xs: number[] = [];
  const ys: number[] = [];
  const add = (c: unknown): void => {
    if (Array.isArray(c) && typeof c[0] === "number") {
      xs.push(c[0] as number);
      ys.push(c[1] as number);
    } else if (Array.isArray(c)) c.forEach(add);
  };
  for (const f of net.features) {
    const p = (f.properties ?? {}) as Record<string, unknown>;
    if ((p.kind === "junction" && p.id === jid) || p.junction_id === jid) add((f.geometry as { coordinates?: unknown }).coordinates);
  }
  if (!xs.length) return null;
  return [[Math.min(...xs), Math.min(...ys)], [Math.max(...xs), Math.max(...ys)]];
}

const fitted = new WeakSet<maplibregl.Map>();

function apply(map: maplibregl.Map, data: NetworkGeoJSON) {
  (map.getSource("net") as GeoJSONSource | undefined)?.setData(data);
  if (!fitted.has(map) && data.bbox) {
    const b = data.bbox as number[];
    map.fitBounds([[b[0], b[1]], [b[2], b[3]]], { padding: 60, duration: 0 });
    fitted.add(map);
  }
}
