/** Formatting helpers - null/undefined always render as an em dash, never as 0. */
export const DASH = "—";

export function num(v: number | null | undefined, digits = 0, unit = ""): string {
  if (v === null || v === undefined || Number.isNaN(v)) return DASH;
  return `${v.toFixed(digits)}${unit ? ` ${unit}` : ""}`;
}

export function pct(v: number | null | undefined, digits = 0): string {
  if (v === null || v === undefined || Number.isNaN(v)) return DASH;
  return `${(v * 100).toFixed(digits)}%`;
}

export function time(iso: string | null | undefined, withSeconds = true): string {
  if (!iso) return DASH;
  return new Date(iso).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: withSeconds ? "2-digit" : undefined,
  });
}

export function dateTime(iso: string | null | undefined): string {
  if (!iso) return DASH;
  return new Date(iso).toLocaleString([], { month: "short", day: "2-digit", hour: "2-digit", minute: "2-digit" });
}

export function ago(iso: string | null | undefined, now = Date.now()): string {
  if (!iso) return "never";
  const s = Math.max(0, Math.round((now - new Date(iso).getTime()) / 1000));
  if (s < 60) return `${s} s ago`;
  if (s < 3600) return `${Math.round(s / 60)} min ago`;
  return `${Math.round(s / 3600)} h ago`;
}

/**
 * Time series for a chart with a null point inserted wherever consecutive samples are more than
 * 3 median steps (and at least `minGapMs`) apart, so the chart (connectNulls: false) shows a GAP
 * where there was no data (e.g. the pipeline was stopped) instead of a straight line across it.
 */
export function withGaps(data: [string, number | null][], minGapMs = 150_000): [string, number | null][] {
  const rows = [...data].sort((a, b) => Date.parse(a[0]) - Date.parse(b[0]));
  if (rows.length < 2) return rows;
  const t = rows.map(([ts]) => Date.parse(ts));
  const steps = t.slice(1).map((v, i) => v - t[i]).filter((d) => d > 0).sort((a, b) => a - b);
  const limit = Math.max(minGapMs, 3 * (steps[Math.floor(steps.length / 2)] ?? 0));
  const out: [string, number | null][] = [rows[0]];
  for (let i = 1; i < rows.length; i++) {
    if (t[i] - t[i - 1] > limit) out.push([new Date((t[i] + t[i - 1]) / 2).toISOString(), null]);
    out.push(rows[i]);
  }
  return out;
}

export function titleCase(s: string | null | undefined): string {
  if (!s) return DASH;
  return s.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}
