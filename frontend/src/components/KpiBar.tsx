import { useEffect, useRef, useState } from "react";
import type { Summary } from "../api/types";
import type { Resource } from "../hooks/useResource";
import { num, time } from "../lib/format";
import { StateMessage } from "./ui";
import { ACCENT, STATUS, TEXT } from "../theme/tokens";

/** One KPI sample per new API data time, kept in this browser tab only (trends start when the page opens). */
interface Sample {
  t: number;
  speed: number | null;
  congested: number;
  incidents: number;
  live: number;
}
const TREND_WINDOW_MS = 60 * 60_000;
const MIN_TREND_MIN = 2;

function useTrend(s: Summary | undefined): Sample[] {
  const [samples, setSamples] = useState<Sample[]>([]);
  const last = useRef<string | null>(null);
  useEffect(() => {
    if (!s?.timestamp || s.timestamp === last.current) return;
    last.current = s.timestamp;
    const t = Date.parse(s.timestamp);
    const add = { t, speed: s.avg_speed_kmh, congested: s.congested_approaches, incidents: s.active_incidents,
                  live: s.fresh_approaches };
    setSamples((prev) => [...prev.filter((p) => t - p.t <= TREND_WINDOW_MS), add]);
  }, [s]);
  return samples;
}

function Sparkline({ values, color }: { values: (number | null)[]; color: string }) {
  const v = values.filter((x): x is number => x !== null);
  if (v.length < 2) return null;
  const lo = Math.min(...v), hi = Math.max(...v), span = hi - lo || 1;
  const pts = v.map((x, i) => `${(i / (v.length - 1)) * 100},${28 - ((x - lo) / span) * 24}`).join(" ");
  return (
    <svg viewBox="0 0 100 30" preserveAspectRatio="none" className="mt-1 h-6 w-full" aria-hidden="true">
      <polyline points={pts} fill="none" stroke={color} strokeWidth="1.5" vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

function delta(values: (number | null)[], unit = "", digits = 0): string | undefined {
  const v = values.filter((x): x is number => x !== null);
  if (v.length < 2) return undefined;
  const d = v[v.length - 1] - v[0];
  const sign = d > 0 ? "▲" : d < 0 ? "▼" : "■";
  return `${sign} ${Math.abs(d).toFixed(digits)}${unit}`;
}

function Kpi({ label, value, sub, accent, trend, change, since }: {
  label: string; value: string; sub?: string; accent?: string; trend?: (number | null)[]; change?: string; since?: string;
}) {
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/70 px-3 py-2">
      <div className="text-[11px] tracking-widest text-slate-500 uppercase">{label}</div>
      <div className="flex items-baseline gap-2">
        <span className="text-2xl font-bold tabular-nums" style={{ color: accent ?? TEXT.primary }}>{value}</span>
        {change && <span className="text-xs text-slate-400 tabular-nums" title={since}>{change}</span>}
      </div>
      {sub && <div className="text-[11px] text-slate-500">{sub}</div>}
      {trend && <Sparkline values={trend} color={accent ?? ACCENT.data} />}
    </div>
  );
}

export function KpiBar({ summary }: { summary: Resource<Summary> }) {
  const s = summary.data;
  const trend = useTrend(s);
  if (summary.loading) return <StateMessage tone="loading">Loading traffic summary…</StateMessage>;
  if (!s) return <StateMessage tone="error">Traffic summary unavailable: {summary.error?.detail}</StateMessage>;
  const live = s.fresh_approaches > 0;
  const sensors = s.speed_only_approaches ?? 0;
  const sensorOnly = sensors > 0 && sensors === s.monitored_approaches;
  const mins = trend.length > 1 ? Math.max(1, Math.round((trend[trend.length - 1].t - trend[0].t) / 60_000)) : 0;
  const since = mins ? `change over the last ${mins} min (since this page opened, up to 60 min)` : undefined;
  // trends appear once there are at least MIN_TREND_MIN minutes of samples (a 10-second "change" means nothing)
  const series = (k: keyof Omit<Sample, "t">) => (mins >= MIN_TREND_MIN ? trend.map((p) => p[k]) : undefined);
  const change = (k: keyof Omit<Sample, "t">, unit = "") => {
    const v = series(k);
    return v ? delta(v, unit) : undefined;
  };
  return (
    <div className="space-y-2">
      <div className={`grid grid-cols-2 gap-3 ${sensorOnly ? "md:grid-cols-3" : "md:grid-cols-4"}`}>
        {sensorOnly ? null : (
          <Kpi label="Vehicles in view" value={live ? num(s.vehicles) : "—"} sub={`${s.fresh_approaches}/${s.monitored_approaches} approaches live`} />
        )}
        <Kpi label="Average speed" value={num(s.avg_speed_kmh, 0, "km/h")} sub="right now"
             trend={series("speed")} change={change("speed", " km/h")} since={since} />
        <Kpi label="Heavy or jammed" value={String(s.congested_approaches)} sub="roads" accent={s.congested_approaches ? STATUS.alert : undefined}
             trend={series("congested")} change={change("congested")} since={since} />
        <Kpi label="Incidents to check" value={String(s.active_incidents)} sub="possible, not yet verified" accent={s.active_incidents ? STATUS.error : undefined}
             trend={series("incidents")} change={change("incidents")} since={since} />
      </div>
      {summary.error && (
        <StateMessage tone="error">
          Not updating - the last poll failed ({summary.error.detail}). The values above are from the last successful poll
          {s.timestamp ? ` (data time ${time(s.timestamp)})` : ""}.
        </StateMessage>
      )}
      {!live && !summary.error && <StateMessage tone="warn">Waiting for traffic data… (start the pipeline: python -m traffic_intel.pipeline run)</StateMessage>}
      {sensors > 0 && (
        <StateMessage tone="info">
          <>Real speeds from {sensors} freeway sensors (recorded, replayed at today's time of day).</>
        </StateMessage>
      )}
      {s.synthetic_approaches > 0 && (
        <StateMessage tone="info">
          {s.synthetic_approaches} approach(es) are fed by SYNTHETIC test scenes (simulated traffic and detections) to
          demonstrate the pipeline - their values are not real-world measurements.
        </StateMessage>
      )}
    </div>
  );
}
