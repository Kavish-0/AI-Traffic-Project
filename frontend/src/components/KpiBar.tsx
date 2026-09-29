import type { Summary } from "../api/types";
import type { Resource } from "../hooks/useResource";
import { num, time } from "../lib/format";
import { StateMessage } from "./ui";

function Kpi({ label, value, sub, accent }: { label: string; value: string; sub?: string; accent?: string }) {
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/70 px-4 py-3">
      <div className="text-[11px] tracking-widest text-slate-500 uppercase">{label}</div>
      <div className="text-2xl font-bold tabular-nums" style={{ color: accent ?? "#f1f5f9" }}>
        {value}
      </div>
      {sub && <div className="text-[11px] text-slate-500">{sub}</div>}
    </div>
  );
}

export function KpiBar({ summary }: { summary: Resource<Summary> }) {
  const s = summary.data;
  if (summary.loading) return <StateMessage>Loading traffic summary…</StateMessage>;
  if (!s) return <StateMessage tone="error">Traffic summary unavailable: {summary.error?.detail}</StateMessage>;
  const live = s.fresh_approaches > 0;
  return (
    <div className="space-y-2">
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <Kpi label="Vehicles in view" value={live ? num(s.vehicles) : "—"} sub={`${s.fresh_approaches}/${s.monitored_approaches} approaches live`} />
        <Kpi label="Avg speed" value={num(s.avg_speed_kmh, 0, "km/h")} sub="vehicle-weighted, live" />
        <Kpi label="Congested" value={String(s.congested_approaches)} sub="approaches HIGH / SEVERE" accent={s.congested_approaches ? "#f97316" : undefined} />
        <Kpi label="Incidents" value={String(s.active_incidents)} sub="active (possible, unverified)" accent={s.active_incidents ? "#ef4444" : undefined} />
      </div>
      {summary.error && (
        <StateMessage tone="error">
          Not updating - the last poll failed ({summary.error.detail}). The values above are from the last successful poll
          {s.timestamp ? ` (data time ${time(s.timestamp)})` : ""}.
        </StateMessage>
      )}
      {!live && !summary.error && <StateMessage tone="warn">Waiting for traffic data… (start the pipeline: python -m traffic_intel.pipeline run)</StateMessage>}
      {s.synthetic_approaches > 0 && (
        <StateMessage tone="info">
          {s.synthetic_approaches} approach(es) are fed by SYNTHETIC test scenes (simulated traffic and detections) to
          demonstrate the pipeline - their values are not real-world measurements.
        </StateMessage>
      )}
    </div>
  );
}
