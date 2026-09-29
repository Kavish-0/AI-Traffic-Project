import type { Incident, Snapshot, TrafficState } from "../api/types";
import type { Resource } from "../hooks/useResource";
import { ago, num, pct, time, titleCase } from "../lib/format";
import { ForecastStrip } from "./ForecastStrip";
import { Badge, Card, LevelBadge, RiskBadge, Stat, StateMessage, SyntheticBadge } from "./ui";

export function scoreLabel(calibrated: boolean): string {
  return calibrated ? "Incident probability" : "Incident probability: UNCALIBRATED";
}

function Evidence({ evidence }: { evidence: Record<string, number> }) {
  const rows = Object.entries(evidence).filter(([k]) => k !== "symptom_strength").sort((a, b) => b[1] - a[1]);
  if (!rows.length) return <p className="text-xs text-slate-500">No contributing evidence.</p>;
  return (
    <ul className="space-y-1">
      {rows.map(([k, v]) => (
        <li key={k} className="flex items-center gap-2 text-xs">
          <span className="w-36 shrink-0 text-slate-400">{titleCase(k)}</span>
          <span className="h-1.5 flex-1 rounded bg-slate-800">
            <span className="block h-1.5 rounded bg-rose-500" style={{ width: `${Math.round(v * 100)}%` }} />
          </span>
          <span className="w-9 text-right text-slate-300 tabular-nums">{v.toFixed(2)}</span>
        </li>
      ))}
    </ul>
  );
}

export function IncidentBlock({ incident, score }: { incident: Incident | null; score: number | null }) {
  if (!incident) {
    return (
      <div className="text-sm text-slate-300">
        No open incident.{" "}
        <span className="text-slate-500">
          {scoreLabel(false)} — {score === null ? "unavailable" : pct(score)}
        </span>
      </div>
    );
  }
  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-center gap-2">
        <Badge color="#ef4444">{incident.label || titleCase(incident.incident_type)}</Badge>
        <span className="text-sm text-slate-200">
          {scoreLabel(incident.calibrated)} — <b className="tabular-nums">{pct(incident.probability)}</b>
        </span>
        <span className="text-xs text-slate-500">
          {titleCase(incident.status)} · since {time(incident.first_detected, false)}
        </span>
      </div>
      <Evidence evidence={incident.evidence} />
      <p className="text-[11px] text-slate-500">{incident.uncertainty_note}</p>
    </div>
  );
}

function current(snap: Snapshot): { s: TrafficState | null; label: string; stale: boolean } {
  if (snap.live && !snap.live_stale) return { s: snap.live, label: `live (${snap.live.window_s.toFixed(0)} s rolling)`, stale: false };
  if (snap.window) return { s: snap.window, label: `last 60 s window ended ${time(snap.window.timestamp, false)}`, stale: snap.window_stale };
  return { s: null, label: "", stale: true };
}

/** Live values come from the snapshot (GET /approaches/{id}/snapshot, polled at the live rate). */
export function RoadPanel({ snapshot }: { snapshot: Resource<Snapshot> }) {
  if (snapshot.loading) return <Card title="Selected road"><StateMessage>Loading road…</StateMessage></Card>;
  const snap = snapshot.data;
  if (!snap) return <Card title="Selected road"><StateMessage tone="error">Road unavailable: {snapshot.error?.detail}</StateMessage></Card>;
  const { s, label, stale } = current(snap);
  const level = s?.traffic_state || null;
  return (
    <Card
      title="Selected road"
      right={snap.synthetic ? <SyntheticBadge /> : null}
    >
      <div className="mb-3">
        <div className="text-base font-semibold text-slate-100">{snap.road?.name ?? snap.approach.name}</div>
        <div className="text-xs text-slate-500">
          {snap.approach.name} · {snap.approach.direction} · into {snap.approach.junction_id}
        </div>
      </div>

      {snapshot.error && (
        <div className="mb-2">
          <StateMessage tone="error">Not updating - {snapshot.error.detail}. Showing the last values received.</StateMessage>
        </div>
      )}
      <h3 className="mb-1 text-[11px] font-semibold tracking-widest text-slate-500 uppercase">Current state</h3>
      {!s ? (
        <StateMessage tone="warn">Waiting for traffic data… (no window for this road yet)</StateMessage>
      ) : (
        <>
          {stale && <StateMessage tone="warn">Data is stale - last update {ago(s.timestamp)}. Camera or pipeline may be offline.</StateMessage>}
          <div className="mt-2 grid grid-cols-3 gap-3">
            <Stat label="Vehicles" value={num(s.vehicle_count)} />
            <Stat label="Avg speed" value={num(s.average_speed_kmh, 0, "km/h")} />
            <Stat label="Density" value={num(s.density_pct, 0, "%")} />
            <Stat label="Occupancy" value={num(s.occupancy_pct, 0, "%")} />
            <Stat label="Flow" value={num(s.flow_veh_h, 0, "veh/h")} />
            <Stat label="Queue" value={num(s.queue_length_m, 0, "m")} />
            <Stat label="Stopped" value={num(s.stopped_vehicle_count)} />
            <Stat label="Data quality" value={pct(s.quality)} />
          </div>
          <p className="mt-1 text-[11px] text-slate-500">{label}</p>
        </>
      )}

      <h3 className="mt-4 mb-1 text-[11px] font-semibold tracking-widest text-slate-500 uppercase">Traffic state</h3>
      <div className="flex items-center gap-2">
        <LevelBadge level={level} stale={stale} />
        {s?.explanation?.[0] && <span className="truncate text-xs text-slate-400">{s.explanation[0]}</span>}
      </div>

      <h3 className="mt-4 mb-1 text-[11px] font-semibold tracking-widest text-slate-500 uppercase">Forecast</h3>
      <ForecastStrip forecast={snap.forecast} currentLevel={snap.window_stale ? null : snap.window?.traffic_state} />

      <h3 className="mt-4 mb-1 text-[11px] font-semibold tracking-widest text-slate-500 uppercase">Incident</h3>
      <IncidentBlock incident={snap.incident} score={snap.incident_score} />

      <h3 className="mt-4 mb-1 text-[11px] font-semibold tracking-widest text-slate-500 uppercase">Risk</h3>
      {snap.risk ? (
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <RiskBadge level={snap.risk.level} />
            <span className="text-sm text-slate-200 tabular-nums">score {snap.risk.risk_score.toFixed(0)} / 100</span>
            <span className="text-xs text-slate-500">{time(snap.risk.timestamp, false)}</span>
          </div>
          <div className="flex flex-wrap gap-1">
            {snap.risk.reason_codes.map((c) => (
              <span key={c} className="rounded bg-slate-800 px-1.5 py-0.5 text-[10px] tracking-wide text-slate-300">{c}</span>
            ))}
          </div>
          <p className="font-mono text-[10px] text-slate-500">{snap.risk.explanation[0]}</p>
        </div>
      ) : (
        <StateMessage>Risk unavailable - no assessment for this road yet.</StateMessage>
      )}
    </Card>
  );
}
