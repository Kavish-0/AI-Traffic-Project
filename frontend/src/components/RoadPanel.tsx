import type { Incident, Snapshot, TrafficState } from "../api/types";
import type { Resource } from "../hooks/useResource";
import { ago, num, pct, time, titleCase } from "../lib/format";
import { ForecastStrip } from "./ForecastStrip";
import { Badge, Card, RiskBadge, Stat, StateMessage, SyntheticBadge } from "./ui";
import { STATUS } from "../theme/tokens";
import { levelLabel } from "../lib/labels";

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
        <Badge color={STATUS.error}>{incident.label || titleCase(incident.incident_type)}</Badge>
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
  if (snap.live && !snap.live_stale)
    return {
      s: snap.live,
      label: snap.speed_only ? `latest ${(snap.live.window_s / 60).toFixed(0)}-min sensor reading` : `live (${snap.live.window_s.toFixed(0)} s rolling)`,
      stale: false,
    };
  if (snap.window) {
    const w = snap.window.window_s >= 120 ? `${(snap.window.window_s / 60).toFixed(0)}-min` : `${snap.window.window_s.toFixed(0)} s`;
    return { s: snap.window, label: `last ${w} window ended ${time(snap.window.timestamp, false)}`, stale: snap.window_stale };
  }
  return { s: null, label: "", stale: true };
}

/** Live values come from the snapshot (GET /approaches/{id}/snapshot, polled at the live rate). */
export function RoadPanel({ snapshot }: { snapshot: Resource<Snapshot> }) {
  if (snapshot.loading) return <Card title="Selected road"><StateMessage tone="loading">Loading road…</StateMessage></Card>;
  const snap = snapshot.data;
  if (!snap) return <Card title="Selected road"><StateMessage tone="error">Road unavailable: {snapshot.error?.detail}</StateMessage></Card>;
  const { s, stale } = current(snap);
  const level = s?.traffic_state || null;
  return (
    <Card
      title={snap.road?.name ?? snap.approach.name}
      subtitle={`${snap.approach.direction} · into ${snap.approach.junction_id}`}
      right={snap.synthetic ? <SyntheticBadge /> : null}
    >
      {snapshot.error && (
        <div className="mb-2">
          <StateMessage tone="error">Not updating - {snapshot.error.detail}. Showing the last values received.</StateMessage>
        </div>
      )}
      {!s ? (
        <StateMessage tone="warn">Waiting for traffic data… (no window for this road yet)</StateMessage>
      ) : (
        <>
          <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1" data-testid="road-headline">
            <span className="text-2xl font-bold text-slate-100">{stale ? "No recent data" : levelLabel(level)}</span>
            <span className="text-lg text-slate-300 tabular-nums">{num(s.average_speed_kmh, 0, "km/h")}</span>
            {snap.road?.speed_limit_kmh ? (
              <span className="text-xs text-slate-500">limit {num(snap.road.speed_limit_kmh, 0, "km/h")}</span>
            ) : null}
          </div>
          {stale && <div className="mt-2"><StateMessage tone="warn">Data is stale - last update {ago(s.timestamp)}.</StateMessage></div>}
          {!snap.speed_only && (
            <div className="mt-2 grid grid-cols-3 gap-2">
              <Stat label="Vehicles" value={num(s.vehicle_count)} />
              <Stat label="Queue" value={num(s.queue_length_m, 0, "m")} />
              <Stat label="Stopped" value={num(s.stopped_vehicle_count)} />
            </div>
          )}
        </>
      )}

      <h3 className="mt-3 mb-1 text-[11px] font-semibold tracking-widest text-slate-500 uppercase">Next 30 minutes</h3>
      <ForecastStrip forecast={snap.forecast} currentLevel={snap.window_stale ? null : snap.window?.traffic_state} />

      <div className="mt-3 grid gap-3 sm:grid-cols-2">
        <div>
          <h3 className="mb-1 text-[11px] font-semibold tracking-widest text-slate-500 uppercase">Incident</h3>
          <IncidentBlock incident={snap.incident} score={snap.incident_score} />
        </div>
        <div>
          <h3 className="mb-1 text-[11px] font-semibold tracking-widest text-slate-500 uppercase">Risk</h3>
          {snap.risk ? (
            <div className="flex items-center gap-2">
              <RiskBadge level={snap.risk.level} />
              <span className="text-sm text-slate-200 tabular-nums">{snap.risk.risk_score.toFixed(0)} / 100</span>
              <span className="text-xs text-slate-500">{time(snap.risk.timestamp, false)}</span>
            </div>
          ) : (
            <p className="text-xs text-slate-500">No assessment for this road yet.</p>
          )}
        </div>
      </div>
    </Card>
  );
}
