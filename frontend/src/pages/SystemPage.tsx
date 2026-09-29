import type { ComponentStatus, SystemStatus } from "../api/types";
import type { Resource } from "../hooks/useResource";
import { POLL } from "../config";
import { useLive } from "../state/live";
import { dateTime, titleCase } from "../lib/format";
import { Card, Dot, StateMessage } from "../components/ui";

function Row({ name, s }: { name: string; s: ComponentStatus | { ok: boolean | null; status: string; detail: string } }) {
  return (
    <li className="flex items-start gap-3 border-t border-slate-800 py-2 first:border-t-0">
      <span className="mt-1.5"><Dot ok={s.ok} /></span>
      <span className="w-32 shrink-0 font-semibold text-slate-200">{name}</span>
      <span className="w-40 shrink-0 text-xs font-bold tracking-wider uppercase" style={{ color: s.ok ? "#6ee7b7" : s.ok === null ? "#94a3b8" : "#fca5a5" }}>
        {s.status.replace(/_/g, " ")}
      </span>
      <span className="min-w-0 text-xs break-words text-slate-400">{s.detail}</span>
    </li>
  );
}

/** How this dashboard receives updates: REST polling of the API (no WebSocket). */
function useUpdatesRow(): { ok: boolean | null; status: string; detail: string } {
  const { connection, pollMs } = useLive();
  const every = `REST polling: live values every ${pollMs / 1000} s, per-window panels on each new window, status every ${POLL.statusMs / 1000} s`;
  if (connection === "connecting") return { ok: null, status: "connecting", detail: every };
  if (connection === "offline") return { ok: false, status: "not updating", detail: `last poll failed - ${every}` };
  return { ok: true, status: "polling", detail: every };
}

/** Status from real backend signals (database ping, heartbeats, registry, freshness) + this dashboard's polling. */
export function SystemStatusCard({ system }: { system: Resource<SystemStatus> }) {
  const updates = useUpdatesRow();
  if (system.loading) return <Card title="System status"><StateMessage>Checking backend…</StateMessage></Card>;
  if (!system.data)
    return (
      <Card title="System status">
        <ul>
          <Row name="Backend" s={{ ok: false, status: "offline", detail: system.error?.detail ?? "unreachable" }} />
          <Row name="Live updates" s={updates} />
        </ul>
      </Card>
    );
  const s = system.data;
  return (
    <Card title="System status" subtitle={`traffic_intel ${s.version} · ${s.env}`}>
      <ul className="text-sm">
        <Row name="Backend" s={s.backend} />
        <Row name="Database" s={s.database} />
        <Row name="Pipeline" s={s.pipeline} />
        <Row name="Forecast model" s={s.forecast_model} />
        <Row name="Cameras" s={s.cameras} />
        <Row name="Weather" s={s.weather} />
        <Row name="Live updates" s={updates} />
      </ul>
    </Card>
  );
}

export function SystemPage({ system }: { system: Resource<SystemStatus> }) {
  return (
    <div className="grid gap-4 p-5 lg:grid-cols-2">
      <SystemStatusCard system={system} />
      <Card title="Production models" subtitle="Model registry (models/<task>/<name>/PRODUCTION) - read-only">
        {!system.data ? (
          <StateMessage>Unavailable.</StateMessage>
        ) : system.data.models.length === 0 ? (
          <StateMessage tone="warn">No production models registered.</StateMessage>
        ) : (
          <table className="w-full text-xs">
            <thead className="text-left text-slate-500"><tr><th>Task</th><th>Model</th><th>Trained on</th><th>Trained</th></tr></thead>
            <tbody>
              {system.data.models.map((m) => (
                <tr key={`${m.task}/${m.name}`} className="border-t border-slate-800">
                  <td className="py-1 text-slate-300">{titleCase(m.task)}</td>
                  <td className="text-slate-100">{m.name}</td>
                  <td className="text-slate-300">{titleCase(m.trained_on)}</td>
                  <td className="text-slate-400">{dateTime(m.trained_at)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        <p className="mt-3 text-[11px] text-slate-500">
          The serving forecaster is trained on the synthetic live feature set; real-data results (METR-LA, PEMS-BAY,
          PEMS08, UA-DETRAC, AI City) and the reasons for each choice are in docs/MODEL_EVALUATION.md. Incident scores
          stay uncalibrated until labelled incidents from this site are available.
        </p>
      </Card>
    </div>
  );
}
