import type { Incident, Risk } from "../api/types";
import type { Resource } from "../hooks/useResource";
import { pct, time, titleCase } from "../lib/format";
import { useSelection } from "../state/selection";
import { RISK_COLORS } from "../theme/congestion";
import { Card, RiskBadge, StateMessage } from "./ui";

/** Compact list of ACTIVE incidents for the dashboard (full view: Incident Center). */
export function IncidentList({ incidents, onOpen }: { incidents: Resource<Incident[]>; onOpen: () => void }) {
  const sel = useSelection();
  return (
    <Card
      title="Incidents"
      subtitle="Possible incidents for operator verification - scores are uncalibrated unless stated"
      right={<button onClick={onOpen} className="text-xs text-sky-400 hover:underline">Incident center →</button>}
    >
      {incidents.loading ? (
        <StateMessage tone="loading">Loading incidents…</StateMessage>
      ) : !incidents.data ? (
        <StateMessage tone="error">Incidents unavailable: {incidents.error?.detail}</StateMessage>
      ) : incidents.data.length === 0 ? (
        <StateMessage>No active incidents.</StateMessage>
      ) : (
        <ul className="divide-y divide-slate-800">
          {incidents.data.map((i) => (
            <li key={i.incident_id}>
              <button
                onClick={() => sel.selectApproach(i.approach_id, i.junction_id)}
                className="flex w-full items-center justify-between gap-2 py-2 text-left hover:bg-slate-800/40"
              >
                <span>
                  <span className="block text-sm font-semibold text-slate-100">{i.label || titleCase(i.incident_type)}</span>
                  <span className="text-xs text-slate-500">
                    {i.road_id ?? i.approach_id} · {time(i.timestamp, false)} · {titleCase(i.status)}
                  </span>
                </span>
                <span className="text-right">
                  <span className="block text-lg font-bold text-rose-300 tabular-nums">{pct(i.probability)}</span>
                  <span className="text-[10px] text-slate-500">{i.calibrated ? "probability" : "score (uncalibrated)"}</span>
                </span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </Card>
  );
}

export function RiskPanel({ risk, names }: { risk: Resource<Risk[]>; names: Record<string, string> }) {
  const sel = useSelection();
  return (
    <Card title="Risk" subtitle="Weighted, documented score (0-100) with reason codes">
      {risk.loading ? (
        <StateMessage tone="loading">Loading risk…</StateMessage>
      ) : !risk.data ? (
        <StateMessage tone="error">Risk unavailable: {risk.error?.detail}</StateMessage>
      ) : risk.data.length === 0 ? (
        <StateMessage>No risk assessments yet.</StateMessage>
      ) : (
        <ul className="space-y-1.5">
          {risk.data.map((r) => (
            <li key={r.approach_id}>
              <button
                onClick={() => sel.selectApproach(r.approach_id, r.junction_id)}
                className={`w-full rounded-md px-2 py-1 text-left ${sel.approachId === r.approach_id ? "bg-slate-800" : "hover:bg-slate-800/50"}`}
              >
                <div className="flex items-center justify-between">
                  <span className="text-sm text-slate-200">{names[r.approach_id] ?? r.approach_id}</span>
                  <span className="flex items-center gap-2">
                    <span className="text-xs text-slate-400 tabular-nums">{r.risk_score.toFixed(0)}</span>
                    <RiskBadge level={r.level} />
                  </span>
                </div>
                <div className="mt-1 h-1 rounded bg-slate-800">
                  <div className="h-1 rounded" style={{ width: `${r.risk_score}%`, background: RISK_COLORS[r.level] }} />
                </div>
                <div className="mt-0.5 truncate text-[10px] text-slate-500">
                  {r.reason_codes.slice(0, 4).join(" · ") || "no reason codes"} · {time(r.timestamp, false)}
                </div>
              </button>
            </li>
          ))}
        </ul>
      )}
    </Card>
  );
}
