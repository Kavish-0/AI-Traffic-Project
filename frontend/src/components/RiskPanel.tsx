import type { Risk } from "../api/types";
import type { Resource } from "../hooks/useResource";
import { time, titleCase } from "../lib/format";
import { useSelection } from "../state/selection";
import { RISK_COLORS } from "../theme/congestion";
import { Card, RiskBadge, StateMessage } from "./ui";

const RISK_ROWS = 8;

export function RiskPanel({ risk, names }: { risk: Resource<Risk[]>; names: Record<string, string> }) {
  const sel = useSelection();
  return (
    <Card title="Riskiest roads" subtitle="Weighted, documented score (0-100) with the reasons behind it">
      {risk.loading ? (
        <StateMessage tone="loading">Loading risk…</StateMessage>
      ) : !risk.data ? (
        <StateMessage tone="error">Risk unavailable: {risk.error?.detail}</StateMessage>
      ) : risk.data.length === 0 ? (
        <StateMessage>No risk assessments yet.</StateMessage>
      ) : (
        <ul className="space-y-1.5">
          {risk.data.slice(0, RISK_ROWS).map((r) => (
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
                  {r.reason_codes.slice(0, 3).map((c) => titleCase(c.toLowerCase())).join(" · ") || "no warning signs"} · {time(r.timestamp, false)}
                </div>
              </button>
            </li>
          ))}
          {risk.data.length > RISK_ROWS && (
            <li className="px-2 pt-1 text-[11px] text-slate-500">+{risk.data.length - RISK_ROWS} lower-risk roads not shown</li>
          )}
        </ul>
      )}
    </Card>
  );
}
