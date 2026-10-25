import type { Approach, Junction } from "../api/types";
import type { Resource } from "../hooks/useResource";
import { LEVEL_RANK, levelLabel } from "../lib/labels";
import { useSelection } from "../state/selection";
import { CONGESTION_COLORS } from "../theme/congestion";
import { STATUS } from "../theme/tokens";
import { Card, StateMessage } from "./ui";

const MAX_ROWS = 8;

/** Worst first: an open incident outranks any traffic level, then Jammed > Heavy > Slow. */
function rank(a: Approach): number {
  return (a.active_incident ? 10 : 0) + (a.level ? LEVEL_RANK[a.level] : -1);
}

/**
 * Simple-view replacement for the full junction / road list: only the roads an operator should look at
 * now (an open incident, or Slow / Heavy / Jammed), worst first. Everything comes from the backend levels.
 */
export function NeedsAttention({ junctions }: { junctions: Resource<Junction[]> }) {
  const sel = useSelection();
  if (junctions.loading) return <StateMessage tone="loading">Loading roads…</StateMessage>;
  if (!junctions.data) return <StateMessage tone="error">Roads unavailable: {junctions.error?.detail}</StateMessage>;
  const corridor = Object.fromEntries(junctions.data.map((j) => [j.id, j.name]));
  const all = junctions.data.flatMap((j) => j.approaches);
  const monitored = all.filter((a) => a.level).length;
  const flagged = all.filter((a) => a.active_incident || (a.level && a.level !== "free")).sort((a, b) => rank(b) - rank(a));
  return (
    <Card title="Needs attention" subtitle={`${flagged.length} of ${monitored} roads with live data`}>
      {flagged.length === 0 ? (
        <StateMessage tone="info">All monitored roads are flowing and no incident is open.</StateMessage>
      ) : (
        <ul className="space-y-1">
          {flagged.slice(0, MAX_ROWS).map((a) => (
            <li key={a.id}>
              <button
                onClick={() => sel.selectApproach(a.id, a.junction_id)}
                className={`flex w-full items-center gap-3 rounded-lg px-2 py-2 text-left hover:bg-slate-800 ${
                  sel.approachId === a.id ? "bg-slate-800" : ""
                }`}
              >
                <span className="h-8 w-1.5 shrink-0 rounded-full"
                      style={{ background: a.active_incident ? STATUS.error : a.level ? CONGESTION_COLORS[a.level] : undefined }} />
                <span className="min-w-0 flex-1">
                  <span className="block truncate text-sm text-slate-100">{a.name.split(" (")[0]}</span>
                  <span className="block truncate text-[11px] text-slate-500">{corridor[a.junction_id]}</span>
                </span>
                <span className="shrink-0 text-xs font-semibold"
                      style={{ color: a.active_incident ? STATUS.error : a.level ? CONGESTION_COLORS[a.level] : undefined }}>
                  {a.active_incident ? "Incident" : levelLabel(a.level)}
                </span>
              </button>
            </li>
          ))}
          {flagged.length > MAX_ROWS && (
            <li className="px-2 pt-1 text-[11px] text-slate-500">
              +{flagged.length - MAX_ROWS} more, shown on the map
            </li>
          )}
        </ul>
      )}
    </Card>
  );
}
