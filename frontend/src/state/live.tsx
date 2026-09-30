import { createContext, useContext, useMemo, type ReactNode } from "react";
import { api } from "../api/client";
import type { Summary, TrafficState } from "../api/types";
import { POLL } from "../config";
import { useResource, type Resource } from "../hooks/useResource";

export type Connection = "connecting" | "online" | "offline";

/**
 * Live dashboard state from REST polling (no WebSocket). Every `POLL.liveMs` the provider fetches
 *   GET /traffic/summary   KPIs from fresh live values + active incident count
 *   GET /traffic/current   latest closed 60 s window per approach (small)
 * and derives `windowKey`, which changes whenever any approach closes a new window or the active
 * incident count changes. Panels that only change per window (map colours, charts, incidents, risk)
 * refetch on that change instead of polling at the live rate.
 */
export interface LiveState {
  connection: Connection; // backend reachability, from the latest live poll
  fresh: boolean; // the backend reports >= 1 approach with fresh live values
  dataAt: string | null; // newest live measurement the backend reports (summary.timestamp)
  summary: Resource<Summary>;
  windowKey: string | null; // changes when new windows / incident changes are detected
  pollMs: number;
}

export const LiveContext = createContext<LiveState | null>(null);

function windowSignature(windows: TrafficState[] | undefined, activeIncidents: number | undefined): string | null {
  if (!windows) return null;
  // per approach: a lagging camera's new (older-timestamped) window must still count as new
  const parts = windows.map((w) => `${w.approach_id}@${w.timestamp}`).sort();
  return `${parts.join(",")}|${activeIncidents ?? "?"}`;
}

export function LiveProvider({ children }: { children: ReactNode }) {
  const summary = useResource(api.summary, "summary", { intervalMs: POLL.liveMs });
  const current = useResource(api.trafficCurrent, "traffic:current", { intervalMs: POLL.liveMs });

  const connection: Connection = summary.error ? "offline" : summary.data ? "online" : "connecting";
  const windowKey = windowSignature(current.data, summary.data?.active_incidents);

  const value = useMemo<LiveState>(
    () => ({
      connection,
      fresh: (summary.data?.fresh_approaches ?? 0) > 0,
      dataAt: summary.data?.timestamp ?? null,
      summary,
      windowKey,
      pollMs: POLL.liveMs,
    }),
    [connection, summary, windowKey],
  );
  return <LiveContext.Provider value={value}>{children}</LiveContext.Provider>;
}

export function useLive(): LiveState {
  const v = useContext(LiveContext);
  if (!v) throw new Error("useLive outside LiveProvider");
  return v;
}
