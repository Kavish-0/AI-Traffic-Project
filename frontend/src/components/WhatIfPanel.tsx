import { useMemo, useState } from "react";
import { ApiError, api } from "../api/client";
import type { Junction, Road, SimulationRequest, SimulationResult } from "../api/types";
import type { Resource } from "../hooks/useResource";
import { pct } from "../lib/format";
import { useSelection } from "../state/selection";
import { CONGESTION_COLORS } from "../theme/congestion";
import { Badge, Card, StateMessage } from "./ui";

type Action = "close" | "lane_closure" | "demand";

const ACTIONS: { id: Action; label: string }[] = [
  { id: "close", label: "Close road" },
  { id: "lane_closure", label: "Lane closure (50% capacity)" },
  { id: "demand", label: "Demand +20% (whole network)" },
];

export function buildRequest(action: Action, roadId: string): SimulationRequest {
  if (action === "close") return { closed_segment_ids: [roadId] };
  if (action === "lane_closure") return { closed_segment_ids: [], capacity_factors: { [roadId]: 0.5 } };
  return { closed_segment_ids: [], demand_factor: 1.2 };
}

/** What-if (static assignment on the road graph, run by the backend). Decision support - not a prediction. */
export function WhatIfPanel({ junctions, roads }: { junctions: Resource<Junction[]>; roads: Resource<Road[]> }) {
  const sel = useSelection();
  const [junction, setJunction] = useState<string>("");
  const [road, setRoad] = useState<string>("");
  const [action, setAction] = useState<Action>("close");
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<SimulationResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const roadOptions = useMemo(() => {
    const all = roads.data ?? [];
    if (!junction) return all;
    const j = junctions.data?.find((x) => x.id === junction);
    const ids = new Set(j?.approaches.map((a) => a.road_id));
    return all.filter((r) => ids.has(r.id) || r.junction_id === junction);
  }, [roads.data, junctions.data, junction]);

  const run = async () => {
    const target = road || roadOptions[0]?.id;
    if (!target && action !== "demand") return;
    setRunning(true);
    setError(null);
    try {
      const res = await api.whatIf(buildRequest(action, target ?? ""));
      setResult(res);
      sel.setWhatIfOverlay(
        Object.fromEntries(res.impacts.filter((i) => Math.abs(i.flow_change_pct) >= 1).map((i) => [i.segment_id, i.flow_change_pct])),
        res.impacts.filter((i) => i.status === "closed").map((i) => i.segment_id),
      );
    } catch (e) {
      setResult(null);
      setError(e instanceof ApiError ? e.detail : String(e));
    } finally {
      setRunning(false);
    }
  };

  const changed = result?.impacts
    .filter((i) => i.status !== "open" || Math.abs(i.flow_change_pct) >= 1)
    .sort((a, b) => Math.abs(b.flow_change_pct) - Math.abs(a.flow_change_pct));

  return (
    <Card
      title="What-if simulation"
      subtitle="Simulation / Decision Support - static macroscopic re-assignment, NOT a guaranteed real-world prediction"
      right={<Badge color="#E879F9">decision support</Badge>}
    >
      {roads.loading || junctions.loading ? (
        <StateMessage>Loading road network…</StateMessage>
      ) : !roads.data ? (
        <StateMessage tone="error">Road network unavailable: {roads.error?.detail}</StateMessage>
      ) : (
        <div className="flex flex-wrap items-end gap-3 text-sm">
          <label className="flex flex-col gap-1 text-xs text-slate-400">
            Junction
            <select value={junction} onChange={(e) => { setJunction(e.target.value); setRoad(""); }}
              className="rounded border border-slate-700 bg-slate-900 px-2 py-1 text-slate-100">
              <option value="">All roads</option>
              {junctions.data?.map((j) => <option key={j.id} value={j.id}>{j.name}</option>)}
            </select>
          </label>
          <label className="flex flex-col gap-1 text-xs text-slate-400">
            Road
            <select value={road} onChange={(e) => setRoad(e.target.value)} disabled={action === "demand"}
              className="rounded border border-slate-700 bg-slate-900 px-2 py-1 text-slate-100 disabled:opacity-40" aria-label="Road">
              {roadOptions.map((r) => <option key={r.id} value={r.id}>{r.name}</option>)}
            </select>
          </label>
          <label className="flex flex-col gap-1 text-xs text-slate-400">
            Action
            <select value={action} onChange={(e) => setAction(e.target.value as Action)}
              className="rounded border border-slate-700 bg-slate-900 px-2 py-1 text-slate-100" aria-label="Action">
              {ACTIONS.map((a) => <option key={a.id} value={a.id}>{a.label}</option>)}
            </select>
          </label>
          <button onClick={run} disabled={running}
            className="rounded-md bg-sky-600 px-4 py-1.5 text-xs font-bold tracking-wider text-white uppercase hover:bg-sky-500 disabled:opacity-50">
            {running ? "Running…" : "Run simulation"}
          </button>
          {result && (
            <button onClick={() => { setResult(null); sel.setWhatIfOverlay(null); }} className="text-xs text-slate-400 hover:underline">
              Clear
            </button>
          )}
        </div>
      )}

      {error && <div className="mt-3"><StateMessage tone="error">Simulation failed: {error}</StateMessage></div>}

      {result && (
        <div className="mt-4 grid gap-4 lg:grid-cols-3">
          <div className="lg:col-span-2">
            <h3 className="mb-1 text-[11px] font-semibold tracking-widest text-slate-500 uppercase">Affected roads</h3>
            {changed && changed.length ? (
              <table className="w-full text-xs">
                <thead className="text-slate-500">
                  <tr><th className="py-1 text-left">Road</th><th className="text-right">Traffic change</th>
                    <th className="text-right">Congestion probability</th><th className="text-right">Level</th></tr>
                </thead>
                <tbody>
                  {changed.map((i) => (
                    <tr key={i.segment_id} className="border-t border-slate-800">
                      <td className="py-1 text-slate-200">{i.name}{!i.baseline_observed && <span className="ml-1 text-slate-500" title="baseline flow assumed (no camera)">*</span>}</td>
                      <td className="text-right tabular-nums" style={{ color: i.status === "closed" ? "#f87171" : i.flow_change_pct > 0 ? "#fbbf24" : "#86efac" }}>
                        {i.status === "closed" ? "CLOSED" : `${i.flow_change_pct > 0 ? "+" : ""}${i.flow_change_pct.toFixed(0)}%`}
                      </td>
                      <td className="text-right text-slate-300 tabular-nums">
                        {pct(i.baseline_congestion_prob)} → {i.status === "closed" ? "—" : pct(i.scenario_congestion_prob)}
                      </td>
                      <td className="text-right uppercase" style={{ color: CONGESTION_COLORS[i.scenario_level] }}>
                        {i.status === "closed" ? "—" : i.scenario_level}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <StateMessage>No material change predicted by the static model.</StateMessage>
            )}
            <p className="mt-1 text-[10px] text-slate-500">* baseline flow assumed (no camera on that road).</p>
          </div>
          <div className="space-y-3">
            <div>
              <h3 className="mb-1 text-[11px] font-semibold tracking-widest text-slate-500 uppercase">Junction congestion probability</h3>
              <ul className="space-y-1 text-xs">
                {result.junctions.map((j) => (
                  <li key={j.junction_id} className="flex justify-between">
                    <span className="text-slate-300">{j.name}</span>
                    <span className="text-slate-200 tabular-nums">{pct(j.baseline_congestion_prob)} → {pct(j.scenario_congestion_prob)}</span>
                  </li>
                ))}
              </ul>
              {result.unserved_demand_veh_h > 0 && (
                <div className="mt-2"><StateMessage tone="warn">{result.unserved_demand_veh_h.toFixed(0)} veh/h have no alternative route.</StateMessage></div>
              )}
            </div>
            <details className="text-xs text-slate-400">
              <summary className="cursor-pointer text-slate-300">Assumptions ({result.assumptions.length})</summary>
              <ul className="mt-1 list-disc space-y-1 pl-4">
                {result.assumptions.map((a) => <li key={a}>{a}</li>)}
              </ul>
            </details>
          </div>
        </div>
      )}
    </Card>
  );
}
