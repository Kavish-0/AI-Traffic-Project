import { useState } from "react";
import { ApiError, api } from "../api/client";
import type { Incident } from "../api/types";
import { POLL } from "../config";
import { useResource } from "../hooks/useResource";
import { dateTime, pct, time, titleCase } from "../lib/format";
import { useLive } from "../state/live";
import { useSelection } from "../state/selection";
import { scoreLabel } from "../components/RoadPanel";
import { Badge, Card, PageLayout, RiskBadge, StateMessage } from "../components/ui";
import { STATUS, TEXT } from "../theme/tokens";

const TYPE_COLOR: Record<string, string> = {
  possible_incident: STATUS.error,
  stopped_vehicle: STATUS.alert,
  wrong_way: STATUS.error,
  abnormal_queue: STATUS.warn,
  sudden_slowdown: STATUS.warn,
  abnormal_trajectory: STATUS.alert,
  normal: STATUS.ok,
};

function Detail({ incident, onChanged }: { incident: Incident; onChanged: () => void }) {
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [note, setNote] = useState("");
  const feedback = async (status: "acknowledged" | "resolved" | "dismissed", label?: string) => {
    setBusy(true);
    setErr(null);
    try {
      await api.incidentFeedback(incident.incident_id, { status, label, note: note || undefined });
      onChanged();
    } catch (e) {
      setErr(e instanceof ApiError ? e.detail : String(e));
    } finally {
      setBusy(false);
    }
  };
  const evidence = Object.entries(incident.evidence).filter(([k]) => k !== "symptom_strength").sort((a, b) => b[1] - a[1]);
  const active = incident.status === "open" || incident.status === "acknowledged";
  return (
    <Card title="Incident detail" subtitle={incident.incident_id}>
      <div className="space-y-3 text-sm">
        <div className="flex flex-wrap items-center gap-2">
          <Badge color={TYPE_COLOR[incident.incident_type]}>{incident.label || titleCase(incident.incident_type)}</Badge>
          <Badge>{incident.status}</Badge>
          <RiskBadge level={incident.risk_level} />
        </div>
        <div>
          <div className="text-xs text-slate-500">{scoreLabel(incident.calibrated)}</div>
          <div className="text-3xl font-bold text-rose-300 tabular-nums">{pct(incident.probability)}</div>
          <div className="text-xs text-slate-500">peak {pct(incident.max_probability)} · severity {incident.severity}</div>
        </div>
        <p className="text-slate-300">{incident.message}</p>
        <StateMessage tone="info">{incident.uncertainty_note}</StateMessage>
        <dl className="grid grid-cols-2 gap-1 text-xs">
          <dt className="text-slate-500">Road</dt><dd className="text-slate-200">{incident.road_id ?? "—"}</dd>
          <dt className="text-slate-500">Junction</dt><dd className="text-slate-200">{incident.junction_id ?? "—"}</dd>
          <dt className="text-slate-500">Location</dt>
          <dd className="text-slate-200">{incident.lat !== null && incident.lon !== null ? `${incident.lat.toFixed(5)}, ${incident.lon.toFixed(5)}` : "approach level"}</dd>
          <dt className="text-slate-500">First detected</dt><dd className="text-slate-200">{dateTime(incident.first_detected)}</dd>
          <dt className="text-slate-500">Last update</dt><dd className="text-slate-200">{dateTime(incident.timestamp)}</dd>
          {incident.operator_label && (<><dt className="text-slate-500">Operator label</dt><dd className="text-slate-200">{titleCase(incident.operator_label)}</dd></>)}
        </dl>
        <div>
          <h3 className="mb-1 text-[11px] font-semibold tracking-widest text-slate-500 uppercase">Evidence (fused signal families)</h3>
          {evidence.map(([k, v]) => (
            <div key={k} className="flex items-center gap-2 text-xs">
              <span className="w-40 text-slate-400">{titleCase(k)}</span>
              <span className="h-1.5 flex-1 rounded bg-slate-800"><span className="block h-1.5 rounded bg-rose-500" style={{ width: `${v * 100}%` }} /></span>
              <span className="w-8 text-right tabular-nums">{v.toFixed(2)}</span>
            </div>
          ))}
        </div>
        <div>
          <h3 className="mb-1 text-[11px] font-semibold tracking-widest text-slate-500 uppercase">Contributing signals ({incident.contributing.length})</h3>
          <ul className="max-h-40 space-y-0.5 overflow-auto text-xs text-slate-400">
            {incident.contributing.map((c, n) => (
              <li key={n}>
                {time(c.timestamp)} · {titleCase(c.type)} ({c.detector}) · score {c.score.toFixed(2)}
                {c.track_ids?.length ? ` · track ${c.track_ids.join(",")}` : ""}
              </li>
            ))}
          </ul>
        </div>
        <div className="border-t border-slate-800 pt-3">
          <h3 className="mb-2 text-[11px] font-semibold tracking-widest text-slate-500 uppercase">Operator feedback (stored for calibration)</h3>
          <input value={note} onChange={(e) => setNote(e.target.value)} placeholder="Note (optional)"
            className="mb-2 w-full rounded border border-slate-700 bg-slate-950 px-2 py-1 text-xs" />
          <div className="flex flex-wrap gap-2">
            <button disabled={busy} onClick={() => feedback("acknowledged", "true_incident")}
              className="rounded bg-emerald-700 px-3 py-1 text-xs font-bold text-white hover:bg-emerald-600 disabled:opacity-50">TRUE INCIDENT</button>
            <button disabled={busy} onClick={() => feedback("dismissed", "false_alarm")}
              className="rounded bg-slate-700 px-3 py-1 text-xs font-bold text-white hover:bg-slate-600 disabled:opacity-50">FALSE ALARM</button>
            {active && (
              <button disabled={busy} onClick={() => feedback("resolved")}
                className="rounded border border-slate-600 px-3 py-1 text-xs text-slate-200 hover:bg-slate-800 disabled:opacity-50">Mark resolved</button>
            )}
          </div>
          {err && <div className="mt-2"><StateMessage tone="error">Feedback failed: {err}</StateMessage></div>}
        </div>
      </div>
    </Card>
  );
}

export function IncidentCenter() {
  const { windowKey } = useLive();
  const sel = useSelection();
  const [filter, setFilter] = useState<"active" | "all">("active");
  const [selected, setSelected] = useState<string | null>(null);
  const list = useResource(
    () => api.incidents({ active_only: filter === "active", limit: 200 }),
    `incidents:${filter}`,
    // incidents change when a window is processed: refetch on a detected new window, slow safety interval
    { intervalMs: POLL.slowMs, refreshToken: windowKey ?? undefined },
  );
  const current = list.data?.find((i) => i.incident_id === selected) ?? null;
  return (
    <PageLayout title="Incidents" description="Possible incidents found by the anomaly detectors, with their evidence. Scores are UNCALIBRATED: each entry is for an operator to verify, and the TRUE / FALSE feedback is stored for calibration.">
    <div className="grid gap-4 lg:grid-cols-3">
      <div className="lg:col-span-2">
        <Card
          title="Incident center"
          subtitle="Every entry is a POSSIBLE incident with an uncalibrated score until labelled-event calibration - verify before acting"
          right={
            <div className="flex gap-1 text-xs">
              {(["active", "all"] as const).map((f) => (
                <button key={f} onClick={() => setFilter(f)} className={`rounded px-2 py-1 ${filter === f ? "bg-slate-700 text-white" : "text-slate-400"}`}>
                  {f === "active" ? "Active" : "All"}
                </button>
              ))}
            </div>
          }
        >
          {list.loading ? (
            <StateMessage tone="loading">Loading incidents…</StateMessage>
          ) : !list.data ? (
            <StateMessage tone="error">Incidents unavailable: {list.error?.detail}</StateMessage>
          ) : list.data.length === 0 ? (
            <StateMessage tone="info">{filter === "active" ? "No active incidents: none of the anomaly detectors has flagged a road." : "No incidents recorded."}</StateMessage>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead className="text-left text-slate-500">
                  <tr>
                    <th className="py-1">Time</th><th>Road</th><th>Junction</th><th>Type</th>
                    <th className="pr-4 text-right">Score</th><th>Risk</th><th>Evidence</th><th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {list.data.map((i) => {
                    const top = Object.entries(i.evidence).filter(([k]) => k !== "symptom_strength").sort((a, b) => b[1] - a[1])[0];
                    return (
                      <tr key={i.incident_id} onClick={() => { setSelected(i.incident_id); sel.selectApproach(i.approach_id, i.junction_id); }}
                        className={`cursor-pointer border-t border-slate-800 ${selected === i.incident_id ? "bg-slate-800/70" : "hover:bg-slate-800/40"}`}>
                        <td className="py-1.5 text-slate-300">{time(i.timestamp, false)}</td>
                        <td className="text-slate-200">{i.road_id ?? i.approach_id}</td>
                        <td className="text-slate-400">{i.junction_id ?? "—"}</td>
                        <td><span style={{ color: TYPE_COLOR[i.incident_type] ?? TEXT.secondary }}>{i.label || titleCase(i.incident_type)}</span></td>
                        <td className="pr-4 text-right text-slate-100 tabular-nums" title={scoreLabel(i.calibrated)}>
                          {pct(i.probability)}{!i.calibrated && <span className="text-slate-500">*</span>}
                        </td>
                        <td><RiskBadge level={i.risk_level} /></td>
                        <td className="text-slate-400">{top ? `${titleCase(top[0])} ${top[1].toFixed(2)}` : "—"}</td>
                        <td className="text-slate-300">{titleCase(i.status)}{i.operator_label ? ` · ${titleCase(i.operator_label)}` : ""}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
              <p className="mt-2 text-[10px] text-slate-500">* incident probability UNCALIBRATED - not yet calibrated against labelled incidents.</p>
            </div>
          )}
        </Card>
      </div>
      <div>
        {current ? (
          <Detail incident={current} onChanged={list.reload} />
        ) : (
          <Card title="Incident detail"><StateMessage>Select an incident to see its evidence and give feedback.</StateMessage></Card>
        )}
      </div>
    </div>
    </PageLayout>
  );
}
