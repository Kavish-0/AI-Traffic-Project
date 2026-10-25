import { api } from "../api/client";
import type { ServingComponent } from "../api/types";
import { Card, PageLayout, StateMessage } from "../components/ui";
import { useResource } from "../hooks/useResource";

function fmt(v: number | null): string {
  if (v === null) return "—";
  if (Number.isInteger(v) && Math.abs(v) >= 10) return String(v);
  return Math.abs(v) <= 1 && !Number.isInteger(v) ? v.toFixed(3) : v.toFixed(2);
}

function ComponentCard({ c }: { c: ServingComponent }) {
  const uncalibrated = c.implementation.includes("UNCALIBRATED");
  return (
    <Card title={c.stage}>
      <div className="text-sm font-semibold text-slate-100">{c.implementation}</div>
      {c.detail && <div className="mt-0.5 text-xs text-slate-400">{c.detail}</div>}
      {c.evidence ? (
        <div className="mt-3">
          <div className="grid grid-cols-2 gap-2">
            {Object.entries(c.evidence.metrics).map(([k, v]) => (
              <div key={k} className="rounded-lg border border-slate-800 bg-slate-950/40 px-3 py-2">
                <div className="text-[10px] tracking-wider text-slate-500 uppercase">{k}</div>
                <div className="text-lg font-bold text-sky-400 tabular-nums">{fmt(v)}</div>
              </div>
            ))}
          </div>
          <p className="mt-2 text-[11px] text-slate-500">
            Measured on {c.evidence.dataset}
            {c.evidence.unit_note ? ` · ${c.evidence.unit_note}` : ""} · record {c.evidence.experiment}
          </p>
        </div>
      ) : (
        <p className="mt-3 text-xs text-slate-400">No accuracy number: {c.no_evidence_reason}.</p>
      )}
      {uncalibrated && (
        <p className="mt-2 text-[11px] text-amber-300/80">
          Incident probability is UNCALIBRATED: the numbers above are an evaluation on real video, not a calibrated
          probability for this site.
        </p>
      )}
    </Card>
  );
}

/** What every pipeline stage runs now and its measured results (GET /models/serving, read from experiment records). */
export function ModelsPage() {
  const serving = useResource(api.servingModels, "models:serving", { intervalMs: 5 * 60_000 });
  return (
    <PageLayout
      title="AI models"
      description="What each stage of the pipeline runs right now, and how well it measured on real data. Every number comes from a saved experiment record; details and the reason for each choice are in docs/MODEL_EVALUATION.md."
    >
      {serving.loading ? (
        <StateMessage tone="loading">Loading models…</StateMessage>
      ) : !serving.data ? (
        <StateMessage tone="error">Models unavailable: {serving.error?.detail}</StateMessage>
      ) : (
        <div className="grid gap-4 md:grid-cols-2 2xl:grid-cols-3">
          {serving.data.map((c) => <ComponentCard key={c.stage} c={c} />)}
        </div>
      )}
    </PageLayout>
  );
}
