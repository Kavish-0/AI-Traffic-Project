import type { SystemStatus } from "../api/types";
import { useLive } from "../state/live";
import { time, titleCase } from "../lib/format";
import { Dot } from "./ui";

export type Page = "dashboard" | "incidents" | "whatif" | "system";

const TABS: { id: Page; label: string }[] = [
  { id: "dashboard", label: "Dashboard" },
  { id: "incidents", label: "Incident Center" },
  { id: "whatif", label: "What-If Simulation" },
  { id: "system", label: "System" },
];

/**
 * From the REST live poll: DISCONNECTED when the last poll failed, LIVE only when the API answered AND
 * reports fresh live values (its own freshness rule, live_stale_s), otherwise WAITING FOR DATA.
 */
export function useLiveIndicator(): { label: string; ok: boolean | null } {
  const { connection, fresh } = useLive();
  if (connection === "connecting") return { label: "CONNECTING", ok: null };
  if (connection === "offline") return { label: "DISCONNECTED", ok: false };
  if (!fresh) return { label: "WAITING FOR DATA", ok: null };
  return { label: "LIVE", ok: true };
}

export function Header({
  page,
  onPage,
  system,
  activeIncidents,
}: {
  page: Page;
  onPage: (p: Page) => void;
  system: SystemStatus | undefined;
  activeIncidents: number | undefined;
}) {
  const { dataAt, pollMs } = useLive();
  const live = useLiveIndicator();
  const w = system?.weather_now;
  return (
    <header className="sticky top-0 z-20 border-b border-slate-800 bg-slate-950/95 backdrop-blur">
      <div className="flex flex-wrap items-center gap-x-6 gap-y-2 px-5 py-3">
        <div>
          <h1 className="text-sm font-bold tracking-[0.2em] text-slate-100 uppercase">
            AI Urban Traffic &amp; Incident Intelligence
          </h1>
          <p className="text-[11px] tracking-wider text-slate-500 uppercase">
            Perception → Understanding → Prediction → Anomaly → Risk → Decision support
          </p>
        </div>
        <div
          className="flex items-center gap-2 text-xs font-semibold tracking-wider"
          data-testid="live-indicator"
          title={`REST polling every ${pollMs / 1000} s`}
        >
          <Dot ok={live.ok} pulse />
          <span className={live.ok ? "text-emerald-300" : live.ok === null ? "text-slate-400" : "text-red-300"}>
            {live.label}
          </span>
        </div>
        <div className="text-xs text-slate-400" title="Time of the newest live measurement reported by the API">
          Last update: <span className="text-slate-200 tabular-nums">{time(dataAt)}</span>
        </div>
        {w && (
          <div className="text-xs text-slate-400" title={`Open-Meteo, observed ${time(w.ts, false)}`}>
            Weather: <span className="text-slate-200">{titleCase(w.weather)}</span>
            {w.temperature_c !== null && <span className="text-slate-200"> · {w.temperature_c.toFixed(1)} °C</span>}
          </div>
        )}
        <nav className="ml-auto flex gap-1">
          {TABS.map((t) => (
            <button
              key={t.id}
              onClick={() => onPage(t.id)}
              className={`rounded-md px-3 py-1.5 text-xs font-semibold tracking-wide ${
                page === t.id ? "bg-sky-600 text-white" : "text-slate-300 hover:bg-slate-800"
              }`}
            >
              {t.label}
              {t.id === "incidents" && activeIncidents ? (
                <span className="ml-1.5 rounded-full bg-red-600 px-1.5 text-[10px] text-white">{activeIncidents}</span>
              ) : null}
            </button>
          ))}
        </nav>
      </div>
    </header>
  );
}
