import type { SystemStatus } from "../api/types";
import { useLive } from "../state/live";
import { time, titleCase } from "../lib/format";
import { Dot } from "./ui";

export type Page = "dashboard" | "incidents" | "whatif" | "models" | "system";

/** Main pages first; technical pages are grouped under "More". */
const TABS: { id: Page; label: string; more?: boolean }[] = [
  { id: "dashboard", label: "Overview" },
  { id: "incidents", label: "Incidents" },
  { id: "whatif", label: "Simulate" },
  { id: "models", label: "AI models", more: true },
  { id: "system", label: "System", more: true },
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
  // One element for every screen size: a top bar on small screens, a fixed left sidebar from lg up
  // (the page content is offset by SIDEBAR_WIDTH in App.tsx).
  return (
    <aside className="sticky top-0 z-20 flex flex-wrap items-center gap-x-6 gap-y-2 border-b border-slate-800 bg-slate-950/95 px-5 py-3 backdrop-blur lg:fixed lg:inset-y-0 lg:left-0 lg:w-60 lg:flex-col lg:flex-nowrap lg:items-stretch lg:gap-0 lg:border-r lg:border-b-0 lg:px-4 lg:py-5">
      <div className="lg:mb-6">
        <h1 className="text-sm font-bold tracking-[0.2em] text-slate-100 uppercase">
          AI Urban Traffic &amp; Incident Intelligence
        </h1>
        <p className="text-[11px] tracking-wider text-slate-500 uppercase lg:mt-1">
          Perception → Understanding → Prediction → Anomaly → Risk → Decision support
        </p>
      </div>
      <nav className="flex gap-1 lg:order-last lg:mt-6 lg:flex-col" aria-label="Pages">
        {TABS.map((t, i) => (
          <div key={t.id} className="contents">
          {t.more && !TABS[i - 1]?.more && (
            <div className="hidden px-3 pt-4 pb-1 text-[10px] tracking-widest text-slate-500 uppercase lg:block">More</div>
          )}
          <button
            key={t.id}
            onClick={() => onPage(t.id)}
            aria-current={page === t.id ? "page" : undefined}
            className={`flex items-center justify-between rounded-md px-3 py-1.5 text-left text-xs font-semibold tracking-wide lg:py-2.5 lg:text-sm ${
              page === t.id ? "bg-sky-600 text-white" : "text-slate-300 hover:bg-slate-800"
            }`}
          >
            {t.label}
            {t.id === "incidents" && activeIncidents ? (
              <span className="ml-1.5 rounded-full bg-red-600 px-1.5 text-[10px] text-white">{activeIncidents}</span>
            ) : null}
          </button>
          </div>
        ))}
      </nav>
      <div className="flex flex-wrap items-center gap-x-6 gap-y-2 lg:flex-col lg:items-start lg:gap-2 lg:rounded-lg lg:border lg:border-slate-800 lg:bg-slate-900/60 lg:p-3">
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
      </div>
    </aside>
  );
}

