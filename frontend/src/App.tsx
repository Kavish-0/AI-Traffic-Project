import { useEffect, useState } from "react";
import { api } from "./api/client";
import { Header, type Page } from "./components/Header";
import { StateMessage } from "./components/ui";
import { WhatIfPanel } from "./components/WhatIfPanel";
import { POLL } from "./config";
import { useResource } from "./hooks/useResource";
import { Dashboard, type SharedData } from "./pages/Dashboard";
import { IncidentCenter } from "./pages/IncidentCenter";
import { SystemPage, SystemStatusCard } from "./pages/SystemPage";
import { LiveProvider, useLive } from "./state/live";
import { SelectionProvider } from "./state/selection";

const PAGES: Page[] = ["dashboard", "incidents", "whatif", "system"];
const pageFromHash = (): Page => {
  const h = location.hash.replace("#", "") as Page;
  return PAGES.includes(h) ? h : "dashboard";
};

function Shell() {
  const live = useLive();
  const [page, setPage] = useState<Page>(pageFromHash);
  useEffect(() => {
    const on = () => setPage(pageFromHash());
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  const go = (p: Page) => {
    location.hash = p;
    setPage(p);
  };

  // REST polling tiers (src/config.ts): live values come from LiveProvider every POLL.liveMs; per-window data
  // is refetched when a new window is detected (live.windowKey) with a slow safety interval; status data
  // (cameras, system) every POLL.statusMs; the road list is near-static.
  const tick = live.windowKey ?? undefined;
  const system = useResource(api.system, "system", { intervalMs: POLL.statusMs });
  const junctions = useResource(api.junctions, "junctions", { intervalMs: POLL.slowMs, refreshToken: tick });
  const roads = useResource(api.roads, "roads", { intervalMs: 5 * 60_000 });
  const network = useResource(api.network, "network", { intervalMs: POLL.slowMs, refreshToken: tick });
  const cameras = useResource(api.cameras, "cameras", { intervalMs: POLL.statusMs });
  const shared: SharedData = { junctions, roads, network, summary: live.summary, cameras: cameras.data ?? [] };
  // the live poll cannot reach the API (network error, not an HTTP error) - panels keep their last data
  const backendDown = live.connection === "offline" && live.summary.error?.status === 0;

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <Header page={page} onPage={go} system={system.data} activeIncidents={live.summary.data?.active_incidents} />
      {backendDown && (
        <div className="px-5 pt-4">
          <StateMessage tone="error">
            Backend unavailable - start the API: <code>uvicorn traffic_intel.api.main:app --reload</code> (in backend/).
          </StateMessage>
        </div>
      )}
      {system.data && !system.data.database.ok && (
        <div className="px-5 pt-4"><StateMessage tone="error">Database unavailable: {system.data.database.detail}</StateMessage></div>
      )}
      {page === "dashboard" && <Dashboard shared={shared} onOpenIncidents={() => go("incidents")} />}
      {page === "incidents" && <IncidentCenter />}
      {page === "whatif" && (
        <div className="p-5">
          <WhatIfPanel junctions={junctions} roads={roads} />
          <p className="mt-2 text-xs text-slate-500">Affected roads are highlighted on the dashboard map after a run.</p>
        </div>
      )}
      {page === "system" && <SystemPage system={system} />}
      {page === "dashboard" && (
        <div className="px-5 pb-6"><SystemStatusCard system={system} /></div>
      )}
    </div>
  );
}

export default function App() {
  return (
    <LiveProvider>
      <SelectionProvider>
        <Shell />
      </SelectionProvider>
    </LiveProvider>
  );
}
