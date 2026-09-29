import { useEffect, useMemo, useState } from "react";
import { api } from "../api/client";
import type { Camera, Junction, NetworkGeoJSON, Road, Summary } from "../api/types";
import { CameraPanel } from "../components/CameraPanel";
import { IncidentList, RiskPanel } from "../components/IncidentList";
import { JunctionTree } from "../components/JunctionTree";
import { KpiBar } from "../components/KpiBar";
import { MapView } from "../components/MapView";
import { RoadPanel } from "../components/RoadPanel";
import { TrendCharts } from "../components/TrendCharts";
import { WhatIfPanel } from "../components/WhatIfPanel";
import { Card, StateMessage } from "../components/ui";
import { POLL } from "../config";
import { useResource, type Resource } from "../hooks/useResource";
import { useLive } from "../state/live";
import { useSelection } from "../state/selection";
import { CONGESTION_COLORS, UNKNOWN_COLOR } from "../theme/congestion";

export interface SharedData {
  junctions: Resource<Junction[]>;
  roads: Resource<Road[]>;
  cameras: Camera[];
  network: Resource<NetworkGeoJSON>;
  summary: Resource<Summary>;
}

function Legend() {
  return (
    <div className="absolute bottom-3 left-3 flex flex-wrap gap-3 rounded-lg bg-slate-950/85 px-3 py-2 text-[11px] text-slate-300">
      {Object.entries(CONGESTION_COLORS).map(([level, color]) => (
        <span key={level} className="flex items-center gap-1 capitalize">
          <span className="inline-block h-1.5 w-4 rounded" style={{ background: color }} />
          {level}
        </span>
      ))}
      <span className="flex items-center gap-1"><span className="inline-block h-1.5 w-4 rounded" style={{ background: UNKNOWN_COLOR }} />no data</span>
      <span className="flex items-center gap-1"><span className="inline-block h-2.5 w-2.5 rounded-full bg-red-500" />incident</span>
      <span className="flex items-center gap-1"><span className="inline-block h-2 w-2 rounded-full bg-cyan-400" />camera</span>
    </div>
  );
}

export function Dashboard({ shared, onOpenIncidents }: { shared: SharedData; onOpenIncidents: () => void }) {
  const live = useLive();
  const sel = useSelection();
  const [horizon, setHorizon] = useState(15);
  const [notice, setNotice] = useState<string | null>(null);
  const windowKey = live.windowKey ?? undefined;
  const aid = sel.approachId;

  // default selection: the first approach that has a camera
  useEffect(() => {
    if (aid || !shared.junctions.data) return;
    const all = shared.junctions.data.flatMap((j) => j.approaches);
    const first = all.find((a) => a.level) ?? all.find((a) => a.cameras.length) ?? all[0];
    if (first) sel.selectApproach(first.id, first.junction_id);
  }, [aid, shared.junctions.data, sel]);

  // selected road: live values (1 s rolling) at the live polling rate
  const snapshot = useResource(aid ? () => api.snapshot(aid) : null, `snap:${aid}`, { intervalMs: POLL.liveMs });
  // per-window data: refetched when the live poll detects a new window, plus a slow safety interval
  const perWindow = { intervalMs: POLL.slowMs, refreshToken: windowKey };
  const history = useResource(aid ? () => api.history(aid, 120) : null, `hist:${aid}`, perWindow);
  const riskHist = useResource(aid ? () => api.riskHistory(aid) : null, `riskh:${aid}`, perWindow);
  const fva = useResource(aid ? () => api.forecastVsActual(aid, horizon) : null, `fva:${aid}:${horizon}`, perWindow);
  const incidents = useResource(() => api.incidents({ active_only: true, limit: 20 }), "incidents:active", perWindow);
  const risk = useResource(api.risk, "risk", perWindow);

  const names = useMemo(
    () => Object.fromEntries((shared.junctions.data ?? []).flatMap((j) => j.approaches.map((a) => [a.id, `${a.name} → ${j.name}`]))),
    [shared.junctions.data],
  );
  const camera = useMemo(() => {
    if (sel.cameraId) return shared.cameras.find((c) => c.id === sel.cameraId) ?? null;
    const mine = shared.cameras.filter((c) => c.approach_id === aid);
    return mine.find((c) => c.status === "online") ?? mine.find((c) => c.preview_available) ?? mine[0] ?? null;
  }, [sel.cameraId, shared.cameras, aid]);

  return (
    <div className="space-y-4 p-5">
      <KpiBar summary={shared.summary} />

      <div className="grid gap-4 xl:grid-cols-3">
        {/* on wide screens the map fills the row (as tall as the junction list + camera beside it) */}
        <div className="relative h-[520px] overflow-hidden rounded-xl border border-slate-800 xl:col-span-2 xl:h-auto xl:min-h-[520px]">
          {shared.network.error && !shared.network.data ? (
            <div className="p-4"><StateMessage tone="error">Map data unavailable: {shared.network.error.detail}</StateMessage></div>
          ) : (
            <MapView
              network={shared.network.data}
              selectedApproach={aid}
              selectedJunction={sel.junctionId}
              whatIfChanges={sel.whatIfChanges}
              closedRoads={sel.closedRoads}
              onRoad={(approachId, roadId) => {
                if (approachId) {
                  setNotice(null);
                  sel.selectApproach(approachId);
                } else {
                  setNotice(`${roadId}: not monitored - no camera on this road (map shows it for routing / what-if only).`);
                }
              }}
              onJunction={(id) => sel.selectJunction(id)}
              onCamera={(id, approachId) => sel.selectCamera(id, approachId)}
            />
          )}
          <Legend />
          {notice && (
            <div className="absolute top-3 left-3 max-w-sm rounded bg-slate-950/90 px-3 py-2 text-xs text-slate-300" role="status">
              {notice}
            </div>
          )}
        </div>
        <div className="space-y-4">
          <JunctionTree junctions={shared.junctions} cameras={shared.cameras} />
          <CameraPanel camera={camera} />
        </div>
      </div>

      <div className="grid gap-4 xl:grid-cols-3">
        <div className="xl:col-span-1">
          {aid ? <RoadPanel snapshot={snapshot} /> :<Card title="Selected road"><StateMessage>Select a road on the map or in the junction list.</StateMessage></Card>}
        </div>
        <div className="space-y-4 xl:col-span-2">
          {aid && <TrendCharts history={history} risk={riskHist} fva={fva} horizon={horizon} onHorizon={setHorizon} />}
          <div className="grid gap-4 md:grid-cols-2">
            <IncidentList incidents={incidents} onOpen={onOpenIncidents} />
            <RiskPanel risk={risk} names={names} />
          </div>
        </div>
      </div>

      <WhatIfPanel junctions={shared.junctions} roads={shared.roads} />
    </div>
  );
}
