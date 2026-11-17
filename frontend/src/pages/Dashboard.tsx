import { useEffect, useMemo, useState } from "react";
import { api } from "../api/client";
import type { Camera, Junction, NetworkGeoJSON, Road, Summary } from "../api/types";
import { CameraPanel } from "../components/CameraPanel";
import { RiskPanel } from "../components/RiskPanel";
import { KpiBar } from "../components/KpiBar";
import { MapView } from "../components/MapView";
import { RoadPanel } from "../components/RoadPanel";
import { TrendCharts } from "../components/TrendCharts";
import { Card, StateMessage } from "../components/ui";
import { POLL } from "../config";
import { useResource, type Resource } from "../hooks/useResource";
import { useLive } from "../state/live";
import { useSelection } from "../state/selection";
import { CONGESTION_COLORS, UNKNOWN_COLOR } from "../theme/congestion";
import { LEVEL_LABEL, NO_DATA_LABEL } from "../lib/labels";
import { NeedsAttention } from "../components/NeedsAttention";
import { SampleFeedPanel } from "../components/SampleFeedPanel";

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
        <span key={level} className="flex items-center gap-1">
          <span className="inline-block h-1.5 w-4 rounded" style={{ background: color }} />
          {LEVEL_LABEL[level as keyof typeof LEVEL_LABEL]}
        </span>
      ))}
      <span className="flex items-center gap-1"><span className="inline-block h-1.5 w-4 rounded" style={{ background: UNKNOWN_COLOR }} />{NO_DATA_LABEL}</span>
      <span className="flex items-center gap-1"><span className="inline-block h-2.5 w-2.5 rounded-full bg-red-500" />incident</span>
      <span className="flex items-center gap-1"><span className="inline-block h-2 w-2 rounded-full bg-cyan-400" />camera</span>
    </div>
  );
}

export function Dashboard({ shared }: { shared: SharedData }) {
  const live = useLive();
  const sel = useSelection();
  const [horizon, setHorizon] = useState(15);
  const [notice, setNotice] = useState<string | null>(null);
  // clicking a monitored road on the map opens the sample camera feed (video mode) when that road has no
  // camera of its own; the default selection on load does not, so no camera is shown until a click
  const [sampleFeed, setSampleFeed] = useState(false);
  const windowKey = live.windowKey ?? undefined;
  const aid = sel.approachId;

  // default selection: real footage first (an online, non-synthetic camera), then any road with data
  useEffect(() => {
    if (aid || !shared.junctions.data) return;
    const all = shared.junctions.data.flatMap((j) => j.approaches);
    const realCam = shared.cameras.find((c) => c.status === "online" && !c.synthetic && c.source_kind !== "sensor_replay");
    const first =
      (realCam && all.find((a) => a.id === realCam.approach_id)) ?? all.find((a) => a.level) ?? all.find((a) => a.cameras.length) ?? all[0];
    if (first) sel.selectApproach(first.id, first.junction_id);
  }, [aid, shared.junctions.data, shared.cameras, sel]);

  // selected road: live values (1 s rolling) at the live polling rate
  const snapshot = useResource(aid ? () => api.snapshot(aid) : null, `snap:${aid}`, { intervalMs: POLL.liveMs });
  // per-window data: refetched when the live poll detects a new window, plus a slow safety interval
  const perWindow = { intervalMs: POLL.slowMs, refreshToken: windowKey };
  const history = useResource(aid ? () => api.history(aid, 120) : null, `hist:${aid}`, perWindow);
  const riskHist = useResource(aid ? () => api.riskHistory(aid) : null, `riskh:${aid}`, perWindow);
  const fva = useResource(aid ? () => api.forecastVsActual(aid, horizon) : null, `fva:${aid}:${horizon}`, perWindow);
  const risk = useResource(api.risk, "risk", perWindow);

  const names = useMemo(
    () => Object.fromEntries((shared.junctions.data ?? []).flatMap((j) => j.approaches.map((a) => [a.id, `${a.name} → ${j.name}`]))),
    [shared.junctions.data],
  );
  const camera = useMemo(() => {
    if (sel.cameraId) return shared.cameras.find((c) => c.id === sel.cameraId) ?? null;
    // the sensor-replay pseudo-camera has no video and belongs to no single road
    const mine = shared.cameras.filter((c) => c.approach_id === aid && c.source_kind !== "sensor_replay");
    return mine.find((c) => c.status === "online") ?? mine.find((c) => c.preview_available) ?? mine[0] ?? null;
  }, [sel.cameraId, shared.cameras, aid]);

  const video = camera ? <CameraPanel camera={camera} /> : sampleFeed ? <SampleFeedPanel onClose={() => setSampleFeed(false)} /> : null;
  const trends = aid ? <TrendCharts history={history} risk={riskHist} fva={fva} horizon={horizon} onHorizon={setHorizon} /> : null;

  return (
    <div className="space-y-3 p-4">
      <KpiBar summary={shared.summary} />

      {/* row 1: the map is the main view; the junction / road list scrolls beside it */}
      <div className="grid gap-3 xl:grid-cols-4">
        <div className="relative h-[62vh] min-h-[480px] overflow-hidden rounded-xl border border-slate-800 xl:col-span-3">
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
                  setSampleFeed(true);
                  sel.selectApproach(approachId);
                } else {
                  setNotice(`${roadId}: not monitored - no camera or sensor on this road (map shows it for routing / what-if only).`);
                }
              }}
              onJunction={(id) => sel.selectJunction(id)}
              onCamera={(id, approachId) => sel.selectCamera(id, approachId)}
            />
          )}
          <Legend />
          {shared.network.data?.geo_referenced === false && (
            <div className="absolute top-3 left-3 max-w-md rounded bg-slate-950/90 px-3 py-2 text-xs text-slate-300" role="note">
              {shared.network.data.location_note ?? "Schematic view: location unknown, so no city map is shown."}
            </div>
          )}
          {notice && (
            <div className="absolute top-3 left-3 max-w-sm rounded bg-slate-950/90 px-3 py-2 text-xs text-slate-300" role="status">
              {notice}
            </div>
          )}
        </div>
        <div className="max-h-[62vh] min-h-[480px] overflow-y-auto">
          <NeedsAttention junctions={shared.junctions} />
        </div>
      </div>

      {/* row 2: the selected road beside its video (its own camera, or the sample feed after a map click) - or
          beside its speed trend when there is no video, so no space goes to an empty video box */}
      <div className="grid gap-3 xl:grid-cols-5">
        <div className="xl:col-span-2">
          {aid ? <RoadPanel snapshot={snapshot} junctionName={shared.junctions.data?.find((j) => j.approaches.some((a) => a.id === aid))?.name} /> : <Card title="Selected road"><StateMessage>Select a road on the map or in the list.</StateMessage></Card>}
        </div>
        <div className="xl:col-span-3">
          {video ?? trends}
        </div>
      </div>

      {/* row 3 (roads with a camera): the speed trend; then the riskiest roads. Incidents have their own page,
          the what-if its own Simulate page and the component health the System page. */}
      {video && trends}
      <RiskPanel risk={risk} names={names} />
    </div>
  );
}
