import { useState } from "react";
import type { Junction, NetworkGeoJSON, Road } from "../api/types";
import { MapView } from "../components/MapView";
import { PageLayout, StateMessage } from "../components/ui";
import { WhatIfPanel } from "../components/WhatIfPanel";
import type { Resource } from "../hooks/useResource";
import { useSelection } from "../state/selection";
import { CHANGE, LEVEL } from "../theme/tokens";
import { AURORA } from "../theme/congestion";

/** Key for the what-if overlay drawn on the map (colours: theme/tokens.ts). */
function WhatIfLegend() {
  const items: [string, string, string][] = [
    ["Selected road", AURORA.data, "h-2"],
    ["More traffic", CHANGE.up, "h-2"],
    ["Less traffic", CHANGE.down, "h-2"],
    ["Closed", LEVEL.severe, "h-1 border-t-2 border-dashed"],
  ];
  return (
    <div className="absolute bottom-3 left-3 flex flex-wrap gap-3 rounded-lg bg-slate-950/85 px-3 py-2 text-[11px] text-slate-300">
      {items.map(([label, color]) => (
        <span key={label} className="flex items-center gap-1.5">
          <span className="inline-block h-1.5 w-5 rounded" style={{ background: color }} />
          {label}
        </span>
      ))}
    </div>
  );
}

/**
 * Road closures, lane closures and demand changes, re-assigned over the network by the backend simulator.
 * Pick the road in the list or click it on the map; after a run, the map shows the closure, where the traffic
 * goes (orange = more, teal = less, wider = bigger change) and the assumed detours that carry it.
 */
export function WhatIfPage({ junctions, roads, network }: {
  junctions: Resource<Junction[]>;
  roads: Resource<Road[]>;
  network: Resource<NetworkGeoJSON>;
}) {
  const sel = useSelection();
  const [road, setRoad] = useState<string>("");
  return (
    <PageLayout
      title="Simulate"
      description="Close a road, close a lane or raise demand, and see where traffic goes. Click a road on the map (or pick it in the list), then run. Static macroscopic re-assignment for decision support, not a guaranteed real-world prediction."
    >
      <div className="grid gap-4 xl:grid-cols-5">
        <div className="xl:col-span-2">
          <WhatIfPanel junctions={junctions} roads={roads} road={road} onRoadChange={setRoad} />
        </div>
        <div className="relative h-[60vh] min-h-[440px] overflow-hidden rounded-xl border border-slate-800 xl:col-span-3">
          {network.error && !network.data ? (
            <div className="p-4"><StateMessage tone="error">Map data unavailable: {network.error.detail}</StateMessage></div>
          ) : (
            <MapView
              network={network.data}
              selectedApproach={null}
              selectedJunction={null}
              whatIfChanges={sel.whatIfChanges}
              closedRoads={sel.closedRoads}
              pickedRoad={road || null}
              // assumed connectors / detours exist only for rerouting: only real roads can be picked
              onRoad={(_approachId, roadId) => roadId && !/^(conn_|detour_)/.test(roadId) && setRoad(roadId)}
              onJunction={() => {}}
              onCamera={() => {}}
            />
          )}
          <WhatIfLegend />
        </div>
      </div>
    </PageLayout>
  );
}
