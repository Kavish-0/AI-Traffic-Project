import type { Camera, Junction } from "../api/types";
import type { Resource } from "../hooks/useResource";
import { useSelection } from "../state/selection";
import { CONGESTION_COLORS, UNKNOWN_COLOR } from "../theme/congestion";
import { Card, LevelBadge, StateMessage } from "./ui";

const CAM_COLOR: Record<string, string> = { online: "#E879F9", offline: "#FDE047", error: "#F43F5E", never_started: "#8A83AD" };

/** Junction -> approaches (roads) -> cameras. Selecting updates map, panels, charts and video. */
export function JunctionTree({ junctions, cameras }: { junctions: Resource<Junction[]>; cameras: Camera[] }) {
  const sel = useSelection();
  if (junctions.loading) return <StateMessage>Loading junctions…</StateMessage>;
  if (!junctions.data) return <StateMessage tone="error">Junctions unavailable: {junctions.error?.detail}</StateMessage>;
  const camById = Object.fromEntries(cameras.map((c) => [c.id, c]));
  return (
    <Card title="Junctions · roads · cameras">
      <ul className="space-y-2 text-sm">
        {junctions.data.map((j) => (
          <li key={j.id}>
            <button
              onClick={() => sel.selectJunction(sel.junctionId === j.id ? null : j.id)}
              className={`flex w-full items-center justify-between rounded-md px-2 py-1 text-left ${
                sel.junctionId === j.id ? "bg-slate-800" : "hover:bg-slate-800/60"
              }`}
            >
              <span className="font-semibold text-slate-100">{j.name}</span>
              <span className="flex items-center gap-2">
                {j.active_incidents > 0 && (
                  <span className="rounded bg-red-600/80 px-1.5 text-[10px] font-bold text-white">{j.active_incidents} incident</span>
                )}
                <LevelBadge level={j.worst_level} />
              </span>
            </button>
            {(sel.junctionId === j.id || j.approaches.some((a) => a.id === sel.approachId)) && (
              <ul className="mt-1 ml-3 space-y-1 border-l border-slate-800 pl-3">
                {j.approaches.map((a) => (
                  <li key={a.id}>
                    <button
                      onClick={() => sel.selectApproach(a.id, j.id)}
                      className={`flex w-full items-center justify-between rounded px-2 py-0.5 text-left ${
                        sel.approachId === a.id ? "bg-sky-900/50 text-sky-100" : "text-slate-300 hover:bg-slate-800/60"
                      }`}
                    >
                      <span className="flex items-center gap-2">
                        <span
                          className="inline-block h-1.5 w-4 rounded"
                          style={{ background: a.level ? CONGESTION_COLORS[a.level] : UNKNOWN_COLOR }}
                        />
                        {a.name} <span className="text-xs text-slate-500">({a.direction})</span>
                      </span>
                      {a.active_incident && <span className="text-[10px] font-bold text-red-400">INCIDENT</span>}
                    </button>
                    {a.cameras.length > 0 && (
                      <ul className="ml-6">
                        {a.cameras.map((cid) => {
                          const c = camById[cid];
                          return (
                            <li key={cid}>
                              <button
                                onClick={() => sel.selectCamera(cid, a.id, j.id)}
                                className={`flex items-center gap-2 rounded px-1 text-xs ${
                                  sel.cameraId === cid ? "text-cyan-200" : "text-slate-400 hover:text-slate-200"
                                }`}
                              >
                                <span className="inline-block h-2 w-2 rounded-full" style={{ background: CAM_COLOR[c?.status ?? "never_started"] }} />
                                {c?.name ?? cid}
                                {c?.synthetic && <span className="text-[10px] text-purple-300">synthetic</span>}
                              </button>
                            </li>
                          );
                        })}
                      </ul>
                    )}
                  </li>
                ))}
              </ul>
            )}
          </li>
        ))}
      </ul>
    </Card>
  );
}
