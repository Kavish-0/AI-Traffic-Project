import type {
  AnomalySignal,
  Camera,
  Forecast,
  ForecastVsActual,
  Incident,
  IncidentStatus,
  Junction,
  NetworkGeoJSON,
  Risk,
  Road,
  SimulationRequest,
  SimulationResult,
  Snapshot,
  Summary,
  SystemStatus,
  TrafficState,
} from "./types";

export const API_BASE = "/api/v1";

/** HTTP error with the backend's explanation (FastAPI `detail`). */
export class ApiError extends Error {
  constructor(
    public status: number,
    public detail: string,
    public path: string,
  ) {
    super(`${status}: ${detail}`);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, { headers: { "Content-Type": "application/json" }, ...init });
  } catch {
    throw new ApiError(0, "Backend unavailable - is the API running (uvicorn traffic_intel.api.main:app)?", path);
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = (await res.json()) as { detail?: unknown };
      if (typeof body.detail === "string") detail = body.detail;
      else if (body.detail) detail = JSON.stringify(body.detail);
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(res.status, detail, path);
  }
  return (await res.json()) as T;
}

const q = (params: Record<string, string | number | boolean | undefined | null>) => {
  const s = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) if (v !== undefined && v !== null && v !== "") s.set(k, String(v));
  const str = s.toString();
  return str ? `?${str}` : "";
};

export const api = {
  health: () => request<{ status: string; version: string; database: string }>("/health"),
  system: () => request<SystemStatus>("/system/status"),
  network: () => request<NetworkGeoJSON>("/network"),
  junctions: () => request<Junction[]>("/junctions"),
  roads: () => request<Road[]>("/roads"),
  cameras: () => request<Camera[]>("/cameras"),
  summary: () => request<Summary>("/traffic/summary"),
  /** Latest closed 60 s window per approach (small; polled to detect new windows). */
  trafficCurrent: () => request<TrafficState[]>("/traffic/current"),
  snapshot: (approachId: string) => request<Snapshot>(`/approaches/${encodeURIComponent(approachId)}/snapshot`),
  history: (approachId: string, minutes = 120) =>
    request<TrafficState[]>(`/approaches/${encodeURIComponent(approachId)}/history${q({ minutes })}`),
  forecast: (approachId: string) => request<Forecast>(`/approaches/${encodeURIComponent(approachId)}/forecast`),
  forecastVsActual: (approachId: string, horizon = 15, target = "avg_speed_kmh", minutes = 180) =>
    request<ForecastVsActual>(
      `/approaches/${encodeURIComponent(approachId)}/forecast/history${q({ horizon_min: horizon, target, minutes })}`,
    ),
  riskHistory: (approachId: string, limit = 240) =>
    request<Risk[]>(`/risk/${encodeURIComponent(approachId)}/history${q({ limit })}`),
  risk: () => request<Risk[]>("/risk"),
  incidents: (params: { active_only?: boolean; status?: IncidentStatus; junction_id?: string; limit?: number } = {}) =>
    request<Incident[]>(`/incidents${q(params)}`),
  incident: (id: string) => request<Incident>(`/incidents/${encodeURIComponent(id)}`),
  incidentFeedback: (
    id: string,
    body: { status: "acknowledged" | "resolved" | "dismissed"; label?: string; note?: string },
  ) => request<Incident>(`/incidents/${encodeURIComponent(id)}`, { method: "PATCH", body: JSON.stringify(body) }),
  anomalies: (params: { approach_id?: string; junction_id?: string; limit?: number } = {}) =>
    request<AnomalySignal[]>(`/anomalies${q(params)}`),
  whatIf: (body: SimulationRequest) =>
    request<SimulationResult>("/simulation/what-if", { method: "POST", body: JSON.stringify(body) }),
  cameraFrameUrl: (cameraId: string, bust: number) =>
    `${API_BASE}/cameras/${encodeURIComponent(cameraId)}/frame.jpg?t=${bust}`,
};
