// Mirrors backend/traffic_intel/api/schemas.py (the REST contract) and the parts of
// core/schemas.py the dashboard uses. Keep in sync; values that are unavailable are null, never
// placeholders.

export type CongestionLevel = "free" | "moderate" | "high" | "severe";
export type RiskLevel = "low" | "medium" | "high" | "critical";
export type IncidentStatus = "open" | "acknowledged" | "resolved" | "dismissed";
export type SourceKind = "file" | "webcam" | "stream" | "synthetic" | "sensor_replay" | "unknown";
export type CameraStatus = "online" | "offline" | "error" | "never_started";

export interface TrafficState {
  timestamp: string;
  window_start: string;
  window_s: number;
  is_live: boolean;
  camera_id: string | null;
  approach_id: string;
  junction_id: string | null;
  road_id: string | null;
  vehicle_count: number;
  car_count: number;
  motorcycle_count: number;
  bus_count: number;
  truck_count: number;
  bicycle_count: number;
  pedestrian_count: number;
  average_speed_kmh: number | null;
  median_speed_kmh: number | null;
  density_veh_km: number;
  density_pct: number;
  occupancy_pct: number;
  flow_veh_h: number;
  inflow: number;
  outflow: number;
  queue_length_m: number;
  stopped_vehicle_count: number;
  traffic_state: CongestionLevel | null;
  congestion_score: number | null;
  explanation: string[];
  quality: number;
}

export interface Evidence {
  dataset: string;
  experiment: string;
  metrics: Record<string, number | null>;
  unit_note: string | null;
}

export interface ServingComponent {
  stage: string;
  implementation: string;
  detail: string;
  evidence: Evidence | null;
  no_evidence_reason: string | null;
}

export interface ModelInfo {
  task: string;
  name: string;
  version: string;
  trained_on: string | null;
  resolution_s: number | null;
  required_history: number | null;
  trained_at: string | null;
  metrics: Record<string, unknown>;
}

export interface ForecastHorizon {
  horizon_min: number;
  target_time: string;
  level: CongestionLevel | null;
  values: Record<string, number>;
}

export type ForecastStatus = "ok" | "insufficient_history" | "no_model" | "stale" | "no_data";

export interface Forecast {
  approach_id: string;
  status: ForecastStatus;
  message: string;
  model: ModelInfo | null;
  issued_at: string | null;
  current_level: CongestionLevel | null;
  horizons: ForecastHorizon[];
  history_windows: number;
}

export interface ForecastVsActual {
  approach_id: string;
  horizon_min: number;
  target: string;
  forecast: { issued_at: string; target_time: string; value: number; level: string | null }[];
  actual: { timestamp: string; value: number | null }[];
}

export interface AnomalySignal {
  id: number;
  approach_id: string;
  camera_id: string | null;
  junction_id: string | null;
  road_id: string | null;
  timestamp: string;
  type: string;
  scope: string;
  detector: string;
  score: number;
  track_ids: number[] | null;
  lon: number | null;
  lat: number | null;
  details: Record<string, unknown> | null;
  incident_id: string | null;
}

export interface ContributingSignal {
  type: string;
  detector: string;
  score: number;
  timestamp: string;
  track_ids?: number[];
  details?: Record<string, unknown>;
}

export interface Incident {
  incident_id: string;
  timestamp: string;
  first_detected: string | null;
  resolved_at: string | null;
  approach_id: string;
  junction_id: string | null;
  road_id: string | null;
  incident_type: string;
  label: string;
  probability: number;
  max_probability: number;
  calibrated: boolean;
  score_label: string;
  severity: string;
  risk_level: RiskLevel | null;
  status: IncidentStatus;
  message: string;
  evidence: Record<string, number>;
  lon: number | null;
  lat: number | null;
  location_description: string | null;
  operator_label: string | null;
  operator_note: string | null;
  contributing: ContributingSignal[];
  uncertainty_note: string;
}

export interface Risk {
  approach_id: string;
  timestamp: string;
  junction_id: string | null;
  road_id: string | null;
  risk_score: number;
  level: RiskLevel;
  reason_codes: string[];
  components: Record<string, number>;
  explanation: string[];
  incident_score: number | null;
}

export interface Camera {
  id: string;
  name: string;
  approach_id: string;
  junction_id: string | null;
  source_kind: SourceKind;
  synthetic: boolean;
  configured: boolean;
  lon: number | null;
  lat: number | null;
  status: CameraStatus;
  state: string | null;
  last_frame_at: string | null;
  updated_at: string | null;
  processing_fps: number | null;
  frames: number | null;
  detector: string | null;
  device: string | null;
  error: string | null;
  preview_available: boolean;
}

export interface Approach {
  id: string;
  name: string;
  direction: string;
  junction_id: string;
  road_id: string;
  cameras: string[];
  level: CongestionLevel | null;
  risk_level: RiskLevel | null;
  active_incident: string | null;
}

export interface Junction {
  id: string;
  name: string;
  lon: number;
  lat: number;
  signalized: boolean;
  approaches: Approach[];
  active_incidents: number;
  worst_level: CongestionLevel | null;
}

export interface Road {
  id: string;
  name: string;
  from_node: string;
  to_node: string;
  lanes: number;
  speed_limit_kmh: number;
  capacity_veh_h: number | null;
  importance: number | null;
  approach_id: string | null;
  junction_id: string | null;
  monitored: boolean;
  level: CongestionLevel | null;
  stale: boolean;
  coordinates: number[][];
}

export interface Snapshot {
  approach: Approach;
  road: Road | null;
  cameras: Camera[];
  live: TrafficState | null;
  live_stale: boolean;
  window: TrafficState | null;
  window_stale: boolean;
  forecast: Forecast;
  incident: Incident | null;
  incident_score: number | null;
  risk: Risk | null;
  synthetic: boolean;
  /** road sensor (e.g. PEMS-BAY loop detector): speed measured; counts, density, occupancy, queue are not */
  speed_only?: boolean;
  source_note?: string | null;
}

export interface Summary {
  timestamp: string | null;
  vehicles: number | null;
  avg_speed_kmh: number | null;
  congested_approaches: number;
  monitored_approaches: number;
  fresh_approaches: number;
  active_incidents: number;
  synthetic_approaches: number;
  speed_only_approaches?: number;
  sensor_source?: string | null;
}

export interface ComponentStatus {
  ok: boolean;
  status: string;
  detail: string;
}

export interface SystemStatus {
  backend: ComponentStatus;
  database: ComponentStatus;
  pipeline: ComponentStatus;
  forecast_model: ComponentStatus;
  cameras: ComponentStatus;
  weather: ComponentStatus;
  websocket_clients: number; // optional backend push channel (off by default); not used by this dashboard
  version: string;
  env: string;
  models: ModelInfo[];
  weather_now: {
    ts: string;
    weather: string | null;
    temperature_c: number | null;
    precipitation_mm: number | null;
    visibility_km: number | null;
    source: string;
  } | null;
}

export interface SimulationRequest {
  closed_segment_ids: string[];
  capacity_factors?: Record<string, number>;
  demand_factor?: number;
}

export interface SegmentImpact {
  segment_id: string;
  name: string;
  status: "open" | "closed" | "reduced";
  baseline_observed: boolean;
  baseline_flow_veh_h: number;
  scenario_flow_veh_h: number;
  flow_change_pct: number;
  baseline_congestion_prob: number;
  scenario_congestion_prob: number;
  scenario_level: CongestionLevel;
}

export interface JunctionImpact {
  junction_id: string;
  name: string;
  baseline_congestion_prob: number;
  scenario_congestion_prob: number;
}

export interface SimulationResult {
  scenario_id: string;
  request: SimulationRequest;
  created_at: string;
  impacts: SegmentImpact[];
  unserved_demand_veh_h: number;
  assumptions: string[];
  junctions: JunctionImpact[];
  summary: string[];
}

export type NetworkGeoJSON = GeoJSON.FeatureCollection & {
  bbox?: [number, number, number, number] | null;
  /** false: schematic coordinates (real location unknown) - draw on a plain background, never a city map */
  geo_referenced?: boolean;
  location_note?: string | null;
};
