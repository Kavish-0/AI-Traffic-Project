// Test fixtures shaped like real API responses (backend api/schemas.py). Used ONLY by tests.
import type { Forecast, Incident, Junction, Snapshot, Summary, SystemStatus, TrafficState } from "../api/types";

export const state = (over: Partial<TrafficState> = {}): TrafficState => ({
  timestamp: "2026-09-24T10:15:00Z",
  window_start: "2026-09-24T10:14:00Z",
  window_s: 60,
  is_live: false,
  camera_id: "synth_a_north",
  approach_id: "junction_a_north",
  junction_id: "junction_a",
  road_id: "road_a_sb",
  vehicle_count: 28,
  car_count: 20,
  motorcycle_count: 4,
  bus_count: 1,
  truck_count: 3,
  bicycle_count: 0,
  pedestrian_count: 0,
  average_speed_kmh: 9.3,
  median_speed_kmh: 8,
  density_veh_km: 110,
  density_pct: 73,
  occupancy_pct: 61,
  flow_veh_h: 720,
  inflow: 12,
  outflow: 12,
  queue_length_m: 48,
  stopped_vehicle_count: 6,
  traffic_state: "severe",
  congestion_score: 0.89,
  explanation: ["speed 9 km/h < 25% of free flow"],
  quality: 1,
  ...over,
});

export const forecast = (over: Partial<Forecast> = {}): Forecast => ({
  approach_id: "junction_a_north",
  status: "insufficient_history",
  message: "Forecast unavailable - insufficient historical data (3/30 consecutive 1-min windows).",
  model: { task: "forecasting", name: "xgboost", version: "v1", trained_on: "synthetic", resolution_s: 60, required_history: 30, trained_at: null, metrics: {} },
  issued_at: null,
  current_level: "severe",
  horizons: [],
  history_windows: 3,
  ...over,
});

export const incident = (over: Partial<Incident> = {}): Incident => ({
  incident_id: "inc-1",
  timestamp: "2026-09-24T10:15:00Z",
  first_detected: "2026-09-24T10:12:00Z",
  resolved_at: null,
  approach_id: "junction_a_north",
  junction_id: "junction_a",
  road_id: "road_a_sb",
  incident_type: "possible_incident",
  label: "Possible incident",
  probability: 0.87,
  max_probability: 0.9,
  calibrated: false,
  score_label: "Incident probability: UNCALIBRATED",
  severity: "high",
  risk_level: "high",
  status: "open",
  message: "Possible incident - incident probability 87% (uncalibrated estimate ...)",
  evidence: { stopped_vehicle: 0.95, queue_growth: 1, symptom_strength: 1 },
  lon: null,
  lat: null,
  location_description: null,
  operator_label: null,
  operator_note: null,
  contributing: [{ type: "stopped_vehicle", detector: "stopped_vehicle", score: 0.95, timestamp: "2026-09-24T10:14:30Z", track_ids: [7] }],
  uncertainty_note: "Incident probability is UNCALIBRATED (an estimate from combined signals); verify.",
  ...over,
});

export const snapshot = (over: Partial<Snapshot> = {}): Snapshot => ({
  approach: { id: "junction_a_north", name: "Road A", direction: "southbound", junction_id: "junction_a", road_id: "road_a_sb", cameras: ["synth_a_north"], level: "severe", risk_level: "high", active_incident: "inc-1" },
  road: null,
  cameras: [],
  live: null,
  live_stale: true,
  window: state(),
  window_stale: false,
  forecast: forecast(),
  incident: incident(),
  incident_score: 0.87,
  risk: {
    approach_id: "junction_a_north", timestamp: "2026-09-24T10:15:00Z", junction_id: "junction_a", road_id: "road_a_sb",
    risk_score: 68.3, level: "high", reason_codes: ["SEVERE_CONGESTION_NOW", "STOPPED_VEHICLE"], components: {},
    explanation: ["risk = 100 x 0.73 x (1 - 0.3 + 0.3 x 0.80) = 68"], incident_score: 0.87,
  },
  synthetic: true,
  ...over,
});

export const summary = (over: Partial<Summary> = {}): Summary => ({
  timestamp: null, vehicles: null, avg_speed_kmh: null, congested_approaches: 0, monitored_approaches: 5,
  fresh_approaches: 0, active_incidents: 0, synthetic_approaches: 4, ...over,
});

export const junctions: Junction[] = [
  { id: "junction_a", name: "Junction A", lon: 77.59, lat: 12.97, signalized: true, active_incidents: 1, worst_level: "severe",
    approaches: [{ id: "junction_a_north", name: "Road A", direction: "southbound", junction_id: "junction_a", road_id: "road_a_sb", cameras: ["synth_a_north"], level: "severe", risk_level: "high", active_incident: "inc-1" }] },
];

const ok = (status: string, detail = "") => ({ ok: true, status, detail });
export const system: SystemStatus = {
  backend: ok("online"), database: ok("connected", "sqlite"), pipeline: ok("running"), forecast_model: ok("available"),
  cameras: ok("online", "4/5"), weather: ok("available"), websocket_clients: 1, version: "0.1.0", env: "dev", models: [], weather_now: null,
};
