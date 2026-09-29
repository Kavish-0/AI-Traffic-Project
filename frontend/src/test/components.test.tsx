import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ReactNode } from "react";
import { describe, expect, it, vi } from "vitest";
import { api } from "../api/client";
import { ForecastStrip } from "../components/ForecastStrip";
import { useLiveIndicator } from "../components/Header";
import { KpiBar } from "../components/KpiBar";
import { RoadPanel } from "../components/RoadPanel";
import { WhatIfPanel, buildRequest } from "../components/WhatIfPanel";
import type { Snapshot } from "../api/types";
import type { Resource } from "../hooks/useResource";
import { IncidentCenter } from "../pages/IncidentCenter";
import { LiveContext, type LiveState } from "../state/live";
import { SelectionProvider } from "../state/selection";
import { forecast, incident, junctions, snapshot, state, summary } from "./fixtures";

const res = <T,>(data: T | undefined, over: Partial<Resource<T>> = {}): Resource<T> => ({
  data, error: null, loading: false, reload: () => {}, ...over,
});

const liveState = (over: Partial<LiveState> = {}): LiveState => ({
  connection: "online", fresh: false, dataAt: null, summary: res(summary()), windowKey: null, pollMs: 3000, ...over,
});

function Wrap({ children, live = liveState() }: { children: ReactNode; live?: LiveState }) {
  return (
    <LiveContext.Provider value={live}>
      <SelectionProvider>{children}</SelectionProvider>
    </LiveContext.Provider>
  );
}

describe("RoadPanel", () => {
  it("marks an uncalibrated incident value as UNCALIBRATED, never as a plain probability", () => {
    render(<Wrap><RoadPanel snapshot={res(snapshot())} /></Wrap>);
    expect(screen.getByText(/^Incident probability: UNCALIBRATED —/)).toBeInTheDocument();
    expect(screen.queryByText(/^Incident probability —/)).not.toBeInTheDocument();
    expect(screen.getByText("87%")).toBeInTheDocument();
    expect(screen.getByText(/synthetic test data/i)).toBeInTheDocument();
  });

  it("calls it a probability only when the backend marks it calibrated", () => {
    const calibrated = incident({ calibrated: true, uncertainty_note: "Probability is calibrated on labelled incidents." });
    render(<Wrap><RoadPanel snapshot={res(snapshot({ incident: calibrated }))} /></Wrap>);
    expect(screen.getByText(/^Incident probability —/)).toBeInTheDocument();
    expect(screen.queryByText(/UNCALIBRATED/)).not.toBeInTheDocument();
  });

  it("shows the forecast's reason instead of values when unavailable", () => {
    render(<Wrap><RoadPanel snapshot={res(snapshot())} /></Wrap>);
    expect(screen.getByText(/insufficient historical data \(3\/30/)).toBeInTheDocument();
  });

  it("warns about stale data and shows waiting text without a window", () => {
    const { rerender } = render(<Wrap><RoadPanel snapshot={res(snapshot({ window_stale: true }))} /></Wrap>);
    expect(screen.getByText(/Data is stale/)).toBeInTheDocument();
    rerender(<Wrap><RoadPanel snapshot={res(snapshot({ window: null, incident: null, risk: null, incident_score: null }))} /></Wrap>);
    expect(screen.getByText(/Waiting for traffic data/)).toBeInTheDocument();
    expect(screen.getByText(/Risk unavailable/)).toBeInTheDocument();
    expect(screen.getByText(/UNCALIBRATED — unavailable/)).toBeInTheDocument();
  });

  it("shows the backend error", () => {
    render(<Wrap><RoadPanel snapshot={res<Snapshot>(undefined, { error: { status: 503, detail: "Database unavailable" } as never })} /></Wrap>);
    expect(screen.getByText(/Database unavailable/)).toBeInTheDocument();
  });
});

describe("ForecastStrip", () => {
  it("renders horizon levels when the forecast is ok", () => {
    const f = forecast({
      status: "ok", message: "ok", issued_at: "2026-09-24T10:15:00Z",
      horizons: [5, 15, 30].map((h, i) => ({
        horizon_min: h, target_time: "2026-09-24T10:20:00Z", level: (["high", "severe", "severe"] as const)[i],
        values: { avg_speed_kmh: 12 - i },
      })),
    });
    render(<ForecastStrip forecast={f} currentLevel="high" />);
    expect(screen.getAllByText("severe")).toHaveLength(2);
    expect(screen.getByText("+15 min")).toBeInTheDocument();
    expect(screen.getByText(/trained on/)).toHaveTextContent(/Synthetic/);
  });
});

describe("KpiBar", () => {
  it("shows no numbers when no approach is live", () => {
    render(<KpiBar summary={res(summary())} />);
    expect(screen.getByText(/Waiting for traffic data/)).toBeInTheDocument();
    expect(screen.getAllByText("—").length).toBeGreaterThanOrEqual(2);
    expect(screen.getByText(/SYNTHETIC test scenes/)).toBeInTheDocument();
  });
  it("shows live KPIs", () => {
    render(<KpiBar summary={res(summary({ fresh_approaches: 4, vehicles: 57, avg_speed_kmh: 23.4, congested_approaches: 2, active_incidents: 1 }))} />);
    expect(screen.getByText("57")).toBeInTheDocument();
    expect(screen.getByText("23 km/h")).toBeInTheDocument();
  });
});

describe("live indicator (REST polling)", () => {
  function Probe() {
    const l = useLiveIndicator();
    return <span>{l.label}</span>;
  }
  it("is LIVE only when the last poll succeeded AND the backend reports fresh live values", () => {
    const { rerender } = render(<Wrap live={liveState({ connection: "connecting" })}><Probe /></Wrap>);
    expect(screen.getByText("CONNECTING")).toBeInTheDocument();
    rerender(<Wrap live={liveState({ connection: "offline", fresh: true })}><Probe /></Wrap>);
    expect(screen.getByText("DISCONNECTED")).toBeInTheDocument();
    rerender(<Wrap live={liveState({ connection: "online", fresh: false })}><Probe /></Wrap>);
    expect(screen.getByText("WAITING FOR DATA")).toBeInTheDocument();
    rerender(<Wrap live={liveState({ connection: "online", fresh: true })}><Probe /></Wrap>);
    expect(screen.getByText("LIVE")).toBeInTheDocument();
  });
});

describe("disconnected state", () => {
  const down = { status: 0, detail: "Backend unavailable" } as never;
  it("KPIs say they are not updating instead of passing old values off as current", () => {
    render(<KpiBar summary={res(summary({ fresh_approaches: 2, vehicles: 9, timestamp: "2026-09-24T10:15:00Z" }), { error: down })} />);
    expect(screen.getByText(/Not updating - the last poll failed/)).toBeInTheDocument();
    expect(screen.queryByText(/Waiting for traffic data/)).not.toBeInTheDocument();
  });
  it("the road panel flags values from before a failed poll", () => {
    render(<Wrap><RoadPanel snapshot={res(snapshot(), { error: down })} /></Wrap>);
    expect(screen.getByText(/Not updating - Backend unavailable/)).toBeInTheDocument();
  });
});

describe("IncidentCenter", () => {
  it("lists incidents and stores operator feedback through the API", async () => {
    vi.spyOn(api, "incidents").mockResolvedValue([incident()]);
    const fb = vi.spyOn(api, "incidentFeedback").mockResolvedValue(incident({ status: "acknowledged", operator_label: "true_incident" }));
    render(<Wrap><IncidentCenter /></Wrap>);
    const row = await screen.findByText("road_a_sb");
    await userEvent.click(row);
    expect(screen.getByText("Incident probability: UNCALIBRATED")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "TRUE INCIDENT" }));
    expect(fb).toHaveBeenCalledWith("inc-1", { status: "acknowledged", label: "true_incident", note: undefined });
    await userEvent.click(screen.getByRole("button", { name: "FALSE ALARM" }));
    expect(fb).toHaveBeenLastCalledWith("inc-1", { status: "dismissed", label: "false_alarm", note: undefined });
  });

  it("shows an explicit empty state", async () => {
    vi.spyOn(api, "incidents").mockResolvedValue([]);
    render(<Wrap><IncidentCenter /></Wrap>);
    expect(await screen.findByText("No active incidents.")).toBeInTheDocument();
  });
});

describe("WhatIfPanel", () => {
  const roads = [
    { id: "road_a_sb", name: "Road A (southbound)", from_node: "n", to_node: "a", lanes: 2, speed_limit_kmh: 50, capacity_veh_h: null, importance: 0.8, approach_id: "junction_a_north", junction_id: "junction_a", monitored: true, level: null, stale: true, coordinates: [] },
    { id: "road_b_wb", name: "Road B (westbound)", from_node: "d", to_node: "a", lanes: 2, speed_limit_kmh: 50, capacity_veh_h: null, importance: 0.8, approach_id: "junction_a_east", junction_id: "junction_a", monitored: true, level: null, stale: true, coordinates: [] },
  ];

  it("builds the backend request for each action", () => {
    expect(buildRequest("close", "road_b_wb")).toEqual({ closed_segment_ids: ["road_b_wb"] });
    expect(buildRequest("lane_closure", "road_b_wb")).toEqual({ closed_segment_ids: [], capacity_factors: { road_b_wb: 0.5 } });
    expect(buildRequest("demand", "")).toEqual({ closed_segment_ids: [], demand_factor: 1.2 });
  });

  it("runs the simulation on the backend and shows affected roads", async () => {
    const run = vi.spyOn(api, "whatIf").mockResolvedValue({
      scenario_id: "s1", request: { closed_segment_ids: ["road_b_wb"] }, created_at: "", unserved_demand_veh_h: 0,
      assumptions: ["Static macroscopic assignment on a small road graph - NOT a microscopic or dynamic traffic simulator."],
      junctions: [{ junction_id: "junction_a", name: "Junction A", baseline_congestion_prob: 0.03, scenario_congestion_prob: 0.34 }],
      summary: [],
      impacts: [
        { segment_id: "road_b_wb", name: "Road B (westbound)", status: "closed", baseline_observed: true, baseline_flow_veh_h: 900, scenario_flow_veh_h: 0, flow_change_pct: -100, baseline_congestion_prob: 0.03, scenario_congestion_prob: 0, scenario_level: "free" },
        { segment_id: "road_a_sb", name: "Road A (southbound)", status: "open", baseline_observed: false, baseline_flow_veh_h: 900, scenario_flow_veh_h: 1098, flow_change_pct: 22, baseline_congestion_prob: 0.03, scenario_congestion_prob: 0.08, scenario_level: "free" },
      ],
    });
    render(<Wrap><WhatIfPanel junctions={res(junctions)} roads={res(roads)} /></Wrap>);
    expect(screen.getByText(/NOT a guaranteed real-world prediction/)).toBeInTheDocument();
    await userEvent.selectOptions(screen.getByLabelText("Road"), "road_b_wb");
    await userEvent.click(screen.getByRole("button", { name: /run simulation/i }));
    await waitFor(() => expect(run).toHaveBeenCalledWith({ closed_segment_ids: ["road_b_wb"] }));
    expect(await screen.findByText("+22%")).toBeInTheDocument();
    expect(screen.getByText("CLOSED")).toBeInTheDocument();
    expect(screen.getByText("3% → 34%")).toBeInTheDocument();
  });

  it("reports a failed simulation", async () => {
    vi.spyOn(api, "whatIf").mockRejectedValue(new (await import("../api/client")).ApiError(422, "unknown segment id(s): ['x']", "/simulation/what-if"));
    render(<Wrap><WhatIfPanel junctions={res(junctions)} roads={res(roads)} /></Wrap>);
    await userEvent.click(screen.getByRole("button", { name: /run simulation/i }));
    expect(await screen.findByText(/Simulation failed: unknown segment/)).toBeInTheDocument();
  });
});

describe("fixtures sanity", () => {
  it("uses schema-shaped states", () => {
    expect(state().traffic_state).toBe("severe");
  });
});
