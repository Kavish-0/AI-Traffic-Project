import { act, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi, type MockInstance } from "vitest";
import App from "../App";
import { POLL } from "../config";
import { junctions, snapshot, state, summary, system } from "./fixtures";
import { wsAttempts } from "./setup";

// WebGL map and canvas charts do not run in jsdom; everything else is the real UI.
vi.mock("../components/MapView", () => ({ MapView: () => <div data-testid="map" /> }));
vi.mock("echarts-for-react/lib/core", () => ({ default: () => <div data-testid="chart" /> }));

let routes: Record<string, unknown>;
let fetchSpy: MockInstance<typeof fetch>;

beforeEach(() => {
  wsAttempts.length = 0;
  routes = {
    "/api/v1/system/status": system,
    "/api/v1/junctions": junctions,
    "/api/v1/roads": [],
    "/api/v1/network": { type: "FeatureCollection", features: [], bbox: null },
    "/api/v1/cameras": [],
    "/api/v1/traffic/summary": summary({ fresh_approaches: 1, vehicles: 28, avg_speed_kmh: 9.3, congested_approaches: 1, active_incidents: 1 }),
    "/api/v1/traffic/current": [state({ approach_id: "junction_a_north", timestamp: "2026-09-24T10:15:00Z" })],
    "/api/v1/approaches/junction_a_north/snapshot": snapshot(),
    "/api/v1/risk": [snapshot().risk],
  };
  fetchSpy = vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const url = String(input).split("?")[0];
    if (url.endsWith("/history")) return new Response("[]", { status: 200 });
    if (url.startsWith("/api/v1/incidents")) return new Response(JSON.stringify([]), { status: 200 });
    if (url in routes) return new Response(JSON.stringify(routes[url]), { status: 200 });
    return new Response(JSON.stringify({ detail: `not mocked: ${url}` }), { status: 404 });
  });
});

afterEach(() => {
  vi.useRealTimers();
});

const calls = (prefix: string) => fetchSpy.mock.calls.filter(([u]) => String(u).split("?")[0].startsWith(prefix)).length;

describe("App (UI integration against mocked backend responses)", () => {
  it("renders the dashboard from API data and selects the first road", async () => {
    render(<App />);
    expect(await screen.findByText("AI Urban Traffic & Incident Intelligence")).toBeInTheDocument();
    expect((await screen.findAllByText("Junction A")).length).toBeGreaterThan(0);
    // KPI from /traffic/summary
    expect(await screen.findByText("28")).toBeInTheDocument();
    // road panel from /approaches/<id>/snapshot (auto-selected approach)
    expect(await screen.findByText(/Incident probability: UNCALIBRATED/)).toBeInTheDocument();
    // trend charts: no windows -> explicit waiting message, not an empty chart
    expect(await screen.findByText(/no closed 60 s window/)).toBeInTheDocument();
    expect(screen.getByTestId("map")).toBeInTheDocument();
    expect(screen.getByText("System status")).toBeInTheDocument();
    expect(await screen.findByText("LIVE")).toBeInTheDocument();
  });

  it("says so when the backend is down (and shows no values)", async () => {
    fetchSpy.mockRejectedValue(new TypeError("Failed to fetch"));
    render(<App />);
    expect(await screen.findByText(/Backend unavailable - start the API/)).toBeInTheDocument();
    expect(await screen.findByText("DISCONNECTED")).toBeInTheDocument();
    expect(screen.queryByText("28")).not.toBeInTheDocument();
  });

  it("updates by REST polling: live endpoints at the live rate, per-window data only on a new window - no WebSocket", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    render(<App />);
    expect(await screen.findByText("28")).toBeInTheDocument();
    await screen.findByText(/Incident probability: UNCALIBRATED/);
    await act(() => vi.advanceTimersByTimeAsync(200));

    const before = {
      summary: calls("/api/v1/traffic/summary"),
      current: calls("/api/v1/traffic/current"),
      snapshot: calls("/api/v1/approaches/junction_a_north/snapshot"),
      roads: calls("/api/v1/roads"),
      network: calls("/api/v1/network"),
      incidents: calls("/api/v1/incidents"),
    };

    // one live interval, nothing new in the database: only the live endpoints are polled again
    await act(() => vi.advanceTimersByTimeAsync(POLL.liveMs + 100));
    expect(calls("/api/v1/traffic/summary")).toBeGreaterThan(before.summary);
    expect(calls("/api/v1/traffic/current")).toBeGreaterThan(before.current);
    expect(calls("/api/v1/approaches/junction_a_north/snapshot")).toBeGreaterThan(before.snapshot);
    expect(calls("/api/v1/roads")).toBe(before.roads);
    expect(calls("/api/v1/network")).toBe(before.network);
    expect(calls("/api/v1/incidents")).toBe(before.incidents);

    // new values are rendered in place (no page reload)
    routes["/api/v1/traffic/summary"] = summary({ fresh_approaches: 1, vehicles: 41, avg_speed_kmh: 12, active_incidents: 1 });
    await act(() => vi.advanceTimersByTimeAsync(POLL.liveMs + 100));
    expect(await screen.findByText("41")).toBeInTheDocument();

    // a new 60 s window appears -> map and incidents are refetched without waiting for their slow interval
    routes["/api/v1/traffic/current"] = [state({ approach_id: "junction_a_north", timestamp: "2026-09-24T10:16:00Z" })];
    const network = calls("/api/v1/network");
    const incidents = calls("/api/v1/incidents");
    await act(() => vi.advanceTimersByTimeAsync(POLL.liveMs + 100));
    expect(calls("/api/v1/network")).toBeGreaterThan(network);
    expect(calls("/api/v1/incidents")).toBeGreaterThan(incidents);

    expect(wsAttempts).toEqual([]);
  });
});
