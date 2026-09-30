import { describe, expect, it, vi } from "vitest";
import { ApiError, api } from "../api/client";
import { ago, num, pct, withGaps } from "../lib/format";
import { lineMidpoint, mapLabels } from "../lib/mapLabels";

const jsonResponse = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });

describe("api client", () => {
  it("calls the versioned backend routes", async () => {
    const f = vi.spyOn(globalThis, "fetch").mockImplementation(async () => jsonResponse([]));
    await api.forecastVsActual("junction a/north", 30, "flow_veh_h", 60);
    expect(f.mock.calls[0][0]).toBe(
      "/api/v1/approaches/junction%20a%2Fnorth/forecast/history?horizon_min=30&target=flow_veh_h&minutes=60",
    );
    await api.incidents({ active_only: true });
    expect(f.mock.calls[1][0]).toBe("/api/v1/incidents?active_only=true");
  });

  it("surfaces the backend's explanation on errors", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse({ detail: "road 'x' is not monitored" }, 404));
    await expect(api.snapshot("x")).rejects.toMatchObject({ status: 404, detail: "road 'x' is not monitored" });
  });

  it("reports an unreachable backend as status 0", async () => {
    vi.spyOn(globalThis, "fetch").mockRejectedValue(new TypeError("Failed to fetch"));
    const err = await api.summary().catch((e: unknown) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect((err as ApiError).status).toBe(0);
    expect((err as ApiError).detail).toMatch(/Backend unavailable/);
  });

  it("sends operator feedback as PATCH", async () => {
    const f = vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse({}));
    await api.incidentFeedback("inc-1", { status: "dismissed", label: "false_alarm" });
    const [url, init] = f.mock.calls[0];
    expect(url).toBe("/api/v1/incidents/inc-1");
    expect(init?.method).toBe("PATCH");
    expect(JSON.parse(init?.body as string)).toEqual({ status: "dismissed", label: "false_alarm" });
  });
});

describe("format", () => {
  it("never renders missing values as zero", () => {
    expect(num(null)).toBe("—");
    expect(num(undefined, 1, "km/h")).toBe("—");
    expect(pct(null)).toBe("—");
    expect(num(0)).toBe("0");
    expect(pct(0.874)).toBe("87%");
    expect(ago(null)).toBe("never");
  });
});

describe("chart gaps", () => {
  it("breaks the line where the pipeline produced no data, never inventing values", () => {
    const at = (min: number, v: number): [string, number] => [new Date(Date.UTC(2026, 0, 1, 0, min)).toISOString(), v];
    const out = withGaps([at(0, 1), at(1, 2), at(2, 3), at(40, 4), at(41, 5)]);
    expect(out).toHaveLength(6);
    expect(out[3][1]).toBeNull();                       // one break inside the 38-min gap
    expect(out.filter((r) => r[1] !== null).map((r) => r[1])).toEqual([1, 2, 3, 4, 5]);
    expect(withGaps([at(0, 1), at(1, 2), at(2, 3)])).toHaveLength(3);   // regular series unchanged
  });
});

describe("map labels", () => {
  it("labels monitored roads with the backend level and junctions by name", () => {
    expect(lineMidpoint([[0, 0], [2, 0], [2, 2]])).toEqual([2, 0]);
    const road = (id: string, extra: Record<string, unknown>) => ({
      geometry: { type: "LineString", coordinates: [[0, 0], [0, 2]] },
      properties: { kind: "road", id, name: `${id} (southbound)`, approach_id: `${id}_ap`, ...extra },
    });
    const labels = mapLabels([
      road("Road A", { monitored: true, level: "severe", stale: false }),
      road("Road B", { monitored: true, level: "high", stale: true }),     // stale -> never shown as a level
      road("Road X", { monitored: false, level: null }),                  // context road: no label
      { geometry: { type: "Point", coordinates: [1, 1] }, properties: { kind: "junction", id: "j", name: "Junction A" } },
    ]);
    expect(labels.map((l) => l.text)).toEqual(["Road A · Severe", "Road B · no data", "Junction A"]);
    expect(labels[0].lngLat).toEqual([0, 1]);
  });
});
