import ReactEChartsCore from "echarts-for-react/lib/core";
import { LineChart } from "echarts/charts";
import { GridComponent, LegendComponent, TooltipComponent } from "echarts/components";
import * as echarts from "echarts/core";
import { CanvasRenderer } from "echarts/renderers";
import type { ForecastVsActual, Risk, TrafficState } from "../api/types";
import type { Resource } from "../hooks/useResource";
import { withGaps } from "../lib/format";
import { Card, StateMessage } from "./ui";

echarts.use([LineChart, GridComponent, TooltipComponent, LegendComponent, CanvasRenderer]);

type Series = { name: string; color: string; data: [string, number | null][]; dashed?: boolean };

function Chart({ title, unit, series, yMax }: { title: string; unit: string; series: Series[]; yMax?: number }) {
  const option = {
    backgroundColor: "transparent",
    animation: false,
    grid: { left: 44, right: 12, top: 30, bottom: 26 },
    tooltip: { trigger: "axis", valueFormatter: (v: number | null) => (v === null ? "—" : `${v.toFixed(1)} ${unit}`) },
    legend: series.length > 1 ? { top: 0, right: 0, textStyle: { color: "#94a3b8", fontSize: 10 } } : undefined,
    xAxis: { type: "time", axisLabel: { color: "#64748b", fontSize: 10 }, axisLine: { lineStyle: { color: "#334155" } } },
    yAxis: {
      type: "value",
      max: yMax,
      name: unit,
      nameTextStyle: { color: "#64748b", fontSize: 10 },
      axisLabel: { color: "#64748b", fontSize: 10 },
      splitLine: { lineStyle: { color: "#1e293b" } },
    },
    series: series.map((s) => ({
      name: s.name,
      type: "line",
      showSymbol: s.data.length < 30,
      connectNulls: false,
      data: withGaps(s.data),
      lineStyle: { color: s.color, width: 2, type: s.dashed ? "dashed" : "solid" },
      itemStyle: { color: s.color },
    })),
  };
  return (
    <div className="rounded-lg border border-slate-800 bg-slate-950/40 p-2">
      <div className="px-1 text-[11px] font-semibold tracking-widest text-slate-400 uppercase">{title}</div>
      <ReactEChartsCore echarts={echarts} option={option} style={{ height: 180 }} notMerge lazyUpdate />
    </div>
  );
}

const HISTORY_MINUTES = 120;

const pick = (h: TrafficState[], f: (s: TrafficState) => number | null): [string, number | null][] =>
  h.map((s) => [s.timestamp, f(s)]);

export function TrendCharts({
  history,
  risk,
  fva,
  horizon,
  onHorizon,
}: {
  history: Resource<TrafficState[]>;
  risk: Resource<Risk[]>;
  fva: Resource<ForecastVsActual>;
  horizon: number;
  onHorizon: (h: number) => void;
}) {
  if (history.loading) return <Card title="Traffic trends"><StateMessage>Loading history…</StateMessage></Card>;
  if (!history.data)
    return <Card title="Traffic trends"><StateMessage tone="error">History unavailable: {history.error?.detail}</StateMessage></Card>;
  const h = history.data;
  if (h.length === 0)
    return <Card title="Traffic trends"><StateMessage tone="warn">Waiting for traffic data… no closed 60 s window for this road in the last {HISTORY_MINUTES} min.</StateMessage></Card>;
  const insufficient = h.length < 3;
  const fc = fva.data;
  const riskRows = risk.data ?? [];
  return (
    <Card
      title="Traffic trends"
      subtitle={`Closed 60 s windows, last ${HISTORY_MINUTES} min (${h.length} windows) · gaps = no data`}
      right={
        <label className="text-xs text-slate-400">
          Forecast horizon{" "}
          <select
            value={horizon}
            onChange={(e) => onHorizon(Number(e.target.value))}
            className="rounded border border-slate-700 bg-slate-900 px-1 text-slate-200"
          >
            {[5, 15, 30].map((x) => (
              <option key={x} value={x}>+{x} min</option>
            ))}
          </select>
        </label>
      }
    >
      {insufficient && <StateMessage tone="warn">Insufficient data - only {h.length} window(s) so far; trends become meaningful after a few minutes.</StateMessage>}
      <div className="mt-2 grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-3">
        <Chart title="Speed vs time" unit="km/h" series={[{ name: "avg speed", color: "#38bdf8", data: pick(h, (s) => s.average_speed_kmh) }]} />
        <Chart title="Density vs time" unit="%" series={[{ name: "density", color: "#f97316", data: pick(h, (s) => s.density_pct) }]} />
        <Chart title="Vehicle count vs time" unit="veh" series={[{ name: "vehicles", color: "#a3e635", data: pick(h, (s) => s.vehicle_count) }]} />
        <Chart title="Queue length vs time" unit="m" series={[{ name: "queue", color: "#f43f5e", data: pick(h, (s) => s.queue_length_m) }]} />
        <Chart title="Traffic flow vs time" unit="veh/h" series={[{ name: "flow", color: "#c084fc", data: pick(h, (s) => s.flow_veh_h) }]} />
        {fc && fc.forecast.length > 0 ? (
          <Chart
            title={`Forecast vs actual speed (+${horizon} min)`}
            unit="km/h"
            series={[
              { name: "actual", color: "#38bdf8", data: fc.actual.map((a) => [a.timestamp, a.value]) },
              { name: `forecast +${horizon}`, color: "#facc15", dashed: true, data: fc.forecast.map((f) => [f.target_time, f.value]) },
            ]}
          />
        ) : (
          <div className="rounded-lg border border-slate-800 p-3">
            <div className="text-[11px] font-semibold tracking-widest text-slate-400 uppercase">Forecast vs actual</div>
            <div className="mt-2">
              <StateMessage tone="warn">
                {fva.error ? `Unavailable: ${fva.error.detail}` : "No forecasts issued for this road yet (forecasts need enough history)."}
              </StateMessage>
            </div>
          </div>
        )}
        {riskRows.some((r) => r.incident_score !== null) ? (
          <Chart
            title="Incident score vs time (uncalibrated)"
            unit="score"
            yMax={1}
            series={[{ name: "incident score", color: "#ef4444", data: riskRows.map((r) => [r.timestamp, r.incident_score]) }]}
          />
        ) : (
          <div className="rounded-lg border border-slate-800 p-3">
            <div className="text-[11px] font-semibold tracking-widest text-slate-400 uppercase">Incident score vs time</div>
            <div className="mt-2"><StateMessage>Incident score unavailable - no assessments yet.</StateMessage></div>
          </div>
        )}
      </div>
    </Card>
  );
}
