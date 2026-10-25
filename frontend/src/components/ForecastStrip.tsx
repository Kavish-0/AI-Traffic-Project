import type { CongestionLevel, Forecast } from "../api/types";
import { CONGESTION_COLORS, UNKNOWN_COLOR } from "../theme/congestion";
import { num } from "../lib/format";
import { StateMessage } from "./ui";
import { TEXT } from "../theme/tokens";
import { levelLabel } from "../lib/labels";

function Tile({ label, level, sub }: { label: string; level: CongestionLevel | null | undefined; sub?: string }) {
  const color = level ? CONGESTION_COLORS[level] : UNKNOWN_COLOR;
  return (
    <div className="rounded-lg border px-3 py-2 text-center" style={{ borderColor: `${color}88`, background: `${color}14` }}>
      <div className="text-[11px] tracking-widest text-slate-400 uppercase">{label}</div>
      <div className="text-base font-bold uppercase" style={{ color: level ? color : TEXT.muted }}>
        {level ? levelLabel(level) : "—"}
      </div>
      {sub && <div className="text-[11px] text-slate-400 tabular-nums">{sub}</div>}
    </div>
  );
}

/** Current / +5 / +15 / +30 min levels from the backend forecast; unavailable -> explicit reason. */
export function ForecastStrip({ forecast, currentLevel }: { forecast: Forecast; currentLevel?: CongestionLevel | null }) {
  const byH = Object.fromEntries(forecast.horizons.map((h) => [h.horizon_min, h]));
  const ok = forecast.status === "ok";
  return (
    <div className="space-y-2">
      <div className="grid grid-cols-4 gap-2">
        <Tile label="Current" level={currentLevel ?? forecast.current_level} />
        {[5, 15, 30].map((h) => (
          <Tile
            key={h}
            label={`+${h} min`}
            level={ok ? byH[h]?.level : null}
            sub={ok && byH[h] ? num(byH[h].values.avg_speed_kmh, 0, "km/h") : undefined}
          />
        ))}
      </div>
      {!ok && <StateMessage tone="warn">{forecast.message}</StateMessage>}
    </div>
  );
}
