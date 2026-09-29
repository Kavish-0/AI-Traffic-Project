import type { ReactNode } from "react";
import type { CongestionLevel, RiskLevel } from "../api/types";
import { CONGESTION_COLORS, RISK_COLORS, UNKNOWN_COLOR } from "../theme/congestion";

export function Card({
  title,
  subtitle,
  right,
  children,
  className = "",
}: {
  title?: ReactNode;
  subtitle?: ReactNode;
  right?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={`rounded-xl border border-slate-800 bg-slate-900/70 p-4 ${className}`}>
      {(title || right) && (
        <header className="mb-3 flex items-start justify-between gap-3">
          <div>
            {title && <h2 className="text-xs font-semibold tracking-widest text-slate-400 uppercase">{title}</h2>}
            {subtitle && <p className="mt-0.5 text-xs text-slate-500">{subtitle}</p>}
          </div>
          {right}
        </header>
      )}
      {children}
    </section>
  );
}

type Tone = "info" | "warn" | "error" | "muted";
const TONES: Record<Tone, string> = {
  info: "border-sky-800/60 bg-sky-950/40 text-sky-200",
  warn: "border-amber-800/60 bg-amber-950/40 text-amber-200",
  error: "border-red-800/60 bg-red-950/40 text-red-200",
  muted: "border-slate-800 bg-slate-900/40 text-slate-400",
};

/** Loading / empty / error / unavailable states - explicit text, never fake values. */
export function StateMessage({ tone = "muted", children }: { tone?: Tone; children: ReactNode }) {
  return (
    <div role="status" className={`rounded-lg border px-3 py-2 text-sm ${TONES[tone]}`}>
      {children}
    </div>
  );
}

export function Badge({ children, color, title }: { children: ReactNode; color?: string; title?: string }) {
  return (
    <span
      title={title}
      className="inline-flex items-center gap-1 rounded-md px-2 py-0.5 text-[11px] font-semibold tracking-wide uppercase"
      style={{ background: `${color ?? UNKNOWN_COLOR}26`, color: color ?? "#C7C0E6", border: `1px solid ${color ?? UNKNOWN_COLOR}66` }}
    >
      {children}
    </span>
  );
}

export function LevelBadge({ level, stale = false }: { level: CongestionLevel | null | undefined; stale?: boolean }) {
  if (!level || stale) return <Badge title="no recent data">{stale && level ? "stale" : "no data"}</Badge>;
  return <Badge color={CONGESTION_COLORS[level]}>{level}</Badge>;
}

export function RiskBadge({ level }: { level: RiskLevel | null | undefined }) {
  if (!level) return <Badge>no risk data</Badge>;
  return <Badge color={RISK_COLORS[level]}>{level}</Badge>;
}

export function SyntheticBadge() {
  return (
    <Badge color="#a855f7" title="Test fixture: simulated traffic and simulated detections - not real-world measurements">
      synthetic test data
    </Badge>
  );
}

export function Stat({ label, value, hint }: { label: string; value: ReactNode; hint?: ReactNode }) {
  return (
    <div className="min-w-0">
      <div className="text-[11px] tracking-wider text-slate-500 uppercase">{label}</div>
      <div className="truncate text-lg font-semibold text-slate-100 tabular-nums">{value}</div>
      {hint && <div className="text-[11px] text-slate-500">{hint}</div>}
    </div>
  );
}

export function Dot({ ok, pulse = false }: { ok: boolean | null; pulse?: boolean }) {
  const color = ok === null ? "bg-slate-500" : ok ? "bg-emerald-400" : "bg-red-500";
  return (
    <span className="relative inline-flex h-2.5 w-2.5">
      {pulse && ok && <span className={`absolute inline-flex h-full w-full animate-ping rounded-full ${color} opacity-60`} />}
      <span className={`relative inline-flex h-2.5 w-2.5 rounded-full ${color}`} />
    </span>
  );
}
