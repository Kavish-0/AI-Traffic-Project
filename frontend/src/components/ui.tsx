import type { ReactNode } from "react";
import type { CongestionLevel, RiskLevel } from "../api/types";
import { CONGESTION_COLORS, RISK_COLORS, UNKNOWN_COLOR } from "../theme/congestion";
import { ACCENT, TEXT, alpha } from "../theme/tokens";
import { LEVEL_LABEL, RISK_LABEL } from "../lib/labels";

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
    <section className={`fade-in rounded-xl border border-slate-800 bg-slate-900/70 p-3 transition-colors hover:border-slate-700 ${className}`}>
      {(title || right) && (
        <header className="mb-2 flex items-start justify-between gap-3">
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

type Tone = "info" | "warn" | "error" | "muted" | "loading";
const TONES: Record<Tone, string> = {
  info: "border-sky-800/60 bg-sky-950/40 text-sky-200",
  warn: "border-amber-800/60 bg-amber-950/40 text-amber-200",
  error: "border-red-800/60 bg-red-950/40 text-red-200",
  muted: "border-slate-800 bg-slate-900/40 text-slate-400",
  loading: "border-slate-800 bg-slate-900/40 text-slate-400",
};

/**
 * The one way every panel shows a state that is not data: loading, empty, error or unavailable.
 * Explicit text, never placeholder numbers. ``loading`` adds a pulsing marker and aria-busy.
 */
export function StateMessage({ tone = "muted", children }: { tone?: Tone; children: ReactNode }) {
  return (
    <div role="status" aria-busy={tone === "loading" || undefined}
         className={`fade-in flex items-center gap-2 rounded-lg border px-3 py-2 text-sm ${TONES[tone]}`}>
      {tone === "loading" && <span className="h-2 w-2 shrink-0 animate-pulse rounded-full bg-sky-400" aria-hidden="true" />}
      <span className="min-w-0">{children}</span>
    </div>
  );
}

export function Badge({ children, color, title }: { children: ReactNode; color?: string; title?: string }) {
  return (
    <span
      title={title}
      className="inline-flex items-center gap-1 rounded-md px-2 py-0.5 text-[11px] font-semibold tracking-wide uppercase"
      style={{ background: alpha(color ?? UNKNOWN_COLOR, 0.15), color: color ?? TEXT.secondary, border: `1px solid ${alpha(color ?? UNKNOWN_COLOR, 0.4)}` }}
    >
      {children}
    </span>
  );
}

export function LevelBadge({ level, stale = false }: { level: CongestionLevel | null | undefined; stale?: boolean }) {
  if (!level || stale) return <Badge title="no recent data">{stale && level ? "stale" : "no data"}</Badge>;
  return <Badge color={CONGESTION_COLORS[level]}>{LEVEL_LABEL[level]}</Badge>;
}

export function RiskBadge({ level }: { level: RiskLevel | null | undefined }) {
  if (!level) return <Badge>no risk data</Badge>;
  return <Badge color={RISK_COLORS[level]}>{RISK_LABEL[level]}</Badge>;
}

export function SyntheticBadge() {
  return (
    <Badge color={ACCENT.synthetic} title="Test fixture: simulated traffic and simulated detections - not real-world measurements">
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

/** Small text button for a card header that dismisses the card. */
export function CloseButton({ label, onClick }: { label: string; onClick: () => void }) {
  return (
    <button type="button" onClick={onClick} aria-label={label}
            className="rounded-md px-2 py-0.5 text-xs text-slate-400 hover:bg-slate-800 hover:text-slate-100">
      Close
    </button>
  );
}

/** Every page starts the same way: a title, one sentence on what it shows, optional actions on the right. */
export function PageLayout({ title, description, actions, children }: {
  title: string; description?: ReactNode; actions?: ReactNode; children: ReactNode;
}) {
  return (
    <div className="space-y-3 p-4">
      <header className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold text-slate-100">{title}</h2>
          {description && <p className="mt-0.5 max-w-3xl text-xs text-slate-400">{description}</p>}
        </div>
        {actions}
      </header>
      {children}
    </div>
  );
}
