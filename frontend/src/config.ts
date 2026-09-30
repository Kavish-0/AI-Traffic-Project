/**
 * Dashboard refresh intervals. The dashboard gets all data by polling the REST API (no WebSocket).
 * Override in frontend/.env.local (milliseconds); values below 1000 ms are ignored.
 */
function ms(value: string | undefined, fallback: number): number {
  const n = Number(value);
  return Number.isFinite(n) && n >= 1000 ? n : fallback;
}

export const POLL = {
  /** Live traffic: KPIs, the selected road's live values, and new-window detection (2-5 s recommended). */
  liveMs: ms(import.meta.env.VITE_POLL_LIVE_MS, 3000),
  /** Camera and system status (camera heartbeats every 2 s; offline after 15 s). */
  statusMs: ms(import.meta.env.VITE_POLL_STATUS_MS, 10_000),
  /**
   * Safety refresh for panels that only change when a 60 s window closes (map colours, charts, incidents,
   * risk). They are refetched as soon as the live poll sees a new window; this interval only covers
   * changes that detection cannot see (e.g. another operator's feedback).
   */
  slowMs: ms(import.meta.env.VITE_POLL_SLOW_MS, 30_000),
};
