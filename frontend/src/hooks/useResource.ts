import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError } from "../api/client";

export interface Resource<T> {
  data: T | undefined;
  error: ApiError | null;
  loading: boolean; // true only until the first response (success or error)
  reload: () => void;
}

/**
 * Fetches `fetcher()` on mount / when `key` changes, optionally re-fetching every `intervalMs` (REST
 * polling) and whenever `refreshToken` changes (e.g. the live poll detected a new 60 s window).
 * Keeps the last good data on a transient error so the UI does not flicker, but always exposes the
 * error. Polling pauses while the browser tab is hidden and catches up when it is shown again, and a
 * poll tick is skipped while the previous request is still running (no pile-up on a slow backend).
 */
export function useResource<T>(
  fetcher: (() => Promise<T>) | null,
  key: string,
  opts: { intervalMs?: number; refreshToken?: unknown } = {},
): Resource<T> {
  const [data, setData] = useState<T | undefined>(undefined);
  const [error, setError] = useState<ApiError | null>(null);
  const [loading, setLoading] = useState<boolean>(fetcher !== null);
  const fetcherRef = useRef(fetcher);
  const seq = useRef(0);
  const inFlight = useRef(false);

  useEffect(() => {
    fetcherRef.current = fetcher;
  });

  const run = useCallback(() => {
    const f = fetcherRef.current;
    if (!f) return;
    const mine = ++seq.current;
    inFlight.current = true;
    f()
      .then((d) => {
        if (mine !== seq.current) return;
        setData(d);
        setError(null);
      })
      .catch((e: unknown) => {
        if (mine !== seq.current) return;
        setError(e instanceof ApiError ? e : new ApiError(0, String(e), key));
      })
      .finally(() => {
        if (mine !== seq.current) return;
        inFlight.current = false;
        setLoading(false);
      });
  }, [key]);

  useEffect(() => {
    setData(undefined);
    setError(null);
    setLoading(fetcherRef.current !== null);
    run();
  }, [key, run]);

  // refetch when the token changes (not on mount - the effect above already fetched)
  const lastToken = useRef(opts.refreshToken);
  useEffect(() => {
    if (Object.is(lastToken.current, opts.refreshToken)) return;
    lastToken.current = opts.refreshToken;
    run();
  }, [opts.refreshToken, run]);

  useEffect(() => {
    if (!opts.intervalMs) return;
    const tick = () => {
      if (!document.hidden && !inFlight.current) run();
    };
    const onVisible = () => {
      if (!document.hidden) run();
    };
    const id = setInterval(tick, opts.intervalMs);
    document.addEventListener("visibilitychange", onVisible);
    return () => {
      clearInterval(id);
      document.removeEventListener("visibilitychange", onVisible);
    };
  }, [opts.intervalMs, run]);

  return { data, error, loading, reload: run };
}
