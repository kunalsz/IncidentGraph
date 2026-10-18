import { useEffect, useState } from "react";

import type { RunSnapshot } from "../types";

/** A server clock older than this is treated as stale (the page was closed). */
const MAX_CLOCK_AGE_MS = 10 * 60 * 1000;

/**
 * Live elapsed time for a run.
 *
 * The clock is anchored to the server's own time, so a browser with a skewed
 * clock still shows the truth, and the anchor is re-taken on every snapshot —
 * which also corrects the offset after a refresh. A finished run freezes at the
 * duration the server recorded rather than at whatever the browser last saw.
 */
export function useElapsed(snapshot: RunSnapshot | null): number | null {
  const [now, setNow] = useState(() => Date.now());
  const [offsetMs, setOffsetMs] = useState(0);
  const serverTime = snapshot?.server_time ?? null;

  useEffect(() => {
    if (serverTime === null) return;
    const candidate = serverTime * 1000 - Date.now();
    // A snapshot restored from storage carries the server time from when it was
    // written; trusting it would offset the clock by the whole idle period.
    if (Math.abs(candidate) > MAX_CLOCK_AGE_MS) return;
    setOffsetMs(candidate);
  }, [serverTime]);

  const running = snapshot?.status === "running";
  const runId = snapshot?.run_id ?? null;

  useEffect(() => {
    if (!running) return;
    setNow(Date.now());
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [running, runId]);

  if (!snapshot) return null;
  if (!running) return snapshot.duration_s ?? null;
  const serverNow = (now + offsetMs) / 1000;
  return Math.max(0, serverNow - snapshot.started_at);
}
