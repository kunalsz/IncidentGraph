import { useCallback, useEffect, useMemo, useState } from "react";

import { api } from "../api/client";
import type { RunSnapshot } from "../types";

const STORAGE_KEY = "incidentgraph.runs.v1";

export type RunStore = Record<string, RunSnapshot>;

function readStorage(): RunStore {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return {};
    const parsed: unknown = JSON.parse(raw);
    return parsed && typeof parsed === "object" ? (parsed as RunStore) : {};
  } catch {
    return {};
  }
}

/**
 * Per-target run state: one snapshot per case or custom target, kept in
 * localStorage so a refresh (or another page) shows the same picture. Runs
 * themselves are single-flight on the server; this store mirrors that.
 */
export function useRunStore(activeRunId: string | null) {
  const [runs, setRuns] = useState<RunStore>(readStorage);
  const [error, setError] = useState<string | null>(null);
  const [reconciled, setReconciled] = useState(false);

  useEffect(() => {
    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(runs));
    } catch {
      /* storage full or unavailable: state simply stops persisting */
    }
  }, [runs]);

  const put = useCallback((snapshot: RunSnapshot) => {
    setRuns((previous) => ({ ...previous, [snapshot.case_id]: snapshot }));
  }, []);

  const forget = useCallback((key: string) => {
    setRuns((previous) => {
      const next = { ...previous };
      delete next[key];
      return next;
    });
  }, []);

  // A run left "running" in storage may have finished while this page was
  // closed: re-read those snapshots (the API falls back to disk) so the UI
  // never shows a stuck run.
  useEffect(() => {
    const stale = Object.values(readStorage()).filter((item) => item.status === "running");
    if (stale.length === 0) {
      setReconciled(true);
      return;
    }
    let cancelled = false;
    Promise.all(
      stale.map((item) =>
        api
          .run(item.run_id)
          .then((snapshot) => snapshot)
          .catch(() => ({ ...item, status: "failed" as const, error: "this run is no longer available" })),
      ),
    ).then((snapshots) => {
      if (cancelled) return;
      setRuns((previous) => {
        const next = { ...previous };
        for (const snapshot of snapshots) next[snapshot.case_id] = snapshot;
        return next;
      });
      setReconciled(true);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  // The server is the authority on what is running right now.
  useEffect(() => {
    if (!activeRunId) return;
    let cancelled = false;
    api
      .run(activeRunId)
      .then((snapshot) => {
        if (!cancelled) put(snapshot);
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [activeRunId, put]);

  const running = useMemo(
    () => Object.values(runs).find((item) => item.status === "running") ?? null,
    [runs],
  );

  const start = useCallback(
    async (starter: () => Promise<RunSnapshot>) => {
      setError(null);
      const snapshot = await starter();
      put(snapshot);
      return snapshot;
    },
    [put],
  );

  // Stable identity: consumers put this object in effect dependencies, so a
  // fresh object on every render would re-run (and abort) their fetches.
  return useMemo(
    () => ({ runs, running, ready: reconciled, error, setError, put, start, forget }),
    [runs, running, reconciled, error, put, start, forget],
  );
}

export type RunStoreApi = ReturnType<typeof useRunStore>;
