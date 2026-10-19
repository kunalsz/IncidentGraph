import { useCallback, useEffect, useState } from "react";

import { api } from "../api/client";
import type { Health } from "../types";

/** Polls database/provider readiness so the Run button can be gated honestly. */
export function useHealth(intervalMs = 15000) {
  const [health, setHealth] = useState<Health | null>(null);
  const [checked, setChecked] = useState(false);

  const refresh = useCallback(async () => {
    try {
      setHealth(await api.health());
    } catch {
      setHealth(null);
    } finally {
      setChecked(true);
    }
  }, []);

  useEffect(() => {
    void refresh();
    const timer = window.setInterval(() => void refresh(), intervalMs);
    return () => window.clearInterval(timer);
  }, [refresh, intervalMs]);

  return { health, checked, refresh };
}
