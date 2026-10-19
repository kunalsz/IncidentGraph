import { useCallback } from "react";

import { NavBar } from "./components/NavBar";
import { useHealth } from "./hooks/useHealth";
import { useRunStore } from "./hooks/useRunStore";
import { useRunStream } from "./hooks/useRunStream";
import { useRoute } from "./lib/router";
import { DashboardPage } from "./pages/DashboardPage";
import { DemoPage } from "./pages/DemoPage";
import { LandingPage } from "./pages/LandingPage";

export default function App() {
  const [route, navigate] = useRoute();
  const { health, refresh } = useHealth();
  const store = useRunStore(health?.active_run ?? null);

  const onStreamError = useCallback(
    (message: string) => store.setError(message),
    [store],
  );
  useRunStream(store.running?.run_id ?? null, store.put, onStreamError);

  // A finished run changes database state and the stored-run list.
  const onFinished = useCallback(() => void refresh(), [refresh]);

  return (
    <div className="shell">
      <NavBar
        route={route}
        onNavigate={navigate}
        runningKind={(store.running?.kind as "benchmark" | "custom" | undefined) ?? null}
      />
      {route === "landing" && <LandingPage onNavigate={navigate} />}
      {route === "dashboard" && (
        <DashboardPage health={health} store={store} onFinished={onFinished} />
      )}
      {route === "demo" && <DemoPage health={health} store={store} onFinished={onFinished} />}
    </div>
  );
}
