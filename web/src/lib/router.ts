import { useEffect, useState } from "react";

export type Route = "landing" | "dashboard" | "demo";

const ROUTES: Route[] = ["landing", "dashboard", "demo"];

const parse = (): Route => {
  const hash = window.location.hash.replace(/^#\/?/, "");
  return (ROUTES.find((route) => route === hash) ?? "landing") as Route;
};

/** Minimal hash router: no dependency, works with the server's SPA fallback. */
export function useRoute(): [Route, (route: Route) => void] {
  const [route, setRoute] = useState<Route>(parse);

  useEffect(() => {
    const onChange = () => setRoute(parse());
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);

  const navigate = (next: Route) => {
    window.location.hash = next === "landing" ? "/" : `/${next}`;
    setRoute(next);
  };

  return [route, navigate];
}
