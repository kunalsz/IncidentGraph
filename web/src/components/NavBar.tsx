import { Logo } from "./Logo";
import type { Route } from "../lib/router";

const PAGES: Array<{ route: Route; label: string }> = [
  { route: "landing", label: "Overview" },
  { route: "dashboard", label: "Dashboard" },
  { route: "demo", label: "Demo" },
];

interface NavBarProps {
  route: Route;
  onNavigate: (route: Route) => void;
  /** Kind of the run in flight, so the pill can reopen the page it started on. */
  runningKind?: "benchmark" | "custom" | null;
}

export function NavBar({ route, onNavigate, runningKind }: NavBarProps) {
  return (
    <header className="nav">
      <button type="button" className="nav__brand" onClick={() => onNavigate("landing")}>
        <Logo />
        <span>IncidentGraph</span>
      </button>
      <nav className="nav__pages">
        {PAGES.map((page) => (
          <button
            key={page.route}
            type="button"
            className="nav__page"
            aria-current={route === page.route}
            onClick={() => onNavigate(page.route)}
          >
            {page.label}
          </button>
        ))}
      </nav>
      {runningKind && (
        <button
          type="button"
          className="nav__running"
          onClick={() => onNavigate(runningKind === "custom" ? "dashboard" : "demo")}
        >
          a run is in progress
        </button>
      )}
    </header>
  );
}
