import type { CSSProperties, ReactNode } from "react";

import { useSweep } from "../hooks/useSweep";
import type { Route } from "../lib/router";

interface LandingPageProps {
  onNavigate: (route: Route) => void;
}

const STAGES: Array<{ name: string; what: string }> = [
  { name: "resolve", what: "Resolve the incident context: anchor service, time window, authoritative onset clock." },
  { name: "scope", what: "Admit a bounded, immutable slice of metrics, logs and traces under explicit caps." },
  { name: "rank", what: "Rank candidate services deterministically; the model has no influence on the baseline." },
  { name: "investigate", what: "Form hypotheses that may cite only evidence IDs from the admitted bundle." },
  { name: "validate", what: "Check every proposal against exactly the evidence it cited." },
  { name: "select", what: "Select from predeclared slots under strict gates; abstain when nothing is proven." },
  { name: "report", what: "Report deterministically: verdict, evidence strength, and what was not checked." },
];

const IDEAS: Array<{ title: string; body: string; mark: ReactNode }> = [
  {
    title: "Bounded evidence admission",
    body: "Hard caps on nodes, series and queries. Everything omitted is recorded as a coverage gap, never dropped silently.",
    mark: (
      <svg viewBox="0 0 24 24" className="idea__mark" aria-hidden="true">
        <path d="M3 18 H21" strokeWidth="1.2" />
        <rect x="4" y="11" width="3" height="7" />
        <rect x="9" y="7" width="3" height="11" />
        <rect x="14" y="13" width="3" height="5" />
        <path d="M4 4 H20" strokeDasharray="2 2" strokeWidth="1.2" />
      </svg>
    ),
  },
  {
    title: "Provenance-bound hypotheses",
    body: "Every hypothesis must cite evidence IDs from the bundle it received. Uncited reasoning is recorded as inference, not evidence.",
    mark: (
      <svg viewBox="0 0 24 24" className="idea__mark" aria-hidden="true">
        <circle cx="6" cy="7" r="2.4" />
        <circle cx="18" cy="17" r="2.4" />
        <path d="M7.6 8.6 L16.4 15.4" strokeWidth="1.2" />
        <path d="M6 17.5 V12" strokeDasharray="2 2" strokeWidth="1.2" />
      </svg>
    ),
  },
  {
    title: "Falsification before confirmation",
    body: "A refuted claim is terminal. A missing signal remains a coverage gap; it never becomes proof of absence.",
    mark: (
      <svg viewBox="0 0 24 24" className="idea__mark" aria-hidden="true">
        <path d="M5 13 L10 18 L20 6" strokeWidth="1.4" />
        <path d="M4 4 L20 20" strokeWidth="1" strokeDasharray="2 3" opacity="0.6" />
      </svg>
    ),
  },
  {
    title: "Abstention over speculation",
    body: "When no candidate clears the validation gate, the report states that outcome with its reasons instead of asserting a guess.",
    mark: (
      <svg viewBox="0 0 24 24" className="idea__mark" aria-hidden="true">
        <circle cx="12" cy="12" r="7" strokeWidth="1.3" />
        <path d="M12 1.5 V5" strokeWidth="1.3" />
        <path d="M12 19 V22.5" strokeWidth="1.3" opacity="0.5" />
      </svg>
    ),
  },
];

const NODES = [
  { x: 38, y: 34, label: "frontend" },
  { x: 152, y: 58, label: "checkout" },
  { x: 258, y: 26, label: "payments" },
  { x: 268, y: 118, label: "currency" },
  { x: 60, y: 138, label: "cart" },
  { x: 162, y: 176, label: "cause", cause: true },
];

const EDGES: Array<[number, number]> = [
  [0, 1],
  [1, 2],
  [1, 3],
  [0, 4],
  [1, 5],
  [4, 5],
];

/** The idea in one picture: a symptom at the edge, traced to one service. */
function HeroGraph({ highlight }: { highlight: boolean }) {
  const cause = NODES[5];
  const path = `M${NODES[0].x},${NODES[0].y} L${NODES[1].x},${NODES[1].y} L${cause.x},${cause.y}`;
  return (
    <svg viewBox="0 0 300 210" className="hero__graph" role="img"
         aria-label="A service graph where one service is identified as the cause">
      {EDGES.map(([from, to]) => (
        <line
          key={`${from}-${to}`}
          x1={NODES[from].x}
          y1={NODES[from].y}
          x2={NODES[to].x}
          y2={NODES[to].y}
          className="hero__edge"
        />
      ))}
      <path d={path} className="hero__trace" />
      {NODES.map((node) => (
        <g key={node.label}>
          <circle
            cx={node.x}
            cy={node.y}
            r={node.cause ? 7 : 5}
            className={node.cause && highlight ? "hero__node hero__node--cause" : "hero__node"}
          />
          <text x={node.x} y={node.y - (node.cause ? 12 : 10)} className="hero__label">
            {node.label === "cause" ? "root cause" : node.label}
          </text>
        </g>
      ))}
      <circle r="2.6" className="hero__pulse">
        <animateMotion dur="4.6s" repeatCount="indefinite" path={path} />
      </circle>
    </svg>
  );
}

export function LandingPage({ onNavigate }: LandingPageProps) {
  const sweep = useSweep(STAGES.length);

  return (
    <div className="page landing">
      <section className="hero">
        <div className="hero__copy">
          <h1>Evidence-constrained root cause analysis for microservice incidents.</h1>
          <p>
            IncidentGraph admits a bounded slice of your telemetry, directs an agent to form
            hypotheses it can only ground in that evidence, attempts to refute them, and reports a
            conclusion it can defend. Where the evidence does not clear its gates, it abstains and
            states why.
          </p>
          <div className="hero__actions">
            <button type="button" className="run" onClick={() => onNavigate("demo")}>
              Run a demo case
            </button>
            <button
              type="button"
              className="run run--ghost"
              onClick={() => onNavigate("dashboard")}
            >
              Investigate my own graph
            </button>
          </div>
        </div>
        <HeroGraph highlight />
      </section>

      <section className="timeline">
        <h2>Seven stages, in order</h2>
        <div className="timeline__rail">
          <span
            className="timeline__progress"
            style={{ transform: `scaleX(${Math.min(sweep, STAGES.length) / STAGES.length})` }}
            aria-hidden="true"
          />
          {STAGES.map((stage, index) => (
            <div
              key={stage.name}
              className="timeline__stage"
              data-reached={index < sweep}
              style={{ "--stage": index } as CSSProperties}
            >
              <span className="timeline__dot" aria-hidden="true" />
              <span className="timeline__index">{String(index + 1).padStart(2, "0")}</span>
              <span className="timeline__name">{stage.name}</span>
              <span className="timeline__what">{stage.what}</span>
            </div>
          ))}
        </div>
      </section>

      <section className="ideas">
        {IDEAS.map((idea) => (
          <article key={idea.title} className="idea">
            {idea.mark}
            <h3>{idea.title}</h3>
            <p>{idea.body}</p>
          </article>
        ))}
      </section>

      <section className="closing">
        <p>
          Every figure in this interface comes from an executed run: the demo cases are recorded
          RCAEval incidents, and the dashboard evaluates whatever FalkorDB graph you connect.
        </p>
        <button type="button" className="run" onClick={() => onNavigate("demo")}>
          See it run
        </button>
      </section>
    </div>
  );
}
