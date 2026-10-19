# IncidentGraph web

Three pages over the live runner: an overview of the idea, a dashboard for your
own graph, and the RCAEval demo. React + TypeScript + Vite, no component
library — one token-based stylesheet.

The UI never fabricates a number: every figure comes from a run the service
actually executed.

## Quick start

```bash
# terminal 1 — API on :8000
.venv/bin/python -m uvicorn server.app.main:app --host 127.0.0.1 --port 8000

# terminal 2 — UI on :5173 (proxies /api to :8000)
cd web && npm install && npm run dev
```

Single-origin production view: `npm run build`, then open `http://127.0.0.1:8000/`.

## Pages

| Route | What it shows |
| --- | --- |
| `#/` | the idea: an animated service graph, the seven stages on a running timeline, and four design commitments |
| `#/dashboard` | connect to a FalkorDB you can reach, verify its schema, then preview the ranking (free) or run the full investigation |
| `#/demo` | ten RCAEval cases over their shared dependency graph; each case keeps its own state and streams the pipeline live |

## On the demo page

- **Dependency graph** — the 12 services and 16 `DEPENDS_ON` edges every case is
  built from, laid out by dependency depth. The faulty service for the selected
  case is marked, clicking any service highlights its edges and opens an
  inspector (depth, criticality, what it calls, what calls it).
- **Case evidence** — a disclosure that reads *that case's* parquet files
  (through the same adapter helpers the pipeline uses) and summarises what went
  wrong: the strongest metric excursions with their z-scores, error logs near the
  onset, and per-service span/error/p95 counts. Loaded on first open, cached per
  case, and it never triggers a run.
- **Live run** — the ledger streams one row per stage, with each model call
  reported as it happens and a clock anchored to the server's time.

## State

`useRunStore` keeps one snapshot per target — a case id or `host:port/graph` —
in `localStorage` (`incidentgraph.runs.v1`). Switching cases therefore shows
*that* target's state: never run, running, or its finished result. A run left
running when the page closes is re-read from the service on the next load, and
the service stays the authority on what is actually in flight.

The server enforces one run at a time; the UI mirrors that by locking the run
buttons while anything is running.

## Layout

```
src/
  api/client.ts            typed fetch wrapper
  hooks/                   useHealth, useRunStore, useRunStream, useSweep
  lib/                     hash router, idle pipeline stages
  types.ts                 wire types (mirror of server/app/models.py)
  components/
    NavBar.tsx  Logo.tsx
    CasePicker.tsx         per-case state marks
    RunControls.tsx        identity, run button, blocking reasons
    DependencyGraph.tsx    the shared service topology, per-node inspection
    CaseEvidencePanel.tsx  what the incident looks like in the case's own data
    StageLedger.tsx        the seven stages as expandable ledger rows
    CostLine.tsx           calls, tokens in/out, elapsed, cap status
    Verdict.tsx            closing entry: tier, why, confidence, next, report
    Markdown.tsx           renders the report without injecting HTML
  pages/
    LandingPage.tsx        hero graph, stage timeline, idea grid
    DashboardPage.tsx      target form, schema checks, preview/full runs
    DemoPage.tsx           RCAEval cases, topology, evidence, live streaming
  styles.css               design tokens and the single stylesheet
```

The tab icon is `public/favicon.svg`, the same topology mark as the navbar.

## Design notes

The palette is neutral graphite with one muted ochre accent used only for the
active stage and a confirmed verdict. Structure carries meaning: stages are a
numbered sequence because they are one, and the conclusion is a definition list
because those are facts. Status is a glyph plus a word, never a coloured pill.
Motion is limited to two deliberate places (the hero trace and the running
stage marker) and is disabled under `prefers-reduced-motion`.
