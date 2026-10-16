# IncidentGraph

An **AI incident investigator for distributed systems**, built on
[FalkorDB](https://falkordb.com) for the FalkorDB Hackathon.

Report a symptom ("checkout latency up"); the agent traverses a FalkorDB graph
of services, dependencies, telemetry, deployments, runbooks, teams, and past
incidents to produce a leading root-cause hypothesis, an evidence path, and a
recommended next action with machine-checkable evidence for every claim.

- **Track 01 (primary):** Agents That Act on Connected Data

## Quickstart

```bash
# 1. FalkorDB (Browser UI on http://localhost:3001)
docker compose up -d

# 2. Python env
python3 -m venv .venv && .venv/bin/pip install -e 'backend[dev]'

# 3. Seed the demo graph + inject a fixture incident + run P0
make seed
.venv/bin/python scripts/demo_inject.py --fixture payment_failure --run-p0

# 4. Tests
make test
```

## Graph model (v2)

Graph namespaces: `incidentgraph:demo` (read-only seed), 
`incidentgraph:session:{id}` (per-investigation working copy via `GRAPH.COPY`).

**Nodes:** Service, Instance, Team, Deployment, ConfigChange, Incident, Alert,
LogEvent, MetricAnomaly, Trace, Span, Runbook, RunbookParam, Decision,
Hypothesis, Evidence, EvidenceGap, Artifact, Detector.

**Edges:** `DEPENDS_ON` (structural/frozen), `OWNED_BY`, `DEPLOYED_AT`,
`CHANGED`, `OBSERVED_ON`, `AFFECTS`, `APPLIES_TO`, `REQUIRES_PARAMETER`,
`MITIGATED_BY`, `HAS_EVIDENCE`, `CAUSES` (volatile), `CONTRADICTS`,
`EXPANDED_TO`, `SIMILAR_TO`, … — full schema in
[`backend/app/graph/schema.py`](backend/app/graph/schema.py).

All temporal properties are epoch-ms integers; ids are slugs
(`service:payment`, `alert:INC-1001/a1`). Loaders are MERGE-on-id idempotent.

## Repo layout

```text
config/tuning.py         every constant, single-sourced (k=6, θ=0.6, ε=60, budgets…)
backend/app/
  graph/                 client, schema, queries (FalkorDB layer)
  seed/                  topology.json, fixtures/, idempotent loader
  ingest/normalize.py    §4.7 normalization pipeline
  engine/                p0_resolve.py, linking.py, budget.py, contracts.py
scripts/                 seed_demo, demo_inject, eval, measure_tool_surface
tests/                   unit/ + integration/ (live FalkorDB)
docs/                    GRAPH_MODEL.md, AI_DISCLOSURE.md (written Day 4)
frontend/                Next.js four-pane UI (Day 3)
```

## AI disclosure

Help was taken from AI coding assistants (Command Code). The team can
explain every component; the deterministic scorers and validators carry
hand-computed unit tests. Full disclosure in `docs/AI_DISCLOSURE.md` at
submission.
