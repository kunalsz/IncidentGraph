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

## P1 scope

Run `.venv/bin/python scripts/demo_inject.py --fixture payment_failure --run-p1`
for the P0-to-P1 handoff. P1 projects the anchor and one-hop structural
neighbors, capped at 40 projected nodes and 12 metric series. It computes
windowed severity, direct-child span self-time, three-sigma flags within
60 seconds of onset, and change proximity without LLM calls.

Missing logs or traces and scope truncation produce incident-scoped
`EvidenceGap` records. Service statistics use all matching windowed
observations before output truncation; missing z-scores are not invented.

## P2 initial ranking

Run `.venv/bin/python scripts/demo_inject.py --fixture payment_failure --run-p2`
for P0 through P2. P2 computes latency instability (empirical p99/p50),
self-processing impact, maximum modality severity, anchor-rooted random walk,
change proximity, and indicator gamma scores. Reciprocal rank fusion combines
five rankings; missing signals contribute nothing and ties share average rank.
Top-eight candidates plus every rule-flagged service become P3 branches.
The component table is persisted as an incident-specific `Artifact`.

Call edges point from caller to dependency, so the cause-seeking walk follows
that direction. Dangling-node mass restarts at the anchor. Service span
latencies are the current proxy for incoming-call latency; sparse fixtures
are not production percentile estimates. Gamma is reported as evidence,
not included as an extra fusion vote. Existing PageRank properties break
exact fusion ties. Correlation pruning is not enabled because fixtures do
not provide aligned raw series, so no correlation-based causal claim is made.

Run `make eval` or `.venv/bin/python scripts/eval.py --fixtures --rank-only`
for the deterministic top-three gate. Each case runs in an isolated graph,
which is removed afterward. Payment, cart, and recommendation causes rank
first on these three authored fixtures (seven candidates per case, zero LLM
calls). These are smoke-test results, not external benchmark accuracy;
end-to-end diagnosis and abstention evaluation require P3 through P6.

## Graph model (v2)

Graph namespaces: `incidentgraph:demo` (read-only seed), 
`incidentgraph:session:{id}` (per-investigation working copy via `GRAPH.COPY`).

**Nodes:** Service, Instance, Team, Deployment, ConfigChange, Incident, Alert,
LogEvent, MetricAnomaly, Trace, Span, Runbook, RunbookParam, Decision,
Hypothesis, Evidence, EvidenceGap, Artifact, Detector.

**Edges:** `DEPENDS_ON` (structural/frozen), `OWNED_BY`, `DEPLOYED_AT`,
`CHANGED`, `OBSERVED_ON`, `AFFECTS`, `APPLIES_TO`, `REQUIRES_PARAMETER`,
`MITIGATED_BY`, `HAS_EVIDENCE`, `CAUSES` (volatile), `CONTRADICTS`,
`EXPANDED_TO`, `SIMILAR_TO`, and others. Full schema in
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
