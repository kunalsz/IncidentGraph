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
`EvidenceGap` records. Records that are present but carry no usable z-score
(`unscored_metrics`, `unscored_logs`, `unscored_traces`) are reported too, so a
0 in the severity vector means "not measured" rather than "looked normal";
`get_severity` exposes the same distinction as a per-modality `scored` flag.
Service statistics use all matching windowed observations before output
truncation; missing z-scores are not invented.

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

P2 defaults to `baseline_rrf` (the unchanged five-component fusion). Explicitly
opt into severity-only RRF for a comparison; this does not change scope, evidence,
rule-flag admission, or tie-breaking:

`--rank-policy rrf_anchor_neutral` keeps the same five components and weights but
replaces the anchor-personalised random walk with a uniform-restart walk, for cold
ranking where the anchor is not case-derived (the benchmark fixes it to `frontend`).
Predeclared candidates and their exploratory 90-case numbers are in
`output/anchor-prior-*/RESULTS.md`: baseline 37.8% AC@1, rrf_anchor_neutral 65.6%,
severity_only 77.8%. Default remains `baseline_rrf`; a real incident's own anchor is
still the correct prior, so this policy is for anchorless ranking, not incidents.

```bash
.venv/bin/python scripts/eval.py --fixtures --rank-only --rank-policy severity_only
.venv/bin/python scripts/eval.py --rcaeval data/RE2-OB --subset 6 --rank-policy severity_only --report output/severity-comparison
```

These rank-only commands make no paid provider calls; they require FalkorDB and
RCAEval additionally requires fetched data. `--rank-policy baseline_rrf` is the
explicit default; replay `--policy` is separate. Effective P2 policy/version are
recorded in ranking artifacts and RCAEval case/run provenance, including failures.
For graph-free offline replay of existing records, use
`.venv/bin/python -m backend.app.eval.p2_ablation RUN_DIR`; fidelity uses each
record's declared policy (historical records without one mean baseline RRF).

## Conservative P5 association update (opt-in)

Call `fuse_investigation(..., ranking_policy="conservative_p2_update_v1")` to
select the experimental conservative policy. The default remains
`deterministic_first_shadow`; historical `corrected_h_eligible_first` and
`bounded_p2_update_v1` behavior is unchanged. This policy concerns only the
P2-derived `service_ranking`, not the separate H-score winner/alternatives.
It makes **no causal-superiority claim**.

The permission criterion is at most **one adjacent association-only swap**:
complete, untruncated scoped ranking/investigation; exact scope-bound
`assessment_matches` proofs; every predeclared stable-ID challenger slot present,
all-pass, contradiction-free, and cross-modal `evidence_strength >= 0.85`; an
explicit structured claim with `claim_applicability=pass`. Propagation claims
additionally need a nonempty, contiguous, fully evidence-supported path from the
anchor to the candidate, not merely a supported final edge. Every displaced
incumbent fixed slot must also be present, exactly assessed, and explicitly
refuted on its structured tested claim. Unknown, absent, unsupported, truncated,
provider-failed, late/inapplicable, missing-path, or missing-authority assessments
are **not permission**. Missing slots cannot be rescued by later siblings.

**Currently unsupported positive trigger:** P4 does not expose an unambiguous
structured-claim refutation certificate. In `policy.claim_applicability`, a metric
claim can return `fail` / `cited series contradicts claimed direction` both for
missing direction metadata and for genuinely opposite direction. Other
`claim_applicability=fail` results can inherit `svar_lite` failures; generic
`ValidatorRefute` IDs/contradictions can reflect unrelated vetoes. Neither a fail
verdict, these IDs, nor free-text detail safely certifies explicit refutation.
Therefore this version **always abstains from swaps**, even for a synthetic
opposite-direction claim, until a distinguishable deterministic certificate is
available. It does not turn claim inapplicability into service refutation.

`FuseResult.config.conservative_update` records the abstention reason, each
adjacent pair's challenger/incumbent fixed-slot hypothesis/assessment IDs,
proposal fingerprints, check IDs (`hypothesis:index:check-name`), supported
positive triggers (currently none), and the one-swap/displacement limits.
P2 unknowns retain their baseline order. Focused, database/provider-free checks:

```bash
.venv/bin/python -m pytest tests/unit/test_conservative_p2_update.py tests/unit/test_p5_fuse.py tests/unit/test_evidence_policy.py tests/unit/test_audited_fusion_repairs.py tests/unit/test_contract_report_repairs.py -q
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
docs/                    GRAPH_MODEL.md (graph model, Cypher, algorithms), blog-post.md
frontend/                Next.js four-pane UI (Day 3)
```

## Graph model, Cypher and graph algorithms

The FalkorDB data model, the Cypher queries the product depends on, and the graph
algorithms it implements are documented in [docs/GRAPH_MODEL.md](docs/GRAPH_MODEL.md)
— required by the hackathon submission rules (Rule 10).

## AI disclosure

Help was taken from AI coding assistants (Command Code). The team can
explain every component; the deterministic scorers and validators carry
hand-computed unit tests. Full disclosure in `docs/AI_DISCLOSURE.md` at
submission.
