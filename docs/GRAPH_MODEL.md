# Graph data model, Cypher queries, and graph algorithms

Submission artifact for the FalkorDB hackathon requirement: *"The Cypher queries or
graph algorithms the product depends on."* Every query and algorithm below is in
the shipped source; file references are clickable paths in this repository.

FalkorDB is the system of record, not a cache: topology, incident context,
telemetry, the investigation tree (hypotheses and their evidence edges), the
scope snapshot, and every frozen stage artifact live in the graph. The engine's
phase functions read and write it through one query layer, so the same queries
serve the live demo, the agent's read-only tools, and the offline benchmark.

## 1. Data model (`backend/app/graph/schema.py`)

**Node labels (19):** `Service`, `Instance`, `Team`, `Deployment`, `ConfigChange`,
`Incident`, `Alert`, `LogEvent`, `MetricAnomaly`, `Trace`, `Span`, `Runbook`,
`RunbookParam`, `Decision`, `Hypothesis`, `Evidence`, `EvidenceGap`, `Artifact`,
`Detector`.

**Edge types (19):** `DEPENDS_ON` (Service→Service, frozen topology),
`OBSERVED_ON` (evidence → Service|Instance), `AFFECTS` (Incident→Service),
`OWNED_BY`, `DEPLOYED_AT`, `CHANGED`, `APPLIES_TO`, `REQUIRES_PARAMETER`,
`MITIGATED_BY`, `HAS_EVIDENCE` (Hypothesis→Evidence, with `polarity`/`weight`/`role`),
`CAUSES` (reserved for verified causal claims only), `ASSOCIATED_WITH`,
`CONTRADICTS`, `EXPANDED_TO`, `STATISTICAL_DEPENDENCY`, `MERGED_INTO`,
`DROPPED_BY_PREPROCESSING`, `SIMILAR_TO`, `TRIGGERED_BY`.

**Properties that matter.** Services carry `id/name/aliases/criticality/level/`
`level_convention` plus the scope statistics written by P1 (`sev_m`, `sev_l`,
`sev_tc`, `self_time_ms/flag`, `nsigma_count`, `deploy_proximity`,
`hours_since_change`, `scope_t0/t1`). Evidence nodes carry `ts`, the metric
statistics (`value`, `baseline_mean/std/median/iqr`, `z`, `gamma`, `direction`,
`column`) or span identity (`trace_id`, `span_id`, `parent_id`, `duration_us`,
`start_us`, `status`, `trace_complete`), plus the onset-window summary fields
(`onset_representative_ts`, `onset_qualified_count`, …). Hypotheses carry the
typed `claim_kind`/`claim` and the P5 score components; stage results are stored
as `Artifact` nodes with their integrity handoffs.

**Indexes** (`schema.py:61`, verified against `falkordb/falkordb:latest`):
`Service(id)`, `Instance(id)`, `Instance(service_id)`, `Alert(ts)`, `Alert(kind)`,
`LogEvent(ts)`, `LogEvent(template_id)`, `MetricAnomaly(ts)`, `Span(ts)`,
`Span(trace_id)`, `Span(parent_id)`, `Deployment(ts)`, `ConfigChange(ts)`,
`Incident(id)`. Uniqueness is enforced structurally by MERGE-on-id loaders.

## 2. Cypher the product depends on

### 2.1 Scoped reads — one bounded pull per phase (`graph/queries.py`)

The scope read is the query the whole product hangs off: it fixes the candidate
set, the topology, the in-window evidence and the changes, and it is the only
place raw telemetry crosses into the engine.

```cypher
-- queries.py:SCOPE_SERVICES_FULL (benchmark cold ranking; all services)
MATCH (s:Service)
RETURN <projection of SERVICE_FIELDS> AS s
ORDER BY s.criticality DESC, s.id
```

```cypher
-- queries.py:SCOPE_EDGES
MATCH (a:Service)-[r:DEPENDS_ON]->(b:Service)
WHERE a.id IN $ids AND b.id IN $ids AND coalesce(r.invariant, true) = true
RETURN a.id, b.id, properties(r) AS r
ORDER BY a.id, b.id
```

```cypher
-- queries.py:SCOPE_OBSERVATIONS
MATCH (e)-[:OBSERVED_ON]->(target)
WHERE (target:Service AND target.id IN $ids)
   OR (target:Instance AND target.service_id IN $ids)
WITH e, target, CASE WHEN target:Instance THEN target.service_id ELSE target.id END AS sid
WHERE e.ts >= $t0 AND e.ts <= $t1
  AND (e:Alert OR e:LogEvent OR e:MetricAnomaly OR e:Span OR e:Trace)
RETURN DISTINCT <projection of EVIDENCE_FIELDS with labels(e)[0] AS kind> AS e, sid, target.id
ORDER BY e.ts, e.id, sid
```

```cypher
-- queries.py:SCOPE_CHANGES (deploys/config changes that precede the window)
MATCH (c)-[r]->(target)
WHERE ((c:ConfigChange AND type(r) = 'CHANGED') OR (c:Deployment AND type(r) = 'DEPLOYED_AT'))
  AND ((target:Service AND target.id IN $ids) OR (target:Instance AND target.service_id IN $ids))
  AND c.ts >= $t0 AND c.ts <= $t1
RETURN c, ..., target.id
ORDER BY c.ts DESC, c.id
```

Each `RETURN` is rewritten by `queries.projection(...)` into an explicit property
allowlist whose every field is size-guarded:

```cypher
value: CASE WHEN size(toJSON(e.value)) > $field_chars THEN null ELSE e.value END, ...
__rejected: coalesce(...) OR coalesce(...)          -- any oversized field fails the row
```

A row whose projected payload exceeds the cap is rejected (`PayloadAdmissionError`)
rather than silently truncated, and `SCOPE_SERVICES` (the live-incident variant)
additionally restricts to the anchor and its 1-hop `DEPENDS_ON` neighbourhood.
Field lists live in `queries.py:SERVICE_FIELDS / EVIDENCE_FIELDS / EDGE_FIELDS`.

### 2.2 Scoped writes

```cypher
-- queries.py:UPDATE_SCOPE_STATS
MATCH (s:Service {id: $id})
SET s.sev_m = $sev_m, s.sev_l = $sev_l, s.sev_tc = $sev_tc,
    s.self_time_ms = $self_time_ms, s.self_time_flag = $self_time_flag,
    s.nsigma_count = $nsigma_count, s.deploy_proximity = $deploy_proximity,
    s.hours_since_change = $hours_since_change, s.scope_t0 = $t0, s.scope_t1 = $t1
```

```cypher
-- queries.py:UPSERT_SCOPE_GAP  (missing modality / truncation is a first-class node)
MERGE (g:EvidenceGap {id: $id})
SET g.incident_id = $incident, g.node_id = $node_id, g.missing_kind = $kind,
    g.suggestion = $suggestion, g.attempted_queries = $attempted, g.ts = $ts
```

### 2.3 The remaining queries, by pipeline stage

| Stage / capability | Query purpose | Location |
|---|---|---|
| P0 resolve | upsert `Incident`, link `Alert`→`Incident`, `AFFECTS` to services | `engine/p0_resolve.py:133,143` |
| P0 linking | resolve entity by id/name/alias, abstain when unresolved | `engine/p0_resolve.py`, `ingest/normalize.py` |
| Session isolation | copy the public graph into a private session graph per run; abort if the copy is incomplete | `graph/client.py`, `ingest/publication.py:61` |
| Publication guard | `MERGE (p:Artifact {id: $marker_id}) SET p.source_hash = $generation` — detects live edits during a read | `ingest/publication.py:65` |
| P1 scope | the four scope reads above + stats/gap writes | `engine/p1_scope.py:278`, `graph/gateway.py` |
| P2 rank | persist the ranking stage view as an `Artifact` (handoff) | `engine/p2_rank.py:93` |
| P3 investigate | read the admitted scope snapshot, write `Hypothesis` nodes and `HAS_EVIDENCE` edges, `EXPANDED_TO` for expansion | `engine/p3_investigate.py` (6 sites) |
| P4 validate | read hypothesis-owned evidence, persist `ValidatorRefute` evidence and `CONTRADICTS` edges | `engine/p4_validate.py` (5 sites) |
| P5/P6 report | persist report artifact, incident `diagnostic_state`, final hypothesis decisions | `engine/p6_report.py:250,258,262,267` |
| Agent tools (read-only) | neighbourhood, blast radius, severity vector, series history, error spans/logs, runbook + owner closure, anchor resolution, evidence by id — 20 bounded reads, all scope-bound | `agent/tools.py` |
| Path finding | expand `DEPENDS_ON` from a node, honouring `volatile`/`invariant` filters | `graph/pathfind.py:45` |
| Seed / fixtures | MERGE nodes, MERGE edges, verify endpoint labels/round-trip | `seed/loader.py` (7 sites) |
| Benchmark loader | batched `UNWIND $rows ... MERGE` for services, metric anomalies, logs, spans and their `OBSERVED_ON` edges | `eval/rcaeval_adapter.py` |
| Live transcript replay | re-execute every recorded read through `ReadGateway` and compare result sets, so a scope snapshot cannot be trusted after a live edit | `graph/gateway.py:68,86,98` |

## 3. Graph algorithms the product depends on

**Personalized PageRank over `DEPENDS_ON`** — `graph/scorers.py:random_walk`.
Mass starts on the anchor; each of 20 iterations restarts at the anchor with
φ=0.15 and pushes `(1−φ)·mass/out-degree` along caller→dependency edges; dead
ends return their mass to the anchor. Feeds the `rwr` fusion component.

**Anchor-neutral walk** — `scorers.py:neutral_walk` (added for cold ranking).
Same propagation, but the restart is uniform `φ/n` and dead ends keep their own
mass, so topology centrality is preserved without a privileged service. This
matters because the benchmark has no case-derived symptom: with a hardcoded
anchor the `rwr` component elected `frontend` in 90/90 archived cases and the
true service was never its rank-1.

**Reciprocal rank fusion** — `scorers.py:reciprocal_rank_fusion`. Per component
(`phi1`, `phi3`, `severity`, `rwr`, `deploy_proximity`), candidates with a
positive value get `1/(ε + rank)` with ε=60 and *average* ranks for exact ties;
non-positive values earn nothing. Three selectable policies
(`config/tuning.py:P2_RANK_POLICIES`): `baseline_rrf` (default),
`severity_only`, `rrf_anchor_neutral`.

**Latency structure** — `scorers.py:latency_scores`, percentile interpolation
`scorers.py:percentile`. `phi1 = p99/p50` per service (dispersion);
`phi3 = Σ duration × max(0, 1 − max(mean_callee_duration)/mean_self_duration)`,
i.e. work not explained by the slowest callee — computed on the scoped span
sample, deduplicated by span id.

**Severity aggregation** — `engine/p1_scope.py:service_statistics`. Per modality
`sev = max |z|` over that service's admitted records; unscored records are
reported as `unscored_*` gaps instead of zeros, so "not measured" is never read
as "normal". Log/trace z and metric z now share one baseline policy (§ minute
scoring), so no modality saturates below another.

**Minute-level anomaly scoring** — `eval/rcaeval_adapter.py:_minute_stats`,
`_minute_score`. Per service and minute, count (or p95 latency) against the
pre-injection minutes: measured spread scores a plain z; an unmeasured baseline
uses the shared scale ladder (IQR → std → relative floor, one-count floor for
count series) with a persistence guard, so a single quiet extra line is not an
anomaly. The same ladder scores metric series (`eval/baseline_scale.py`).

**Evidence path over the dependency DAG** — `graph/pathfind.py:find_evidence_path`.
Bounded BFS from anchor to candidate (hop and node caps), skipping `volatile`
edges and requiring `invariant != false`; abnormal nodes are visited first;
path nodes get roles (`local_anomaly`, `propagation`, `context`) and each edge
carries explicit `support` evidence ids. A found path is structural
plausibility — never propagation proof.

**Exclusive self-time** — `engine/p1_scope.py:_exclusive`. Interval-union
subtraction of direct-child span time from the parent span, marked `complete`
only when the child query and the source trace are both proven complete;
otherwise an upper bound is reported and the flag is suppressed.

**Validators (deterministic vetoes)** — `graph/validators.py`:
`level_veto` (reject a cause strictly shallower than its effect),
`cycle_rejection` (a proposed cause→effect edge may not close a loop),
`onset_alignment` (±300 s of onset), `svar_lite` (in-window excursion plus
baseline-spread check and direction stability), `symptom_explanation`,
`no_evidence_cap`, `chain_coherence`, `aggregate_verdict`, confidence caps.

**Topology depth + safe certification** — `eval/rcaeval_adapter.py`:
`dependency_depths` (BFS from services that call but are never called) and
`authoritative_levels`, which certifies a service's authored level only when its
whole weakly connected component is acyclic (Kahn) and has no edge whose BFS
depth decreases — otherwise P4 reports "authoritative level unavailable" rather
than trusting a number that could veto a physically valid direction.

**Bounded trace sampling** — `eval/rcaeval_adapter.py:select_trace_sample`.
Deterministic round-robin over minute×pre/post×service strata, errors first
within a stratum, seeded hashes for ties; whole trace groups only, never sliced.

**Fusion-time hypothesis scoring (P5)** — `engine/p5_fuse.py`:
`H = 0.35·D + 0.25·R + 0.20·CC + 0.20·V` with D the rank-normalized P2 position,
R the reflection blend, CC chain coherence, V the validator verdict; fixed
stable-ID hypothesis slots per service; a failed check can never be talked back
into a winner. `conservative_p2_update_v1` (opt-in) will only swap adjacent
services when an explicit structured claim of the displaced service is
contradicted, and abstains otherwise.

## 4. Capability → query/algorithm map

| Product capability | Depends on |
|---|---|
| "What is broken and what is it connected to?" | `SCOPE_SERVICES_FULL` / `SCOPE_SERVICES`, `SCOPE_EDGES` |
| Incident scope + statistics + gaps | `SCOPE_OBSERVATIONS`, `UPDATE_SCOPE_STATS`, `UPSERT_SCOPE_GAP` |
| Cold candidate ranking | `latency_scores`, `random_walk`/`neutral_walk`, `reciprocal_rank_fusion`, severity aggregation |
| Why this candidate | `find_evidence_path` + role assignment, validators, `svar_lite` |
| Blast radius / neighbourhood / owner closure | agent read-only Cypher over `DEPENDS_ON`, `OWNED_BY`, `APPLIES_TO` |
| "Did a deploy cause this?" | `SCOPE_CHANGES`, `deploy_proximity`, `hours_since_change` |
| Trustworthy replay of a past investigation | scope snapshot + `ReadGateway` transcript re-validation, artifact handoffs |
| Proof that nothing was invented | `EvidenceGap` nodes, `unscored_*` reporting, validator vetoes, provenance hashes |

## 5. Verification and provenance

- `make up && make seed && make demo` runs the whole loop locally against
  FalkorDB; `make test` covers the queries and algorithms (unit + integration on
  a disposable database).
- Evaluated results and their caveats: `output/p2-rebuilt-20261018/RESULTS.md`
  (P2 policy comparison), `output/anchor-prior-*/RESULTS.md` (predeclared
  anchor-prior repair), `output/metric-sampling-audit/RESULTS.md` (evidence
  retention), `output/live-shadow-six-*/RESULTS.md` (real-model runs).
- Every run records source/config/input hashes; the anchor-prior candidates were
  written to `protocol.json` before their numbers were computed.
