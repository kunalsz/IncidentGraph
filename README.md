# IncidentGraph

An **AI incident investigator for distributed systems**, built on
[FalkorDB](https://falkordb.com) for the FalkorDB Hackathon(**Track 01 :** Agents That Act on Connected Data).

Report a symptom ("checkout latency up"); the agent traverses a FalkorDB graph
of services, dependencies, telemetry, deployments, runbooks, teams, and past
incidents to produce a leading root-cause hypothesis, an evidence path, and a
recommended next action with machine-checkable evidence for every claim.

## What runs where

Four processes, two databases. The second exists because the web demo runs
*real* investigations and must not share the seeded demo graph.

| Process | Port | Talks to | Started by |
| --- | --- | --- | --- |
| FalkorDB (demo graph, engine, tests, evals) | 6380 | n/a | `make up` |
| FalkorDB (web service only) | 6381 | n/a | `make ui-db` |
| FastAPI runner service (`server/`) | 8000 | 6381 | `make api` |
| React UI (`web/`, Vite dev server) | 5173 | 8000 via `/api` proxy | `make web` |

The CLI demo, the test suite and the benchmark evals use **6380**; the web
service uses **6381** (`INCIDENTGRAPH_UI_DB_PORT`, see `.env.example`). Building
the UI (`cd web && npm run build`) lets the API serve it from
`http://127.0.0.1:8000/`, so production needs only one port.

## Prerequisites

- Docker (Compose v2 `docker compose`, or legacy `docker-compose`; the Makefile
  detects which one you have)
- Python 3.10+ and `venv`
- Node 18+ with `npm` (web UI only)
- Optional: a model provider key. `mock` runs every stage offline with no key.

## Quickstart: CLI demo (no LLM key needed)

```bash
# 1. Databases and Python env (pandas/pyarrow are used by the RCAEval adapter)
make up
python3 -m venv .venv && .venv/bin/pip install -e 'backend[dev,api,eval]'

# 2. Seed the demo graph, inject a fixture incident, run P0
make seed
make demo          # = demo_inject.py --fixture payment_failure --seed-demo --run-p0

# 3. Step through the pipeline
.venv/bin/python scripts/demo_inject.py --fixture payment_failure --seed-demo --run-p1
.venv/bin/python scripts/demo_inject.py --fixture payment_failure --seed-demo --run-p2

# 4. Tests (needs the 6380 database from step 1)
make test
```

`make demo` must inject the fixture (`--seed-demo`) before running P0: the
fixture's alert only exists once injected, and P0 resolves its anchor from that
alert. Full-scale diagnosis needs P3 to P6, which require a provider; the CLI
refuses to reach a live one implicitly, so P0 to P2 run with `LLM_PROVIDER=none`
and the agent stages are driven programmatically (`rcaeval_runner.run_case(...)`
with an explicit client factory) or through the web service.

## Web demo

Two terminals, plus the UI's own database. `LLM_PROVIDER=mock` gives the fully
offline path (every stage runs, no key, no network); `openai`/`anthropic` use
your key and produce the real-model run.

```bash
# 1. Configure a provider (copy the example and edit, or export in the shell)
cp .env.example .env      # set LLM_PROVIDER=mock for an offline demo

# 2. UI database on 6381 + RCAEval cases (the demo page lists 10 of them)
make ui-db
.venv/bin/python scripts/fetch_rcaeval.py --subset 20

# 3a. Terminal 1: API on :8000
make api
# 3b. Terminal 2: UI on :5173 (installs npm deps on first run)
make web
```

Open `http://127.0.0.1:5173`. Pages: `#/` (the idea), `#/dashboard` (point it at
any reachable FalkorDB, verify the schema, preview the ranking for free or run
the full investigation), `#/demo` (the RCAEval cases over their shared
dependency graph, with per-case evidence read from that case's own parquet files
and a live run that streams the seven stages).

Single-origin production view: `cd web && npm run build`, then open
`http://127.0.0.1:8000/`. The API serves `web/dist` and returns a hint if it has
not been built yet.

### What you should see

- `make demo` prints the P0 Investigation JSON (anchor, window, budget) and exits
  0. A degraded run instead means the fixture was not injected; see
  Troubleshooting.
- The UI's health pill turns green when the API reaches its database; the run
  button then unlocks and streams one ledger row per stage with each model call,
  a cost line (calls, tokens in/out, elapsed, cap status) and a closing verdict
  with the markdown report.
- The case-evidence disclosure on `#/demo` never triggers a run: it reads that
  case's telemetry through the same adapter helpers the pipeline uses.

## Configuration

`.env` lives in the repository root and is read by CLI scripts, `make` targets,
the test suite and the API service. Variables already set in the shell win, so
`export LLM_PROVIDER=mock` overrides the file. Nothing is required, every
setting has a working default. The important ones:

| Variable | Default | Meaning |
| --- | --- | --- |
| `FALKORDB_HOST` / `FALKORDB_PORT` | `localhost` / `6380` | engine, CLI demo, tests, evals |
| `LLM_PROVIDER` | `none` | `none` \| `mock` \| `openai` \| `anthropic`; the web service refuses to start a run while it is `none` |
| `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY(S)` | empty | any OpenAI-compatible endpoint, incl. gateways; keys may be a comma-separated list |
| `INCIDENTGRAPH_UI_DB_HOST` / `INCIDENTGRAPH_UI_DB_PORT` | `127.0.0.1` / `6381` | the web service's database; it pins `FALKORDB_*` for its engine phases, so setting `FALKORDB_PORT` alone does not redirect it |
| `INCIDENTGRAPH_PARALLELISM` | `8` | concurrent P3 branches |

`.env.example` documents every variable, including the accepted key aliases.

## Benchmark data (needed for the web demo and the evals)

The web demo page and the RCAEval evals read benchmark telemetry, which is **not
committed** (`.gitignore`), so fetch it:

```bash
.venv/bin/python scripts/fetch_rcaeval.py --all              # all 90 RE2-OB cases, ~880 MB
.venv/bin/python scripts/fetch_rcaeval.py --subset 20        # a smaller start, ~200 MB
.venv/bin/python scripts/fetch_rcaeval.py --all --no-traces  # skip traces (~10 MB/case)
.venv/bin/python scripts/fetch_rcaeval.py --cases re2ob_checkoutservice_cpu_1,re2ob_carts_delay_1
```

- **Source, licence, revision.** RCAEval RE2-OB from
  [phamquiluan/RCAEval](https://huggingface.co/datasets/phamquiluan/RCAEval)
  (**MIT**, public). The fetcher resolves the requested revision (default
  `main`) to an immutable commit and records it, with per-file SHA256 checksums,
  in `data/RE2-OB/_subset.json`.
- **Verification.** Each file is checked against the dataset's own metadata:
  large files against their LFS **content SHA256**, small ones against Git's
  blob hash. The resolve endpoint's ETag is a chunk hash on HuggingFace's Xet
  backend and is not used as a guarantee. A file failing verification is removed,
  never kept; an already-valid file is skipped without re-downloading, so
  re-running is cheap and safe.
- **Rate limits.** The API allows 500 requests per 5 minutes. The fetcher makes
  one metadata call per case and honours `Retry-After` with backoff on 429 and
  5xx; other 4xx (a missing file, a bad revision) fail immediately rather than
  retrying.
- **`_subset.json` describes the last fetch, and a narrower fetch replaces it.**
  Run `--all` (or `--cases ...`) when you want the full manifest; re-running
  `--subset 20` afterwards rewrites it with those 20 cases.

Without this data the demo page shows an empty case list and `make eval-rcaeval`
reports "dataset absent"; the CLI demo (`make demo`) uses the authored fixtures
and needs no benchmark telemetry.

## How it works

Seven phases, each bounded and independently auditable. `docs/P00.md` to `P06.md`
document each phase; [docs/GRAPH_MODEL.md](docs/GRAPH_MODEL.md) documents the
graph model, the Cypher the product depends on, and the algorithms.

| Phase | What it does |
| --- | --- |
| P0 resolve | anchors the incident from the alert or the symptom, and copies the graph into an isolated session graph |
| P1 scope | builds a bounded, read-only evidence snapshot (anchor plus one-hop neighbours, capped) with per-service statistics |
| P2 rank | deterministic scores plus reciprocal-rank fusion select the services P3 investigates |
| P3 investigate | agent branches propose hypotheses with cited evidence, inside the run budget |
| P4 validate | deterministic checks veto claims their own cited evidence contradicts |
| P5 fuse | combines the surviving hypotheses into one ranked, typed list with a winner and alternatives |
| P6 report | renders the report and re-checks every citation against the live graph |

**Three things it will not do.** It does not invent a measurement: a `0` in the
severity vector means "not measured" and is reported as an `unscored_*` gap,
never as "looked normal", and a missing modality becomes an `EvidenceGap`.
It does not let a weighted score override a veto: a failed P4 check removes the
candidate from winner and alternative eligibility, however high its score.
And it does not claim causality: a dependency edge or a passing validator check
is an observed association, and the report says so.

**Which ranking policy runs.** A real incident has a case-derived anchor (alert
or symptom), so the product default is `baseline_rrf` with its personalised walk.
The RCAEval harness has no symptom and fixes the anchor to an entry service, so
it declares `case_anchor=False` and defaults to `severity_only`, the only policy
that led on both Online Boutique (78.9%) and SockShop (85.0%). Either walk
variant wins on one corpus and loses on the other, so neither is the benchmark
default; all three remain selectable. Tables and mechanism:
`output/replication-2026-10-19/RESULTS.md`.

### The deterministic ranking gate

```bash
make eval                                   # three authored fixtures, no provider
.venv/bin/python scripts/eval.py --fixtures --rank-only --rank-policy severity_only
.venv/bin/python scripts/eval.py --rcaeval data/RE2-OB --subset 6 --rank-policy severity_only
.venv/bin/python -m backend.app.eval.p2_ablation RUN_DIR   # offline replay of stored records
```

Rank-only commands make no paid provider calls. They require FalkorDB, and the
RCAEval ones additionally need fetched data. Effective P2 policy and version are
recorded in ranking artifacts and in RCAEval case/run provenance, including
failures; `p2_ablation` replays each record under its declared policy (records
without one mean baseline RRF). The engine keeps every rule-flagged service as a
P3 branch; the benchmark harness narrows the initial branches to a three-service
coverage floor, so a benchmark case investigates a fixed, equal set. Payment,
cart and recommendation causes rank first on the three authored fixtures (seven
candidates each): a smoke gate, not external benchmark accuracy.

## Benchmark evaluation

Deterministic ranking over the full RE2-OB corpus, **with no model calls**. Every
case record stores its own per-candidate components, so each policy is scored
offline from identical inputs. The candidate list and the selection rule were
written to `protocol.json` before any number was computed
(`backend/app/eval/anchor_prior_experiment.py`).

| Ranker (zero LLM) | Online Boutique, 90 cases | SockShop, 60 cases |
| --- | --- | --- |
| `severity_only` (benchmark default) | **78.9%** AC@1 · 95.6% AC@3 · MRR 0.871 | **85.0%** · 93.3% · 0.903 |
| anchor-neutral walk | 57.8% · 93.3% · 0.755 | 23.3% · 60.0% · 0.477 |
| anchored entry-service walk (pre-repair default) | 24.4% · 90.0% · 0.559 | 43.3% · 83.3% · 0.656 |

**How these were measured.**

- **Data.** RCAEval **RE2-OB** (all 90 cases) and **RE2-SS** (a predeclared
  60-case subset: the first ten per fault class), both at pinned revision
  `afeacb11bcc94dadfd1c8f483ee4377b2b8b614e`, both **MIT**. Fetched and
  checksum-verified by `scripts/fetch_rcaeval.py`; the manifest in
  `data/*/_subset.json` records the revision and per-file hashes.
- **Metric.** Service-level AC@1 / AC@3 / MRR under RCAEval's own evaluation
  semantics (`backend/app/eval/evaluator.py`): ranking strings normalized to
  entities, `-db` stripped, duplicates collapsed. Chance AC@1 is `1/n` with
  n = 12 candidates on Boutique and 15 on SockShop. That is **8.3%** and **6.7%**.
  The evaluator's own chance `Avg@5` on SockShop is 0.200, so `severity_only`
  clears it by a lift of 0.730.
- **Zero model calls.** The ranking is graph-computable and deterministic, so the
  whole table reproduces from the stored records with no provider tokens. See
  `make eval`, or `p2_ablation`/`anchor_prior_experiment` for offline replay.
- **Reference reconstruction.** The anchored baseline is recomputed offline from
  the stored scope edges and verified bit-identical to the masses the pre-repair
  records carried: 1082 services compared, maximum difference 0.0.

**What this table is not.**

- **Development data, not a holdout.** Every case here has informed a design
  decision (the evidence-sampling audit, an earlier ablation's split, the live
  agent runs), so read it as measured evidence on a corpus we have examined
  repeatedly, not as accuracy on unseen data. The two walk variants *reverse
  order* between the corpora, and that is the load-bearing result: a topology
  prior's direction depends on graph shape, so neither variant is a default.
- **SockShop has no traces**, so its rows compare severity and logs against a
  topology prior only; they say nothing about the latency components or about
  propagation.
- **Not a head-to-head with published systems.** GALA reports pod-level AC@1
  (finer than this service-level metric) on its own split, and its implementation
  is withheld from public release, so it cannot be re-run; RCAEval's own
  baselines are not re-run here either. No comparison row is claimed.
- **Status of the exact percentages.** Measured on a substrate collected before
  the trace-status fix (the status column is gRPC-coded; reading it as OTEL hid
  6,481 failing spans, and 122 caller/callee failure pairs, in 19 of 90 cases).
  A regenerated substrate is in progress, so treat the method and the *ordering*
  of the policies as the result, and these percentages as the last complete
  measurement. Full report: `output/replication-2026-10-19/RESULTS.md`.

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

## Repository layout

```text
config/tuning.py         every constant, single-sourced (k=6, θ=0.6, ε=60, budgets…)
backend/app/
  graph/                 client, schema, queries (FalkorDB layer)
  seed/                  topology.json (+ per-corpus RCAEval topologies), fixtures/
  ingest/normalize.py    §4.7 normalization pipeline
  engine/                p0_resolve…p6_report, contracts, budget, policy
  agent/                 LLM client, prompts, bounded bundle builder, tools
  eval/                  RCAEval adapter, runner, ablation, comparisons
server/                  FastAPI runner service for the UI (uvicorn :8000)
web/                     React + TypeScript + Vite UI (:5173 dev)
scripts/                 seed_demo, demo_inject, eval, fetch_rcaeval, measure_tool_surface
tests/                   unit/ + integration/ (live FalkorDB)
docs/                    GRAPH_MODEL.md (graph model, Cypher, algorithms), AI_DISCLOSURE.md, blog-post.md
```

Setup notes specific to each service live next to them: [`web/README.md`](web/README.md)
(pages, state, layout) and [`server/README.md`](server/README.md) (routes, run
model).

## Make targets

| Target | What it does |
| --- | --- |
| `up` / `down` | start / stop the demo FalkorDB (port 6380) |
| `ui-db` / `ui-db-down` | start / stop the web service's FalkorDB (port 6381) |
| `seed` | load the demo topology, teams, runbooks, instances |
| `demo` | seed-checked fixture incident + P0 run (the CLI demo) |
| `api` / `web` | FastAPI runner on :8000 / Vite dev server on :5173 |
| `test` | full pytest suite (needs 6380; unit-only: `pytest tests/unit -q`) |
| `eval` | deterministic fixture ranking gate (no provider) |
| `eval-rcaeval` | P2 over fetched RCAEval data (no provider) |
| `surface` | measure the agent's tool surface |

Live-agent evaluation is deliberately **not** a CLI flag: the CLI refuses
implicit providers, and agent runs are invoked with an explicit client factory
or through the web service.

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `make up` → "unknown command: compose" | Docker Compose v2 plugin missing | the Makefile falls back to `docker-compose`; if neither exists, install one, or run a FalkorDB container by hand on 6380 |
| `make demo` exits non-zero, P0 unresolved | the fixture alert was never injected | `make demo` now passes `--seed-demo`; if you run `demo_inject.py` directly, keep that flag |
| Tests error with `ConnectionError … 6380` | no database on 6380 | `make up` first (`pytest tests/unit -q` needs no database) |
| Web run button disabled / API returns 503 | provider is `none`, or the UI database is unreachable | set `LLM_PROVIDER=mock` for an offline run; check the health pill and `make ui-db` |
| Demo page lists no cases | benchmark data not fetched | `scripts/fetch_rcaeval.py --subset 20` |
| `make api` starts but `GET /` shows a build hint | UI not built (dev server is the alternative) | `cd web && npm run build`, or use `make web` and open :5173 |
| Port 6381 already in use | another FalkorDB is running there | `INCIDENTGRAPH_UI_DB_PORT=6390 make ui-db api` |
| A run's stage rows stop mid-pipeline | budget or wall-clock cap reached (by design) | the ledger shows the stage that stopped and the reason; caps live in `config/tuning.py` |
| `GET /` still shows the build hint after `npm run build` | the API resolves `web/dist` at startup | restart `make api` after building (the dev server does not have this ordering constraint) |
| FalkorDB container disappears (exit 139) under heavy parallel work | FalkorDB runs in a single container; several concurrent investigations can exhaust it | `docker start <container>` and lower `INCIDENTGRAPH_PARALLELISM` / `P3_MAX_PARALLEL` |
| `docker-compose` errors with "Not supported URL scheme http+docker" | legacy Compose v1 against a newer Docker API | `make up` falls back to starting the container directly; nothing to do |

## Graph model, Cypher and graph algorithms

The FalkorDB data model, the Cypher queries the product depends on, and the graph
algorithms it implements are documented in [docs/GRAPH_MODEL.md](docs/GRAPH_MODEL.md)
required by the hackathon submission rules (Rule 10).

## AI disclosure

Help was taken from AI coding assistants (Command Code). The team can
explain every component; the deterministic scorers and validators carry
hand-computed unit tests. Full disclosure, including the corrections made
during the work, in [`docs/AI_DISCLOSURE.md`](docs/AI_DISCLOSURE.md).
