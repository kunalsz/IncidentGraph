# Runner service (FastAPI)

Executes the real IncidentGraph pipeline live and streams progress to the web UI.
It is a separate process from the investigation engine: nothing here edits engine
code, and the only external resources it touches are a FalkorDB instance and the
configured model provider.

## What it does

| Route | Purpose |
| --- | --- |
| `GET /api/health` | database reachability, resolved provider/model, active run |
| `GET /api/cases` | the ten RCAEval cases available locally |
| `POST /api/runs` | start a run on one RCAEval case |
| `GET /api/runs/{run_id}` | snapshot of a live or stored run |
| `GET /api/runs/{run_id}/stream` | server-sent snapshots until the run ends |
| `GET /api/results` | runs already stored on disk, newest first |
| `POST /api/dashboard/probe` | read-only check of a caller-supplied graph |
| `POST /api/dashboard/runs` | run against that graph (preview or full) |
| `GET /` | the built React app from `web/dist` |

Interactive schema: `http://127.0.0.1:8000/docs`.

## Run

```bash
# from the repository root, with a FalkorDB reachable on 127.0.0.1:6381
.venv/bin/python -m uvicorn server.app.main:app --host 127.0.0.1 --port 8000
```

Dependencies: `fastapi`, `uvicorn` (`requirements.txt`).

## How a run is executed

`live.py` composes the engine's public phase functions — the same wrappers the
RCAEval benchmark uses — and adds the report stage the benchmark stops short of:

```
adapter P0 → scope (full) → rank → investigate → validate → select → report
```

* one run at a time (`RunManager` is single-flight; a second start is refused);
* each phase is wrapped so its result is published the moment it returns, which
  is what the UI renders row by row;
* a per-run `BudgetMeter` bounds the work (75 calls / 220k tokens / 540 s) and
  supplies the call and token totals shown live;
* the session graph is deleted in a `finally` block, and P1 deletes its own read
  snapshot.

## Custom targets (`dashboard.py`)

A caller supplies `host`, `port`, `graph`, `anchor_service`, a window and an
authoritative onset. Two modes:

* **preview** — P0→P2 only: deterministic ranking, no provider call is reachable;
* **full** — P0→P6 with the live agent, bounded by the same budget.

Isolation and safety:

* the caller's graph is only read: the run happens on a `GRAPH.COPY` session
  that is deleted afterwards;
* the engine's client singleton is rebound to the caller's instance for the
  duration of the run and restored afterwards — runs are single-flight, which is
  what makes that safe;
* only loopback and private addresses are accepted (`INCIDENTGRAPH_UI_ALLOWED_HOSTS`
  extends the allowlist), so the service cannot be used to probe other hosts.

The engine requires an **authoritative onset**; it will not invent a clock, so a
run without one is refused with a clear message.

## Modules

- `app/main.py` — FastAPI routes and the static SPA mount
- `app/live.py` — run record, phase composition, meter accounting, run manager
- `app/dashboard.py` — host policy, connection probe, schema verification, custom runs
- `app/models.py` — pydantic response contract the UI mirrors
- `app/config.py` — budget caps, featured cases, stage metadata
