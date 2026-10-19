"""FastAPI service for live IncidentGraph runs.

Serves the built React app plus a small API that executes one RCAEval case
through the real P0-P6 pipeline on demand and streams progress back. It is a
separate process from the investigation engine: nothing here edits engine code,
and the only external resources it touches are the disposable FalkorDB instance
and the configured provider.
"""
from __future__ import annotations

import asyncio
import json
import os

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import config, dashboard, evidence, live
from . import topology as topology_source
from .models import (CaseEvidence, CaseInfo, CustomRunRequest, Health, ProbeRequest,
                     ProbeResult, ResultsPage, RunSnapshot, StartRunRequest, Topology)

app = FastAPI(
    title="IncidentGraph live runner",
    version="0.2.0",
    description="Runs the P0-P6 investigation pipeline on demand and reports each stage.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Idle seconds before an SSE stream emits a comment line. Proxies and mobile
# networks commonly drop connections that are silent for 15-30s, and a long
# model call can be exactly that. Lower it if your proxy is more aggressive.
STREAM_HEARTBEAT_S = float(os.environ.get("INCIDENTGRAPH_UI_STREAM_HEARTBEAT_S", "10"))


@app.get("/api/health", response_model=Health)
def health() -> Health:
    """Database and provider readiness; the UI gates the Run button on this."""
    return Health(
        database=live.database_status(),
        provider=live.provider_status(),
        active_run=live.MANAGER.active_run_id(),
    )


@app.get("/api/cases", response_model=list[CaseInfo])
def cases() -> list[CaseInfo]:
    return [CaseInfo(**case) for case in config.available_cases()]


@app.get("/api/topology", response_model=Topology)
def topology() -> Topology:
    """The dependency map drawn on the demo page. Identical for every case."""
    return Topology.model_validate(topology_source.topology())


@app.get("/api/cases/{case_id}/evidence", response_model=CaseEvidence)
def case_evidence(case_id: str) -> CaseEvidence:
    """Bounded, read-only view of a case's raw telemetry: what went wrong."""
    payload = evidence.case_evidence(case_id)
    if payload is None:
        raise HTTPException(status_code=404, detail=f"unknown case {case_id!r}")
    return CaseEvidence.model_validate(payload)


@app.post("/api/runs", response_model=RunSnapshot)
def start_run(request: StartRunRequest) -> RunSnapshot:
    """Start a live run. Refuses when the database or provider is unavailable."""
    database = live.database_status()
    if database["state"] != "ok":
        raise HTTPException(status_code=503, detail=f"database offline at {database['detail']}")
    provider = live.provider_status()
    if not provider["available"]:
        raise HTTPException(
            status_code=503,
            detail=f"no live provider configured (resolved to {provider['name']})",
        )
    try:
        record = live.MANAGER.start(request.case_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return RunSnapshot.model_validate(record.snapshot())


@app.post("/api/dashboard/probe", response_model=ProbeResult)
def dashboard_probe(request: ProbeRequest) -> ProbeResult:
    """Read-only check that a caller's database holds what the engine needs."""
    return ProbeResult.model_validate(dashboard.probe(request.model_dump()))


@app.post("/api/dashboard/runs", response_model=RunSnapshot)
def dashboard_run(request: CustomRunRequest) -> RunSnapshot:
    """Run against a caller's graph. Preview mode stays deterministic and free."""
    spec = request.model_dump()
    verdict = dashboard.probe(spec)
    if verdict["verdict"] != "ready":
        raise HTTPException(status_code=422,
                            detail=verdict["detail"] or "the graph is not ready to investigate")
    if spec["mode"] == "full":
        provider = live.provider_status()
        if not provider["available"]:
            raise HTTPException(status_code=503,
                                detail=f"no live provider is configured ({provider['name']})")
    if not spec.get("onset_ts"):
        raise HTTPException(
            status_code=422,
            detail="an authoritative onset is required; the engine will not invent a clock",
        )
    try:
        record = live.MANAGER.start_custom(spec)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return RunSnapshot.model_validate(record.snapshot())


@app.get("/api/runs/{run_id}", response_model=RunSnapshot)
def run_snapshot(run_id: str) -> RunSnapshot:
    record = live.MANAGER.get(run_id)
    if record is not None:
        return RunSnapshot.model_validate(record.snapshot())
    stored = live.load_stored(run_id)
    if stored is None:
        raise HTTPException(status_code=404, detail=f"unknown run {run_id!r}")
    return RunSnapshot.model_validate(stored)


@app.get("/api/runs/{run_id}/stream")
async def stream_run(run_id: str) -> StreamingResponse:
    """Server-sent snapshots, pushed when the run changes.

    Event-driven rather than polled: the worker hands each new snapshot to the
    stream. A heartbeat comment every few seconds keeps proxies and mobile
    networks from dropping a stream that is idle during a long model call.
    """
    record = live.MANAGER.get(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"unknown run {run_id!r}")

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue(maxsize=1)  # newest state supersedes
    record.subscribe(loop, queue)

    async def events():
        try:
            while True:
                try:
                    snapshot = await asyncio.wait_for(queue.get(), timeout=STREAM_HEARTBEAT_S)
                except asyncio.TimeoutError:
                    yield ": ping\n\n"
                    continue
                yield f"data: {json.dumps(snapshot)}\n\n"
                if snapshot["status"] in live.TERMINAL_STATUSES:
                    break
        finally:
            record.unsubscribe(loop, queue)

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.get("/api/results", response_model=ResultsPage)
def results() -> ResultsPage:
    """Past runs stored on disk, newest first (no re-spend to read them)."""
    return ResultsPage(runs=live.stored_runs())


if config.WEB_DIST.is_dir():
    assets = config.WEB_DIST / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str) -> FileResponse:
        candidate = config.WEB_DIST / path
        if path and candidate.is_file() and not path.endswith(".html"):
            return FileResponse(candidate)
        # index.html references content-hashed assets, so it must never be
        # cached: a stale copy would ask for a bundle that no longer exists.
        return FileResponse(config.WEB_DIST / "index.html",
                            headers={"Cache-Control": "no-store, must-revalidate"})

else:

    @app.get("/", include_in_schema=False)
    def missing_build() -> JSONResponse:
        return JSONResponse(
            content={
                "detail": "web build not found",
                "hint": "run `npm install && npm run build` in web/, or use the Vite dev server",
            }
        )


def main() -> None:
    uvicorn.run("server.app.main:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
