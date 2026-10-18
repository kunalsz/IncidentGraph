"""Live P0-P6 pipeline runner.

Executes one RCAEval case through the real investigation engine: bounded scope
admission, deterministic ranking, live agent investigation, validation,
selection and reporting. Only public engine entry points are used — this module
never edits the engine, eval, config or test code.

Guarantees:
  * one run at a time, in a worker thread, with a bounded budget;
  * a disposable session graph that is deleted when the run ends;
  * a per-run meter, so the reported call/token totals are that run's alone.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from . import config

logger = logging.getLogger(__name__)

# The engine resolves database and provider configuration from the process
# environment on first use, and load_env_file() never overrides variables that
# are already set, so pin the run environment before importing anything from
# backend/.
os.environ["FALKORDB_HOST"] = config.DB_HOST
os.environ["FALKORDB_PORT"] = str(config.DB_PORT)
os.environ.setdefault("INCIDENTGRAPH_DEPLOYMENT_MODE", "local")
os.environ["INCIDENTGRAPH_PARALLELISM"] = str(config.PARALLELISM)
os.environ["P3_MAX_PARALLEL"] = str(config.PARALLELISM)

# The engine's default .env lookup resolves one directory above the repository,
# so point it at this repository's file explicitly. Existing variables always
# win, which is why the database overrides above are set first.
ENV_FILE = config.ROOT / ".env"


def load_environment() -> None:
    if not ENV_FILE.is_file():
        return
    try:
        from backend.app.envfile import load_env_file

        load_env_file(str(ENV_FILE))
    except Exception:
        logger.warning("could not load %s; relying on the process environment", ENV_FILE,
                       exc_info=True)


load_environment()

TERMINAL_STATUSES = ("completed", "failed")


def _error_text(exc: BaseException) -> str:
    """Provider errors can embed request context; keep only a safe summary."""
    name = type(exc).__name__
    if isinstance(exc, (ValueError, RuntimeError, KeyError)):
        return f"{name}: {str(exc)[:200]}"
    return name


# ---------------------------------------------------------------------------
# health
# ---------------------------------------------------------------------------

def database_status() -> dict[str, Any]:
    """Ping the disposable database; never raises."""
    try:
        from backend.app.graph.client import client

        client().execute_command("PING")
        graphs = client().execute_command("GRAPH.LIST") or []
        return {"state": "ok", "detail": f"{config.DB_HOST}:{config.DB_PORT}",
                "graphs": len(graphs)}
    except Exception as exc:
        # Drop the cached handle so a later retry reconnects instead of reusing
        # a dead socket.
        try:
            from backend.app.graph import client as client_module

            client_module._client = None
        except Exception:
            logger.debug("could not reset the cached graph client", exc_info=True)
        return {"state": "offline", "graphs": None,
                "detail": f"{type(exc).__name__}: cannot reach {config.DB_HOST}:{config.DB_PORT} "
                          "(set INCIDENTGRAPH_UI_DB_HOST / INCIDENTGRAPH_UI_DB_PORT to point elsewhere)"}


def provider_status() -> dict[str, Any]:
    """Report which provider a run would use; refuses to guess."""
    try:
        from backend.app.agent.llm import provider_from_env

        load_environment()
        provider = provider_from_env()
        name = getattr(provider, "name", type(provider).__name__)
        available = type(provider).__name__ != "NoneProvider"
        model = os.environ.get("LLM_MODEL") or getattr(provider, "DEFAULT_MODEL", None)
        return {"name": name, "model": model, "available": available, "detail": None}
    except Exception as exc:
        return {"name": "unavailable", "model": None, "available": False,
                "detail": type(exc).__name__}


# ---------------------------------------------------------------------------
# per-stage summaries
# ---------------------------------------------------------------------------

def _service(value: Any) -> str | None:
    return None if value is None else str(value).replace("service:", "")


def _stage_p0(investigation: Any, loaded: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    window = getattr(investigation, "window", None)
    onset = getattr(investigation, "onset_ts", None)
    anchor = _service(getattr(investigation, "anchor_service", None))
    line = f"anchor {anchor}, onset {'from the dataset' if onset else 'not supplied'}"
    detail = [
        {"k": "incident", "v": getattr(investigation, "incident_id", "")},
        {"k": "window", "v": f"{getattr(window, 't0', '?')} to {getattr(window, 't1', '?')}"},
        {"k": "onset", "v": str(onset) if onset else "unknown"},
        {"k": "graph", "v": getattr(investigation, "session_graph", "")},
    ]
    counts = loaded.get("counts") or {}
    if counts:
        detail.append({"k": "loaded", "v": ", ".join(f"{k} {v}" for k, v in counts.items())})
    return line, detail


def _stage_p1(scope: Any) -> tuple[str, list[dict[str, Any]]]:
    services = len(getattr(scope, "services", []) or [])
    series = getattr(scope, "n_series_in_scope", None)
    gaps = len(getattr(scope, "evidence_gaps", []) or [])
    truncated = bool(getattr(scope, "truncated", False))
    line = f"{services} services, {series} series" + (", truncated" if truncated else "")
    detail = [
        {"k": "services", "v": str(services)},
        {"k": "series", "v": str(series)},
        {"k": "evidence gaps", "v": str(gaps)},
        {"k": "truncated", "v": "yes" if truncated else "no"},
    ]
    return line, detail


def _stage_p2(ranking: Any) -> tuple[str, list[dict[str, Any]]]:
    candidates = list(getattr(ranking, "candidates", []) or [])
    ordered = sorted(candidates, key=lambda c: getattr(c, "rank", 99))
    top = ordered[0] if ordered else None
    truncated = bool(getattr(ranking, "truncated", False))
    line = (f"top {_service(getattr(top, 'id', None))} of {len(candidates)} candidates"
            + (", truncated" if truncated else ""))
    detail = [{"k": "candidates", "v": str(len(candidates))},
              {"k": "branches", "v": str(len(getattr(ranking, "branch_ids", []) or []))},
              {"k": "truncated", "v": "yes" if truncated else "no"}]
    for candidate in ordered[:4]:
        detail.append({"k": f"rank {getattr(candidate, 'rank', '?')}",
                       "v": f"{_service(getattr(candidate, 'id', None))} "
                            f"({getattr(candidate, 'score', 0.0):.4f})"})
    return line, detail


def _stage_p3(invest: Any) -> tuple[str, list[dict[str, Any]]]:
    branches = getattr(invest, "n_branches", None)
    hypotheses = getattr(invest, "n_hypotheses", None)
    truncated = bool(getattr(invest, "truncated", False))
    line = f"{branches} branches, {hypotheses} hypotheses" + (", truncated" if truncated else "")
    detail = [
        {"k": "branches", "v": str(branches)},
        {"k": "hypotheses", "v": str(hypotheses)},
        {"k": "expansions", "v": str(len(getattr(invest, "expansions", []) or []))},
        {"k": "truncated", "v": "yes" if truncated else "no"},
    ]
    return line, detail


def _stage_p4(validated: Any) -> tuple[str, list[dict[str, Any]]]:
    assessments = list((getattr(validated, "hypotheses", {}) or {}).values())
    tally = {"pass": 0, "partial": 0, "fail": 0}
    timed = 0
    multimodal = 0
    for assessment in assessments:
        verdict = getattr(assessment, "verdict", None)
        if verdict in tally:
            tally[verdict] += 1
        if getattr(assessment, "qualified_timing", False):
            timed += 1
        if len(getattr(assessment, "modalities", []) or []) >= 2:
            multimodal += 1
    line = f"{tally['pass']} pass, {tally['partial']} partial, {tally['fail']} fail"
    detail = [
        {"k": "assessments", "v": str(len(assessments))},
        {"k": "qualified timing", "v": str(timed)},
        {"k": "two or more modalities", "v": str(multimodal)},
    ]
    return line, detail


def _stage_p5(fused: Any) -> tuple[str, list[dict[str, Any]]]:
    winner = getattr(fused, "winner_candidate", None)
    line = f"winner {_service(winner)}" if winner else "no winner, the selector abstained"
    detail = [
        {"k": "presentation", "v": str(getattr(fused, "presentation", ""))},
        {"k": "fused hypotheses", "v": str(len(getattr(fused, "fused", []) or []))},
        {"k": "alternatives", "v": str(len(getattr(fused, "alternatives", []) or []))},
    ]
    return line, detail


def _stage_p6(report: Any) -> tuple[str, list[dict[str, Any]]]:
    reasons = (((getattr(report, "config", {}) or {}).get("gate") or {}).get("reasons")) or []
    line = f"{getattr(report, 'tier', '?')} — {reasons[0] if reasons else getattr(report, 'verdict', '')}"
    detail = [
        {"k": "verdict", "v": str(getattr(report, "verdict", ""))},
        {"k": "confidence", "v": f"{getattr(report, 'confidence', 0.0)} "
                                f"({getattr(report, 'confidence_rule', '')})"},
        {"k": "winner", "v": _service(getattr(report, "winner_candidate", None)) or "none"},
    ]
    for reason in reasons[1:4]:
        detail.append({"k": "also", "v": str(reason)})
    return line, detail


_SUMMARISERS = (_stage_p0, _stage_p1, _stage_p2, _stage_p3, _stage_p4, _stage_p5, _stage_p6)


def _cost(meter: Any) -> dict[str, Any]:
    data = meter.as_dict()
    return {
        "calls": data.get("provider_attempts", 0),
        "tokens_in": data.get("tokens_in", 0),
        "tokens_out": data.get("tokens_out", 0),
        "tokens_total": data.get("tokens_total", 0),
        "tokens_charged": data.get("tokens_charged", 0),
        "wall_s": data.get("wall_s", 0.0),
        "unknown_usage_attempts": data.get("unknown_usage_attempts", 0),
        "compliant": bool(data.get("compliant", False)),
        "budget": data.get("budget", {}),
    }


def cost_snapshot(meter: Any) -> dict[str, Any]:
    """Public alias used by the custom-target runner too."""
    return _cost(meter)


def stage_summary(index: int, value: Any, *, investigation: Any = None,
                  loaded: dict[str, Any] | None = None) -> tuple[str, list[dict[str, Any]]]:
    """One-line summary plus expandable detail for a finished phase."""
    if index == 0:
        return _stage_p0(investigation, loaded or {})
    return _SUMMARISERS[index](value)


def tracked_phase(record: "RunRecord", index: int, fn: Any, meter: Any, *,
                  investigation: Any = None, loaded: dict[str, Any] | None = None) -> Any:
    """Wrap a phase so its result is published the moment it returns."""
    def call(*args: Any, **kwargs: Any) -> Any:
        record.set_stage(index, "running")
        value = fn(*args, **kwargs)
        line, detail = stage_summary(index, value, investigation=investigation, loaded=loaded)
        record.set_stage(index, "done", line, detail)
        record.set_cost(cost_snapshot(meter))
        return value
    return call


# ---------------------------------------------------------------------------
# run record
# ---------------------------------------------------------------------------

class RunRecord:
    """Mutable run state, read concurrently by the API while the worker writes."""

    def __init__(self, run_id: str, case_id: str, *, kind: str = "benchmark",
                 mode: str = "full") -> None:
        self.run_id = run_id
        self.case_id = case_id
        self.kind = kind
        self.mode = mode
        self.status = "running"
        self.started_at = time.time()
        self.finished_at: float | None = None
        self.error: str | None = None
        self.session_graph: str | None = None
        self.cleanup: str | None = None
        self.stages: list[dict[str, Any]] = [
            {"id": stage_id, "name": name, "status": "idle", "line": None, "detail": []}
            for stage_id, name in config.STAGES
        ]
        self.cost: dict[str, Any] | None = None
        self.result: dict[str, Any] | None = None
        self.events: list[dict[str, Any]] = []
        self.activity: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        # (event loop, asyncio.Queue) of every open progress stream.
        self._subscribers: list[tuple[Any, Any]] = []

    # -- change notification ------------------------------------------------

    def subscribe(self, loop: Any, queue: Any) -> None:
        """Register a progress stream; it receives the current state at once."""
        with self._lock:
            self._subscribers.append((loop, queue))
        self._publish()

    def unsubscribe(self, loop: Any, queue: Any) -> None:
        with self._lock:
            self._subscribers = [item for item in self._subscribers if item != (loop, queue)]

    @staticmethod
    def _offer(queue: Any, snapshot: dict[str, Any]) -> None:
        """Runs on the event loop. Coalesces: a newer snapshot wins."""
        try:
            queue.put_nowait(snapshot)
        except asyncio.QueueFull:
            try:
                queue.get_nowait()
            except Exception:
                logger.debug("progress queue was already drained", exc_info=True)
            try:
                queue.put_nowait(snapshot)
            except Exception:
                logger.debug("progress snapshot dropped", exc_info=True)

    def _publish(self) -> None:
        """Hand the latest state to every subscriber.

        Must never be called with ``self._lock`` held: it takes the lock again
        to copy the subscriber list and to build the snapshot.
        """
        with self._lock:
            subscribers = list(self._subscribers)
        if not subscribers:
            return
        snapshot = self.snapshot()
        for loop, queue in subscribers:
            try:
                if loop.is_closed():
                    self.unsubscribe(loop, queue)
                    continue
                loop.call_soon_threadsafe(self._offer, queue, snapshot)
            except Exception:
                logger.debug("dropping a progress subscriber", exc_info=True)
                self.unsubscribe(loop, queue)

    # -- worker-side updates ------------------------------------------------

    def set_stage(self, index: int, status: str, line: str | None = None,
                  detail: list[dict[str, Any]] | None = None) -> None:
        with self._lock:
            stage = self.stages[index]
            stage["status"] = status
            if line is not None:
                stage["line"] = line
            if detail is not None:
                stage["detail"] = detail
        self._publish()

    def set_cost(self, cost: dict[str, Any]) -> None:
        with self._lock:
            self.cost = cost
        self._publish()

    def on_event(self, event: dict[str, Any]) -> None:
        """Engine event sink. Runs in-process on the cooperative path."""
        with self._lock:
            self.events.append(event)
            del self.events[:-40]

    def on_call(self, info: dict[str, Any]) -> None:
        """One model call finished: record it against whichever stage is running."""
        with self._lock:
            stage = next((item["id"] for item in self.stages
                          if item["status"] == "running"), None)
            self.activity.append({"seq": len(self.activity) + 1, "stage": stage, **info})
            del self.activity[:-80]
        self._publish()

    def fail(self, message: str) -> None:
        with self._lock:
            self.status = "failed"
            self.error = message
            self.finished_at = time.time()
            for stage in self.stages:
                if stage["status"] in ("idle", "running"):
                    stage["status"] = "failed"
                    break
        self._publish()

    # -- reader side --------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            finished = self.finished_at or time.time()
            return {
                "run_id": self.run_id,
                "case_id": self.case_id,
                "kind": self.kind,
                "mode": self.mode,
                "status": self.status,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                # The client needs the server's own clock to show a live elapsed
                # time that is not skewed by the browser's clock.
                "server_time": time.time(),
                "duration_s": round(finished - self.started_at, 1),
                "stages": json.loads(json.dumps(self.stages)),
                "activity": json.loads(json.dumps(self.activity[-40:])),
                "cost": json.loads(json.dumps(self.cost)) if self.cost else None,
                "result": json.loads(json.dumps(self.result)) if self.result else None,
                "error": self.error,
                "cleanup": self.cleanup,
            }

    def is_active(self) -> bool:
        with self._lock:
            return self.status not in TERMINAL_STATUSES

    def persist(self) -> str | None:
        """Store the finished run so it can be read again without re-spending."""
        try:
            config.RUNS_DIR.mkdir(parents=True, exist_ok=True)
            path = config.RUNS_DIR / f"{self.run_id}.json"
            path.write_text(json.dumps(self.snapshot(), indent=1))
            return str(path)
        except Exception:
            return None


# ---------------------------------------------------------------------------
# execution
# ---------------------------------------------------------------------------

def make_reporter(record: "RunRecord", meter: Any):
    """Receipt sink: records each model call and refreshes the running totals."""
    def report(info: dict[str, Any]) -> None:
        record.on_call(info)
        try:
            record.set_cost(cost_snapshot(meter))
        except Exception:
            logger.debug("cost snapshot failed after a model call", exc_info=True)
    return report


def _execute(record: RunRecord, case_dir: Path) -> None:
    from backend.app.agent.llm import LLMClient, provider_from_env
    from backend.app.engine import p1_scope, p2_rank, p3_investigate, p4_validate, p5_fuse, p6_report
    from backend.app.engine.budget import BudgetMeter
    from backend.app.engine.contracts import Investigation, InvestigationWindow
    from backend.app.engine.runner import run_investigation
    from backend.app.eval import rcaeval_adapter as adapter
    from backend.app.graph.client import client, graph_for
    from backend.app.settings import SCOPE_BENCHMARK_SETTINGS
    from config.tuning import InvestigationBudget
    from .receipts import reporting_provider

    graph_name = f"incidentgraph:session:ui-{record.run_id}"
    record.session_graph = graph_name
    meter = BudgetMeter(InvestigationBudget(**config.BUDGET_CAPS))
    db = None
    created = False
    try:
        db = client()
        graph = graph_for(graph_name)
        created = True
        loaded = adapter.load_case(graph, case_dir)
        investigation = Investigation(
            incident_id=loaded["incident_id"],
            session_graph=graph_name,
            symptom=f"live run of {record.case_id}",
            anchor_service="service:frontend",
            budget={},
            window=InvestigationWindow(t0=loaded["t0_ms"], t1=loaded["t1_ms"]),
            onset_ts=loaded.get("onset_ts"),
        )

        # Phase functions mirror the documented benchmark protocol, plus the
        # report stage the benchmark stops short of.
        def phase_p0(symptom: str, **kwargs: Any) -> Any:
            return investigation

        def phase_p1(inv: Any) -> Any:
            return p1_scope.scope_investigation(
                inv, full_scope=True, max_nodes=10_000, max_series=10_000,
                settings=SCOPE_BENCHMARK_SETTINGS)

        def phase_p2(inv: Any, scope: Any) -> Any:
            return p2_rank.rank_investigation(inv, scope)

        def phase_p3(inv: Any, scope: Any, ranking: Any, meter: Any = None, llm: Any = None) -> Any:
            return p3_investigate.investigate_investigation(inv, scope, ranking, meter=meter, llm=llm)

        def phase_p5(inv: Any, ranking: Any, invest: Any, validated: Any) -> Any:
            return p5_fuse.fuse_investigation(inv, ranking, invest, validated)

        raw_phases = (phase_p0, phase_p1, phase_p2, phase_p3,
                      p4_validate.validate_investigation, phase_p5, p6_report.report_investigation)

        def tracked(index: int, fn: Any) -> Any:
            """Publish a stage summary the moment its phase returns."""
            def call(*args: Any, **kwargs: Any) -> Any:
                record.set_stage(index, "running")
                value = fn(*args, **kwargs)
                if index == 0:
                    line, detail = _stage_p0(investigation, loaded)
                else:
                    line, detail = _SUMMARISERS[index](value)
                record.set_stage(index, "done", line, detail)
                record.set_cost(_cost(meter))
                return value
            return call

        phases = tuple(tracked(index, fn) for index, fn in enumerate(raw_phases))
        # Build the client here so every model call reports a receipt to the UI.
        client = LLMClient(meter, provider=reporting_provider(
            provider_from_env(), make_reporter(record, meter)))
        state = run_investigation(
            investigation.symptom, meter=meter, llm=client, until_phase=6,
            phase_functions=phases, hard_deadline=False, event_sink=record.on_event)

        if state.status != "completed":
            first_untouched = next((i for i, stage in enumerate(record.stages)
                                    if stage["status"] in ("idle", "running")), 6)
            record.set_stage(first_untouched, "failed")
            record.error = (state.errors[-1].get("code") if state.errors else state.status)

        report = state.report
        fused = state.fused
        if report is not None:
            gate = (getattr(report, "config", {}) or {}).get("gate") or {}
            record.result = {
                "tier": getattr(report, "tier", None),
                "verdict": getattr(report, "verdict", None),
                "confidence": getattr(report, "confidence", None),
                "confidence_rule": getattr(report, "confidence_rule", None),
                "winner": _service(getattr(report, "winner_candidate", None)),
                "reasons": list(gate.get("reasons") or []),
                "next_action": getattr(report, "next_action", None),
                "summary": list(getattr(report, "summary", []) or [])[:4],
                "abstained": getattr(fused, "winner", "missing") is None,
                "markdown": getattr(report, "markdown", None),
                "session_graph": graph_name,
            }
        with record._lock:
            record.status = "completed" if state.status == "completed" else "failed"
            record.finished_at = time.time()
            record.cost = _cost(meter)
        record._publish()
    finally:
        if created and db is not None:
            try:
                db.execute_command("GRAPH.DELETE", graph_name)
                record.cleanup = "deleted"
            except Exception as exc:
                record.cleanup = _error_text(exc)
        elif not created:
            record.cleanup = "not created"
        record._publish()  # the last snapshot carries the cleanup outcome


# ---------------------------------------------------------------------------
# manager
# ---------------------------------------------------------------------------

class RunManager:
    """Single-flight run manager; snapshots are safe to read while running."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._runs: dict[str, RunRecord] = {}
        self._active: str | None = None

    def start(self, case_id: str) -> RunRecord:
        known = {case["case"] for case in config.available_cases()}
        if case_id not in known:
            raise ValueError(f"unknown case {case_id!r}")
        case_dir = config.DATA_DIR / case_id
        record = self._reserve(case_id, kind="benchmark", mode="full")
        threading.Thread(target=self._worker, args=(record, case_dir),
                         name=f"run-{record.run_id}", daemon=True).start()
        return record

    def start_custom(self, spec: dict[str, Any]) -> RunRecord:
        """Run against a caller-supplied FalkorDB graph."""
        label = spec.get("label") or spec["graph"]
        record = self._reserve(label, kind="custom", mode=spec.get("mode", "full"))

        def worker() -> None:
            from . import dashboard

            try:
                dashboard.execute_custom(record, spec)
            except Exception as exc:
                record.fail(_error_text(exc))
            finally:
                record.persist()

        threading.Thread(target=worker, name=f"run-{record.run_id}", daemon=True).start()
        return record

    def _reserve(self, label: str, *, kind: str, mode: str) -> RunRecord:
        with self._lock:
            if self._active is not None and self._runs[self._active].is_active():
                raise RuntimeError("a run is already in progress")
            record = RunRecord(uuid.uuid4().hex[:12], label, kind=kind, mode=mode)
            self._runs[record.run_id] = record
            self._active = record.run_id
        return record

    def _worker(self, record: RunRecord, case_dir: Path) -> None:
        try:
            _execute(record, case_dir)
        except Exception as exc:
            record.fail(_error_text(exc))
        finally:
            record.persist()

    def get(self, run_id: str) -> RunRecord | None:
        return self._runs.get(run_id)

    def active_run_id(self) -> str | None:
        with self._lock:
            if self._active is not None and self._runs[self._active].is_active():
                return self._active
        return None


MANAGER = RunManager()


def load_stored(run_id: str) -> dict[str, Any] | None:
    """Read a persisted run so it can be reviewed without spending again."""
    if not run_id.isalnum():
        return None
    path = config.RUNS_DIR / f"{run_id}.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def stored_runs(limit: int = 20) -> list[dict[str, Any]]:
    if not config.RUNS_DIR.is_dir():
        return []
    summaries: list[dict[str, Any]] = []
    for path in sorted(config.RUNS_DIR.glob("*.json"),
                       key=lambda item: item.stat().st_mtime, reverse=True)[:limit]:
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        result = data.get("result") or {}
        cost = data.get("cost") or {}
        summaries.append({
            "run_id": data.get("run_id", path.stem),
            "case_id": data.get("case_id", ""),
            "status": data.get("status", ""),
            "started_at": data.get("started_at", 0.0),
            "duration_s": data.get("duration_s"),
            "tier": result.get("tier"),
            "winner": result.get("winner"),
            "calls": cost.get("calls"),
            "tokens_total": cost.get("tokens_total"),
        })
    return summaries
