"""Custom-target support: connect to a caller-supplied FalkorDB and run there.

Isolation contract
------------------
* the caller's graph is only read; the run happens on a ``GRAPH.COPY`` session
  graph that is deleted when the run ends;
* the engine's client singleton is rebound to the caller's instance for the
  duration of the run and restored afterwards. Runs are single-flight, which is
  what makes that rebinding safe;
* only localhost / private-network hosts are accepted, so this service cannot be
  used to probe arbitrary internet hosts.
"""
from __future__ import annotations

import ipaddress
import os
import re
import socket
import time
import uuid
from typing import Any

from . import config, live

GRAPH_NAME = re.compile(r"^[A-Za-z0-9_:.\-]{1,128}$")

# Drawing bounds: a graph larger than this is summarised, never silently sliced.
MAX_DRAWN_NODES = 60
MAX_DRAWN_EDGES = 200
MAX_DERIVED_DEPTH = 32


def _derive_levels(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> dict[str, int]:
    """Breadth-first depth from the entry services, cycle-safe and bounded.

    Matches the engine's own convention (`level` = shortest dependency depth
    over DEPENDS_ON). A graph that is all cycles has no entry service, so the
    first unvisited node seeds a level-0 column and the walk continues — no
    relaxation loop, so depth can never inflate.
    """
    from collections import deque

    identifiers = [node["id"] for node in nodes]
    outgoing: dict[str, list[str]] = {node_id: [] for node_id in identifiers}
    incoming: dict[str, int] = {node_id: 0 for node_id in identifiers}
    for edge in edges:
        if edge["source"] in outgoing and edge["target"] in incoming:
            outgoing[edge["source"]].append(edge["target"])
            incoming[edge["target"]] += 1

    levels: dict[str, int] = {}
    roots = sorted(node_id for node_id in identifiers if incoming[node_id] == 0)
    for root in roots:
        levels[root] = 0
    queue: deque[str] = deque(roots)
    while len(levels) < len(identifiers):
        while queue:
            current = queue.popleft()
            for target in outgoing[current]:
                if target not in levels:
                    levels[target] = min(levels[current] + 1, MAX_DERIVED_DEPTH)
                    queue.append(target)
        remaining = [node_id for node_id in sorted(identifiers) if node_id not in levels]
        if not remaining:
            break
        levels[remaining[0]] = 0
        queue.append(remaining[0])
    return {node_id: levels.get(node_id, 0) for node_id in identifiers}


def graph_topology(db: Any, graph_name: str, anchor: str | None = None) -> dict[str, Any]:
    """Read the caller's own service graph, drawn-ready. Never writes."""
    graph = db.select_graph(graph_name)
    node_rows = graph.query(
        "MATCH (s:Service) RETURN s.id, s.name, s.level, s.criticality "
        f"LIMIT {MAX_DRAWN_NODES + 1}").result_set
    truncated = len(node_rows) > MAX_DRAWN_NODES
    nodes: list[dict[str, Any]] = []
    for row in node_rows[:MAX_DRAWN_NODES]:
        identifier = row[0]
        if not identifier:
            continue
        nodes.append({
            "id": str(identifier),
            "name": str(row[1] or str(identifier).removeprefix("service:")),
            "level": int(row[2]) if isinstance(row[2], (int, float)) else None,
            "criticality": float(row[3]) if isinstance(row[3], (int, float)) else 0.0,
        })
    edge_rows = graph.query(
        "MATCH (a:Service)-[r:DEPENDS_ON]->(b:Service) RETURN a.id, b.id "
        f"LIMIT {MAX_DRAWN_EDGES + 1}").result_set
    edges = [{"source": str(row[0]), "target": str(row[1])}
             for row in edge_rows[:MAX_DRAWN_EDGES] if row[0] and row[1]]

    # The graph's own levels are authoritative where present; only what it does
    # not state is derived, so a hand-edited graph keeps its own answers.
    missing = [node for node in nodes if node["level"] is None]
    derived = bool(missing)
    if derived:
        levels = _derive_levels(nodes, edges)
        for node in missing:
            node["level"] = levels.get(node["id"], 0)

    notes = []
    if derived:
        notes.append(f"{len(missing)} of {len(nodes)} services carry no level property, "
                     "so their depth is derived from the edges")
    if truncated or len(edge_rows) > MAX_DRAWN_EDGES:
        notes.append(f"the drawing is bounded to {MAX_DRAWN_NODES} services and "
                     f"{MAX_DRAWN_EDGES} dependencies")
    return {"nodes": nodes, "edges": edges, "derived_levels": derived,
            "note": "; ".join(notes) or None}

# Labels the engine reads, and the properties each one must carry.
SCHEMA = (
    ("Service", ("id",), "service identity and anchor for the investigation"),
    ("MetricAnomaly", ("ts",), "metric excursions with a millisecond timestamp"),
    ("LogEvent", ("ts",), "log lines with a millisecond timestamp"),
    ("Span", ("ts",), "trace spans with a millisecond timestamp"),
)

HOST_HELP = (
    "FalkorDB must be reachable from this service on a loopback or private "
    "address (for example 127.0.0.1:6379, or host.docker.internal for a "
    "container)."
)


def allowed_host(host: str) -> bool:
    """Only loopback and private addresses, plus an explicit env allowlist."""
    extra = {item.strip() for item in os.environ.get("INCIDENTGRAPH_UI_ALLOWED_HOSTS", "").split(",")
             if item.strip()}
    if host in extra or host == "localhost":
        return True
    try:
        address = ipaddress.ip_address(socket.gethostbyname(host))
    except Exception:
        return False
    return address.is_loopback or address.is_private


def _connect(spec: dict[str, Any]):
    from falkordb import FalkorDB

    kwargs: dict[str, Any] = {"host": spec["host"], "port": int(spec["port"])}
    if spec.get("username"):
        kwargs["username"] = spec["username"]
    if spec.get("password"):
        kwargs["password"] = spec["password"]
    return FalkorDB(**kwargs)


def _check(name: str, status: str, detail: str) -> dict[str, str]:
    return {"name": name, "status": status, "detail": detail}


def _count(graph: Any, query: str) -> int:
    result = graph.query(query)
    rows = getattr(result, "result_set", None) or []
    return int(rows[0][0]) if rows and rows[0] else 0


def _sample_properties(graph: Any, label: str) -> dict[str, Any]:
    result = graph.query(f"MATCH (n:{label}) RETURN n LIMIT 1")
    rows = getattr(result, "result_set", None) or []
    if not rows or not rows[0]:
        return {}
    node = rows[0][0]
    return dict(getattr(node, "properties", {}) or {})


def probe(spec: dict[str, Any]) -> dict[str, Any]:
    """Read-only verification of a caller's database and graph."""
    host = (spec.get("host") or "").strip()
    if not host:
        return {"state": "error", "verdict": "blocked", "checks": [],
                "detail": "a database host is required", "graphs": []}
    if not allowed_host(host):
        return {"state": "error", "verdict": "blocked", "checks": [], "graphs": [],
                "detail": f"{host} is not a loopback or private address. {HOST_HELP}"}
    graph_name = (spec.get("graph") or "").strip()
    if not GRAPH_NAME.match(graph_name):
        return {"state": "error", "verdict": "blocked", "checks": [], "graphs": [],
                "detail": "a graph name is required (letters, digits, and _ : . - only)"}

    checks: list[dict[str, str]] = []
    try:
        db = _connect(spec)
        db.execute_command("PING")
        graphs = [name for name in (db.execute_command("GRAPH.LIST") or [])]
    except Exception as exc:
        return {"state": "error", "verdict": "unreachable", "checks": [], "graphs": [],
                "detail": f"{type(exc).__name__}: cannot reach {host}:{spec.get('port')}"}

    checks.append(_check("connection", "pass", f"{host}:{spec.get('port')} answered PING"))
    if graph_name not in graphs:
        checks.append(_check("graph", "fail", f"{graph_name!r} is not one of {graphs[:8] or 'no graphs'}"))
        return {"state": "ok", "verdict": "incomplete", "checks": checks, "graphs": graphs,
                "detail": f"graph {graph_name!r} not found"}
    checks.append(_check("graph", "pass", f"{graph_name!r} exists"))

    graph = db.select_graph(graph_name)
    counts: dict[str, int] = {}
    for label, required, note in SCHEMA:
        try:
            total = _count(graph, f"MATCH (n:{label}) RETURN count(n)")
        except Exception as exc:
            checks.append(_check(label, "fail", f"query failed: {type(exc).__name__}"))
            continue
        counts[label] = total
        if total == 0:
            checks.append(_check(label, "warn", f"no :{label} nodes found; {note}"))
            continue
        properties = _sample_properties(graph, label)
        missing = [key for key in required if key not in properties]
        if missing:
            checks.append(_check(label, "fail",
                                 f"{total} nodes, but the sample is missing {', '.join(missing)}"))
        else:
            checks.append(_check(label, "pass", f"{total} nodes; sample has {', '.join(required)}"))

    anchor = (spec.get("anchor_service") or "").strip()
    if anchor:
        try:
            rows = graph.query("MATCH (s:Service {id: $id}) RETURN s.id", {"id": anchor}).result_set
            checks.append(_check("anchor", "pass" if rows else "fail",
                                 f"{anchor} found" if rows else f"no :Service with id {anchor!r}"))
        except Exception as exc:
            checks.append(_check("anchor", "fail", f"query failed: {type(exc).__name__}"))

    try:
        linked = _count(graph, "MATCH (e)-[:OBSERVED_ON]->(t) RETURN count(e)")
        checks.append(_check("observation links", "pass" if linked else "fail",
                             f"{linked} records attached with OBSERVED_ON"
                             if linked else "no OBSERVED_ON edges; telemetry cannot be scoped"))
    except Exception as exc:
        checks.append(_check("observation links", "fail", f"query failed: {type(exc).__name__}"))

    try:
        edges = _count(graph, "MATCH (:Service)-[r:DEPENDS_ON]->(:Service) RETURN count(r)")
        checks.append(_check("dependency edges", "pass" if edges else "warn",
                             f"{edges} DEPENDS_ON edges"
                             if edges else "no DEPENDS_ON edges; scope will contain the anchor only"))
    except Exception as exc:
        checks.append(_check("dependency edges", "warn", f"query failed: {type(exc).__name__}"))

    # A usable default window saves the caller from guessing timestamps.
    time_range: dict[str, int] | None = None
    try:
        rows = graph.query(
            "MATCH (e) WHERE e:MetricAnomaly OR e:LogEvent OR e:Span "
            "RETURN min(e.ts), max(e.ts)").result_set
        if rows and rows[0] and rows[0][0] is not None and rows[0][1] is not None:
            time_range = {"t0": int(rows[0][0]), "t1": int(rows[0][1])}
            span_hours = (time_range["t1"] - time_range["t0"]) / 3_600_000
            checks.append(_check("time range", "pass",
                                 f"telemetry spans {span_hours:.1f} hours"))
    except Exception as exc:
        checks.append(_check("time range", "warn", f"query failed: {type(exc).__name__}"))

    failures = [item for item in checks if item["status"] == "fail"]
    telemetry = any(counts.get(label, 0) > 0 for label in ("MetricAnomaly", "LogEvent", "Span"))
    verdict = "ready" if not failures and counts.get("Service", 0) > 0 and telemetry else "incomplete"
    drawing = None
    if counts.get("Service", 0) > 0:
        try:
            drawing = graph_topology(db, graph_name, anchor or None)
        except Exception as exc:
            checks.append(_check("topology view", "warn", f"query failed: {type(exc).__name__}"))
    return {"state": "ok", "verdict": verdict, "checks": checks, "graphs": graphs,
            "counts": counts, "anchor": anchor or None, "host": host,
            "port": int(spec.get("port") or 0), "graph": graph_name, "time_range": time_range,
            "topology": drawing,
            "detail": None if verdict == "ready" else "the graph does not yet satisfy the required schema"}


def execute_custom(record: live.RunRecord, spec: dict[str, Any]) -> None:
    """Run the pipeline against the caller's graph, on a disposable copy."""
    from backend.app.agent.llm import LLMClient, provider_from_env
    from backend.app.engine import p1_scope, p2_rank, p3_investigate, p4_validate, p5_fuse, p6_report
    from backend.app.engine.budget import BudgetMeter
    from backend.app.engine.contracts import Investigation, InvestigationWindow
    from backend.app.engine.runner import run_investigation
    from backend.app.graph import client as client_module
    from config.tuning import InvestigationBudget

    from .receipts import reporting_provider

    preview = spec.get("mode") == "preview"
    session = f"incidentgraph:session:ui-{record.run_id}"
    meter = BudgetMeter(InvestigationBudget(**config.BUDGET_CAPS))
    custom = _connect(spec)
    previous = client_module._client
    copied = False
    record.session_graph = session
    try:
        # The engine talks to the caller's instance for the length of this run.
        client_module._client = custom
        custom.execute_command("GRAPH.COPY", spec["graph"], session)
        copied = True

        investigation = Investigation(
            incident_id=spec.get("incident_id") or f"incident:custom:{uuid.uuid4().hex[:8]}",
            session_graph=session,
            symptom=spec.get("symptom") or "custom investigation",
            anchor_service=spec["anchor_service"],
            budget={},
            window=InvestigationWindow(t0=int(spec["window_t0"]), t1=int(spec["window_t1"])),
            onset_ts=int(spec["onset_ts"]) if spec.get("onset_ts") else None,
        )

        def phase_p0(symptom: str, **kwargs: Any) -> Any:
            return investigation

        def phase_p1(inv: Any) -> Any:
            # Product path: expand from the anchor over DEPENDS_ON.
            return p1_scope.scope_investigation(inv)

        def phase_p2(inv: Any, scope: Any) -> Any:
            return p2_rank.rank_investigation(inv, scope)

        def phase_p3(inv: Any, scope: Any, ranking: Any, meter: Any = None, llm: Any = None) -> Any:
            return p3_investigate.investigate_investigation(inv, scope, ranking, meter=meter, llm=llm)

        def phase_p5(inv: Any, ranking: Any, invest: Any, validated: Any) -> Any:
            return p5_fuse.fuse_investigation(inv, ranking, invest, validated)

        raw = (phase_p0, phase_p1, phase_p2, phase_p3,
               p4_validate.validate_investigation, phase_p5, p6_report.report_investigation)
        if preview:
            # Deterministic ranking only: no provider call is reachable here.
            raw = raw[:3]
        phases = tuple(live.tracked_phase(record, index, fn, meter, investigation=investigation)
                       for index, fn in enumerate(raw))
        # Receipts come from the client we build here, so the UI can show calls
        # as they happen (preview mode never reaches a call).
        client = LLMClient(meter, provider=reporting_provider(
            provider_from_env(), live.make_reporter(record, meter)))

        state = run_investigation(
            investigation.symptom, meter=meter, llm=client, until_phase=len(phases) - 1,
            phase_functions=phases, hard_deadline=False, event_sink=record.on_event)

        if state.status != "completed":
            first_untouched = next((i for i, stage in enumerate(record.stages)
                                    if stage["status"] in ("idle", "running")), len(phases) - 1)
            record.set_stage(first_untouched, "failed")
            record.error = state.errors[-1].get("code") if state.errors else state.status

        if preview:
            ranking = state.ranking
            ordered = sorted(getattr(ranking, "candidates", []) or [],
                             key=lambda item: getattr(item, "rank", 99)) if ranking else []
            record.result = {
                "mode": "preview",
                "tier": None,
                "verdict": "ranking only",
                "reasons": ["deterministic ranking; no provider calls were made"],
                "candidates": [
                    {"id": str(getattr(item, "id", "")).replace("service:", ""),
                     "rank": getattr(item, "rank", None),
                     "score": round(float(getattr(item, "score", 0.0)), 4)}
                    for item in ordered[:8]
                ],
                "truncated": bool(getattr(ranking, "truncated", False)) if ranking else False,
                "next_action": "Run the full investigation to validate a candidate.",
                "abstained": False, "confidence": None, "confidence_rule": None,
                "winner": None, "summary": [], "markdown": None, "session_graph": session,
            }
        else:
            report = state.report
            fused = state.fused
            if report is not None:
                gate = (getattr(report, "config", {}) or {}).get("gate") or {}
                record.result = {
                    "mode": "full",
                    "tier": getattr(report, "tier", None),
                    "verdict": getattr(report, "verdict", None),
                    "confidence": getattr(report, "confidence", None),
                    "confidence_rule": getattr(report, "confidence_rule", None),
                    "winner": live._service(getattr(report, "winner_candidate", None)),
                    "reasons": list(gate.get("reasons") or []),
                    "next_action": getattr(report, "next_action", None),
                    "summary": list(getattr(report, "summary", []) or [])[:4],
                    "abstained": getattr(fused, "winner", "missing") is None,
                    "markdown": getattr(report, "markdown", None),
                    "session_graph": session,
                }
        with record._lock:
            record.status = "completed" if state.status == "completed" else "failed"
            record.finished_at = time.time()
            record.cost = live.cost_snapshot(meter)
        record._publish()
    finally:
        if copied:
            try:
                custom.execute_command("GRAPH.DELETE", session)
                record.cleanup = "deleted"
            except Exception as exc:
                record.cleanup = live._error_text(exc)
        else:
            record.cleanup = "not created"
        client_module._client = previous
        record._publish()  # the last snapshot carries the cleanup outcome
