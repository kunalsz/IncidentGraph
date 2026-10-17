"""Tier-1 MCP tool surface (§4.5): typed, read-only, token-budgeted.

Every tool touches the graph and returns JSON-able dicts; tools never raise —
errors come back as observations (paper_02's lesson) the agent can act on.

Read-only boundary (structural, not classifier-based — Aurora's lesson):
  * import-time assertion: no dispatchable name may match a write pattern
    (`create_`, `delete_`, `merge_`, `set_`) — `assert_allowlist_safe()`;
  * every tool issues parameter-bound, template Cypher only (no free-form
    Cypher from the model — the model picks templates, the layer binds params);
  * graph credentials are read-only at deployment.

`check_hypothesis` (P4 validators) and `get_evidence_path` (P4 path finder)
register here when M3 lands — the surface budget check covers the full
registry, so the gate is measured against exactly what ships.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass

from backend.app.agent.llm import estimate_tokens
from backend.app.engine import linking

WRITE_PATTERN = re.compile(r"^(create_|delete_|merge_|set_|update_|upsert_)")

MAX_NEIGHBORHOOD_NODES = 40
MAX_BLAST_NODES = 50


@dataclass(frozen=True)
class ToolSpec:
    name: str
    signature: str
    description: str
    tier: int
    fn: object  # callable(graph, **params) -> dict


# ---------------------------------------------------------------------------
# Tool implementations (all read-only Cypher; params always bound)
# ---------------------------------------------------------------------------

def t_resolve_entity(graph, name: str) -> dict:
    """Entity linking with alias/fuzzy match (§4.5)."""
    return linking.link_entity(name, graph)


def _neighbors(graph, service_id: str, direction: str) -> list[dict]:
    if direction == "out":
        pattern = "(s:Service {id: $sid})-[:DEPENDS_ON]->(n:Service)"
    elif direction == "in":
        pattern = "(n:Service)-[:DEPENDS_ON]->(s:Service {id: $sid})"
    else:
        pattern = "(s:Service {id: $sid})-[:DEPENDS_ON]-(n:Service)"
    rows = graph.query(
        f"MATCH {pattern} WHERE n.id <> $sid "
        f"RETURN n.id, n.name, n.criticality "
        f"ORDER BY n.criticality DESC, n.id",
        params={"sid": service_id},
    ).result_set
    return [{"id": row[0], "name": row[1], "criticality": row[2] or 0.0}
            for row in rows]


def t_get_neighborhood(graph, service_id: str, hops: int = 1,
                       direction: str = "both") -> dict:
    """Annotated adjacency for scope traversal (hops capped at 2)."""
    if direction not in ("both", "out", "in"):
        return {"error": f"invalid direction {direction!r}"}
    hops = max(1, min(int(hops), 2))
    frontier = {service_id}
    seen = {service_id}
    nodes, edges = [], []
    for _ in range(hops):
        next_frontier = set()
        for node_id in sorted(frontier):
            for neighbor in _neighbors(graph, node_id, direction):
                edges.append({"source": node_id, "target": neighbor["id"],
                              "type": "DEPENDS_ON"})
                if neighbor["id"] not in seen:
                    seen.add(neighbor["id"])
                    nodes.append(neighbor)
                    next_frontier.add(neighbor["id"])
                if len(seen) >= MAX_NEIGHBORHOOD_NODES:
                    return {"nodes": nodes, "edges": edges, "truncated": True,
                            "n_nodes": len(seen)}
        frontier = next_frontier
        if not frontier:
            break
    return {"nodes": nodes, "edges": edges, "truncated": False,
            "n_nodes": len(seen)}


def t_get_severity(graph, node_id: str, window_min: int = 30) -> dict:
    """TORAI-style severity vector + top gamma series for a service."""
    rows = graph.query(
        "MATCH (s:Service {id: $sid}) RETURN s.sev_m, s.sev_l, s.sev_tc, "
        "s.self_time_ms, s.self_time_flag, s.deploy_proximity, s.hours_since_change",
        params={"sid": node_id},
    ).result_set
    if not rows:
        return {"error": f"service {node_id} not found"}
    sev_m, sev_l, sev_tc, self_time, self_flag, proximity, hours = rows[0]
    latest = graph.query(
        "MATCH (m:MetricAnomaly)-[:OBSERVED_ON]->(s:Service {id: $sid}) "
        "RETURN max(m.ts)", params={"sid": node_id},
    ).result_set[0][0]
    t0 = (latest or 0) - window_min * 60_000
    series = graph.query(
        "MATCH (m:MetricAnomaly)-[:OBSERVED_ON]->(s:Service {id: $sid}) "
        "WHERE m.ts >= $t0 RETURN m.metric, m.gamma, m.z, m.ts, m.direction "
        "ORDER BY m.gamma DESC LIMIT 5",
        params={"sid": node_id, "t0": t0},
    ).result_set
    return {
        "node_id": node_id,
        "vector": {"metric": sev_m or 0.0, "log": sev_l or 0.0,
                   "trace": sev_tc or 0.0},
        "self_time_ms": self_time or 0.0, "self_time_flag": bool(self_flag),
        "deploy_proximity": proximity or 0.0, "hours_since_change": hours,
        "top_series": [{"metric": row[0], "gamma": row[1], "z": row[2],
                        "ts": row[3], "direction": row[4]} for row in series],
    }


def t_get_recent_changes(graph, service_id: str, window_min: int = 120) -> dict:
    """Deployments/config changes with the exp(-Δt/24h) proximity prior."""
    latest_rows = graph.query(
        "MATCH (c:ConfigChange)-[:CHANGED]->(s:Service {id: $sid}) "
        "RETURN max(c.ts)", params={"sid": service_id},
    ).result_set
    latest = latest_rows[0][0] if latest_rows else None
    anchor = latest or 0
    t0 = anchor - window_min * 60_000
    changes = []
    for row in graph.query(
        "MATCH (c:ConfigChange)-[:CHANGED]->(s:Service {id: $sid}) "
        "WHERE c.ts >= $t0 RETURN c ORDER BY c.ts DESC",
        params={"sid": service_id, "t0": t0},
    ).result_set:
        props = dict(row[0].properties)
        changes.append({"id": props.get("id"), "kind": "ConfigChange",
                        "ts": props.get("ts"), "flag": props.get("flag"),
                        "value": props.get("value")})
    for row in graph.query(
        "MATCH (d:Deployment)-[:DEPLOYED_AT]->(i:Instance) "
        "WHERE i.service_id = $sid AND d.ts >= $t0 "
        "RETURN d ORDER BY d.ts DESC",
        params={"sid": service_id, "t0": t0},
    ).result_set:
        props = dict(row[0].properties)
        changes.append({"id": props.get("id"), "kind": "Deployment",
                        "ts": props.get("ts"), "version": props.get("version"),
                        "status": props.get("status")})
    hours = None
    proximity = 0.0
    if changes and anchor:
        newest = max(change["ts"] for change in changes if change.get("ts"))
        hours = round((anchor - newest) / 3_600_000, 2)
        proximity = math.exp(-hours / 24) if hours is not None else 0.0
    return {"service_id": service_id, "changes": changes,
            "hours_since_change": hours, "proximity_score": proximity}


def t_get_evidence(graph, node_id: str, kinds: list | None = None,
                   window_min: int = 30, budget_tokens: int = 2500) -> dict:
    """Token-budgeted evidence bundle for one node (kinds select modalities)."""
    kinds = kinds or ["alert", "log", "metric", "trace"]
    label_for = {"alert": "Alert", "log": "LogEvent", "metric": "MetricAnomaly",
                 "trace": "Span"}
    latest_rows = graph.query(
        "MATCH (e)-[:OBSERVED_ON]->(s:Service {id: $sid}) "
        "WHERE e:Alert OR e:LogEvent OR e:MetricAnomaly OR e:Span OR e:Trace "
        "RETURN max(e.ts)", params={"sid": node_id},
    ).result_set
    latest = latest_rows[0][0] if latest_rows else 0
    t0 = (latest or 0) - window_min * 60_000
    evidence = []
    for kind in kinds:
        label = label_for.get(kind)
        if label is None:
            continue
        rows = graph.query(
            f"MATCH (e:{label})-[:OBSERVED_ON]->(s:Service {{id: $sid}}) "
            f"WHERE e.ts >= $t0 RETURN e ORDER BY e.ts DESC LIMIT 60",
            params={"sid": node_id, "t0": t0},
        ).result_set
        for row in rows:
            props = dict(row[0].properties)
            entry = {"id": props.get("id"), "kind": kind, "ts": props.get("ts"),
                     "z": props.get("z")}
            for key in ("metric", "value", "direction", "template_id", "message",
                        "level", "is_error", "gamma", "duration_us", "op",
                        "status"):
                if key in props:
                    entry[key] = props[key]
            evidence.append(entry)
    if evidence:
        # trim the flat list until it fits the token budget (never below 1)
        while len(evidence) > 1 and estimate_tokens(
                json.dumps(evidence)) > budget_tokens:
            evidence.pop()
    return {"node_id": node_id, "kinds": kinds, "window_min": window_min,
            "evidence": evidence, "n_evidence": len(evidence)}


def t_blast_radius(graph, service_id: str, depth: int = 3) -> dict:
    """Dependents (who calls me), bounded BFS ≤3 hops."""
    depth = max(1, min(int(depth), 3))
    dependents, seen = [], {service_id}
    frontier = [service_id]
    for level in range(1, depth + 1):
        next_frontier = []
        for node_id in frontier:
            rows = graph.query(
                "MATCH (caller:Service)-[:DEPENDS_ON]->(n:Service {id: $sid}) "
                "RETURN caller.id, caller.criticality ORDER BY caller.criticality DESC",
                params={"sid": node_id},
            ).result_set
            for caller_id, criticality in rows:
                if caller_id in seen:
                    continue
                seen.add(caller_id)
                dependents.append({"id": caller_id, "depth": level,
                                   "criticality": criticality or 0.0})
                next_frontier.append(caller_id)
                if len(seen) >= MAX_BLAST_NODES:
                    return {"dependents": dependents, "truncated": True}
        frontier = next_frontier
        if not frontier:
            break
    return {"dependents": dependents, "truncated": False}


def t_get_runbook_and_owner(graph, service_id: str) -> dict:
    """Retrieved operational closure: runbook steps + owner team (Track 03)."""
    from backend.app.graph.queries import projection
    from backend.app.graph.gateway import admit_payload, PayloadAdmissionError
    runbooks = []
    for row in graph.query(
        "MATCH (r:Runbook)-[:APPLIES_TO]->(s:Service {id: $sid}) "
        "RETURN " + projection('r', ('id', 'title', 'steps', 'approved', 'approval_status', 'approved_by', 'last_reviewed')) + " AS r ORDER BY r.id",
        params={"sid": service_id},
    ).result_set:
        node = row[0]
        props = dict(node.properties if hasattr(node, 'properties') else node)
        if props.pop('__rejected', False):
            raise PayloadAdmissionError('oversized runbook property')
        props = {k: v for k, v in props.items() if v is not None}
        admit_payload(props)
        param_rows = graph.query(
            'MATCH (r:Runbook {id:$rid})-[:REQUIRES_PARAMETER]->(p:RunbookParam) RETURN '
            + projection('p', ('id', 'name', 'kind')) + ' AS p ORDER BY p.id',
            params={'rid': props['id']}).result_set
        params = []
        for (param,) in param_rows:
            p = dict(param.properties if hasattr(param, 'properties') else param)
            if p.pop('__rejected', False):
                raise PayloadAdmissionError('oversized runbook parameter')
            admit_payload(p)
            params.append(p)
        steps = props.get("steps")
        if isinstance(steps, str):
            try:
                steps = json.loads(steps)
            except json.JSONDecodeError:
                steps = []
        runbooks.append({
            "id": props.get("id"), "title": props.get("title"),
            **{key: props[key] for key in ("approved", "approval_status", "approved_by", "last_reviewed") if key in props},
            "steps": steps,
            "params_resolved": [{"id": p.get("id"),
                                 "name": p.get("name"),
                                 "kind": p.get("kind")}
                                for p in params if p is not None],
        })
    teams = []
    for row in graph.query(
        "MATCH (s:Service {id: $sid})-[:OWNED_BY]->(t:Team) "
        "RETURN " + ', '.join(f"CASE WHEN size(toJSON(t.{key})) > $field_chars THEN null ELSE t.{key} END" for key in ('id', 'name', 'oncall_handle')),
        params={"sid": service_id},
    ).result_set:
        admit_payload(row)
        if not isinstance(row[0], str) or not row[0]:
            raise PayloadAdmissionError('owner identity rejected')
        teams.append({"id": row[0], "name": row[1], "oncall_handle": row[2]})
    return {"service_id": service_id, "runbooks": runbooks,
            "team": teams[0] if teams else None}


def t_find_related_incidents(graph, service_id: str, pattern: str | None = None,
                             limit: int = 5) -> dict:
    """Past incidents on this service (Track 02 memory recall)."""
    from backend.app.graph.queries import projection
    from backend.app.graph.gateway import admit_payload, PayloadAdmissionError
    incidents, seen = [], set()
    for row in graph.query(
        "MATCH (i:Incident)-[:AFFECTS]->(s:Service {id: $sid}) "
        "WHERE i.status <> 'investigating' "
        "RETURN " + projection('i', ('id', 'symptom', 'start_ts', 'verdict', 'status')) + " AS i ORDER BY i.start_ts DESC LIMIT $limit",
        params={"sid": service_id, "limit": int(limit)},
    ).result_set:
        props = dict(row[0].properties if hasattr(row[0], 'properties') else row[0])
        if props.pop('__rejected', False):
            raise PayloadAdmissionError('oversized incident property')
        admit_payload(props)
        if props.get("id") in seen:
            continue
        seen.add(props.get("id"))
        incidents.append({"id": props.get("id"), "symptom": props.get("symptom"),
                          "start_ts": props.get("start_ts"),
                          "verdict": props.get("verdict"),
                          "status": props.get("status")})
    return {"service_id": service_id, "pattern": pattern, "incidents": incidents}


def t_check_hypothesis(graph, candidate_id: str,
                       evidence_ids: list | None = None) -> dict:
    """Deterministic validators for one candidate (§4.5): the checks that gate
    every confident claim. Uses the candidate's own graph evidence."""
    from backend.app.graph import validators
    rows = graph.query(
        "MATCH (i:Incident) RETURN max(i.start_ts)").result_set
    onset = rows[0][0] if rows and rows[0][0] is not None else 0
    metrics, timestamps = [], []
    for row in graph.query(
        "MATCH (m:MetricAnomaly)-[:OBSERVED_ON]->(s:Service {id: $sid}) "
        "WHERE m.ts >= $t0 RETURN m ORDER BY m.ts DESC LIMIT 20",
        params={"sid": candidate_id, "t0": onset - 3_600_000},
    ).result_set:
        props = dict(row[0].properties)
        metrics.append(props)
        if props.get("ts"):
            timestamps.append(props["ts"])
    for row in graph.query(
        "MATCH (l:LogEvent)-[:OBSERVED_ON]->(s:Service {id: $sid}) "
        "WHERE l.ts >= $t0 RETURN l.ts ORDER BY l.ts DESC LIMIT 20",
        params={"sid": candidate_id, "t0": onset - 3_600_000},
    ).result_set:
        timestamps.append(row[0])
    error_spans = graph.query(
        "MATCH (sp:Span)-[:OBSERVED_ON]->(s:Service {id: $sid}) "
        "WHERE sp.ts >= $t0 AND sp.status <> 0 RETURN count(sp)",
        params={"sid": candidate_id, "t0": onset - 3_600_000},
    ).result_set[0][0]
    logs_errors = graph.query(
        "MATCH (l:LogEvent)-[:OBSERVED_ON]->(s:Service {id: $sid}) "
        "WHERE l.ts >= $t0 AND (l.is_error = true OR l.level IN ['major','critical']) "
        "RETURN count(l)",
        params={"sid": candidate_id, "t0": onset - 3_600_000},
    ).result_set[0][0]
    checks = [
        validators.onset_alignment(timestamps, onset),
        validators.svar_lite(metrics),
        validators.no_evidence_cap(bool(logs_errors),
                                   any(abs(validators.finite(m.get("z"))) >= 3.0
                                       for m in metrics),
                                   bool(error_spans)),
    ]
    return {"candidate_id": candidate_id, "verdict": validators.aggregate_verdict(checks),
            "checks": checks,
            "confidence_cap": validators.confidence_cap_from_checks(checks)}


def t_get_evidence_path(graph, from_id: str, to_id: str) -> dict:
    """The report artifact: bounded, role-labelled evidence path (§4.5, L10)."""
    from backend.app.engine.contracts import InvestigationWindow
    from backend.app.graph import pathfind
    from config import tuning
    rows = graph.query(
        "MATCH (e)-[:OBSERVED_ON]->(s:Service {id: $sid}) "
        "RETURN max(e.ts)", params={"sid": to_id},
    ).result_set
    anchor_ts = rows[0][0] if rows and rows[0][0] is not None else 0
    window = InvestigationWindow(t0=anchor_ts - 2 * tuning.WINDOW_PAD_MS,
                                 t1=anchor_ts + tuning.WINDOW_PAD_MS)
    return pathfind.find_evidence_path(graph, from_id, to_id, window)


def t_draft_investigation_plan(graph, symptom: str | None = None,
                               anchor: str | None = None) -> dict:
    """Plan-first tool (paper_02): forces a stated plan before diagnostic calls."""
    anchor = anchor or "unresolved"
    return {
        "symptom": symptom, "anchor": anchor,
        "steps": [
            {"step": "confirm_scope", "tool": "get_neighborhood",
             "why": "bound the blast radius before ranking"},
            {"step": "measure_severity", "tool": "get_severity",
             "why": "windowed severity vector per candidate"},
            {"step": "check_changes", "tool": "get_recent_changes",
             "why": "deployment/config proximity prior"},
            {"step": "gather_evidence", "tool": "get_evidence",
             "why": "bounded evidence per candidate"},
            {"step": "validate", "tool": "check_hypothesis",
             "why": "deterministic checks before any confident claim"},
            {"step": "close_operationally", "tool": "get_runbook_and_owner",
             "why": "retrieved remediation, never generated"},
        ],
        "note": "deterministic skeleton; the agent may add bounded steps",
    }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DiagnosticContext:
    investigation: object
    scope: object
    onset_ts: int

    def __post_init__(self):
        if self.investigation.incident_id != self.scope.incident_id:
            raise ValueError('diagnostic context incident mismatch')
        if self.investigation.window != self.scope.window:
            raise ValueError('diagnostic context window mismatch')


def _contextual_tool(name, graph, context, args):
    from backend.app.engine.scope_policy import admitted_records
    from backend.app.agent.bundle import assemble_bundle, bundle_evidence_ids
    import copy
    from backend.app.engine.scope_policy import recalled_scope
    snapshot = recalled_scope(graph, context.scope.window)
    scope = snapshot[0] if snapshot else copy.deepcopy(context.scope)
    if scope.scope_ref != context.scope.scope_ref:
        raise ValueError('diagnostic context differs from admitted snapshot')
    allowed = {s.id for s in scope.services}
    requested = {args[k] for k in ('service_id', 'node_id', 'candidate_id', 'from_id', 'to_id', 'anchor')
                 if args.get(k) is not None}
    if not requested <= allowed:
        return {'error': 'scope expansion required', 'code': 'expansion_required',
                'requested_ids': sorted(requested - allowed)}
    records = admitted_records(scope)
    sid = args.get('service_id') or args.get('node_id') or args.get('candidate_id')
    own = [r for r in records if r['service_id'] == sid]
    if name == 'resolve_entity':
        nodes = [dict(n['properties'], type='Service') for n in scope.nodes if n['type'] == 'Service']
        best = linking.pick_best(args['name'], nodes)
        return {'query': args['name'], 'matched': best is not None, 'nodes': [best] if best else [],
                'anchor_service': best['id'] if best else None}
    if name == 'get_evidence':
        kinds = args.get('kinds') or ['alert', 'metric', 'log', 'trace']
        mapping = {'Alert': 'alert', 'MetricAnomaly': 'metric', 'LogEvent': 'log', 'Span': 'trace', 'Trace': 'trace'}
        evidence = [dict(r['properties'], kind=mapping[r['type']]) for r in own if mapping.get(r['type']) in kinds]
        cap = args.get('budget_tokens', 2500)
        if type(cap) is not int or cap < 2:
            return {'error': 'invalid evidence budget', 'code': 'invalid_arguments'}
        while evidence and estimate_tokens(json.dumps(evidence)) > cap:
            evidence.pop()
        return {'node_id': sid, 'evidence': evidence, 'n_evidence': len(evidence),
                'truncated': len(evidence) < len(own), 'token_accounting': 'character_estimate'}
    if name == 'get_severity':
        stat = next(s for s in scope.services if s.id == sid)
        result = dict(stat.model_dump(), node_id=sid)
        result['vector'] = {'metric': stat.sev_m, 'log': stat.sev_l, 'trace': stat.sev_tc}
        # True: scored; False: records present but unscored (0 means not measured); None: no records.
        from backend.app.engine.scope_policy import scored_modalities
        scored = scored_modalities(own)
        result['scored'] = {m: scored.get(m) for m in ('metric', 'log', 'trace')}
        result['top_series'] = [r['properties'] for r in own if r['type'] == 'MetricAnomaly'][:5]
        return result
    if name == 'get_recent_changes':
        stat = next(s for s in scope.services if s.id == sid)
        return {'service_id': sid, 'changes': [dict(r['properties'], kind=r['type']) for r in own
                if r['type'] in ('ConfigChange', 'Deployment') and r['properties']['ts'] <= context.onset_ts],
                'hours_since_change': stat.hours_since_change, 'proximity_score': stat.deploy_proximity}
    if name == 'get_evidence_path':
        from backend.app.graph.pathfind import find_evidence_path
        return find_evidence_path(graph, args['from_id'], args['to_id'], scope.window,
                                  approved_edges=scope.edges, approved_services=allowed)
    if name in ('get_neighborhood', 'blast_radius'):
        direction = 'in' if name == 'blast_radius' else args.get('direction', 'both')
        if direction not in ('in', 'out', 'both'):
            return {'error': 'invalid direction'}
        depth = min(3 if name == 'blast_radius' else 2, max(1, int(args.get('depth', args.get('hops', 1)))))
        seen, frontier, links = {sid}, {sid}, []
        depths = {sid: 0}
        for level in range(1, depth + 1):
            following = set()
            for edge in scope.edges:
                if edge['type'] != 'DEPENDS_ON':
                    continue
                a, b = edge['source'], edge['target']
                target = b if a in frontier and direction in ('both', 'out') else a if b in frontier and direction in ('both', 'in') else None
                if target:
                    links.append(edge)
                    if target not in seen:
                        following.add(target)
                        depths[target] = level
            seen |= following
            frontier = following
        nodes = [{'id': s, 'depth': depths[s]} for s in sorted(seen - {sid})]
        return {'nodes': nodes, 'dependents': nodes, 'edges': links, 'n_nodes': len(seen), 'truncated': scope.truncated}
    if name == 'check_hypothesis':
        from backend.app.engine.contracts import HypothesisRecord
        from backend.app.engine.p4_validate import assess_hypothesis
        refs = args.get('evidence_ids') or []
        approved = {r['id'] for r in records}
        if not set(refs) <= approved:
            return {'error': 'evidence expansion required', 'code': 'expansion_required'}
        hypothesis = HypothesisRecord(id='tool:check', candidate=sid, text='Service abnormality', evidence=refs, scope_ref=scope.scope_ref)
        result = assess_hypothesis(graph, context.investigation, scope, hypothesis, context.onset_ts)
        output = result.model_dump() if hasattr(result, 'model_dump') else result
        return dict(output, candidate_id=sid)
    # Operational/history reads remain explicit bounded templates, never telemetry.
    if name == 'find_related_incidents':
        args = dict(args, limit=min(20, max(1, int(args.get('limit', 5)))))
    from backend.app.graph.gateway import ReadGateway, admit_payload
    operational_gateway = ReadGateway(graph)
    class BoundedOperationalGraph:
        def query(self, query, params=None):
            from types import SimpleNamespace
            from backend.app.graph.gateway import ReadGateway
            # Existing fixed LIMIT is replaced by gateway's server-bound limit.
            query = re.sub(r' LIMIT \$limit\s*$', '', query)
            rows, overflow = operational_gateway.read(query, params, cap=40)
            return SimpleNamespace(result_set=rows, overflow=overflow)
    result = _LEGACY_FUNCTIONS[name](BoundedOperationalGraph(), **args)
    # Each retrieved record is admitted above; the assembled envelope adds two
    # container levels and is serialized separately for finite-value validation.
    json.dumps(result, allow_nan=False)
    result['limits'] = {'rows_per_query': 40, 'query_timeout_ms': 2000,
                        'queries': len(operational_gateway.diagnostics),
                        'truncated': any(d.get('overflow') for d in operational_gateway.diagnostics)}
    return result


_CONTEXT_NAMES = ('resolve_entity', 'get_neighborhood', 'get_severity', 'get_recent_changes',
                  'get_evidence', 'get_evidence_path', 'blast_radius', 'check_hypothesis',
                  'get_runbook_and_owner', 'find_related_incidents')
_LEGACY_FUNCTIONS = {name: globals()['t_' + name] for name in _CONTEXT_NAMES}


def _bind_context(name):
    def bound(graph, **args):
        context = args.pop('context', None)
        if context is None:
            from backend.app.engine.scope_policy import recalled_scope
            snapshot = recalled_scope(graph)
            if snapshot:
                # Runtime scope is server-owned. No newest graph event can move its clock.
                from types import SimpleNamespace
                scope, onset = snapshot
                inv = SimpleNamespace(incident_id=scope.incident_id, window=scope.window,
                                      session_graph=getattr(graph, 'name', ''), anchor_service=scope.anchor_service)
                context = DiagnosticContext(inv, scope, onset)
        if context is None:
            return {'error': 'diagnostic context required', 'code': 'context_required'}
        return _contextual_tool(name, graph, context, args)
    return bound


for _name in _CONTEXT_NAMES:
    globals()['t_' + _name] = _bind_context(_name)


def _spec(name, signature, description, tier, fn) -> ToolSpec:
    return ToolSpec(name=name, signature=signature, description=description,
                    tier=tier, fn=fn)


TOOLS: dict[str, ToolSpec] = {spec.name: spec for spec in [
    _spec("resolve_entity", "(name: str) -> {nodes: [{id, type, score}]}",
          "Entity linking with alias/fuzzy match", 1, t_resolve_entity),
    _spec("get_neighborhood",
          '(service_id, hops=1, direction="both") -> {nodes, edges}',
          "Annotated adjacency / scope traversal", 1, t_get_neighborhood),
    _spec("get_severity", "(node_id, window_min=30) -> {vector, scored, top_series}",
          "TORAI-style severity vector + gamma-pinned series", 1, t_get_severity),
    _spec("get_recent_changes",
          "(service_id, window_min=120) -> {changes, proximity_score}",
          "Deployment/config change priors", 1, t_get_recent_changes),
    _spec("get_evidence",
          "(node_id, kinds, window_min=30, budget_tokens=2500) -> {evidence}",
          "Token-budgeted evidence bundle", 1, t_get_evidence),
    _spec("get_evidence_path", "(from_id, to_id) -> {path, edges}",
          "Role-labelled evidence path (the report artifact)", 1,
          t_get_evidence_path),
    _spec("blast_radius", "(service_id, depth=3) -> {dependents}",
          "Impact analysis over reverse dependencies", 1, t_blast_radius),
    _spec("get_runbook_and_owner", "(service_id) -> {runbooks, team}",
          "Retrieved remediation steps and owner team", 1, t_get_runbook_and_owner),
    _spec("find_related_incidents", "(service_id, pattern, limit=5) -> {incidents}",
          "Past-incident memory recall", 1, t_find_related_incidents),
    _spec("check_hypothesis",
          "(candidate_id, evidence_ids) -> {verdict, checks, confidence_cap}",
          "Deterministic validators gating confident claims", 1, t_check_hypothesis),
    _spec("draft_investigation_plan",
          "(symptom, anchor) -> {steps}",
          "Plan-first tool: state the plan before diagnostic calls", 0,
          t_draft_investigation_plan),
]}


def assert_allowlist_safe() -> None:
    """Import-time boundary (Aurora's lesson): no write-pattern tool names."""
    offenders = [name for name in TOOLS if WRITE_PATTERN.match(name)]
    if offenders:
        raise AssertionError(f"write-pattern tool names refused: {offenders}")
    if "call_tool" in TOOLS or "search_tools" in TOOLS:
        raise AssertionError("meta tools must not be dispatchable through call_tool")


def surface_tokens(tiers: tuple[int, ...] = (1,)) -> int:
    """Measured schema budget: name + signature + description per tool."""
    return sum(estimate_tokens(f"{spec.name}{spec.signature}{spec.description}")
               for spec in TOOLS.values() if spec.tier in tiers)


def search_tools(query: str, filters: dict | None = None) -> list[dict]:
    """Aurora's search-rank mechanics, simplified (tokenize, match, no filter
    drop needed at this surface size)."""
    tokens = [t for t in re.split(r"[^a-z0-9]+", (query or "").lower()) if len(t) >= 3]
    scored = []
    for spec in TOOLS.values():
        haystack_name = spec.name.lower()
        haystack_desc = spec.description.lower()
        score = sum(2 for t in tokens if t in haystack_name)
        score += sum(1 for t in tokens if t in haystack_desc)
        if filters and "tier" in filters and spec.tier != filters["tier"]:
            continue
        if score > 0 or not tokens:
            scored.append((score, spec))
    scored.sort(key=lambda item: (-item[0], item[1].name))
    return [{"name": spec.name, "signature": spec.signature,
             "description": spec.description, "tier": spec.tier,
             "callable_now": True} for _, spec in scored]


def call_tool(name: str, graph, args: dict | None = None, *, context: DiagnosticContext | None = None) -> dict:
    """Template-bound dispatch; unknown/invalid calls return observations."""
    spec = TOOLS.get(name)
    if spec is None:
        return {"error": f"unknown tool {name!r}",
                "known": sorted(TOOLS)}
    if WRITE_PATTERN.match(name):
        return {"error": f"tool {name!r} refused by the read-only boundary"}
    try:
        params = dict(args or {})
        if context is not None and name in _CONTEXT_NAMES:
            params["context"] = context
        result = spec.fn(graph, **params)
    except TypeError as exc:
        return {"error": f"invalid arguments for {name}: {exc}"}
    except Exception as exc:  # tools never raise to the caller (paper_02)
        return {"error": f"{name} failed: {exc}"}
    return result if isinstance(result, dict) else {"result": result}


assert_allowlist_safe()
