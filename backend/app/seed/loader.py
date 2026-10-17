"""Validated correction-aware MERGE-on-id ingestion.

Topology/teams/runbooks/history go to `incidentgraph:demo`; fixture
`graph_writes` go into the demo graph too (they are the showcase data judges
poke at), while investigations write only to `incidentgraph:session:*`.
"""

import json
import math
import re
import hashlib
from pathlib import Path
from datetime import datetime

from backend.app.graph import schema
from backend.app.graph.client import graph_for, DEMO_GRAPH
from backend.app.ingest import normalize
from backend.app.ingest.publication import PUBLICATION_MARKER_ID, publish_import

SEED_DIR = Path(__file__).resolve().parent
FIXTURES_DIR = SEED_DIR / "fixtures"


PROPERTY_KEYS = set("approved approval_status approved_by node id name aliases service_id criticality level level_convention ts start_ts end_ts kind severity value threshold unit template_id message is_error metric column z direction baseline_median baseline_iqr baseline_mean baseline_std gamma delta duration_us start_us trace_id span_id parent_id trace_complete status op flag actor detail version commit title steps last_reviewed oncall_handle symptom verdict action outcome note risk undo validate_cmd source_hash normalizer_version schema_version src dst invariant calls_per_min p99_ms weight mode".split())


def _validate_props(label, props):
    if label not in schema.NODE_LABELS:
        raise ValueError(f'unknown node label {label!r}')
    if not isinstance(props, dict) or not isinstance(props.get('id'), str) or not props['id'] or len(props['id']) > 512:
        raise ValueError('record requires bounded nonempty string id')
    if props['id'] == PUBLICATION_MARKER_ID:
        raise ValueError('reserved publication marker id')
    for key, value in props.items():
        if key not in PROPERTY_KEYS or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', key):
            raise ValueError(f'unsupported property {key!r}')
        if key in {'z', 'gamma', 'threshold', 'duration_us', 'start_us',
                   'baseline_mean', 'baseline_std', 'baseline_median', 'baseline_iqr',
                   'criticality', 'calls_per_min', 'p99_ms', 'weight'} and value is not None:
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f'invalid measurement {key}')
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f'nonfinite property {key}')
        if value is not None and type(value) not in (str, int, float, bool):
            raise ValueError(f'property {key} must be scalar or serialized JSON')
        if type(value) is int and not -(2**63) <= value < 2**63:
            raise ValueError(f'out-of-range integer {key}')
        if isinstance(value, str) and len(value) > 65536:
            raise ValueError(f'oversized property {key}')


def _props_str(props):
    return ', '.join(f'n.{k} = $node_{k}' for k in props)


def merge_node(graph, label, props):
    """Supplied fields replace; omitted fields remain; null removes property."""
    props = dict(props)
    for key in ('ts', 'start_ts', 'end_ts', 'last_reviewed'):
        if props.get(key) is not None:
            props[key] = normalize.to_epoch_ms(props[key])
    _validate_props(label, props)
    fields = {k: v for k, v in props.items() if k != 'id'}
    suffix = ' SET ' + _props_str(fields) if fields else ''
    graph.query(f'MERGE (n:{label} {{id: $node_id}})' + suffix,
                params={f'node_{k}': v for k, v in props.items()})


_ENDPOINTS = {
    schema.DEPENDS_ON: ({schema.SERVICE}, {schema.SERVICE}),
    schema.OWNED_BY: ({schema.SERVICE}, {schema.TEAM}),
    schema.OBSERVED_ON: ({schema.ALERT, schema.LOG_EVENT, schema.METRIC_ANOMALY, schema.SPAN, schema.TRACE}, {schema.SERVICE, schema.INSTANCE}),
    schema.CHANGED: ({schema.CONFIG_CHANGE}, {schema.SERVICE, schema.INSTANCE}),
    schema.DEPLOYED_AT: ({schema.DEPLOYMENT}, {schema.INSTANCE, schema.SERVICE}),
    schema.AFFECTS: ({schema.INCIDENT}, {schema.SERVICE}),
    schema.APPLIES_TO: ({schema.RUNBOOK}, {schema.SERVICE}),
    schema.REQUIRES_PARAMETER: ({schema.RUNBOOK}, {schema.RUNBOOK_PARAM}),
    schema.MITIGATED_BY: ({schema.INCIDENT}, {schema.DECISION}),
}


def _endpoint_types(graph, node_id):
    rows = graph.query('MATCH (n {id: $endpoint_id}) RETURN labels(n)',
                       params={'endpoint_id': node_id}).result_set
    if len(rows) != 1:
        raise ValueError(f'missing or duplicate endpoint {node_id!r}')
    return set(rows[0][0])


def merge_edge(graph, src_id, rel, dst_id, props=None):
    if rel not in _ENDPOINTS:
        raise ValueError(f'unsupported import relationship {rel!r}')
    props = dict(props or {})
    _validate_props(schema.SERVICE, {'id': 'edge', **props})
    src_types, dst_types = _endpoint_types(graph, src_id), _endpoint_types(graph, dst_id)
    expected_src, expected_dst = _ENDPOINTS[rel]
    if not src_types.intersection(expected_src) or not dst_types.intersection(expected_dst):
        raise ValueError(f'invalid endpoint types for {rel}')
    if rel in (schema.OBSERVED_ON, schema.CHANGED, schema.DEPLOYED_AT, schema.OWNED_BY):
        old = graph.query(f'MATCH (a {{id: $endpoint_src}})-[:{rel}]->(b) RETURN b.id',
                          params={'endpoint_src': src_id}).result_set
        if any(row[0] != dst_id for row in old):
            raise ValueError('ownership reattachment requires an explicit migration')
    suffix = ' SET ' + ', '.join(f'e.{k} = $edge_{k}' for k in props) if props else ''
    rows = graph.query(f'MATCH (a {{id: $endpoint_src}}), (b {{id: $endpoint_dst}}) '
                       f'MERGE (a)-[e:{rel}]->(b){suffix} RETURN count(e)',
                       params={'endpoint_src': src_id, 'endpoint_dst': dst_id,
                               **{f'edge_{k}': v for k, v in props.items()}}).result_set
    if not rows or rows[0][0] != 1:
        raise ValueError('relationship write did not match exactly one endpoint pair')


def _load_topology(graph, topology: dict) -> dict:
    """Load services, deps, teams, instances, runbooks, deployments, history."""
    counts = {"services": 0, "depends_on": 0, "teams": 0, "owned_by": 0,
              "instances": 0, "runbooks": 0, "runbook_params": 0,
              "applies_to": 0, "requires_parameter": 0,
              "deployments": 0, "deployed_at": 0,
              "incidents": 0, "affects": 0, "mitigated_by": 0}

    for svc in topology["services"]:
        props = dict(svc)
        if 'aliases' in svc and svc['aliases'] is not None:
            props['aliases'] = json.dumps(svc['aliases'])
        merge_node(graph, schema.SERVICE, props)
        counts["services"] += 1

    for dep in topology["depends_on"]:
        merge_edge(graph, dep["from"], schema.DEPENDS_ON, dep["to"],
                   {k: v for k, v in dep.items() if k not in ("from", "to")})
        counts["depends_on"] += 1

    for team in topology["teams"]:
        merge_node(graph, schema.TEAM, team)
        counts["teams"] += 1

    for svc_id, team_id in topology["owned_by"].items():
        merge_edge(graph, svc_id, schema.OWNED_BY, team_id)
        counts["owned_by"] += 1

    for inst in topology["instances"]:
        merge_node(graph, schema.INSTANCE, inst)
        counts["instances"] += 1

    for rb in topology["runbooks"]:
        params = [dict(p) for p in rb.get("params", [])]
        steps = json.dumps(rb["steps"])
        merge_node(graph, schema.RUNBOOK,
                   {"id": rb["id"], "title": rb["title"], "steps": steps,
                    "last_reviewed": rb.get("last_reviewed"),
                    **{key: rb[key] for key in ("approved", "approval_status", "approved_by") if key in rb}})
        counts["runbooks"] += 1
        for p in params:
            merge_node(graph, schema.RUNBOOK_PARAM, p)
            counts["runbook_params"] += 1
            merge_edge(graph, rb["id"], schema.REQUIRES_PARAMETER, p["id"])
            counts["requires_parameter"] += 1
        for target in rb["applies_to"]:
            merge_edge(graph, rb["id"], schema.APPLIES_TO, target)
            counts["applies_to"] += 1

    for dep in topology["deployments"]:
        props = dict(dep)
        instance = props.pop("instance", None)
        merge_node(graph, schema.DEPLOYMENT, props)
        counts["deployments"] += 1
        if instance:
            merge_edge(graph, dep["id"], schema.DEPLOYED_AT, instance)
            counts["deployed_at"] += 1

    for inc in topology.get("past_incidents", []):
        d = inc.get("decision")
        props = {k: v for k, v in inc.items()
                 if k not in ("affects", "decision")}
        merge_node(graph, schema.INCIDENT, props)
        counts["incidents"] += 1
        merge_edge(graph, inc["id"], schema.AFFECTS, inc["affects"])
        counts["affects"] += 1
        if d:
            merge_node(graph, schema.DECISION, d)
            merge_edge(graph, inc["id"], schema.MITIGATED_BY, d["id"])
            counts["mitigated_by"] += 1

    return counts


def _prepared_node(label, raw):
    """Normalize only supplied correction fields; metadata hashes raw input."""
    props = dict(raw)
    original = dict(props)
    for key in ('ts', 'start_ts', 'end_ts', 'last_reviewed'):
        if props.get(key) is not None:
            props[key] = normalize.to_epoch_ms(props[key])
    _validate_props(label, props)
    fn = {schema.ALERT: normalize.normalize_alert, schema.LOG_EVENT: normalize.normalize_log,
          schema.METRIC_ANOMALY: normalize.normalize_metric_anomaly,
          schema.SPAN: normalize.normalize_span}.get(label)
    if fn:
        # Missing fields on corrections are not defaults to persist. Required
        # normalizer inputs are placeholders, filtered out below.
        inputs = {'ts': 0, 'metric': '', **props}
        if inputs['ts'] is None:
            inputs['ts'] = 0
        normalized = fn(inputs)
        derived = set()
        if label == schema.LOG_EVENT:
            if 'level' in props:
                derived.add('is_error')
            if 'message' in props and 'template_id' not in props:
                derived.add('template_id')
        if label == schema.SPAN and 'status' in props:
            derived.add('is_error')
        for key, value in normalized.items():
            if key in props and props[key] is not None or key in derived:
                props[key] = value
        if props.get('level', False) is None or props.get('status', False) is None:
            props['is_error'] = None
    props.update(source_hash=hashlib.sha256(json.dumps(original, sort_keys=True, allow_nan=False,
                     default=lambda value: value.isoformat() if isinstance(value, datetime) else None).encode()).hexdigest(),
                 normalizer_version='3', schema_version='2')
    _validate_props(label, props)
    return props


def _check_identity(graph, label, props, *, references=True):
    rows = graph.query('MATCH (n {id: $endpoint_id}) RETURN labels(n), n.service_id',
                       params={'endpoint_id': props['id']}).result_set
    if len(rows) > 1 or rows and set(rows[0][0]) != {label}:
        raise ValueError('duplicate id or conflicting node label')
    if rows and 'service_id' in props and rows[0][1] is not None and props['service_id'] != rows[0][1]:
        raise ValueError('ownership reattachment requires explicit migration')
    if references and props.get('service_id') is not None:
        if _endpoint_types(graph, props['service_id']) != {schema.SERVICE}:
            raise ValueError('service_id must reference a Service')


def _verify_records(graph, records, edges):
    """Read back supplied properties and exact endpoint multiplicity before publish."""
    for label, props in records:
        rows = graph.query('MATCH (n {id: $endpoint_id}) RETURN n',
                           params={'endpoint_id': props['id']}).result_set
        if len(rows) != 1 or set(rows[0][0].labels) != {label}:
            raise ValueError('staged node verification failed')
        stored = rows[0][0].properties
        if any(stored.get(key) != value for key, value in props.items()):
            raise ValueError('staged property verification failed')
    for src, rel, dst, props in edges:
        rows = graph.query(f'MATCH (a {{id: $endpoint_src}})-[e:{rel}]->(b {{id: $endpoint_dst}}) RETURN properties(e)',
                           params={'endpoint_src': src, 'endpoint_dst': dst}).result_set
        if len(rows) != 1 or any(rows[0][0].get(key) != value for key, value in props.items()):
            raise ValueError('staged relationship verification failed')


def load_fixture(graph, fixture: dict) -> int:
    """Load a fixture's graph_writes (telemetry/change evidence) into `graph`.

    Normalization/preflight precede private staging. Supplied fields replace,
    omitted fields remain, null removes. Verified staging is atomically renamed
    over the live key, guarded against concurrent mutations. Returns node count.
    """
    if not isinstance(fixture, dict):
        raise ValueError('import envelope must be an object')
    writes = fixture.get('graph_writes')
    if not isinstance(writes, list) or len(writes) > 20000:
        raise ValueError('invalid or oversized import envelope')
    version = fixture.get('fixture_version', 1)
    if type(version) not in (int, str) or version not in (1, 2, '1', '2', 'v1', 'v2'):
        raise ValueError('unsupported fixture version')
    prepared, ids = [], set()
    for write in writes:
        if (not isinstance(write, dict) or not isinstance(write.get('props'), dict)
                or not isinstance(write.get('node'), str)):
            raise ValueError('invalid import record')
        label, props, edge = write['node'], dict(write['props']), write.get('edge')
        if not isinstance(props.get('id'), str):
            raise ValueError('record requires string id')
        if props.get('id') in ids:
            raise ValueError('duplicate import id')
        ids.add(props.get('id'))
        props = _prepared_node(label, props)
        if edge is not None and (not isinstance(edge, (list, tuple)) or len(edge) != 2
                or not isinstance(edge[0], str) or edge[0] not in _ENDPOINTS
                or not isinstance(edge[1], str) or not edge[1] or len(edge[1]) > 512):
            raise ValueError('invalid import edge')
        prepared.append((label, props, edge))
    records = [(label, props) for label, props, _ in prepared]
    edges = [(props['id'], edge[0], edge[1], {}) for _, props, edge in prepared if edge]
    def apply(stage):
        # Reject conflicting existing IDs before adding any nodes, then validate
        # service references with all forward-imported endpoints present.
        for label, props in records:
            _check_identity(stage, label, props, references=False)
            merge_node(stage, label, props)
        for label, props in records:
            _check_identity(stage, label, props)
        for src, rel, dst, props in edges:
            merge_edge(stage, src, rel, dst, props)
        return len(prepared)
    return publish_import(graph, apply, lambda stage: _verify_records(stage, records, edges))


def seed_demo(force: bool = False) -> dict:
    """Seed the demo graph from topology.json (idempotent)."""
    graph = graph_for(DEMO_GRAPH)
    topology = json.loads((SEED_DIR / "topology.json").read_text())
    counts = load_topology(graph, topology, replace=force, ensure_indexes=True)
    return {"graph": DEMO_GRAPH, **counts}


def load_fixture_by_name(name: str, graph=None) -> dict:
    """Load a fixture file (name without .json) into the demo graph."""
    known = {p.stem for p in FIXTURES_DIR.glob("*.json")}
    if name not in known:
        raise ValueError("unknown fixture name")
    path = FIXTURES_DIR / f"{name}.json"
    fixture = json.loads(path.read_text())
    target = graph or graph_for(DEMO_GRAPH)
    written = load_fixture(target, fixture)
    return {"fixture": name, "nodes_written": written,
            "incident_id": fixture["incident"]["id"]}


def load_topology(graph, topology: dict, *, replace=False, ensure_indexes=False) -> dict:
    """Prevalidate topology, stage corrections/indexes, verify and publish atomically."""
    class Plan:
        def __init__(self):
            self.nodes = {}
            self.edges = []
        def query(self, query, params=None):
            from types import SimpleNamespace
            p = params or {}
            if query.startswith('MERGE (n:'):
                label = query.split(':', 1)[1].split(' ', 1)[0]
                node_id = p['node_id']
                if node_id in self.nodes:
                    raise ValueError('duplicate topology id')
                self.nodes[node_id] = label
                self.edges.append((query, p))
                return SimpleNamespace(result_set=[])
            if 'RETURN labels(n)' in query:
                label = self.nodes.get(p['endpoint_id'])
                return SimpleNamespace(result_set=[[[label]]] if label else [])
            if 'RETURN b.id' in query:
                return SimpleNamespace(result_set=[])
            self.edges.append((query, p))
            return SimpleNamespace(result_set=[[1]])
    plan = Plan()
    counts = _load_topology(plan, topology)
    records, edges = [], []
    for query, params in plan.edges:
        if query.startswith('MERGE (n:'):
            label = query.split(':', 1)[1].split(' ', 1)[0]
            records.append((label, _prepared_node(label, {k[5:]: v for k, v in params.items()})))
        else:
            rel = query.split('[e:', 1)[1].split(']', 1)[0]
            edges.append((params['endpoint_src'], rel, params['endpoint_dst'],
                          {k[5:]: v for k, v in params.items() if k.startswith('edge_')}))
    if len(records) > 20000 or len(edges) > 40000:
        raise ValueError('oversized topology')
    def apply(stage):
        for label, props in records:
            _check_identity(stage, label, props, references=False)
            merge_node(stage, label, props)
        for label, props in records:
            _check_identity(stage, label, props)
        for src, rel, dst, props in edges:
            merge_edge(stage, src, rel, dst, props)
        if ensure_indexes:
            schema.ensure_indexes(stage)
        return counts
    return publish_import(graph, apply, lambda stage: _verify_records(stage, records, edges), replace=replace)
