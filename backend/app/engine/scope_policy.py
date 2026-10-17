"""Canonical admitted snapshots shared by ranking, bundles and tools.

The in-process registry is deliberately not a durable resume mechanism. A new
process must rerun P1, never fall back to graph-wide evidence reads.
"""
import copy
import hashlib
import json
from collections import OrderedDict
from threading import RLock

_SNAPSHOTS = OrderedDict()
_LOCK = RLock()


def series_identity(record):
    props = record['properties']
    return record['service_id'], props.get('column') or props.get('metric') or record['id']


def scope_fingerprint(scope, settings_hash):
    value = {'incident': scope.incident_id, 'window': scope.window.model_dump(),
             'nodes': scope.nodes, 'edges': scope.edges, 'settings': settings_hash}
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def remember_scope(graph, scope, onset, *, transcript=None, settings=None):
    name = getattr(graph, 'name', None)
    # Serialized canonical bytes cannot be mutated by callers holding ScopeResult.
    value = (scope.model_dump_json(), onset, copy.deepcopy(transcript or []), settings)
    with _LOCK:
        _SNAPSHOTS[(name, scope.window.t0, scope.window.t1)] = value
        _SNAPSHOTS.move_to_end((name, scope.window.t0, scope.window.t1))
        while len(_SNAPSHOTS) > 128:
            _SNAPSHOTS.popitem(last=False)
    try:
        graph._incidentgraph_scope = value
    except (AttributeError, TypeError):
        pass


def recalled_scope(graph, window=None):
    def restore(value):
        from backend.app.engine.contracts import ScopeResult
        if len(value) == 2:  # Legacy test adapters only, never created by P1.
            return copy.deepcopy(value)
        payload, onset, transcript, settings = value
        scope = ScopeResult.model_validate_json(payload)
        if transcript:
            from backend.app.graph.gateway import ReadGateway
            gateway = ReadGateway(graph, settings=settings)
            gateway.transcript = transcript
            gateway.validate()
        return scope, onset
    value = getattr(graph, '_incidentgraph_scope', None)
    if isinstance(value, tuple):
        restored = restore(value)
        if window is None or restored[0].window == window:
            return restored
    name = getattr(graph, 'name', None)
    if name is None:
        return None
    with _LOCK:
        candidates = [v for (n, t0, t1), v in _SNAPSHOTS.items()
                      if n == name and (window is None or (t0, t1) == (window.t0, window.t1))]
        value = candidates[-1] if candidates else None
    return restore(value) if value else None


def admitted_records(scope, service_id=None):
    owners = {e['source']: e['target'] for e in scope.edges
              if e['type'] in ('OBSERVED_ON', 'CHANGED', 'DEPLOYED_AT')}
    return [{'id': n['id'], 'type': n['type'], 'properties': n['properties'],
             'service_id': owners[n['id']]} for n in scope.nodes
            if n['id'] in owners and (service_id is None or owners[n['id']] == service_id)]
