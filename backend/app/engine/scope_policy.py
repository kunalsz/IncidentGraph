"""
Canonical admitted snapshots shared by ranking, bundles and tools.
The in-process registry.
"""
import copy
import hashlib
import json
import math
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


_SEVERITY_MODALITY = {'MetricAnomaly': 'metric', 'LogEvent': 'log', 'Span': 'trace', 'Trace': 'trace'}


def _finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _has_usable_z(record):
    """Mirror of the z rules in p1_scope.service_statistics (kept in sync by tests)."""
    props = record['properties']
    if _finite_number(props.get('z')):
        return True
    if record['type'] == 'MetricAnomaly':
        value, mean, std = (props.get(k) for k in ('value', 'baseline_mean', 'baseline_std'))
        return all(_finite_number(v) for v in (value, mean, std)) and std > 0
    return False


def scored_modalities(records, service_id=None):
    """Which severity modalities could actually be scored.

    Returns {modality: bool} only for modalities that have records: True when at
    least one record carries a usable z, False when records exist but none can be
    scored. A modality with no records is omitted, so callers can tell
    "not measured" (False) from "absent" (missing) from "measured" (True). A zero
    severity in the first case does not mean the modality looked normal.
    """
    result = {}
    for record in records:
        modality = _SEVERITY_MODALITY.get(record['type'])
        if modality is None or (service_id is not None and record['service_id'] != service_id):
            continue
        result[modality] = result.get(modality, False) or _has_usable_z(record)
    return result


def admitted_records(scope, service_id=None):
    owners = {e['source']: e['target'] for e in scope.edges
              if e['type'] in ('OBSERVED_ON', 'CHANGED', 'DEPLOYED_AT')}
    return [{'id': n['id'], 'type': n['type'], 'properties': n['properties'],
             'service_id': owners[n['id']]} for n in scope.nodes
            if n['id'] in owners and (service_id is None or owners[n['id']] == service_id)]
