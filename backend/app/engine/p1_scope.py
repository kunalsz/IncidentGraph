"""
P1 turns single anchor service from P0 into a small, bounded, read-only evidence picture.
"""

import math
from collections import defaultdict

from backend.app.engine.contracts import Investigation, ScopeResult, ScopeService
from backend.app.engine.scope_policy import series_identity, scope_fingerprint, remember_scope, scored_modalities
from backend.app.graph import queries
from backend.app.graph.client import graph_for
from backend.app.graph.gateway import ReadGateway, admit_payload, PayloadAdmissionError


def _properties(value, settings):
    props = dict(value.properties if hasattr(value, 'properties') else value)
    if props.pop('__rejected', False):
        raise PayloadAdmissionError('oversized projected property')
    kind = props.pop('__kind', None)
    props = {k: v for k, v in props.items() if v is not None}
    admit_payload(props, field_chars=settings.field_chars)
    if not isinstance(props.get('id'), str) or not props['id']:
        raise PayloadAdmissionError('missing canonical identity')
    numeric = {'criticality', 'level', 'value', 'baseline_mean', 'baseline_std', 'z', 'gamma',
               'duration_us', 'start_us'}
    # Severity level is textual; service structural level is numeric.
    for key in numeric.intersection(props):
        if (key == 'level' and kind in ('LogEvent', 'Alert')
                or key == 'value' and kind in ('ConfigChange', 'Deployment')):
            continue
        if type(props[key]) not in (int, float):
            if key == 'level' and isinstance(props[key], str):
                continue
            raise PayloadAdmissionError('invalid numeric property: ' + key)
    for key in {'trace_complete', 'is_error'}.intersection(props):
        if type(props[key]) is not bool:
            raise PayloadAdmissionError('invalid boolean property: ' + key)
    for key in {'metric', 'column', 'trace_id', 'span_id', 'parent_id', 'message', 'template_id'}.intersection(props):
        if not isinstance(props[key], str):
            raise PayloadAdmissionError('invalid text property: ' + key)
    return props, kind
from backend.app.settings import SCOPE_SETTINGS
from config import tuning


def _exclusive(parent, spans, query_complete=True):
    """Integer-us interval union, clipped to parent; unknown source => upper bound."""
    p = parent['properties']
    duration = p.get('duration_us')
    if type(duration) not in (int, float) or not math.isfinite(duration) or duration < 0:
        return None, 'unknown'
    start = p.get('start_us', p.get('ts', 0) * 1000)
    end = start + int(duration)
    intervals, seen, conflict = [], {}, False
    for child in spans:
        c = child['properties']
        key = (c.get('trace_id'), c.get('span_id'))
        if not p.get('trace_id') or not p.get('span_id') or c.get('trace_id') != p['trace_id']:
            continue
        if key in seen and seen[key] != c:
            conflict = True
        seen[key] = c
    for c in seen.values():
        if c.get('parent_id') != p.get('span_id'):
            continue
        d = c.get('duration_us')
        if type(d) not in (int, float) or not math.isfinite(d) or d < 0:
            conflict = True
            continue
        a = c.get('start_us', c.get('ts', 0) * 1000)
        a, b = max(start, a), min(end, a + int(d))
        if b > a:
            intervals.append((a, b))
    covered, right = 0, start
    for a, b in sorted(intervals):
        covered += max(0, b - max(a, right))
        right = max(right, b)
    quality = 'complete' if query_complete and p.get('trace_complete') is True and not conflict else 'partial'
    return max(0, int(duration) - covered) / 1000, quality


def service_statistics(service_id, observations, changes, trigger_ts, child_sums=None,
                       trace_context=None, trace_query_complete=True,
                       change_context_ms=SCOPE_SETTINGS.change_context_ms):
    records = [r for r in observations if r['service_id'] == service_id]
    vectors = {'MetricAnomaly': 0.0, 'LogEvent': 0.0, 'Span': 0.0}
    nsigma, exact, upper, qualities = [], [], [], []
    spans = [r for r in observations + (trace_context or []) if r['type'] == 'Span']
    for record in records:
        props, kind = record['properties'], record['type']
        modality = 'Span' if kind == 'Trace' else kind
        z = props.get('z')
        if z is None and kind == 'MetricAnomaly':
            value, mean, std = (props.get(k) for k in ('value', 'baseline_mean', 'baseline_std'))
            if all(type(v) in (int, float) and math.isfinite(v) for v in (value, mean, std)):
                if std > 0:
                    z = (value - mean) / std
                elif std == 0 and value != mean and abs(props['ts'] - trigger_ts) <= tuning.NSIGMA_DELTA_MS:
                    nsigma.append(record['id'])
        if type(z) in (int, float) and math.isfinite(z) and modality in vectors:
            vectors[modality] = max(vectors[modality], abs(z))
            if kind == 'MetricAnomaly' and abs(z) > tuning.NSIGMA_N and abs(props['ts'] - trigger_ts) <= tuning.NSIGMA_DELTA_MS:
                nsigma.append(record['id'])
        if kind == 'Span':
            value, quality = _exclusive(record, spans, trace_query_complete)
            qualities.append(quality)
            if value is not None:
                upper.append(value)
                if quality == 'complete':
                    exact.append(value)
    latest = max((r['properties']['ts'] for r in changes if r['service_id'] == service_id
                  and trigger_ts - change_context_ms <= r['properties']['ts'] <= trigger_ts), default=None)
    hours = (trigger_ts - latest) / 3_600_000 if latest is not None else None
    quality = 'complete' if qualities and all(q == 'complete' for q in qualities) else 'partial' if upper else 'unknown'
    self_time = max(exact, default=0.0) if quality == 'complete' else 0.0
    extra = {'self_time_upper_bound_ms': max(upper) if upper else None, 'self_time_quality': quality}
    return ScopeService(id=service_id, sev_m=vectors['MetricAnomaly'], sev_l=vectors['LogEvent'],
                        sev_tc=vectors['Span'], self_time_ms=self_time,
                        self_time_flag=quality == 'complete' and self_time > tuning.SELF_TIME_THRESHOLD_MS,
                        nsigma_evidence=sorted(set(nsigma)), hours_since_change=hours,
                        deploy_proximity=math.exp(-hours / 24) if hours is not None else 0.0,
                        **{k: v for k, v in extra.items() if k in ScopeService.model_fields})


def scope_investigation(investigation: Investigation, *, full_scope=False,
                        max_nodes=None, max_series=None, settings=None) -> ScopeResult:
    """Read a private immutable GRAPH.COPY; publish only after live validation."""
    import uuid
    from falkordb import Graph
    if not investigation.session_graph.startswith('incidentgraph:session:'):
        raise ValueError('P1 requires an isolated session graph')
    graph = graph_for(investigation.session_graph)
    snapshot = None
    try:
        if isinstance(graph, Graph):
            snapshot = graph.copy('incidentgraph:read:' + uuid.uuid4().hex)
        return _scope_investigation(investigation, full_scope=full_scope,
                                   max_nodes=max_nodes, max_series=max_series,
                                   settings=settings, graph=graph, read_graph=snapshot or graph)
    finally:
        if snapshot is not None:
            snapshot.delete()


def _scope_investigation(investigation: Investigation, *, full_scope=False,
                         max_nodes=None, max_series=None, settings=None,
                         graph, read_graph) -> ScopeResult:
    settings = settings or SCOPE_SETTINGS
    if not investigation.session_graph.startswith('incidentgraph:session:'):
        raise ValueError('P1 requires an isolated session graph')
    node_cap = tuning.SCOPE_MAX_NODES if max_nodes is None else max_nodes
    series_cap = tuning.SCOPE_MAX_SERIES if max_series is None else max_series
    if type(node_cap) is not int or type(series_cap) is not int or node_cap < 1 or series_cap < 0:
        raise ValueError('scope limits must allow anchor and nonnegative series')
    gateway = ReadGateway(read_graph, settings=settings)
    publication_rows, publication_overflow = gateway.read(
        'MATCH (a:Artifact {id: $id}) RETURN CASE WHEN size(toJSON(a.source_hash)) > $field_chars THEN null ELSE a.source_hash END',
        {'id': '__incidentgraph_import_publication__'}, cap=1)
    if publication_overflow:
        raise ValueError('ambiguous source publication marker')
    publication_version = publication_rows[0][0] if publication_rows else None
    service_query = queries.SCOPE_SERVICES_FULL if full_scope else queries.SCOPE_SERVICES
    if full_scope:
        service_query = service_query.replace('ORDER BY s.criticality', 'ORDER BY s.id = $anchor DESC, s.criticality')
    rows, overflow = gateway.read(service_query, {'anchor': investigation.anchor_service}, cap=node_cap)
    services = [_properties(row[0], settings)[0] for row in rows]
    if investigation.anchor_service not in {s['id'] for s in services}:
        raise ValueError('anchor service absent from bounded scope')
    ids = [s['id'] for s in services]
    window = investigation.window
    onset = getattr(investigation, 'onset_ts', None)
    if onset is None:
        rows, _ = gateway.read('MATCH (i:Incident {id: $id}) RETURN i.start_ts',
                               {'id': investigation.incident_id}, cap=1)
        onset = rows[0][0] if rows else None
    if onset is None:
        rows, _ = gateway.read(
            'MATCH (a:Alert)-[:OBSERVED_ON]->(target) '
            'WHERE (target:Service AND target.id = $sid) OR '
            '(target:Instance AND target.service_id = $sid) RETURN max(a.ts)',
            {'sid': investigation.anchor_service}, cap=1)
        onset = rows[0][0] if rows else None
    if onset is None:
        raise ValueError('scope requires authoritative incident onset')
    failures = []
    def records(query, params, cap):
        nonlocal overflow
        try:
            rows, truncated = gateway.read(query, params, cap=cap)
            overflow |= truncated
            result = {}
            for node, sid, target in rows:
                props, kind = _properties(node, settings)
                admit_payload([sid, target], field_chars=settings.field_chars)
                if sid not in ids or not isinstance(target, str) or not target:
                    raise PayloadAdmissionError('invalid canonical target')
                if type(props.get('ts')) is not int or props['ts'] < 0:
                    raise PayloadAdmissionError('invalid evidence clock')
                record = {'id': props['id'], 'type': kind or node.labels[0],
                          'properties': props, 'service_id': sid, 'target_id': target}
                if record['id'] in ids or any(isinstance(v, str) and len(v) > settings.field_chars
                                               for k, v in record['properties'].items()):
                    failures.append('oversized_or_identity_collision')
                    overflow = True
                    continue
                old = result.get(record['id'])
                if record['id'] in result and old is None:
                    continue
                if old and (old['service_id'] != sid or old['properties'] != record['properties']):
                    failures.append('conflicting_ownership_or_duplicate')
                    overflow = True
                    result[record['id']] = None
                elif record['id'] not in result:
                    result[record['id']] = record
            return [r for r in result.values() if r is not None]
        except Exception as exc:
            failures.append(type(exc).__name__)
            overflow = True
            return []
    # Separate service/modality/pre-post fetch buckets before any shared cap.
    # Sampling is onset-local within each side; no earliest-first all-modality prefix.
    observations = []
    bucket_count = len(ids) * 5 * 2
    quota = max(1, settings.raw_observations // max(1, bucket_count))
    for sid in ids:
        for label in ('MetricAnomaly', 'LogEvent', 'Span', 'Trace', 'Alert'):
            for before in (False, True):
                remaining = settings.raw_observations - len(observations)
                if remaining <= 0:
                    overflow = True
                    continue
                query = queries.SCOPE_OBSERVATIONS.replace(
                    'AND (e:Alert OR e:LogEvent OR e:MetricAnomaly OR e:Span OR e:Trace)',
                    f'AND e:{label} AND e.ts ' + ('< $onset' if before else '>= $onset'))
                query = query.replace('ORDER BY e.ts, e.id, sid',
                                      'ORDER BY e.ts ' + ('DESC' if before else 'ASC') + ', e.id, sid')
                observations.extend(records(query, {'ids': [sid], 't0': window.t0,
                                            't1': window.t1, 'onset': onset}, min(quota, remaining)))
    changes = records(queries.SCOPE_CHANGES, {'ids': ids, 't0': max(0, onset-settings.change_context_ms),
                                             't1': onset}, settings.changes)
    canonical = {}
    for record in observations + changes:
        key = record['id']
        old = canonical.get(key)
        if key in canonical and (old is None or old['service_id'] != record['service_id']
                                 or old['properties'] != record['properties']):
            canonical[key] = None
            overflow = True
            failures.append('conflicting_canonical_ownership')
        elif key not in canonical:
            canonical[key] = record
    observations = [r for r in canonical.values() if r and r['type'] not in ('Deployment', 'ConfigChange')]
    changes = [r for r in canonical.values() if r and r['type'] in ('Deployment', 'ConfigChange')]
    priorities = {sid: i for i, sid in enumerate(ids)}
    series = sorted({series_identity(r) for r in observations if r['type'] == 'MetricAnomaly'},
                    key=lambda x: (priorities[x[0]], x[1]))
    allowed = set(series[:series_cap])
    eligible = [r for r in observations if r['type'] != 'MetricAnomaly' or series_identity(r) in allowed]
    per_service = defaultdict(int)
    for r in sorted(changes, key=lambda r: (-r['properties']['ts'], r['id'])):
        if per_service[r['service_id']] < settings.changes_per_service:
            eligible.append(r)
            per_service[r['service_id']] += 1
    # Round-robin service/modality buckets prevent one log-heavy service monopolizing nodes.
    buckets = defaultdict(list)
    for r in sorted(eligible, key=lambda r: (r['id'] != investigation.anchor_alert, -r['properties']['ts'], r['id'])):
        buckets[(priorities[r['service_id']], r['type'], r['properties']['ts'] < onset)].append(r)
    admitted = []
    while buckets and len(admitted) + len(services) < node_cap:
        for key in sorted(list(buckets)):
            if len(admitted) + len(services) >= node_cap:
                break
            admitted.append(buckets[key].pop(0))
            if not buckets[key]:
                del buckets[key]
    selected_obs = [r for r in admitted if r['type'] not in ('ConfigChange', 'Deployment')]
    selected_changes = [r for r in admitted if r['type'] in ('ConfigChange', 'Deployment')]
    edges_rows, edge_overflow = gateway.read(queries.SCOPE_EDGES, {'ids': ids}, cap=max(1, node_cap * node_cap))
    edges = []
    for a, b, p in edges_rows:
        p = dict(p)
        if p.pop('__rejected', False):
            raise PayloadAdmissionError('oversized dependency property')
        p = {k: v for k, v in p.items() if v is not None}
        admit_payload([a, b, p], field_chars=settings.field_chars)
        edges.append({'source': a, 'target': b, 'type': 'DEPENDS_ON', 'properties': p})
    edges += [{'source': r['id'], 'target': r['service_id'],
               'type': 'CHANGED' if r['type'] == 'ConfigChange' else 'DEPLOYED_AT' if r['type'] == 'Deployment' else 'OBSERVED_ON',
               'properties': {'projected': True, 'target_id': r['target_id']}} for r in admitted]
    # Trace context is not evidence/candidate admission. Fetch only children of admitted spans.
    parents = [r for r in selected_obs if r['type'] == 'Span' and r['properties'].get('span_id')]
    trace_context = []
    trace_complete = not overflow
    if parents:
        try:
            context_rows, context_overflow = gateway.read(
                'MATCH (sp:Span) WHERE sp.trace_id IN $traces AND sp.parent_id IN $parents RETURN ' + queries.projection('sp', queries.EVIDENCE_FIELDS) + ' AS sp ORDER BY sp.ts, sp.id',
                {'traces': sorted({r['properties'].get('trace_id', '') for r in parents}),
                 'parents': sorted({r['properties']['span_id'] for r in parents})}, cap=settings.trace_context)
            for (n,) in context_rows:
                props, _ = _properties(n, settings)
                trace_context.append({'id': props['id'], 'type': 'Span', 'properties': props})
            trace_complete &= not context_overflow
        except Exception as exc:
            failures.append('trace_context:' + type(exc).__name__)
            trace_complete = False
    gateway.graph = graph
    transcript = gateway.validate()
    truncated = overflow or edge_overflow or len(admitted) < len(observations) + len(changes) or len(allowed) < len(series)
    nodes = [{'id': s['id'], 'type': 'Service', 'properties': dict(s)} for s in services]
    nodes += [{k: r[k] for k in ('id', 'type', 'properties')} for r in admitted]
    stats, gaps = [], []
    for service in services:
        sid = service['id']
        coverage = {r['type'] for r in selected_obs if r['service_id'] == sid}
        for kind, labels in [('no_traces', {'Span', 'Trace'}), ('no_logs', {'LogEvent'})]:
            if not coverage.intersection(labels):
                gaps.append({'node_id': sid, 'kind': kind, 'coverage': 'unknown' if truncated else 'absent',
                             'suggestion': 'Request bounded additional telemetry.'})
        # Records exist but none can be scored: a 0 severity here means "not measured", not "normal".
        scored = scored_modalities(selected_obs, sid)
        for modality, kind in (('metric', 'unscored_metrics'), ('log', 'unscored_logs'), ('trace', 'unscored_traces')):
            if scored.get(modality) is False:
                gaps.append({'node_id': sid, 'kind': kind, 'coverage': 'present_unscored',
                             'suggestion': f'{modality.capitalize()} records are present but carry no z-score; '
                                           'severity for this modality is not measured. Provide baseline statistics at ingest.'})
        stat = service_statistics(sid, selected_obs, selected_changes, onset, trace_context=trace_context, trace_query_complete=trace_complete, change_context_ms=settings.change_context_ms)
        stat.criticality = service.get('criticality', 0.0)
        stats.append(stat)
        next(n for n in nodes if n['id'] == sid)['properties'].update(stat.model_dump(exclude={'id', 'nsigma_evidence'}))
        params = {k: v for k, v in stat.model_dump().items() if k in ('id', 'sev_m', 'sev_l', 'sev_tc', 'self_time_ms', 'self_time_flag', 'hours_since_change', 'deploy_proximity')}
        graph.query(queries.UPDATE_SCOPE_STATS, params={**params, 'nsigma_count': len(stat.nsigma_evidence), 't0': window.t0, 't1': window.t1}, timeout=settings.query_timeout_ms)
    if truncated:
        gaps.append({'node_id': investigation.anchor_service, 'kind': 'scope_truncation', 'suggestion': 'Approve a bounded scope revision.'})
    for gap in gaps:
        gap['id'] = f"gap:{investigation.incident_id}:{gap['node_id']}:{gap['kind']}"
        graph.query(queries.UPSERT_SCOPE_GAP, params={**gap, 'incident': investigation.incident_id, 'ts': onset, 'attempted': ['bounded_scope']})
    result = ScopeResult(incident_id=investigation.incident_id, anchor_service=investigation.anchor_service,
                         window=window, services=stats, nodes=nodes, edges=edges, evidence_gaps=gaps,
                         n_series_in_scope=len({series_identity(r) for r in selected_obs if r['type'] == 'MetricAnomaly'}), truncated=truncated)
    config = {'policy': settings.version, 'settings_hash': settings.fingerprint,
              'raw_observation_cap': settings.raw_observations, 'admitted_node_cap': node_cap,
              'series_cap': series_cap, 'sampling': 'service-modality-prepost-onset-local-v2',
              'fingerprint': scope_fingerprint(result, settings.fingerprint), 'queries': gateway.diagnostics,
              'omitted_records': len(observations)+len(changes)-len(admitted), 'failures': failures,
              'trace_context_records': len(trace_context), 'trace_query_complete': trace_complete,
              'snapshot_consistency': 'immutable-graph-copy-live-repeat-validation-v1' if read_graph is not graph else 'adapter-repeat-read-nontransactional-v1',
              'property_projection': 'allowlist-json-field-guard-v1',
              'source_publication_version': publication_version}
    if 'config' in ScopeResult.model_fields:
        payload = result.model_dump()
        payload.update(config=config, scope_ref=None)
        result = ScopeResult(**payload)
    remember_scope(graph, result, onset, transcript=transcript, settings=settings)
    return result
