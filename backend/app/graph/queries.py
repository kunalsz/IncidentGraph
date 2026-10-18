# Explicit allowlists, shared by initial fetch, trace context and live validation.
# toJSON is supported by the pinned server for scalar and array properties.
# CASE prevents oversized values from crossing the wire; rejected rows fail closed.
SERVICE_FIELDS = ('id', 'name', 'aliases', 'criticality', 'level', 'level_convention', 'service_id')
EVIDENCE_FIELDS = ('id', 'ts', 'metric', 'column', 'value', 'baseline_mean', 'baseline_std',
                   'z', 'gamma', 'direction', 'level', 'is_error', 'template_id', 'message',
                   'duration_us', 'start_us', 'trace_id', 'span_id', 'parent_id', 'trace_complete',
                   'op', 'status', 'flag', 'version', 'kind', 'name', 'description', 'reason',
                   'onset_clock_available', 'onset_valid_count', 'onset_qualified_count',
                   'onset_longest_same_direction_run', 'onset_qualified_directions',
                   'onset_representative_ts', 'onset_representative_direction',
                   'post_deviation_directions')
EDGE_FIELDS = ('invariant', 'weight', 'protocol', 'criticality')


def projection(variable, fields, *, kind=False):
    checks = [f'size(toJSON({variable}.{key})) > $field_chars' for key in fields]
    values = [f'{key}: CASE WHEN {checks[i]} THEN null ELSE {variable}.{key} END'
              for i, key in enumerate(fields)]
    values.append('__rejected: ' + ' OR '.join(f'coalesce({c}, false)' for c in checks))
    if kind:
        values.append(f'__kind: labels({variable})[0]')
    return '{' + ', '.join(values) + '}'


SCOPE_SERVICES = """
MATCH (a:Service {id: $anchor})
OPTIONAL MATCH (a)-[r:DEPENDS_ON]-(n:Service)
WHERE coalesce(r.invariant, true) = true
WITH a, collect(DISTINCT n) + [a] AS candidates
UNWIND candidates AS s
RETURN DISTINCT s
ORDER BY s.id = $anchor DESC, s.criticality DESC, s.id
"""

# Benchmark mode (RCAEval adapter): cold ranking over every service in the
# graph — RCAEval methods rank all candidates with no reported symptom.
SCOPE_SERVICES_FULL = """
MATCH (s:Service)
RETURN s
ORDER BY s.criticality DESC, s.id
"""

SCOPE_EDGES = """
MATCH (a:Service)-[r:DEPENDS_ON]->(b:Service)
WHERE a.id IN $ids AND b.id IN $ids AND coalesce(r.invariant, true) = true
RETURN a.id, b.id, properties(r)
ORDER BY a.id, b.id
"""

SCOPE_OBSERVATIONS = """
MATCH (e)-[:OBSERVED_ON]->(target)
WHERE (target:Service AND target.id IN $ids)
   OR (target:Instance AND target.service_id IN $ids)
WITH e, target, CASE WHEN target:Instance THEN target.service_id ELSE target.id END AS sid
WHERE e.ts >= $t0 AND e.ts <= $t1
  AND (e:Alert OR e:LogEvent OR e:MetricAnomaly OR e:Span OR e:Trace)
RETURN DISTINCT e, sid, target.id
ORDER BY e.ts, e.id, sid
"""

SCOPE_CHANGES = """
MATCH (c)-[r]->(target)
WHERE ((c:ConfigChange AND type(r) = 'CHANGED') OR (c:Deployment AND type(r) = 'DEPLOYED_AT'))
  AND ((target:Service AND target.id IN $ids) OR (target:Instance AND target.service_id IN $ids))
  AND c.ts >= $t0 AND c.ts <= $t1
RETURN c, CASE WHEN target:Instance THEN target.service_id ELSE target.id END, target.id
ORDER BY c.ts DESC, c.id
"""

SCOPE_SERVICES = SCOPE_SERVICES.replace('RETURN DISTINCT s\n', 'RETURN DISTINCT ' + projection('s', SERVICE_FIELDS) + ' AS s\n')
SCOPE_SERVICES_FULL = SCOPE_SERVICES_FULL.replace('RETURN s\n', 'RETURN ' + projection('s', SERVICE_FIELDS) + ' AS s\n')
SCOPE_OBSERVATIONS = SCOPE_OBSERVATIONS.replace('RETURN DISTINCT e,', 'RETURN DISTINCT ' + projection('e', EVIDENCE_FIELDS, kind=True) + ' AS e,')
SCOPE_CHANGES = SCOPE_CHANGES.replace('RETURN c,', 'RETURN ' + projection('c', EVIDENCE_FIELDS, kind=True) + ' AS c,')
SCOPE_EDGES = SCOPE_EDGES.replace('properties(r)', projection('r', EDGE_FIELDS))
for _name in ('SCOPE_OBSERVATIONS', 'SCOPE_CHANGES'):
    globals()[_name] = globals()[_name].replace(
        ', target.id\n', ", CASE WHEN size(toJSON(target.id)) > $field_chars THEN null ELSE target.id END\n")


UPDATE_SCOPE_STATS = """
MATCH (s:Service {id: $id})
SET s.sev_m = $sev_m, s.sev_l = $sev_l, s.sev_tc = $sev_tc,
    s.self_time_ms = $self_time_ms, s.self_time_flag = $self_time_flag,
    s.nsigma_count = $nsigma_count, s.deploy_proximity = $deploy_proximity,
    s.hours_since_change = $hours_since_change,
    s.scope_t0 = $t0, s.scope_t1 = $t1
"""

UPSERT_SCOPE_GAP = """
MERGE (g:EvidenceGap {id: $id})
SET g.incident_id = $incident, g.node_id = $node_id,
    g.missing_kind = $kind, g.suggestion = $suggestion,
    g.attempted_queries = $attempted, g.ts = $ts
"""
