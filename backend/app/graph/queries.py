SCOPE_SERVICES = """
MATCH (a:Service {id: $anchor})
OPTIONAL MATCH (a)-[r:DEPENDS_ON]-(n:Service)
WHERE coalesce(r.invariant, true) = true
WITH a, collect(DISTINCT n) + [a] AS candidates
UNWIND candidates AS s
RETURN DISTINCT s
ORDER BY s.id = $anchor DESC, s.criticality DESC, s.id
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
MATCH (c:ConfigChange)-[:CHANGED]->(s:Service)
WHERE s.id IN $ids AND c.ts <= $t1
RETURN c AS change, s.id AS service_id, s.id AS target_id
UNION
MATCH (d:Deployment)-[:DEPLOYED_AT]->(i:Instance)
WHERE i.service_id IN $ids AND d.ts <= $t1
RETURN d AS change, i.service_id AS service_id, i.id AS target_id
"""

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
