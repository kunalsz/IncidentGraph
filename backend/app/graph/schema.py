"""Node labels and edge types of the IncidentGraph schema v2 (§4.2)."""

# Node labels
SERVICE = "Service"
INSTANCE = "Instance"
TEAM = "Team"
DEPLOYMENT = "Deployment"
CONFIG_CHANGE = "ConfigChange"
INCIDENT = "Incident"
ALERT = "Alert"
LOG_EVENT = "LogEvent"
METRIC_ANOMALY = "MetricAnomaly"
TRACE = "Trace"
SPAN = "Span"
RUNBOOK = "Runbook"
RUNBOOK_PARAM = "RunbookParam"
DECISION = "Decision"
HYPOTHESIS = "Hypothesis"
EVIDENCE = "Evidence"
EVIDENCE_GAP = "EvidenceGap"
ARTIFACT = "Artifact"
DETECTOR = "Detector"

NODE_LABELS = {
    SERVICE, INSTANCE, TEAM, DEPLOYMENT, CONFIG_CHANGE, INCIDENT, ALERT,
    LOG_EVENT, METRIC_ANOMALY, TRACE, SPAN, RUNBOOK, RUNBOOK_PARAM, DECISION,
    HYPOTHESIS, EVIDENCE, EVIDENCE_GAP, ARTIFACT, DETECTOR,
}

# Edge types
DEPENDS_ON = "DEPENDS_ON"            # Service->Service, structural/frozen
OWNED_BY = "OWNED_BY"                # Service->Team
DEPLOYED_AT = "DEPLOYED_AT"          # Deployment->Instance
CHANGED = "CHANGED"                  # ConfigChange->Service
OBSERVED_ON = "OBSERVED_ON"          # Alert/LogEvent/MetricAnomaly/Span -> Service|Instance
AFFECTS = "AFFECTS"                  # Incident->Service
APPLIES_TO = "APPLIES_TO"            # Runbook->Service
REQUIRES_PARAMETER = "REQUIRES_PARAMETER"  # Runbook->RunbookParam
MITIGATED_BY = "MITIGATED_BY"        # Incident->Decision
HAS_EVIDENCE = "HAS_EVIDENCE"        # Hypothesis->Evidence
CAUSES = "CAUSES"                    # Reserved for verified causal claims
ASSOCIATED_WITH = "ASSOCIATED_WITH"  # Hypothesis->Service, observed association
CONTRADICTS = "CONTRADICTS"          # Evidence<->Evidence
EXPANDED_TO = "EXPANDED_TO"          # Hypothesis->Hypothesis
STATISTICAL_DEPENDENCY = "STATISTICAL_DEPENDENCY"
MERGED_INTO = "MERGED_INTO"
DROPPED_BY_PREPROCESSING = "DROPPED_BY_PREPROCESSING"
SIMILAR_TO = "SIMILAR_TO"            # Incident->Incident
TRIGGERED_BY = "TRIGGERED_BY"        # Alert->Detector

EDGE_TYPES = {
    DEPENDS_ON, OWNED_BY, DEPLOYED_AT, CHANGED, OBSERVED_ON, AFFECTS,
    APPLIES_TO, REQUIRES_PARAMETER, MITIGATED_BY, HAS_EVIDENCE, CAUSES, ASSOCIATED_WITH,
    CONTRADICTS, EXPANDED_TO, STATISTICAL_DEPENDENCY, MERGED_INTO,
    DROPPED_BY_PREPROCESSING, SIMILAR_TO, TRIGGERED_BY,
}

# Indexes: verified syntax against falkordb/falkordb:latest (2026-10-16).
# Uniqueness is enforced structurally by MERGE-on-id loaders (GRAPH.CONSTRAINT
# is not exposed through Cypher in this server version).
INDEXES = [
    "CREATE INDEX FOR (s:Service) ON (s.id)",
    "CREATE INDEX FOR (i:Instance) ON (i.id)",
    "CREATE INDEX FOR (a:Alert) ON (a.ts)",
    "CREATE INDEX FOR (a:Alert) ON (a.kind)",
    "CREATE INDEX FOR (l:LogEvent) ON (l.ts)",
    "CREATE INDEX FOR (l:LogEvent) ON (l.template_id)",
    "CREATE INDEX FOR (m:MetricAnomaly) ON (m.ts)",
    "CREATE INDEX FOR (d:Deployment) ON (d.ts)",
    "CREATE INDEX FOR (c:ConfigChange) ON (c.ts)",
    "CREATE INDEX FOR (i:Incident) ON (i.id)",
    "CREATE INDEX FOR (sp:Span) ON (sp.ts)",
    "CREATE INDEX FOR (sp:Span) ON (sp.trace_id)",
    "CREATE INDEX FOR (sp:Span) ON (sp.parent_id)",
    "CREATE INDEX FOR (i:Instance) ON (i.service_id)",
]


def ensure_indexes(graph) -> int:
    """Create indexes (idempotent: duplicate creates are ignored by FalkorDB)."""
    created = 0
    for q in INDEXES:
        try:
            graph.query(q)
            created += 1
        except Exception as exc:
            # Only a server-declared duplicate index is safe to ignore.
            from redis.exceptions import ResponseError
            message = str(exc).lower()
            if not isinstance(exc, ResponseError) or not (
                    'index already exists' in message or 'already indexed' in message):
                raise
    return created
