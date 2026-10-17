"""P0 — TRIGGER & RESOLVE (§4.3).

Inputs: symptom text ("checkout latency up"), optional alert id, optional
trigger from Detector. Entity linking: alias table + fuzzy match against
Service|Instance names; unresolved anchors -> clarify/abstain, never guess.
Output: Investigation{anchor_service, window:[t0-15m, t1+15m], budget};
writes the Incident node and AFFECTS edge into the session graph.

Zero-LLM path: with LLM_PROVIDER=none (default for Day 1) P0 is fully
deterministic — entity linking is lexical, no LLM call is made. The
optional tiny LLM call (symptom parsing fallback) lands with the agent
loop on Day 2.
"""

import logging
import json
import re
import uuid

from backend.app.engine import linking
from backend.app.engine.budget import BudgetMeter
from backend.app.engine.contracts import (Investigation, InvestigationWindow,
                                          ResolvedAnchor)
from backend.app.graph.client import session_graph
from config import tuning

logger = logging.getLogger("incidentgraph.p0")


class MissingOnsetError(ValueError):
    """The resolved anchor has no usable incident clock."""


class UnresolvedAnchorError(Exception):
    """Raised when the symptom's anchor cannot be resolved — abstain, never guess."""

    def __init__(self, anchor: ResolvedAnchor):
        self.anchor = anchor
        super().__init__(
            f"cannot resolve anchor service from symptom {anchor.query!r}")


def resolve(symptom: str, graph, alert_id: str | None = None, *, gateway=None) -> ResolvedAnchor:
    """Resolve the anchor service from symptom text or an alert id.

    Alert-first: if an alert id is given, it carries the authoritative
    OBSERVED_ON target (the RWR seed). Otherwise link from the symptom text.
    """
    from backend.app.graph.gateway import ReadGateway
    from backend.app.settings import LINKING_SETTINGS
    gateway = gateway or ReadGateway(graph, settings=LINKING_SETTINGS)
    if alert_id:
        rows, overflow = gateway.read(
            "MATCH (a:Alert {id: $aid})-[:OBSERVED_ON]->(s) "
            "RETURN CASE WHEN size(toJSON(s.id)) > $field_chars THEN null ELSE s.id END, labels(s)",
            params={"aid": alert_id}, cap=1)
        if not overflow and len(rows) == 1:
            svc, labels = rows[0]
            if not isinstance(svc, str) or not svc or len(labels) != 1:
                return ResolvedAnchor(query=alert_id, matched=False, candidates=[])
            if labels == ["Service"]:
                return ResolvedAnchor(query=alert_id, matched=True,
                                      anchor_service=svc, candidates=[
                    {"id": svc, "type": "Service", "score": 1.0}])
            # Instance anchors resolve through their service, never arbitrary labels.
            if labels != ['Instance']:
                return ResolvedAnchor(query=alert_id, matched=False, candidates=[])
            svc_row, overflow = gateway.read(
                "MATCH (i:Instance {id: $iid}), (s:Service) WHERE s.id = i.service_id RETURN s.id",
                params={"iid": svc}, cap=1)
            if not overflow and svc_row and svc_row[0][0]:
                return ResolvedAnchor(query=alert_id, matched=True,
                                      anchor_service=svc_row[0][0], candidates=[
                    {"id": svc, "type": "Instance", "score": 1.0}],
                    via_instance=svc)
        logger.warning("alert %s not found or has no OBSERVED_ON target", alert_id)

        return ResolvedAnchor(query=alert_id, matched=False, candidates=[])

    linked = linking.link_entity(symptom, graph)
    return ResolvedAnchor(query=symptom, matched=linked['matched'],
                          anchor_service=linked['anchor_service'],
                          via_instance=linked.get('via_instance'), candidates=linked['nodes'])


def trigger_investigation(symptom: str,
                          alert_id: str | None = None,
                          trigger_ts: int | None = None,
                          session_id: str | None = None,
                          meter: BudgetMeter | None = None) -> Investigation:
    """P0 entry point: resolve anchor, create session graph, write Incident.

    The session graph is a GRAPH.COPY of the demo graph (tenant isolation);
    investigation writes never touch `incidentgraph:demo`.
    """
    name, graph = session_graph(session_id)
    budget = meter or BudgetMeter()

    from backend.app.graph.gateway import ReadGateway
    from backend.app.settings import LINKING_SETTINGS
    gateway = ReadGateway(graph, settings=LINKING_SETTINGS)
    anchor = resolve(symptom, graph, alert_id=alert_id, gateway=gateway)
    if not anchor.matched:
        raise UnresolvedAnchorError(anchor)

    from backend.app.ingest.normalize import to_epoch_ms
    provenance = {}
    time_source = 'explicit_trigger'
    alert_ts = None
    if alert_id:
        rows, overflow = gateway.read('MATCH (a:Alert {id: $aid}) RETURN a.ts',
                                      params={'aid': alert_id}, cap=1)
        if overflow:
            raise MissingOnsetError('ambiguous alert clock')
        alert_ts = rows[0][0] if rows else None
    if trigger_ts is None:
        trigger_ts = alert_ts
        time_source = 'supplied_alert'
        if trigger_ts is None:
            rows, _ = gateway.read(
                'MATCH (a:Alert)-[:OBSERVED_ON]->(target) '
                'WHERE (target:Service AND target.id = $sid) OR '
                '(target:Instance AND target.service_id = $sid) '
                'RETURN max(a.ts)', params={'sid': anchor.anchor_service}, cap=1)
            trigger_ts = rows[0][0] if rows else None
            time_source = 'service_alert'
    if trigger_ts is None:
        raise MissingOnsetError('No authoritative onset for resolved service')
    trigger_ts = to_epoch_ms(trigger_ts)
    if alert_ts is not None:
        provenance = {'alert_ts': to_epoch_ms(alert_ts),
                      'override_delta_ms': trigger_ts - to_epoch_ms(alert_ts)}
    window = InvestigationWindow(t0=max(0, trigger_ts - tuning.WINDOW_PAD_MS),
                                 t1=trigger_ts + tuning.WINDOW_PAD_MS)

    incident_id = f"incident:INC-{uuid.uuid4().hex[:6].upper()}"

    # Write Incident + AFFECTS into the session graph.
    graph.query(
        "MERGE (i:Incident {id: $id}) "
        "ON CREATE SET i.symptom = $symptom, i.start_ts = $ts, "
        "i.status = 'investigating', i.verdict = 'open', "
        "i.anchor_service = $anchor, i.time_source = $time_source, "
        "i.clock_provenance = $clock_provenance",
        params={"id": incident_id, "symptom": symptom, "ts": trigger_ts,
                "anchor": anchor.anchor_service, "time_source": time_source,
                "clock_provenance": json.dumps(provenance, sort_keys=True)},
    )
    graph.query(
        "MATCH (i:Incident {id: $id}), (s:Service {id: $svc}) "
        "MERGE (i)-[:AFFECTS {detected_at: $ts}]->(s)",
        params={"id": incident_id, "svc": anchor.anchor_service,
                "ts": trigger_ts},
    )

    return Investigation(
        incident_id=incident_id,
        session_graph=name,
        symptom=symptom,
        anchor_service=anchor.anchor_service,
        window=window,
        budget=budget.as_dict(),
        anchor_alert=alert_id,
        **{k: v for k, v in dict(onset_ts=trigger_ts, time_source=time_source,
                                clock_provenance=provenance).items()
           if k in Investigation.model_fields},
    )
