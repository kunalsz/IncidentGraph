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
import re
import uuid

from backend.app.engine import linking
from backend.app.engine.budget import BudgetMeter
from backend.app.engine.contracts import (Investigation, InvestigationWindow,
                                          ResolvedAnchor)
from backend.app.graph.client import session_graph
from config import tuning

logger = logging.getLogger("incidentgraph.p0")


class UnresolvedAnchorError(Exception):
    """Raised when the symptom's anchor cannot be resolved — abstain, never guess."""

    def __init__(self, anchor: ResolvedAnchor):
        self.anchor = anchor
        super().__init__(
            f"cannot resolve anchor service from symptom {anchor.query!r}")


def resolve(symptom: str, graph, alert_id: str | None = None) -> ResolvedAnchor:
    """Resolve the anchor service from symptom text or an alert id.

    Alert-first: if an alert id is given, it carries the authoritative
    OBSERVED_ON target (the RWR seed). Otherwise link from the symptom text.
    """
    if alert_id:
        rows = graph.query(
            "MATCH (a:Alert {id: $aid})-[:OBSERVED_ON]->(s) "
            "RETURN s.id, labels(s)",
            params={"aid": alert_id},
        ).result_set
        if rows:
            svc, labels = rows[0]
            if labels and labels[0] == "Service":
                return ResolvedAnchor(query=alert_id, matched=True,
                                      anchor_service=svc, candidates=[
                    {"id": svc, "type": "Service", "score": 1.0}])
            # Instance anchors resolve through their service
            svc_row = graph.query(
                "MATCH (i:Instance {id: $iid}) RETURN i.service_id",
                params={"iid": svc}).result_set
            if svc_row and svc_row[0][0]:
                return ResolvedAnchor(query=alert_id, matched=True,
                                      anchor_service=svc_row[0][0], candidates=[
                    {"id": svc, "type": "Instance", "score": 1.0}],
                    via_instance=svc)
        logger.warning("alert %s not found or has no OBSERVED_ON target", alert_id)

    # Symptom text: try the strongest token n-grams against the alias table.
    tokens = re.findall(r"[A-Za-z0-9]+", symptom or "")
    for n in (2, 1):
        for i in range(len(tokens) - n + 1):
            phrase = " ".join(tokens[i:i + n])
            linked = linking.link_entity(phrase, graph)
            if linked["matched"]:
                return ResolvedAnchor(
                    query=phrase, matched=True,
                    anchor_service=linked["anchor_service"],
                    via_instance=linked.get("via_instance"),
                    candidates=linked["nodes"])
    return ResolvedAnchor(query=symptom, matched=False, candidates=[])


def trigger_investigation(symptom: str,
                          alert_id: str | None = None,
                          trigger_ts: int | None = None,
                          session_id: str | None = None) -> Investigation:
    """P0 entry point: resolve anchor, create session graph, write Incident.

    The session graph is a GRAPH.COPY of the demo graph (tenant isolation);
    investigation writes never touch `incidentgraph:demo`.
    """
    name, graph = session_graph(session_id)
    budget = BudgetMeter()

    anchor = resolve(symptom, graph, alert_id=alert_id)
    if not anchor.matched:
        raise UnresolvedAnchorError(anchor)

    # Window: [t_trigger - 15m, t_trigger + 15m] (Δτ = ±15 min, documented).
    if trigger_ts is None:
        if alert_id:
            row = graph.query("MATCH (a:Alert {id: $aid}) RETURN a.ts",
                              params={"aid": alert_id}).result_set
            trigger_ts = row[0][0] if row else None
        if trigger_ts is None:
            # Latest Alert in the graph as the incident clock (fixture world)
            row = graph.query(
                "MATCH (a:Alert) RETURN a.ts ORDER BY a.ts DESC LIMIT 1"
            ).result_set
            trigger_ts = row[0][0] if row else 0
    window = InvestigationWindow(
        t0=trigger_ts - tuning.WINDOW_PAD_MS,
        t1=trigger_ts + tuning.WINDOW_PAD_MS,
    )

    incident_id = f"incident:INC-{uuid.uuid4().hex[:6].upper()}"

    # Write Incident + AFFECTS into the session graph.
    graph.query(
        "MERGE (i:Incident {id: $id}) "
        "ON CREATE SET i.symptom = $symptom, i.start_ts = $ts, "
        "i.status = 'investigating', i.verdict = 'open', "
        "i.anchor_service = $anchor",
        params={"id": incident_id, "symptom": symptom, "ts": trigger_ts,
                "anchor": anchor.anchor_service},
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
    )
