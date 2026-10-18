"""P4: hypothesis-owned deterministic service-abnormality checks.

Candidate summaries are compatibility views only. Generic anomaly validation
never establishes the proposed free-text mechanism or causal propagation.
"""
from __future__ import annotations

import json

from config import tuning
from backend.app.agent.llm import LLMClient, LLMError, load_prompt
from backend.app.engine.budget import BudgetMeter, BudgetExceeded
from backend.app.engine.contracts import (CheckResult, EvidencePath, HypothesisValidation,
    InvestResult, Investigation, RankResult, ScopeResult, ValidatedCandidate, ValidateResult)
from backend.app.engine import policy
from backend.app.graph import pathfind, validators


def _service_levels(scope):
    return {node["id"]: node["properties"]["level"] for node in scope.nodes
            if node.get("type") == "Service" and node.get("properties", {}).get("level") is not None
            and node.get("properties", {}).get("level_convention") == "dependency_depth_v1"}


def _callers_closure(candidate, edges):
    """Callers connected to the candidate by validated propagation links only."""
    callers = {}
    for edge in edges:
        if edge["type"] == "DEPENDS_ON" and edge.get("support"):
            callers.setdefault(edge["target"], set()).add(edge["source"])
    closure, frontier = set(), [candidate]
    while frontier:
        for caller in callers.get(frontier.pop(), ()):
            if caller not in closure:
                closure.add(caller)
                frontier.append(caller)
    return closure


def assess_hypothesis(graph, investigation, scope, hypothesis, onset_ts):
    """Read only, reusable by P6 to recheck live source/relationship integrity."""
    if hypothesis.scope_ref != scope.scope_ref or hypothesis.schema_version != 2:
        return HypothesisValidation(hypothesis_id=hypothesis.id, candidate=hypothesis.candidate,
            scope_ref=scope.scope_ref, verdict="partial", evidence_scope=investigation.session_graph,
            proposal_fingerprint=policy.proposal_fingerprint(hypothesis),
            checks=[CheckResult(name="scope_membership", verdict="partial", detail="proposal lacks matching v2 scope")])
    records = policy.owned_evidence(graph, hypothesis, investigation, scope)
    local = {sid: r for sid, r in records.items() if r["role"] == "local_anomaly"}
    metrics = [r["props"] for r in local.values() if r["kind"] == "MetricAnomaly"]
    if hypothesis.claim_kind == "metric_excursion":
        metrics = [p for p in metrics if hypothesis.claim and
                   (p.get("column") or p.get("metric")) == hypothesis.claim.metric]
    levels = _service_levels(scope)
    candidate, anchor = hypothesis.candidate, investigation.anchor_service
    level = (validators.level_veto(levels[candidate], levels[anchor])
             if candidate in levels and anchor in levels else
             {"name": "level_veto", "verdict": "partial", "detail": "authoritative level unavailable"})
    timing = validators.onset_alignment([r["props"]["ts"] for r in local.values()], onset_ts)
    # Missing evidence is a gap, never manufactured refutation.
    if not local:
        timing = {"name": "onset_alignment", "verdict": "partial", "detail": "no qualified local anomaly timing"}
    checks = [level, timing,
        validators.svar_lite(metrics) if metrics else
        {"name": "svar_lite", "verdict": "not_applicable", "detail": "no cited decisive metric"},
        {"name": "symptom_explanation", "verdict": "pass" if local and candidate != anchor else "partial",
         "detail": "cited local abnormality only; mechanism remains a proposal" if local else "missing decisive local evidence"},
        validators.no_evidence_cap(any(r["kind"] == "LogEvent" for r in local.values()),
                                   bool(metrics), any(r["kind"] == "Span" for r in local.values()))]
    path = pathfind.find_evidence_path(graph, anchor, candidate, investigation.window,
                                      approved_edges=scope.edges,
                                      approved_services={s.id for s in scope.services})
    roles = {role: sorted(sid for sid, r in records.items() if r["role"] == role)
             for role in ("context", "local_anomaly", "refutation")}
    roles["propagation"] = []
    for edge in path["edges"]:
        edge["support"] = policy.propagation_support(records, edge)
        roles["propagation"].extend(edge["support"])
    roles["propagation"] = sorted(set(roles["propagation"]))
    # Context annotations must not masquerade as propagation support.
    path["nodes"] = [node for node in path["nodes"] if node["role"] != "corroborating"]
    path["nodes"].extend({"id": sid, "role": "corroborating"} for sid in roles["propagation"])
    checks.append({"name": "scoped_path", "verdict": "pass" if path["found"] else "partial",
                   "detail": "scope-approved structural plausibility; not propagation proof" if path["found"] else "bounded path unavailable"})
    # A chosen explanation has its own service DAG; competing hypotheses are not unioned.
    causal = []
    cycle = {"name": "cycle_rejection", "verdict": "pass", "detail": "hypothesis-local service DAG"}
    for edge in reversed(path["edges"]):
        proposed = (edge["target"], edge["source"])
        cycle = validators.cycle_rejection(causal, proposed)
        if cycle["verdict"] == "fail":
            break
        causal.append(proposed)
    checks.append(cycle)
    checks.append(policy.claim_applicability(graph, hypothesis, investigation, scope, records, path, onset_ts))
    roles["change_association"] = sorted(sid for sid, r in records.items() if r["role"] == "change_association")
    return HypothesisValidation(
        hypothesis_id=hypothesis.id, candidate=candidate, scope_ref=scope.scope_ref, claim_type=hypothesis.claim_kind,
        proposal_fingerprint=policy.proposal_fingerprint(hypothesis),
        checks=[CheckResult(**{k: v for k, v in c.items() if k in CheckResult.model_fields}) for c in checks], evidence_path=EvidencePath(**path),
        verdict=validators.aggregate_verdict(checks), confidence_cap=validators.confidence_cap_from_checks(checks),
        qualified_timing=bool(local and timing["verdict"] == "pass"),
        plausible_path=path["found"], modalities=sorted({r["modality"] for r in local.values()
            if abs(r["props"]["ts"] - onset_ts) <= tuning.ONSET_DELTA_MS}),
        evidence_roles=roles, source_fingerprints={sid: r["fingerprint"] for sid, r in records.items()},
        evidence_scope=investigation.session_graph)


def _persist_refute(graph, hypothesis_id, candidate, incident_id, checks):
    evidence_id = f"ev:refute:{incident_id}:{hypothesis_id}"
    graph.query(
        "MERGE (e:Evidence {id: $id}) SET e.incident_id = $incident, "
        "e.kind = 'ValidatorRefute', e.source_node = $candidate, e.polarity = 'refute', "
        "e.weight = 1.0, e.stat = $stat "
        "WITH e MATCH (h:Hypothesis {id: $hid}) "
        "MERGE (h)-[r:HAS_EVIDENCE]->(e) SET r.polarity = 'refute', r.weight = 1.0, r.role = 'refutation'",
        params={"id": evidence_id, "incident": incident_id, "candidate": candidate,
                "hid": hypothesis_id, "stat": json.dumps([c for c in checks if c["verdict"] == "fail"])})
    return evidence_id


def validate_investigation(investigation: Investigation, scope: ScopeResult,
                           ranking: RankResult, invest: InvestResult,
                           meter: BudgetMeter | None = None,
                           llm: LLMClient | None = None) -> ValidateResult:
    if not investigation.session_graph.startswith("incidentgraph:session:"):
        raise ValueError("P4 requires an isolated session graph")
    from backend.app.graph.client import graph_for
    graph = graph_for(investigation.session_graph)
    meter = meter or BudgetMeter()
    # Critique is strictly opt-in: deterministic P4 needs no provider configuration.
    if any(r.incident_id != investigation.incident_id or r.schema_version != 2 for r in (scope, ranking, invest)):
        raise ValueError("phase results do not match investigation")
    policy.verify_scope(scope, investigation)
    if ranking.scope_ref != scope.scope_ref:
        raise ValueError("ranking does not match scope")
    if invest.scope_ref != scope.scope_ref:
        raise ValueError("investigation proposals do not match scope")
    allowed = {s.id for s in scope.services} & {c.id for c in ranking.candidates}
    onset_ts = investigation.onset_ts
    if onset_ts is None:
        onset_rows = graph.query("MATCH (i:Incident {id: $id}) RETURN i.start_ts",
                                params={"id": investigation.incident_id}).result_set
        onset_ts = onset_rows[0][0] if onset_rows else None
    if not policy.finite_number(onset_ts) or onset_ts < 0:
        raise ValueError("validation requires authoritative incident onset")
    assessments = {}
    summaries = {}
    for hypothesis in invest.hypotheses:
        if hypothesis.candidate not in allowed:
            continue
        if hypothesis.id in assessments:
            raise ValueError("duplicate hypothesis identity")
        assessment = assess_hypothesis(graph, investigation, scope, hypothesis, onset_ts)
        other = {h.candidate for h in invest.hypotheses if h.candidate in allowed and h.candidate != hypothesis.candidate}
        path = assessment.evidence_path
        callers = _callers_closure(hypothesis.candidate, path.edges) if path and path.found else set()
        explained = other & callers
        # No propagation proof is a neutral zero bonus, never a validator veto.
        # Keep the established one/two-branch mapping only for supported chains.
        assessment.chain_coherence = validators.chain_coherence(len(explained), len(other)) if explained else 0.0
        if not assessments and llm is not None:
            try:
                response = llm.complete("critique", load_prompt("critique"),
                    json.dumps({"claim": {"candidate": hypothesis.candidate, "anchor": investigation.anchor_service},
                                "checks": [c.model_dump() for c in assessment.checks]}),
                    tier="default", max_tokens=600)
                assessment.critique = response.json()
            except Exception as exc:
                assessment.critique = {"reason": f"critique unavailable: {type(exc).__name__}"}
            # Critique is review metadata, not a deterministic validator or cap.
        if assessment.verdict == "fail":
            assessment.contradictions = [_persist_refute(graph, hypothesis.id, hypothesis.candidate,
                investigation.incident_id, [c.model_dump() for c in assessment.checks])]
        for role, ids in assessment.evidence_roles.items():
            if not ids:
                continue
            graph.query("MATCH (h:Hypothesis {id: $hid})-[r:HAS_EVIDENCE]->(e:Evidence) "
                        "WHERE e.source_node IN $ids SET r.role = $role, r.rule = $rule",
                        params={"hid": hypothesis.id, "ids": ids, "role": role, "rule": policy.VERSION})
        score, _, eligible = policy.evidence_strength(assessment)
        if eligible and hypothesis.candidate != investigation.anchor_service:
            graph.query("MATCH (h:Hypothesis {id: $hid}), (s:Service {id: $anchor}) "
                        "MERGE (h)-[c:ASSOCIATED_WITH]->(s) SET c.confidence = $score, "
                        "c.provenance = 'rule', c.volatile = true, c.accepted = false, "
                        "c.incident_id = $incident, c.validation_version = $version, c.scope_fingerprint = $scope, c.candidate = $candidate",
                        params={"hid": hypothesis.id, "anchor": investigation.anchor_service,
                                "score": score, "incident": investigation.incident_id, "version": policy.VERSION,
                                "scope": scope.scope_ref.fingerprint, "candidate": hypothesis.candidate})
        assessments[hypothesis.id] = assessment
        summaries.setdefault(hypothesis.candidate, ValidatedCandidate(**{
            key: value for key, value in assessment.model_dump().items()
            if key in ValidatedCandidate.model_fields}))
    result = ValidateResult(incident_id=investigation.incident_id, scope_ref=scope.scope_ref,
        candidates=list(summaries.values()), n_candidates=len(summaries), hypotheses=assessments,
        config={"policy": policy.VERSION, "onset_ts": onset_ts,
                "claim_policy": "structured claim-specific association; causality unverified"})
    from backend.app.engine.contracts import dump_handoff
    graph.query("MERGE (a:Artifact {id: $id}) SET a.kind = 'stage-view', "
                "a.stage = 'validate', a.version = 2, a.incident_id = $incident, a.json = $payload",
                params={"id": f"artifact:{investigation.incident_id}:validate:0",
                        "incident": investigation.incident_id, "payload": dump_handoff("validate", result)})
    return result
