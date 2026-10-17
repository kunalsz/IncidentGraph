"""P6: deterministic facts and approved references, with fail-closed finalization.

No model-written factual prose or actions enter the authoritative report. Optional
LLM composition is deliberately not needed: a deterministic catalog is always
available, including when provider, schema, budget or persistence fails.
"""
from __future__ import annotations

import hashlib
import json

from config import tuning
from backend.app.engine import policy
from backend.app.engine.budget import BudgetMeter, BudgetExceeded
from backend.app.engine.contracts import ReportResult, dump_handoff
from backend.app.engine.p4_validate import assess_hypothesis
from backend.app.report import render, validator


def _per_link_support(records, path):
    return bool(path and path.found and path.edges and all(
        policy.propagation_support(records, edge) for edge in path.edges))


def _reads_available(meter):
    try:
        meter.check()
    except BudgetExceeded:
        return False
    state = meter.as_dict()
    caps = state.get("budget", {})
    # At a hard boundary no final retrieval is authorized. Finalization itself
    # is free of provider calls and can always render the cached audit state.
    return not any(state.get(used, 0) >= caps[cap]
                   for used, cap in (("llm_calls", "max_llm_calls"),
                                     ("tokens_total", "max_tokens"),
                                     ("nodes_visited", "max_nodes_visited"),
                                     ("hypotheses", "max_hypotheses"))
                   if cap in caps)


def _live_assessment(graph, investigation, scope, hypothesis, previous, onset):
    if not policy.assessment_matches(previous, hypothesis, investigation, scope.scope_ref):
        return None, {}
    current = assess_hypothesis(graph, investigation, scope, hypothesis, onset)
    records = policy.owned_evidence(graph, hypothesis, investigation, scope)
    applicability = policy.claim_applicability(graph, hypothesis, investigation, scope, records,
        current.evidence_path.model_dump() if current.evidence_path else {"edges": []}, onset)
    if hypothesis.claim_kind != "service_abnormality" and applicability["verdict"] != "pass":
        return None, {}
    unchanged = {sid for sid, record in records.items()
                 if previous.source_fingerprints.get(sid) == record["fingerprint"]}
    records = {sid: record for sid, record in records.items() if sid in unchanged}
    current.evidence_roles = {role: [sid for sid in ids if sid in unchanged]
                              for role, ids in current.evidence_roles.items()}
    local = [record for record in records.values() if record["role"] == "local_anomaly"
             and abs(record["props"]["ts"] - onset) <= tuning.ONSET_DELTA_MS]
    current.modalities = sorted({r["modality"] for r in local})
    current.qualified_timing = bool(local)
    current.source_fingerprints = {sid: r["fingerprint"] for sid, r in records.items()}
    if current.evidence_path:
        for edge in current.evidence_path.edges:
            edge["support"] = policy.propagation_support(records, edge)
        current.evidence_path.nodes = [n for n in current.evidence_path.nodes
            if n["role"] != "corroborating"] + [{"id": sid, "role": "corroborating"}
            for sid in sorted({sid for edge in current.evidence_path.edges for sid in edge["support"]})]
    if set(previous.source_fingerprints) != unchanged:
        return None, {}
    if not policy.evidence_strength(previous)[2] or not policy.evidence_strength(current)[2]:
        return None, {}
    return current, records


def _approved_closure(closure):
    """Only explicitly approved retrieved runbooks may supply operational steps."""
    runbooks = []
    for entry in closure.get("runbooks", []):
        if entry.get("approved") is not True and entry.get("approval_status") != "approved":
            continue
        if not entry.get("id"):
            continue
        steps = [dict(step) for step in entry.get("steps", []) if isinstance(step, dict)
                 and isinstance(step.get("action"), str)]
        runbooks.append({**entry, "steps": steps})
    return {"runbooks": runbooks}, closure.get("team") or {}


def _catalog(investigation, gated, candidate, assessment, records, owner, gaps):
    facts = [{"id": "fact:symptom", "kind": "symptom", "text": investigation.symptom,
              "evidence_ids": [], "values": {"window": investigation.window.model_dump()}},
             {"id": "fact:decision", "kind": "decision", "text":
              (f"Verified service abnormality on {candidate}; mechanism is not established."
               if candidate else "No candidate has sufficient verified evidence for diagnosis."),
              "evidence_ids": [], "values": {"gate": gated}}]
    if assessment:
        for sid in sorted(assessment.evidence_roles.get("local_anomaly", [])):
            record = records[sid]
            facts.append({"id": "fact:anomaly:" + hashlib.sha256(sid.encode()).hexdigest()[:16],
                          "kind": "anomaly", "text": f"{record['kind']} anomaly observed on {record['service']} at {record['props']['ts']}.",
                          "evidence_ids": [sid], "values": {"service": record["service"],
                          "kind": record["kind"], "ts": record["props"]["ts"],
                          **{k: record["props"][k] for k in ("z", "gamma", "direction", "metric", "status", "level") if k in record["props"]}}})
        for check in assessment.checks:
            facts.append({"id": f"fact:check:{check.name}", "kind": "check",
                          "text": f"{check.name}: {check.verdict}; {check.detail}",
                          "evidence_ids": [], "values": check.model_dump()})
        if assessment.evidence_path:
            for index, edge in enumerate(assessment.evidence_path.edges):
                facts.append({"id": f"fact:path:{index}", "kind": "path",
                              "text": f"Scoped dependency {edge['source']} to {edge['target']}; propagation {'supported' if edge['support'] else 'not verified'}.",
                              "evidence_ids": edge["support"], "values": edge})
    if owner:
        facts.append({"id": "fact:owner", "kind": "owner", "text": f"Retrieved owner: {owner.get('name', 'unspecified')}.", "values": owner, "evidence_ids": []})
    for index, gap in enumerate(gaps):
        facts.append({"id": f"fact:gap:{index}", "kind": "gap", "text": gap["suggestion"],
                      "values": gap, "evidence_ids": []})
    return facts


def report_investigation(investigation, scope, ranking, invest, validated, fused,
                         meter=None, llm=None, *, final_reads_allowed=True):
    """Always return a deterministic report; graph I/O is best effort and bounded.

    Runner must pass final_reads_allowed=False after deadline/resource exhaustion.
    That mode opens no graph handle, performs no reads/writes and calls no provider.
    """
    if not investigation.session_graph.startswith("incidentgraph:session:"):
        raise ValueError("P6 requires an isolated session graph")
    if any(r.incident_id != investigation.incident_id for r in (scope, ranking, invest, validated, fused)):
        raise ValueError("phase results do not match investigation")
    meter = meter or BudgetMeter()
    reads = bool(final_reads_allowed and _reads_available(meter))
    trusted = all(r.schema_version == 2 for r in (investigation, scope, ranking, invest, validated, fused))
    trusted = trusted and validated.scope_ref == scope.scope_ref and invest.scope_ref == scope.scope_ref
    graph, failures, records, current, winner = None, [], {}, None, None
    if not reads:
        failures.append("final retrieval not checked: resource budget exhausted or disabled")
    if reads and trusted:
        try:
            policy.verify_scope(scope, investigation)
            if ranking.scope_ref != scope.scope_ref or fused.scope_ref != scope.scope_ref:
                raise ValueError("phase scope mismatch")
            from backend.app.graph.client import graph_for
            graph = graph_for(investigation.session_graph)
            allowed = {s.id for s in scope.services} & {c.id for c in ranking.candidates}
            winner = next((h for h in fused.fused if h.id == fused.winner
                and h.candidate == fused.winner_candidate and h.candidate in allowed
                and h.status in ("validated", "partial")), None)
            proposal = next((h for h in invest.hypotheses if winner and h.id == winner.id
                             and h.candidate == winner.candidate), None)
            if proposal:
                previous = validated.hypotheses.get(winner.id)
                onset = validated.config.get("onset_ts", (investigation.window.t0 + investigation.window.t1) // 2)
                current, records = _live_assessment(graph, investigation, scope, proposal, previous, onset)
            if current is None:
                winner = None
        except Exception as exc:
            # Resource/client failures never authorize a stale diagnosis. Do not
            # expose provider messages/credentials or raw evidence as prose.
            failures.append(f"live validation unavailable: {type(exc).__name__}")
            winner, current, records = None, None, {}
    confidence, rule, _ = policy.evidence_strength(current)
    if winner:
        confidence = min(confidence, policy.evidence_strength(validated.hypotheses[winner.id])[0])
    else:
        confidence, rule = 0.0, "inference_only"
    gated = validator.gate(has_winner=winner is not None, confidence=confidence,
        coherence=current.chain_coherence if current else 0,
        validator_all_pass=bool(current and current.verdict == "pass"),
        per_link_support=_per_link_support(records, current.evidence_path if current else None),
        validator_verdict=current.verdict if current else None)
    candidate = winner.candidate if winner else None
    runbook, owner = {"runbooks": []}, {}
    if candidate and reads and graph is not None and _reads_available(meter):
        try:
            from backend.app.agent.tools import t_get_runbook_and_owner
            from backend.app.agent.tools import DiagnosticContext
            context = DiagnosticContext(investigation, scope, validated.config["onset_ts"])
            closure = t_get_runbook_and_owner(graph, service_id=candidate, context=context)
            runbook, owner = _approved_closure(closure)
        except Exception as exc:
            failures.append(f"approved closure unavailable: {type(exc).__name__}")
    suggestion = "Collect missing candidate-owned anomaly and compatible trace evidence around the incident onset; request human review."
    gaps = [] if candidate else [{"id": f"gap:{investigation.incident_id}:decisive_evidence",
                                 "missing_kind": "decisive_evidence", "suggestion": suggestion,
                                 "reasons": gated["reasons"]}]
    facts = _catalog(investigation, gated, candidate, current, records, owner, gaps)
    citations = {sid: index for index, sid in enumerate(sorted({sid for f in facts for sid in f["evidence_ids"]}), 1)}
    action_ref = {"kind": "evidence_request", "id": "action:request-evidence", "human_review_required": True}
    next_action = suggestion
    if candidate:
        for entry in runbook["runbooks"]:
            if entry["steps"]:
                step = entry["steps"][0]
                action_ref = {"kind": "runbook_step", "runbook_id": entry["id"],
                              "step": step.get("step", 1), "human_review_required": True}
                next_action = f"Human review required: {step['action']}"
                break
    facts.append({"id": "fact:action", "kind": "action", "text": next_action,
                  "evidence_ids": [], "values": action_ref})
    not_checked = [{"node_id": g.get("node_id"), "reason": g.get("kind", g.get("missing_kind")),
                    "suggestion": g.get("suggestion")} for g in scope.evidence_gaps]
    not_checked.extend({"reason": reason} for reason in failures)
    if scope.truncated:
        not_checked.append({"reason": "scope_truncation"})
    if invest.truncated:
        not_checked.append({"reason": "investigation_budget_truncated"})
    # Bound final alternative reads independently of hypothesis count. Each
    # displayed service must pass the same live ownership check as the winner.
    alternatives, seen = [], {candidate} if candidate else set()
    if candidate and graph is not None:
        attempted = 0
        for item in fused.fused:
            if attempted >= 3 or not _reads_available(meter):
                break
            if item.id not in fused.alternatives or item.candidate in seen:
                continue
            proposal = next((h for h in invest.hypotheses if h.id == item.id and h.candidate == item.candidate), None)
            if proposal is None:
                continue
            attempted += 1
            try:
                assessment, _ = _live_assessment(graph, investigation, scope, proposal,
                    validated.hypotheses.get(item.id), validated.config.get("onset_ts", (investigation.window.t0 + investigation.window.t1) // 2))
                if assessment:
                    seen.add(item.candidate)
                    alternatives.append({"candidate": item.candidate, "hypothesis_id": item.id,
                        "confidence": policy.evidence_strength(assessment)[0], "status": assessment.verdict,
                        "reason": "independently verified service abnormality"})
            except Exception as exc:
                not_checked.append({"reason": f"alternative validation unavailable: {type(exc).__name__}"})
    ruled_out = [{"id": h.candidate, "reason": "; ".join(c.detail for c in h.checks if c.verdict == "fail")}
                 for h in validated.hypotheses.values() if h.verdict == "fail"]
    report = ReportResult(incident_id=investigation.incident_id, scope_ref=scope.scope_ref,
        verdict=gated["verdict"], tier=gated["tier"], confidence=confidence,
        confidence_rule=rule, winner_candidate=candidate,
        summary=[facts[1]["text"]], next_action=next_action,
        evidence_path=current.evidence_path if current else None, facts=facts,
        citations=citations, action_ref=action_ref, runbook=runbook, owner=owner, alternatives=alternatives,
        ruled_out=ruled_out, not_checked=not_checked, budget=meter.as_dict(),
        config={"gate": gated, "composition_payload": {"gate": gated, "fact_ids": [f["id"] for f in facts], "action_ref": action_ref},
                "gaps": gaps, "policy_version": policy.VERSION, "calibrated": False,
                "composition": "deterministic_fact_catalog", "persistence": "not_attempted",
                "final_reads_allowed": reads, "presentation": fused.presentation})
    report.report_version = hashlib.sha256(report.model_dump_json().encode()).hexdigest()[:24]
    report.markdown = render.render_markdown(report)
    if reads and graph is not None and _reads_available(meter):
        try:
            for gap in gaps:
                graph.query("MERGE (g:EvidenceGap {id: $id}) SET g.incident_id = $incident, "
                            "g.node_id = $anchor, g.missing_kind = 'decisive_evidence', "
                            "g.suggestion = $suggestion, g.reasons = $reasons",
                            params={"id": gap["id"], "incident": investigation.incident_id,
                                    "anchor": investigation.anchor_service, "suggestion": gap["suggestion"],
                                    "reasons": gap["reasons"]})
            report.config["persistence"] = "persisted"
            report.markdown = render.render_markdown(report)
            graph.query("MERGE (a:Artifact {id: $id}) SET a.kind = 'report', a.stage = 'report', "
                "a.version = 2, a.incident_id = $incident, a.json = $payload, a.markdown = $markdown",
                params={"id": f"artifact:{investigation.incident_id}:report:v2:{report.report_version}",
                        "incident": investigation.incident_id, "payload": dump_handoff("report", report), "markdown": report.markdown})
            graph.query("MATCH (i:Incident {id: $incident}) SET i.diagnostic_state = $state, "
                        "i.diagnostic_report_version = $version",
                        params={"incident": investigation.incident_id, "state": report.verdict,
                                "version": report.report_version})
            if winner:
                graph.query("MATCH (h:Hypothesis {id: $hid}) SET h.final_decision = 'eligible', "
                            "h.final_report_version = $version "
                            "WITH h OPTIONAL MATCH (h)-[c:ASSOCIATED_WITH]->() "
                            "SET c.accepted = true, c.report_version = $version",
                            params={"hid": winner.id, "version": report.report_version})
        except Exception as exc:
            report.config["persistence"] = "failed"
            report.not_checked.append({"reason": f"report persistence failed: {type(exc).__name__}"})
            report.markdown = render.render_markdown(report)
    return report
