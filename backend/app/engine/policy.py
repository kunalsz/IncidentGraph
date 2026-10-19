"""Version 1 deterministic evidence-strength policy (not calibrated probability).

Only service abnormality is verified here; free-text mechanisms remain proposals.
No model confidence, rule, or critique can promote an evidence class.
"""
from __future__ import annotations

import hashlib
import json
import math

from config import tuning
from backend.app.graph.queries import projection, EVIDENCE_FIELDS
from backend.app.graph.gateway import ReadGateway, admit_payload, PayloadAdmissionError

WRAPPER_FIELDS = ('id', 'source_node', 'incident_id', 'ts', 'kind', 'polarity')
RELATION_FIELDS = ('polarity', 'weight')


def _props(value):
    props = dict(value.properties if hasattr(value, 'properties') else value)
    if props.pop('__rejected', False):
        raise PayloadAdmissionError('oversized live evidence property')
    props = {k: v for k, v in props.items() if v is not None}
    admit_payload(props)
    return props


def _live_projection(variable):
    # Internal identity distinguishes duplicate logical IDs with identical props.
    return projection(variable, EVIDENCE_FIELDS)[:-1] + f', __source_identity: id({variable})' + '}'


def _service_projection():
    return ('CASE WHEN target:Instance THEN CASE WHEN size(toJSON(target.service_id)) > $field_chars '
            'THEN null ELSE target.service_id END ELSE CASE WHEN size(toJSON(target.id)) > $field_chars '
            'THEN null ELSE target.id END END')

VERSION = "owned-evidence-v2"


def verify_scope(scope, investigation):
    # Revalidate mutable compatibility objects at every trust boundary.
    type(scope).model_validate(scope.model_dump())
    if (scope.schema_version != 2 or scope.incident_id != investigation.incident_id
            or scope.anchor_service != investigation.anchor_service
            or scope.window != investigation.window):
        raise ValueError("scope does not match investigation")


def proposal_fingerprint(hypothesis):
    return hashlib.sha256(json.dumps([hypothesis.id, hypothesis.candidate,
        hypothesis.branch_id, hypothesis.claim_kind, hypothesis.claim.model_dump() if hypothesis.claim else None, hypothesis.text,
        sorted(set(hypothesis.evidence)), sorted(set(hypothesis.contradicting))],
        sort_keys=True).encode()).hexdigest()


def assessment_matches(assessment, hypothesis, investigation, scope_ref=None):
    return bool(assessment and assessment.schema_version == 2
        and hypothesis.schema_version == 2 and investigation.schema_version == 2
        and assessment.hypothesis_id == hypothesis.id
        and assessment.candidate == hypothesis.candidate
        and assessment.claim_type == hypothesis.claim_kind
        and assessment.evidence_scope == investigation.session_graph
        and assessment.proposal_fingerprint == proposal_fingerprint(hypothesis)
        and assessment.scope_ref is not None
        and assessment.scope_ref.incident_id == investigation.incident_id
        and (scope_ref is None or assessment.scope_ref == scope_ref)
        and hypothesis.scope_ref == assessment.scope_ref
        and set(assessment.source_fingerprints) <= set(hypothesis.evidence))


def finite_number(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


# Failure codes in BOTH status spaces this system reads. OTEL uses
# 0=UNSET, 1=OK, 2=ERROR; gRPC uses 0=OK, 1=CANCELLED, 2..16 = failure codes
# (2 UNKNOWN, 4 DEADLINE_EXCEEDED, 13 INTERNAL, 14 UNAVAILABLE, ...). On OTEL
# data only 0/1/2 occur, so a 2..16 rule coincides with OTEL semantics there;
# on gRPC data it counts the codes that actually mean failure. The RE2 traces
# are gRPC-coded: reading them with an OTEL-only rule hid 6,481 failing spans
# in 19 of 90 cases, including 122 same-trace caller/callee failure pairs.
GRPC_FAILURE_CODES = frozenset(range(2, 17))


def error_status(value):
    """Did this status value report a failure? Codes 2..16, or a named failure.

    0/UNSET/OK and 1/OK/CANCELLED are not failures: 1 is OK in OTEL and an
    ambiguous client-side cancellation in gRPC. Unparseable values are not
    failures either - they are missing data, which this system reports as a gap
    rather than as evidence.
    """
    if value is None or isinstance(value, bool):
        return False
    if isinstance(value, str):
        key = value.strip().lower()
        if key in {"error", "failed", "failure"}:
            return True
        value = key
    # Duck-typed on purpose: values arrive as Python ints, as strings from
    # stored properties, and as numpy scalars straight out of a DataFrame.
    # isinstance(x, int) is False for numpy integers, which silently marked
    # every status as healthy.
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(number):
        return False
    return int(number) in GRPC_FAILURE_CODES


def modality(kind, props):
    if kind == "MetricAnomaly":
        z, gamma = props.get("z"), props.get("gamma")
        if (finite_number(z) and abs(z) >= tuning.NSIGMA_N
                and (gamma is None or finite_number(gamma) and gamma >= tuning.NSIGMA_N)):
            # A trace-derived metric remains the trace modality.
            return "traces" if props.get("modality") in ("trace", "traces") else "metrics"
    if kind == "LogEvent" and (props.get("is_error") is True or
            str(props.get("level", "")).lower() in {"error", "fatal", "major", "critical"}):
        return "logs"
    if kind == "Span" and error_status(props.get("status")):
        return "traces"
    return None


def fingerprint(kind, props, service):
    return hashlib.sha256(json.dumps([kind, props, service], sort_keys=True,
                                    default=str).encode()).hexdigest()


def owned_evidence(graph, hypothesis, investigation, scope):
    """Resolve cited wrappers AND live source/OBSERVED_ON provenance. Never rescue.

    Incident, source timestamp, positive relationship weight/polarity, candidate
    identity and citation membership are independently checked, not model asserted.
    """
    if len(hypothesis.evidence) > 2048:
        raise ValueError('too many hypothesis evidence references')
    rows, overflow = ReadGateway(graph).read(
        "MATCH (h:Hypothesis {id: $hid})-[r:HAS_EVIDENCE]->(e:Evidence), "
        "(o)-[:OBSERVED_ON]->(target) "
        "WHERE o.id = e.source_node AND e.incident_id = $incident "
        "AND h.candidate = $candidate AND e.source_node IN $ids "
        "AND (target:Service OR target:Instance) "
        "RETURN " + projection('e', WRAPPER_FIELDS) + ", " + projection('r', RELATION_FIELDS)
        + ", " + _live_projection('o') + ", [k IN labels(o) WHERE k IN ['MetricAnomaly','LogEvent','Span']], "
        + _service_projection(),
        params={"hid": hypothesis.id, "incident": investigation.incident_id,
                "candidate": hypothesis.candidate, "ids": list(hypothesis.evidence)}, cap=2048)
    if overflow:
        raise ValueError('live evidence rows exceed bounded validation cap')
    services = {s.id for s in scope.services}
    admitted = {n.get("id") for n in scope.nodes}
    admitted_owners = {}
    for edge in scope.edges:
        if edge.get('type') in ('OBSERVED_ON', 'CHANGED', 'DEPLOYED_AT'):
            admitted_owners.setdefault(edge['source'], set()).add(edge['target'])
    live = {}
    for wrapper, relation, source, labels, service in rows:
        props = _props(source)
        live.setdefault(props.get('id'), set()).add((service, json.dumps(props, sort_keys=True)))
    ambiguous = {sid for sid, identities in live.items() if len(identities) != 1
                 or {owner for owner, _ in identities} != admitted_owners.get(sid, set())}
    records = {}
    for wrapper, relation, source, labels, service in rows:
        e, p = _props(wrapper), _props(source)
        relation = _props(relation or {})
        sid = p.get("id")
        ts = p.get("ts")
        if (sid in ambiguous or sid not in hypothesis.evidence or sid not in admitted or service not in services
                or not finite_number(ts)
                or not investigation.window.t0 <= ts <= investigation.window.t1
                or e.get("ts") != ts):
            continue
        kind = next((k for k in ("MetricAnomaly", "LogEvent", "Span") if k in labels), None)
        if not kind or e.get("kind") != kind:
            continue
        positive = (relation.get("polarity") == "support"
                    and finite_number(relation.get("weight")) and relation["weight"] > 0)
        refute = (relation.get("polarity") == "refute" or e.get("polarity") == "refute"
                  or sid in hypothesis.contradicting)
        m = modality(kind, p) if positive and not refute else None
        role = ("refutation" if refute else "local_anomaly" if m and service == hypothesis.candidate
                else "context")
        records[sid] = {"props": p, "service": service, "kind": kind,
                        "modality": m, "role": role, "positive": positive and not refute,
                        "fingerprint": fingerprint(kind, p, service)}
    return records


def propagation_support(records, edge):
    """Only admitted same-trace caller-parent/callee-child error spans qualify.

    Both spans must be positive winner citations and the child interval must fit
    within its parent. Metrics/logs on the callee cannot prove the link.
    """
    spans = [(sid, r) for sid, r in records.items()
             if r["kind"] == "Span" and r["positive"]]
    support = []
    for child_id, child in spans:
        c = child["props"]
        if child["service"] != edge["target"] or not error_status(c.get("status")):
            continue
        for parent_id, parent in spans:
            p = parent["props"]
            if (parent["service"] != edge["source"] or not error_status(p.get("status")) or not c.get("trace_id")
                    or c.get("trace_id") != p.get("trace_id") or not c.get("parent_id")
                    or c["parent_id"] != p.get("span_id")):
                continue
            durations = [p.get("duration_us"), c.get("duration_us")]
            if not all(finite_number(d) and d > 0 for d in durations):
                continue
            if p["ts"] <= c["ts"] and c["ts"] + durations[1] / 1000 <= p["ts"] + durations[0] / 1000:
                support.extend([parent_id, child_id])
    return sorted(set(support))


def claim_applicability(graph, hypothesis, investigation, scope, records, path, onset_ts=None):
    """Verify explicit associations, never infer a cause from temporal proximity."""
    onset_ts = investigation.onset_ts if onset_ts is None else onset_ts
    kind, claim = hypothesis.claim_kind, hypothesis.claim
    result = {"name": "claim_applicability", "verdict": "partial",
              "detail": "structured claim parameters or qualified support unavailable"}
    if kind == "service_abnormality":
        return {**result, "verdict": "not_applicable", "detail": "generic abnormality; no mechanism asserted"}
    if claim is None:
        return result
    if kind == "metric_excursion" and claim.metric and claim.direction:
        matching = [r["props"] for r in records.values() if r["role"] == "local_anomaly"
                    and r["kind"] == "MetricAnomaly"
                    and (r["props"].get("column") or r["props"].get("metric")) == claim.metric
                    and finite_number(r["props"].get("ts"))
                    and finite_number(onset_ts)
                    and abs(r["props"]["ts"] - onset_ts) <= tuning.ONSET_DELTA_MS]
        if matching:
            from backend.app.graph.validators import svar_lite
            check = svar_lite(matching)
            stable = all(p.get("direction") == claim.direction and
                         (p["z"] > 0 if claim.direction == "up" else p["z"] < 0) for p in matching)
            return {**result, "verdict": check["verdict"] if stable else "fail",
                    "detail": "exact cited metric/direction excursion; no causal mechanism established" if stable else "cited series contradicts claimed direction"}
    if kind == "dependency_propagation" and claim.dependency_source and claim.dependency_target:
        edge = next((e for e in path["edges"] if e["source"] == claim.dependency_source
                     and e["target"] == claim.dependency_target and e["target"] == hypothesis.candidate), None)
        support = propagation_support(records, edge) if edge else []
        if (support and finite_number(onset_ts) and
                all(abs(records[sid]["props"]["ts"] - onset_ts) <= tuning.ONSET_DELTA_MS for sid in support)):
            return {**result, "verdict": "pass", "detail": "same-trace parent/child error association on scoped dependency; causal direction not established"}
    if kind == "change_association" and claim.change_id and claim.change_id in hypothesis.evidence:
        admitted = {n.get("id") for n in scope.nodes}
        if claim.change_id not in admitted:
            return result
        rows, overflow = ReadGateway(graph).read(
            "MATCH (h:Hypothesis {id: $hid})-[r:HAS_EVIDENCE]->(e:Evidence), "
            "(c)-[link]->(target) WHERE e.source_node = c.id AND c.id = $change "
            "AND e.incident_id = $incident AND h.candidate = $candidate "
            "AND (type(link) = 'CHANGED' OR type(link) = 'DEPLOYED_AT') "
            "AND (target:Service OR target:Instance) "
            "RETURN " + projection('e', WRAPPER_FIELDS) + ", " + projection('r', RELATION_FIELDS)
            + ", " + _live_projection('c') + ", [k IN labels(c) WHERE k IN ['ConfigChange','Deployment']], "
            + _service_projection(),
            params={"hid": hypothesis.id, "change": claim.change_id,
                    "incident": investigation.incident_id, "candidate": hypothesis.candidate}, cap=2)
        if overflow:
            return result
        owners = {e['target'] for e in scope.edges if e.get('source') == claim.change_id
                  and e.get('type') in ('CHANGED', 'DEPLOYED_AT')}
        identities = {(service, json.dumps(_props(source), sort_keys=True))
                      for _, _, source, _, service in rows}
        if len(identities) != 1 or {s for s, _ in identities} != owners:
            return result
        for wrapper, relation, source, labels, service in rows:
            p, e, relation = _props(source), _props(wrapper), _props(relation)
            ts = p.get("ts")
            if (service == hypothesis.candidate and finite_number(ts)
                    and finite_number(onset_ts)
                    and investigation.window.t0 <= ts <= onset_ts
                    and e.get("ts") == ts and relation.get("polarity") == "support"
                    and e.get("polarity") != "refute" and claim.change_id not in hypothesis.contradicting
                    and finite_number(relation.get("weight")) and relation["weight"] > 0
                    and any(label in labels for label in ("ConfigChange", "Deployment"))):
                label = next(k for k in ("ConfigChange", "Deployment") if k in labels)
                if e.get("kind") != label:
                    continue
                records[claim.change_id] = {"props": p, "service": service, "kind": label,
                    "modality": None, "role": "change_association", "positive": True,
                    "fingerprint": fingerprint(label, p, service)}
                return {**result, "verdict": "pass", "detail": "cited live change targets candidate and precedes onset; temporal association only, not causation"}
    return result


def evidence_strength(assessment):
    if assessment is None:
        return 0.4, "inference_only", False
    if assessment.claim_type != "service_abnormality" and not any(
            c.name == "claim_applicability" and c.verdict == "pass" for c in assessment.checks):
        return .4, "inference_only", False
    if assessment.verdict == "fail" or any(c.verdict == "fail" for c in assessment.checks):
        return 0.0, "inference_only", False
    local = set(assessment.evidence_roles.get("local_anomaly", []))
    n = len(set(assessment.modalities)) if local and local <= set(assessment.source_fingerprints) else 0
    score, rule = ((.85, "cross_modal_consensus") if n >= 2 else
                   (.7, "single_modality") if n == 1 else (.4, "inference_only"))
    if assessment.verdict == "partial":
        score = min(score, .7)
    if not assessment.qualified_timing:
        score = min(score, .4)
    if assessment.confidence_cap is not None:
        score = min(score, assessment.confidence_cap)
    eligible = bool(n and assessment.qualified_timing and assessment.plausible_path
                    and score >= tuning.CONFIDENCE_GATE_THETA)
    return score, rule, eligible
