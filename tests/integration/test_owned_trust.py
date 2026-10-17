"""Bounded trust milestone: actual graph provenance, no provider assertions."""
import uuid

import pytest

from backend.app.agent.llm import LLMClient, MockProvider
from backend.app.engine.budget import BudgetMeter
from backend.app.engine.contracts import (HypothesisRecord, InvestResult, Investigation,
    InvestigationWindow, ScopeResult, ScopeService)
from backend.app.engine.p4_validate import validate_investigation
from backend.app.engine.p5_fuse import fuse_investigation
from backend.app.engine.p6_report import report_investigation
from backend.app.graph.client import client, graph_for
from tests.unit.test_p5_fuse import _ranking


@pytest.fixture
def world():
    name = f"incidentgraph:session:trust-{uuid.uuid4().hex}"
    graph = graph_for(name)
    inv = Investigation(incident_id="i", session_graph=name, symptom="errors",
        anchor_service="service:s0", window=InvestigationWindow(t0=0, t1=1000000), budget={})
    graph.query("CREATE (:Service {id:'service:s0', level:0}), "
                "(:Service {id:'service:s1', level:1}), (:Service {id:'service:s2', level:1}), "
                "(:Incident {id:'i', start_ts:500000})")
    for sid in ("service:s1", "service:s2"):
        graph.query("MATCH (a:Service {id:'service:s0'}), (b:Service {id:$sid}) "
                    "CREATE (a)-[:DEPENDS_ON]->(b)", params={"sid": sid})
    scope = ScopeResult(incident_id="i", anchor_service=inv.anchor_service, window=inv.window,
        services=[ScopeService(id=f"service:s{i}") for i in range(3)],
        nodes=[{"id": f"service:s{i}", "type": "Service", "properties": {"level": min(i, 1), "level_convention": "dependency_depth_v1"}} for i in range(3)],
        edges=[{"source": "service:s0", "target": f"service:s{i}", "type": "DEPENDS_ON"} for i in (1, 2)],
        evidence_gaps=[], n_series_in_scope=0, truncated=False)
    yield graph, inv, scope
    client().execute_command("GRAPH.DELETE", name)


def source(graph, sid, label, props):
    graph.query(f"CREATE (o:{label}) SET o += $props WITH o "
                "MATCH (s:Service {id:$sid}) CREATE (o)-[:OBSERVED_ON]->(s)",
                params={"sid": sid, "props": props})


def attach(graph, hid, candidate, refs, weight=1, polarity="support", incident="i"):
    graph.query("MERGE (h:Hypothesis {id:$hid}) SET h.candidate=$candidate",
                params={"hid": hid, "candidate": candidate})
    for ref in refs:
        graph.query("MATCH (o {id:$ref}), (h:Hypothesis {id:$hid}) "
            "CREATE (e:Evidence {id:$eid, source_node:$ref, incident_id:$incident, ts:o.ts, kind:$kind}) "
            "CREATE (h)-[:HAS_EVIDENCE {polarity:$polarity, weight:$weight}]->(e)",
            params={"hid": hid, "ref": ref, "eid": f"ev:{hid}:{ref}", "incident": incident,
                    "kind": "Span" if ref.startswith("sp") else "LogEvent" if ref.startswith("l") else "MetricAnomaly",
                    "polarity": polarity, "weight": weight})
    return HypothesisRecord(id=hid, candidate=candidate, text="proposed abnormality",
                            evidence=refs, confidence=1, confidence_rule="cross_modal_consensus", reward=.8)


def run(world, hypotheses):
    graph, inv, scope = world
    ranking = _ranking([.7, .2, .1])
    # Explicitly admit live observation snapshot before attaching proposals.
    observations = graph.query("MATCH (o)-[:OBSERVED_ON]->() RETURN DISTINCT o, labels(o)").result_set
    ownership = graph.query("MATCH (o)-[:OBSERVED_ON]->(s) WHERE s:Service OR s:Instance RETURN o.id, CASE WHEN s:Instance THEN s.service_id ELSE s.id END").result_set
    scope.edges = [e for e in scope.edges if e['type'] != 'OBSERVED_ON'] + [
        {'source': sid, 'target': owner, 'type': 'OBSERVED_ON'} for sid, owner in ownership]
    service_nodes = [n for n in scope.nodes if n.get("type") == "Service"]
    scope.nodes = service_nodes + [{"id": dict(o.properties)["id"], "type": labels[0], "properties": dict(o.properties)} for o, labels in observations]
    data = scope.model_dump()
    data["scope_ref"] = None
    rebound = ScopeResult(**data)
    scope.scope_ref = rebound.scope_ref
    ranking.scope_ref = scope.scope_ref
    for h in hypotheses:
        h.scope_ref = scope.scope_ref
    invest = InvestResult(incident_id="i", scope_ref=scope.scope_ref, hypotheses=hypotheses, branches=[])
    meter = BudgetMeter()
    llm = LLMClient(meter, provider=MockProvider())
    val = validate_investigation(inv, scope, ranking, invest, meter=meter, llm=llm)
    fused = fuse_investigation(inv, ranking, invest, val)
    report = report_investigation(inv, scope, ranking, invest, val, fused, meter=meter, llm=llm)
    return val, fused, report


def metric(graph, sid="service:s1", mid="m", **updates):
    props = dict(id=mid, ts=500000, z=5, gamma=7, direction="up", metric="cpu")
    props.update(updates)
    source(graph, sid, "MetricAnomaly", props)


@pytest.mark.integration
def test_same_service_hypotheses_cannot_share_or_branch_rescue(world):
    graph, _, _ = world
    metric(graph)
    good = attach(graph, "good", "service:s1", ["m"])
    bad = attach(graph, "bad", "service:s1", [])
    val, fused, _ = run(world, [bad, good])
    assert set(val.hypotheses) == {"bad", "good"}  # no consolidated branches needed
    assert val.hypotheses["bad"].modalities == []
    assert val.hypotheses["good"].modalities == ["metrics"]
    assert fused.winner == "good" and "bad" not in fused.alternatives
    assert {h.id: h.confidence for h in fused.fused} == {"good": .7, "bad": .4}


@pytest.mark.integration
@pytest.mark.parametrize('mutation', ['extra_owner', 'outside_owner', 'duplicate_id'])
def test_late_ownership_ambiguity_cannot_reuse_cached_scope(world, mutation):
    from backend.app.engine import policy
    graph, inv, scope = world
    metric(graph)
    h = attach(graph, 'owner-check', 'service:s1', ['m'])
    run(world, [h])
    assert policy.owned_evidence(graph, h, inv, scope)
    if mutation == 'duplicate_id':
        source(graph, 'service:s1', 'MetricAnomaly', dict(id='m', ts=500000, z=5, gamma=7, direction='up', metric='cpu'))
    else:
        owner = 'service:s2' if mutation == 'extra_owner' else 'outside'
        graph.query("MERGE (s:Service {id:$owner}) WITH s MATCH (m:MetricAnomaly {id:'m'}) CREATE (m)-[:OBSERVED_ON]->(s)", params={'owner': owner})
    assert not policy.owned_evidence(graph, h, inv, scope)


@pytest.mark.integration
def test_live_projection_excludes_unknown_payload_and_rejects_known_oversize(world):
    from backend.app.engine import policy
    graph, inv, scope = world
    metric(graph)
    h = attach(graph, "bounded", "service:s1", ["m"])
    run(world, [h])
    graph.query("MATCH (m:MetricAnomaly {id:'m'}) SET m.arbitrary_body=$body", params={"body": "x" * 200000})
    records = policy.owned_evidence(graph, h, inv, scope)
    assert "arbitrary_body" not in records["m"]["props"]
    graph.query("MATCH (m:MetricAnomaly {id:'m'}) SET m.metric=$body", params={"body": "x" * 200000})
    with pytest.raises(ValueError, match="oversized"):
        policy.owned_evidence(graph, h, inv, scope)


@pytest.mark.integration
@pytest.mark.parametrize("metric_name,direction,expected",  [("cpu", "up", True), ("latency", "up", False), ("cpu", "down", False)])
def test_exact_structured_metric_claim_live_graph(world, metric_name, direction, expected):
    from backend.app.engine.contracts import StructuredClaim
    graph, _, _ = world
    metric(graph)
    h = attach(graph, "structured", "service:s1", ["m"])
    h.claim_kind = "metric_excursion"
    h.claim = StructuredClaim(metric=metric_name, direction=direction)
    val, fused, report = run(world, [h])
    check = next(c for c in val.hypotheses[h.id].checks if c.name == "claim_applicability")
    assert (check.verdict == "pass") is expected
    assert (fused.winner is not None) is expected
    assert (report.winner_candidate is not None) is expected


@pytest.mark.integration
@pytest.mark.parametrize("case", ["unrelated", "normal_metric", "normal_log", "refute", "stale", "zero", "wrong_incident", "missing_source", "missing_relationship"])
def test_bad_support_cannot_diagnose_or_confirm(world, case):
    graph, _, _ = world
    sid = "service:s2" if case == "unrelated" else "service:s1"
    mid = "l" if case == "normal_log" else "m"
    if case == "normal_log":
        source(graph, sid, "LogEvent", {"id": mid, "ts": 500000, "level": "info", "is_error": False})
    else:
        metric(graph, sid, mid, **({"z": .1} if case == "normal_metric" else
            {"ts": -1} if case == "stale" else {}))
    h = attach(graph, "h", "service:s1", [mid], weight=0 if case == "zero" else 1,
               polarity="refute" if case == "refute" else "support",
               incident="other" if case == "wrong_incident" else "i")
    if case == "missing_source":
        graph.query("MATCH (o:MetricAnomaly {id:'m'}) DETACH DELETE o")
    if case == "missing_relationship":
        graph.query("MATCH (:Hypothesis {id:'h'})-[r:HAS_EVIDENCE]->() DELETE r")
    val, fused, report = run(world, [h])
    assert fused.winner is None and report.tier == "INCONCLUSIVE"
    assert not val.hypotheses["h"].evidence_roles["local_anomaly"]
    assert not val.hypotheses["h"].contradictions  # gaps are not disproof


@pytest.mark.integration
def test_normal_near_onset_cannot_rescue_old_anomaly(world):
    graph, _, _ = world
    metric(graph, ts=0)  # in-window, but outside the onset tolerance
    source(graph, "service:s1", "LogEvent", {"id": "l", "ts": 500000, "level": "info"})
    _, fused, report = run(world, [attach(graph, "h", "service:s1", ["m", "l"])])
    assert fused.winner is None and report.tier == "INCONCLUSIVE"
    assert fused.fused[0].confidence == .4


@pytest.mark.integration
def test_crossmodal_local_support_is_not_propagation(world):
    graph, _, _ = world
    metric(graph)
    source(graph, "service:s1", "LogEvent", {"id": "l", "ts": 500000, "level": "error", "is_error": True})
    h = attach(graph, "h", "service:s1", ["m", "l"])
    h.confidence = .01
    h.confidence_rule = "inference_only"
    val, fused, report = run(world, [h])
    assert val.hypotheses["h"].modalities == ["logs", "metrics"]
    assert fused.fused[0].confidence == .85
    assert fused.fused[0].confidence_rule == "cross_modal_consensus"
    assert report.tier == "LIKELY"
    assert report.evidence_path.edges[0]["support"] == []


@pytest.mark.integration
@pytest.mark.parametrize("fault", [None, "other_hypothesis", "refute", "zero", "wrong_parent", "wrong_trace", "normal", "normal_parent", "bad_timing"])
def test_only_owned_qualified_span_relation_confirms(world, fault):
    graph, _, _ = world
    source(graph, "service:s0", "Span", {"id": "sp-parent", "ts": 500000, "trace_id": "trace",
        "span_id": "parent", "parent_id": "", "status": "ok" if fault == "normal_parent" else "error", "duration_us": 100000})
    source(graph, "service:s1", "Span", {"id": "sp-child", "ts": 500010,
        "trace_id": "different" if fault == "wrong_trace" else "trace", "span_id": "child",
        "parent_id": "wrong" if fault == "wrong_parent" else "parent",
        "status": "ok" if fault == "normal" else "error",
        "duration_us": 200000 if fault == "bad_timing" else 50000})
    metric(graph)
    refs = ["m", "sp-child"] + ([] if fault == "other_hypothesis" else ["sp-parent"])
    h = attach(graph, "h", "service:s1", refs)
    if fault == "other_hypothesis":
        attach(graph, "other", "service:s0", ["sp-parent"])
    if fault in ("refute", "zero"):
        graph.query("MATCH (:Hypothesis {id:'h'})-[r:HAS_EVIDENCE]->(e:Evidence {source_node:'sp-parent'}) "
                    "SET r.polarity=$polarity, r.weight=$weight",
                    params={"polarity": "refute" if fault == "refute" else "support", "weight": 0 if fault == "zero" else 1})
    _, _, report = run(world, [h])
    assert report.tier == ("CONFIRMED" if fault is None else "LIKELY")


@pytest.mark.integration
@pytest.mark.parametrize("mutation", ["source_delete", "source_edit", "edge_delete", "refute", "zero"])
def test_p6_rechecks_live_grounding_not_frozen_provider_assertions(world, mutation):
    graph, inv, scope = world
    metric(graph)
    h = attach(graph, "h", "service:s1", ["m"])
    val, fused, _ = run(world, [h])
    assert fused.winner == "h"
    if mutation == "source_delete":
        graph.query("MATCH (o:MetricAnomaly {id:'m'}) DETACH DELETE o")
    elif mutation == "source_edit":
        graph.query("MATCH (o:MetricAnomaly {id:'m'}) SET o.z=6")
    elif mutation == "edge_delete":
        graph.query("MATCH (:Hypothesis {id:'h'})-[r:HAS_EVIDENCE]->() DELETE r")
    else:
        graph.query("MATCH (:Hypothesis {id:'h'})-[r:HAS_EVIDENCE]->() SET r.polarity=$polarity, r.weight=$weight",
                    params={"polarity": "refute" if mutation == "refute" else "support", "weight": 0 if mutation == "zero" else 1})
    meter = BudgetMeter()
    report = report_investigation(inv, scope, _ranking([.7, .2, .1]),
        InvestResult(incident_id="i", scope_ref=scope.scope_ref, branches=[], hypotheses=[h]), val, fused,
        meter=meter, llm=LLMClient(meter, provider=MockProvider()))
    assert report.tier == "INCONCLUSIVE" and report.winner_candidate is None


@pytest.mark.integration
def test_independent_grounded_causes_survive_zero_coherence(world):
    graph, _, _ = world
    metric(graph, "service:s1", "m1")
    metric(graph, "service:s2", "m2")
    val, fused, report = run(world, [attach(graph, "h1", "service:s1", ["m1"]),
                                    attach(graph, "h2", "service:s2", ["m2"])])
    assert all(a.chain_coherence == 0 for a in val.hypotheses.values())
    assert fused.winner == "h1" and fused.alternatives == ["h2"]
    assert report.tier == "LIKELY" and report.alternatives[0]["candidate"] == "service:s2"


@pytest.mark.integration
@pytest.mark.parametrize("case", ["missing", "volatile", "outside"])
def test_plausible_path_is_required_and_scope_bounded(world, case):
    graph, _, scope = world
    metric(graph)
    if case == "missing":
        scope.edges = []
    elif case == "volatile":
        scope.edges[0]["volatile"] = True
    else:
        scope.services = [s for s in scope.services if s.id != "service:s2"]
        scope.edges = [{"source": "service:s0", "target": "service:s2", "type": "DEPENDS_ON"},
                       {"source": "service:s2", "target": "service:s1", "type": "DEPENDS_ON"}]
        graph.query("MATCH (a:Service {id:'service:s2'}), (b:Service {id:'service:s1'}) CREATE (a)-[:DEPENDS_ON]->(b)")
    val, fused, report = run(world, [attach(graph, "h", "service:s1", ["m"])])
    assert not val.hypotheses["h"].plausible_path
    assert fused.winner is None and report.tier == "INCONCLUSIVE"


@pytest.mark.integration
def test_anchor_local_does_not_vacuously_confirm(world):
    graph, _, _ = world
    metric(graph, "service:s0")
    _, _, report = run(world, [attach(graph, "h", "service:s0", ["m"])])
    assert report.tier == "LIKELY"


@pytest.mark.integration
def test_old_error_log_cannot_create_crossmodal_class(world):
    graph, _, _ = world
    metric(graph)
    source(graph, "service:s1", "LogEvent", {"id": "l", "ts": 0, "level": "error"})
    _, fused, report = run(world, [attach(graph, "h", "service:s1", ["m", "l"])])
    assert fused.fused[0].confidence == .7
    assert fused.fused[0].confidence_rule == "single_modality"
    assert report.tier == "LIKELY"


@pytest.mark.integration
def test_instance_observation_matches_service_support(world):
    graph, _, _ = world
    graph.query("CREATE (i:Instance {id:'instance:worker', service_id:'service:s1'}), "
                "(m:MetricAnomaly {id:'m', ts:500000, z:5, gamma:7, direction:'up', metric:'cpu'}) "
                "CREATE (m)-[:OBSERVED_ON]->(i)")
    _, fused, report = run(world, [attach(graph, "h", "service:s1", ["m"])])
    assert fused.winner == "h" and report.tier == "LIKELY"
    assert report.citations == {"m": 1}


@pytest.mark.integration
def test_report_persistence_failure_still_returns_structured_report(world, monkeypatch):
    graph, inv, scope = world
    metric(graph)
    h = attach(graph, "h", "service:s1", ["m"])
    val, fused, _ = run(world, [h])
    original = graph.query
    def query(text, *args, **kwargs):
        if "a.kind = 'report'" in text:
            raise RuntimeError("persistence unavailable")
        return original(text, *args, **kwargs)
    monkeypatch.setattr(graph, "query", query)
    monkeypatch.setattr("backend.app.graph.client.graph_for", lambda _: graph)
    ranking = _ranking([.7, .2, .1])
    ranking.scope_ref = scope.scope_ref
    report = report_investigation(inv, scope, ranking,
        InvestResult(incident_id="i", scope_ref=scope.scope_ref, branches=[], hypotheses=[h]), val, fused)
    assert report.tier == "LIKELY" and report.config["persistence"] == "failed"
    assert report.markdown and report.model_dump_json()
