import json
import uuid

import pytest

from backend.app.engine.contracts import Investigation, InvestigationWindow
from backend.app.engine.p1_scope import scope_investigation
from backend.app.graph.client import client, graph_for
from backend.app.seed import loader
from config import tuning


@pytest.fixture
def investigation():
    name = f"incidentgraph:session:p1-test-{uuid.uuid4().hex}"
    graph = graph_for(name)
    topology = json.loads((loader.SEED_DIR / "topology.json").read_text())
    loader.load_topology(graph, topology)
    loader.load_fixture_by_name("payment_failure", graph=graph)
    graph.query("CREATE (:Incident {id: 'incident:p1-test', start_ts: 1792141200000})")
    inv = Investigation(
        incident_id="incident:p1-test", session_graph=name, symptom="checkout latency up",
        anchor_service="service:checkout", anchor_alert="alert:INC-1001/a1",
        window=InvestigationWindow(t0=1792140300000, t1=1792142100000), budget={},
    )
    yield inv, graph
    client().execute_command("GRAPH.DELETE", name)


@pytest.mark.integration
def test_payment_scope_and_statistics(investigation):
    inv, graph = investigation
    result = scope_investigation(inv)
    stats = {s.id: s for s in result.services}
    assert "service:frontend" in stats
    assert "service:payment" in stats
    assert "service:cart-cache" not in stats
    assert "service:recommendation" not in stats
    assert stats["service:payment"].sev_m == 6.1
    assert stats["service:checkout"].self_time_ms == 22
    assert stats["service:payment"].self_time_ms == 1498
    assert stats["service:payment"].hours_since_change == pytest.approx(5 / 60)
    assert result.n_series_in_scope == 3
    assert not result.truncated
    assert len(result.nodes) <= tuning.SCOPE_MAX_NODES
    row = graph.query("MATCH (s:Service {id:'service:payment'}) RETURN s.sev_m").result_set
    assert row == [[6.1]]
    assert any(g["kind"] == "no_traces" and g["node_id"] == "service:currency"
               for g in result.evidence_gaps)


@pytest.mark.integration
def test_instance_evidence_and_window_filter(investigation):
    inv, graph = investigation
    graph.query("""
        MATCH (i:Instance {id: 'instance:payment-7d9f4-x2v4b'})
        CREATE (:LogEvent {id:'log:instance', ts:1792141200000, z:4})-[:OBSERVED_ON]->(i)
    """)
    graph.query("""
        MATCH (s:Service {id:'service:payment'})
        CREATE (:MetricAnomaly {id:'metric:old', ts:1, z:99})-[:OBSERVED_ON]->(s)
    """)
    result = scope_investigation(inv)
    payment = next(s for s in result.services if s.id == "service:payment")
    assert payment.sev_l == 4
    assert payment.sev_m == 6.1
    assert "metric:old" not in {n["id"] for n in result.nodes}
    assert any(e["properties"].get("target_id") == "instance:payment-7d9f4-x2v4b"
               for e in result.edges)


@pytest.mark.integration
def test_caps_keep_anchor_and_persist_truncation(investigation, monkeypatch):
    inv, graph = investigation
    monkeypatch.setattr(tuning, "SCOPE_MAX_NODES", 3)
    monkeypatch.setattr(tuning, "SCOPE_MAX_SERIES", 1)
    result = scope_investigation(inv)
    assert result.nodes[0]["id"] == inv.anchor_service
    assert len(result.nodes) == 3
    assert result.n_series_in_scope <= 1
    assert result.truncated
    assert {s.id for s in result.services} == {
        "service:checkout", "service:payment", "service:frontend"}
    assert any(g["kind"] == "scope_truncation" for g in result.evidence_gaps)
    graph_ids = {n["id"] for n in result.nodes}
    assert all(e["source"] in graph_ids and e["target"] in graph_ids for e in result.edges)
    before = graph.query("MATCH (g:EvidenceGap) RETURN count(g)").result_set
    scope_investigation(inv)
    assert graph.query("MATCH (g:EvidenceGap) RETURN count(g)").result_set == before


@pytest.mark.integration
def test_excludes_volatile_dependencies(investigation):
    inv, graph = investigation
    graph.query("""
        MATCH (a:Service {id:'service:checkout'}), (b:Service {id:'service:recommendation'})
        CREATE (a)-[:DEPENDS_ON {invariant:false}]->(b)
    """)
    result = scope_investigation(inv)
    assert "service:recommendation" not in {s.id for s in result.services}


@pytest.mark.integration
def test_series_limit_and_isolated_anchor(investigation, monkeypatch):
    inv, graph = investigation
    monkeypatch.setattr(tuning, "SCOPE_MAX_SERIES", 1)
    result = scope_investigation(inv)
    assert result.n_series_in_scope == 1
    assert result.truncated
    graph.query("CREATE (:Service {id:'service:isolated', criticality:0.1})")
    inv.anchor_service = "service:isolated"
    result = scope_investigation(inv)
    assert [s.id for s in result.services] == ["service:isolated"]
    assert len(result.nodes) == 1
    assert not result.truncated
    assert {g["kind"] for g in result.evidence_gaps} == {"no_logs", "no_traces"}


def test_public_demo_is_not_writable(investigation):
    inv, _ = investigation
    inv.session_graph = "incidentgraph:demo"
    with pytest.raises(ValueError, match="isolated session"):
        scope_investigation(inv)
