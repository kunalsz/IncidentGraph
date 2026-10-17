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
    assert stats["service:checkout"].self_time_ms == 0
    assert not stats["service:checkout"].self_time_flag
    assert stats["service:payment"].self_time_ms == 0
    assert not stats["service:payment"].self_time_flag
    assert stats["service:payment"].hours_since_change == pytest.approx(5 / 60)
    assert result.n_series_in_scope == 3
    assert not result.truncated
    assert len(result.nodes) <= tuning.SCOPE_MAX_NODES
    row = graph.query("MATCH (s:Service {id:'service:payment'}) RETURN s.sev_m").result_set
    assert row == [[6.1]]
    assert any(g["kind"] == "no_traces" and g["node_id"] == "service:currency"
               for g in result.evidence_gaps)


@pytest.mark.integration
def test_unscored_modalities_are_gaps_not_silent_zeros(investigation):
    inv, graph = investigation
    result = scope_investigation(inv)
    gaps = {(g["node_id"], g["kind"]): g for g in result.evidence_gaps}
    # The authored fixture's spans and logs carry no z, so their severity is not measured.
    for sid in ("service:checkout", "service:payment"):
        assert gaps[(sid, "unscored_traces")]["coverage"] == "present_unscored"
        assert gaps[(sid, "unscored_logs")]["coverage"] == "present_unscored"
    stats = {s.id: s for s in result.services}
    assert stats["service:payment"].sev_l == 0 and stats["service:payment"].sev_tc == 0
    # Metrics carry z, so they are scored; a service with no logs is a blind spot, not unscored.
    assert ("service:payment", "unscored_metrics") not in gaps
    assert ("service:cart", "unscored_logs") not in gaps and ("service:cart", "no_logs") in gaps
    row = graph.query("MATCH (g:EvidenceGap {id: $id}) RETURN g.missing_kind", params={
        "id": f"gap:{inv.incident_id}:service:payment:unscored_traces"}).result_set
    assert row == [["unscored_traces"]]


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


@pytest.mark.integration
def test_projection_rejects_oversized_field_without_wire_value(investigation):
    from backend.app.graph.gateway import ReadGateway
    from backend.app.graph import queries
    inv, graph = investigation
    huge = 'X' * 50000
    graph.query("MATCH (s:Service {id:'service:checkout'}) CREATE (:LogEvent {id:'log:huge', ts:1792141200000, message:$message, extra:$message})-[:OBSERVED_ON]->(s)", params={'message': huge})
    rows, _ = ReadGateway(graph).read(queries.SCOPE_OBSERVATIONS,
        {'ids': [inv.anchor_service], 't0': inv.window.t0, 't1': inv.window.t1}, cap=100)
    projected = next(row[0] for row in rows if row[0]['id'] == 'log:huge')
    assert projected['message'] is None and projected['__rejected'] is True
    assert 'extra' not in projected
    assert huge not in repr(rows)
    result = scope_investigation(inv)
    assert 'log:huge' not in {n['id'] for n in result.nodes}
    assert result.truncated


@pytest.mark.integration
def test_live_publication_change_during_private_read_rejected_and_cleaned(investigation, monkeypatch):
    from falkordb import Graph
    inv, graph = investigation
    original_copy = Graph.copy
    def copy_then_correct(self, name):
        frozen = original_copy(self, name)
        graph.query("MATCH (s:Service {id:$id}) SET s.name='concurrent correction'",
                    params={'id': inv.anchor_service})
        return frozen
    monkeypatch.setattr(Graph, 'copy', copy_then_correct)
    with pytest.raises(ValueError, match='source changed'):
        scope_investigation(inv)
    assert not [n for n in client().execute_command('GRAPH.LIST') if n.startswith('incidentgraph:read:')]


@pytest.mark.integration
def test_snapshot_is_immutable_and_detects_live_corrections(investigation):
    from backend.app.engine.scope_policy import recalled_scope
    inv, graph = investigation
    result = scope_investigation(inv)
    original = result.nodes[0]['properties']['id']
    result.nodes[0]['properties']['id'] = 'mutated'
    restored, _ = recalled_scope(graph, inv.window)
    assert restored.nodes[0]['properties']['id'] == original
    graph.query("MATCH (s:Service {id:$id}) SET s.name='corrected'", params={'id': original})
    with pytest.raises(ValueError, match='source changed'):
        recalled_scope(graph, inv.window)
    assert not [n for n in client().execute_command('GRAPH.LIST') if n.startswith('incidentgraph:read:')]


def test_public_demo_is_not_writable(investigation):
    inv, _ = investigation
    inv.session_graph = "incidentgraph:demo"
    with pytest.raises(ValueError, match="isolated session"):
        scope_investigation(inv)
