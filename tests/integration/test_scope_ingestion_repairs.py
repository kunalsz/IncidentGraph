"""Only run against the caller-selected disposable database."""
import json
import uuid
import pytest
from backend.app.graph.client import graph_for, client
from backend.app.seed import loader
from backend.app.engine.contracts import Investigation, InvestigationWindow, ScopeResult
from backend.app.engine.p1_scope import scope_investigation
from backend.app.agent.bundle import fetch_branch_bundle
from backend.app.agent.tools import call_tool


@pytest.fixture
def isolated_graph():
    name = 'incidentgraph:session:scope-repair-' + uuid.uuid4().hex
    graph = graph_for(name)
    loader.load_topology(graph, json.loads((loader.SEED_DIR / 'topology.json').read_text()))
    try:
        yield name, graph
    finally:
        client().execute_command('GRAPH.DELETE', name)


@pytest.mark.integration
def test_instance_parity_scope_closure_corrections_and_fingerprint(isolated_graph):
    name, graph = isolated_graph
    iid = 'instance:payment-7d9f4-x2v4b'
    # Use an explicitly imported verified instance, not pod-name inference.
    loader.merge_node(graph, 'Instance', {'id': iid, 'service_id': 'service:payment', 'name': 'payment-pod'})
    loader.load_fixture(graph, {'graph_writes': [
        {'node': 'MetricAnomaly', 'props': {'id': 'repair:m', 'ts': 1000, 'metric': 'cpu',
         'column': 'cpu.user', 'z': 9, 'gamma': 10}, 'edge': ['OBSERVED_ON', iid]},
        {'node': 'LogEvent', 'props': {'id': 'repair:l', 'ts': 1000, 'level': 'error', 'message': 'bad'},
         'edge': ['OBSERVED_ON', iid]},
        {'node': 'Span', 'props': {'id': 'repair:p', 'ts': 1000, 'duration_us': 100000,
         'trace_id': 't', 'span_id': 'p', 'status': 2}, 'edge': ['OBSERVED_ON', iid]}]})
    # A partial correction must preserve the old measured z/gamma and ownership.
    loader.load_fixture(graph, {'graph_writes': [
        {'node': 'MetricAnomaly', 'props': {'id': 'repair:m', 'ts': 1000, 'metric': 'cpu', 'value': 5}}]})
    props = graph.query("MATCH (m:MetricAnomaly {id: 'repair:m'}) RETURN m").result_set[0][0].properties
    assert props['z'] == 9 and props['gamma'] == 10 and props['value'] == 5
    inv = Investigation(incident_id='repair:i', session_graph=name, symptom='payment',
                        anchor_service='service:payment', onset_ts=1000,
                        window=InvestigationWindow(t0=0, t1=2000), budget={})
    scope = scope_investigation(inv, max_nodes=50, max_series=10)
    # Serialization validates the reference/config together, not a stale pre-config hash.
    assert ScopeResult.model_validate(scope.model_dump()).scope_ref == scope.scope_ref
    assert next(s for s in scope.services if s.id == 'service:payment').sev_m == 9
    bundle = fetch_branch_bundle(graph, 'service:payment', inv.window)
    assert {'repair:m', 'repair:l', 'repair:p'} <= set(bundle['evidence_ids'])
    projected = next(e for e in scope.edges if e['source'] == 'repair:m')
    assert projected['properties']['target_id'] == iid
    assert call_tool('get_evidence', graph, {'node_id': 'service:payment'})['n_evidence'] == 3
    # zero-series scope cannot borrow high metrics through bundle or tools.
    trimmed = scope_investigation(inv, max_nodes=50, max_series=0)
    assert next(s for s in trimmed.services if s.id == 'service:payment').sev_m == 0
    assert 'repair:m' not in fetch_branch_bundle(graph, 'service:payment', inv.window)['evidence_ids']
    assert trimmed.scope_ref != scope.scope_ref


@pytest.mark.integration
def test_import_namespace_id_only_and_reattachment_rejection(isolated_graph):
    _, graph = isolated_graph
    loader.merge_node(graph, 'Service', {'id': 'repair:service', 'level': 2,
                                          'level_convention': 'dependency_depth_v1'})
    assert graph.query("MATCH (s:Service {id:'repair:service'}) RETURN s.level_convention").result_set == [['dependency_depth_v1']]
    loader.merge_edge(graph, 'service:payment', 'DEPENDS_ON', 'repair:service', {'src': 'evil', 'dst': 'evil'})
    assert graph.query("MATCH (:Service {id: 'service:payment'})-[r:DEPENDS_ON]->(:Service {id: 'repair:service'}) RETURN r.src").result_set == [['evil']]
    loader.load_fixture(graph, {'graph_writes': [
        {'node': 'LogEvent', 'props': {'id': 'repair:l', 'ts': 1000, 'level': 'major'},
         'edge': ['OBSERVED_ON', 'service:payment']}]})
    with pytest.raises(ValueError, match='reattachment'):
        loader.load_fixture(graph, {'graph_writes': [
            {'node': 'LogEvent', 'props': {'id': 'repair:l', 'ts': 1000, 'level': 'minor'},
             'edge': ['OBSERVED_ON', 'service:checkout']}]})
    assert graph.query("MATCH (l:LogEvent {id:'repair:l'}) RETURN l.level").result_set == [['major']]
