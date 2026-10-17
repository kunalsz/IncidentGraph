"""WP02/WP03 adversarial regressions; never connect to a database."""
from types import SimpleNamespace
from unittest.mock import Mock
import pytest

from backend.app.engine import linking
from backend.app.engine.p0_resolve import resolve
from backend.app.engine.p1_scope import _exclusive, service_statistics, scope_investigation
from backend.app.engine.contracts import Investigation, InvestigationWindow
from backend.app.graph.gateway import ReadGateway
from backend.app.graph import schema
from backend.app.ingest import normalize
from backend.app.seed import loader
from backend.app.agent.bundle import assemble_bundle, fetch_branch_bundle
from backend.app.agent.tools import call_tool


def node(kind, identity, **props):
    return SimpleNamespace(labels=[kind], properties=dict(id=identity, **props))


def span(identity, start, duration, **props):
    return {'id': identity, 'type': 'Span', 'service_id': 'service:a',
            'properties': dict(id=identity, ts=0, start_us=start, duration_us=duration,
                               trace_id='t', span_id=identity, **props)}


def test_severity_idempotence_and_timestamp_rejection():
    for severity in normalize.SEVERITY_ORDER:
        assert normalize.canonical_severity(severity) == severity
    raw = {'id': 'l', 'ts': '2026-10-17T00:00:00Z', 'level': 'error', 'message': 'bad'}
    once = normalize.normalize_log(raw)
    assert normalize.normalize_log(once) == once
    for value in [True, -1, float('nan'), float('inf'), '1960-01-01']:
        with pytest.raises(ValueError):
            normalize.to_epoch_ms(value)
    assert normalize.normalize_span({'id': 's', 'ts': 0, 'status': 2})['is_error']
    assert normalize.normalize_span({'id': 's', 'ts': 0, 'status': 'unknown'})['status'] == 'unknown'


def test_non_nested_mentions_abstain_nested_alias_wins():
    nodes = [dict(id='service:cart', type='Service', name='cart', aliases=[]),
             dict(id='service:cache', type='Service', name='cache', aliases=['cart cache'])]
    assert linking.pick_best('cart cache latency', nodes)['id'] == 'service:cache'
    assert linking.pick_best('cart and cache errors', nodes) is None
    assert linking.pick_best('CPU is high', nodes) is None
    assert linking.score_match('pay', 'payment') == 0
    assert linking.score_match('pay', 'payment', ['pay']) == 1


def test_invalid_explicit_alert_never_falls_back():
    graph = Mock()
    graph.query.return_value.result_set = []
    assert not resolve('payment', graph, 'not-found').matched
    assert graph.query.call_count == 1


def test_union_clipping_duplicates_and_unknown_completeness():
    parent = span('p', 1000, 10000, trace_complete=True)
    a = span('a', 0, 5000, parent_id='p')
    b = span('b', 4000, 5000, parent_id='p')
    assert _exclusive(parent, [a, b, b]) == (2.0, 'complete')
    conflict = span('b', 4000, 1000, parent_id='p')
    assert _exclusive(parent, [a, b, conflict])[1] == 'partial'
    parent['properties'].pop('trace_complete')
    assert _exclusive(parent, [a, b]) == (2.0, 'partial')
    assert not service_statistics('service:a', [parent], [], 0).self_time_flag


def test_parameter_namespaces_and_id_only_merge():
    graph = Mock()
    loader.merge_node(graph, 'Service', {'id': 'a'})
    query, = graph.query.call_args.args
    assert 'SET' not in query
    graph.query.side_effect = [SimpleNamespace(result_set=[[['Service']]]),
                               SimpleNamespace(result_set=[[['Service']]]),
                               SimpleNamespace(result_set=[[1]])]
    loader.merge_edge(graph, 'a', 'DEPENDS_ON', 'b', {'src': 'evil', 'dst': 'evil'})
    params = graph.query.call_args.kwargs['params']
    assert params['endpoint_src'] == 'a' and params['endpoint_dst'] == 'b'
    assert params['edge_src'] == 'evil'
    for label, props in [('Service) DELETE n', {'id': 'a'}), ('Service', {'id': 'a', 'evil.key': 1})]:
        with pytest.raises(ValueError):
            loader.merge_node(graph, label, props)


def test_invalid_import_rejected_before_any_write():
    graph = Mock()
    with pytest.raises(ValueError):
        loader.load_fixture(graph, {'graph_writes': [
            {'node': 'LogEvent', 'props': {'id': 'one', 'ts': 0}},
            {'node': 'MetricAnomaly', 'props': {'id': 'two', 'ts': 1, 'metric': 'cpu', 'z': float('nan')}}]})
    graph.query.assert_not_called()
    with pytest.raises(ValueError):
        loader.load_fixture_by_name('../topology', graph)


def test_gateway_limit_timeout_and_overflow():
    graph = Mock()
    graph.query.return_value.result_set = [[1], [2], [3]]
    rows, overflow = ReadGateway(graph).read('MATCH (n) RETURN n', cap=2)
    assert rows == [[1], [2]] and overflow
    assert graph.query.call_args.kwargs['timeout'] == 2000
    assert graph.query.call_args.kwargs['params']['read_limit'] == 3
    graph.query.side_effect = TimeoutError('deadline')
    with pytest.raises(TimeoutError):
        ReadGateway(graph).read('MATCH (n) RETURN n', cap=2)


def test_index_errors_not_swallowed():
    graph = Mock()
    graph.query.side_effect = RuntimeError('permission denied')
    with pytest.raises(RuntimeError):
        schema.ensure_indexes(graph)


def test_zero_series_has_zero_statistical_effect_and_closed_bundle(monkeypatch):
    inv = Investigation(incident_id='i', session_graph='incidentgraph:session:bounded',
                        symptom='s', anchor_service='service:a', budget={},
                        window=InvestigationWindow(t0=0, t1=2000))
    graph = Mock()
    graph.name = inv.session_graph
    def query(q, params=None, **kwargs):
        if ' AS s\n' in q:
            rows = [[node('Service', 'service:a')]]
        elif 'RETURN i.start_ts' in q:
            rows = [[1000]]
        elif ' AS e,' in q:
            rows = [[node('MetricAnomaly', 'm', ts=1000, metric='cpu', z=9), 'service:a', 'service:a']]
        else:
            rows = []
        return SimpleNamespace(result_set=rows)
    graph.query.side_effect = query
    monkeypatch.setattr('backend.app.engine.p1_scope.graph_for', lambda _: graph)
    result = scope_investigation(inv, max_series=0)
    assert result.services[0].sev_m == 0
    assert result.services[0].nsigma_evidence == []
    assert result.n_series_in_scope == 0
    assert 'm' not in {n['id'] for n in result.nodes}
    graph._incidentgraph_scope = (result, 1000)
    built = fetch_branch_bundle(graph, 'service:a', inv.window)
    assert built['evidence_ids'] == []
    assert call_tool('get_evidence', graph, {'node_id': 'service:outside'})['code'] == 'expansion_required'


def test_bundle_may_remove_all_evidence_and_reject_tiny_envelope():
    bundle, _ = assemble_bundle('a', [{'id': 'huge', 'metric': 'x'*50000, 'z': 9}], [], [], token_cap=100)
    assert not bundle['metrics']
    with pytest.raises(ValueError):
        assemble_bundle('a', [], [], [], token_cap=1)


def test_payload_admission_and_startup_settings_are_strict():
    from backend.app.graph.gateway import admit_payload, PayloadAdmissionError
    from backend.app.settings import validate_startup_settings
    for value in ([1] * 129, {'x': float('nan')}, {'x': 'a' * 4097}, [[[[[[1]]]]]]):
        with pytest.raises(PayloadAdmissionError):
            admit_payload(value)
    settings = validate_startup_settings({'FALKORDB_PORT': '12345', 'API_KEY': 'secret'})
    assert settings.db_port == 12345
    assert 'secret' not in repr(settings.redacted_snapshot)
    with pytest.raises(TypeError):
        settings.redacted_snapshot['db_port'] = 3
    for env in ({'FALKORDB_PORT': '0'}, {'INCIDENTGRAPH_PARALLELISM': '999'},
                {'FALKORDB_HOST': 'bad host'}, {'INCIDENTGRAPH_DEPLOYMENT_MODE': 'api'}):
        with pytest.raises(ValueError):
            validate_startup_settings(env)


def test_authoritative_level_convention_is_importable():
    graph = Mock()
    loader.merge_node(graph, 'Service', {'id': 's', 'level': 2,
                                      'level_convention': 'dependency_depth_v1'})
    assert graph.query.call_args.kwargs['params']['node_level_convention'] == 'dependency_depth_v1'
    assert 'n.level_convention = $node_level_convention' in graph.query.call_args.args[0]


def test_catalog_overflow_abstains_even_with_exact_retained_match():
    from backend.app.settings import LinkingSettings
    graph = Mock()
    graph.query.return_value.result_set = [
        ['service:a', ['Service'], 'payment', [], None],
        ['service:b', ['Service'], 'payment', [], None]]
    result = linking.link_entity('payment', graph, settings=LinkingSettings(max_catalog=1))
    assert not result['matched'] and result['anchor_service'] is None
    assert result['nodes'] == [] and result['reason'] == 'catalog_overflow'
    assert result['truncated'] is True
    call = graph.query.call_args
    assert call.kwargs['timeout'] == 2000
    assert call.kwargs['params']['read_limit'] == 2
    assert call.args[0].endswith('ORDER BY n.id LIMIT $read_limit')


def test_complete_bounded_catalog_resolves_and_settings_are_validated():
    from backend.app.settings import LinkingSettings
    graph = Mock()
    graph.query.return_value.result_set = [
        ['service:a', ['Service'], 'payment', [], None]]
    assert linking.link_entity('payment', graph, settings=LinkingSettings(max_catalog=1))['anchor_service'] == 'service:a'
    for cap in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            LinkingSettings(max_catalog=cap)
