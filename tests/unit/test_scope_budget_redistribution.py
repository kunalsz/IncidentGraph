"""Bounded evidence allocation and source-relative completeness, no model/DB calls."""
import re
from types import SimpleNamespace

import pandas as pd
import pytest

from backend.app.engine import p1_scope
from backend.app.engine.contracts import Investigation, InvestigationWindow
from backend.app.eval import rcaeval_adapter as adapter
from backend.app.settings import ScopeSettings


class ScopeGraph:
    name = 'incidentgraph:session:budget'

    def __init__(self, buckets=None, context=(), mutate=False):
        self.buckets = buckets or {}
        self.context = context
        self.calls = []
        self.mutate = mutate
        self.observation_reads = {}

    def query(self, query, params=None, **kwargs):
        params = params or {}
        self.calls.append((query, params, kwargs))
        if '__incidentgraph_import_publication__' in params.values():
            rows = [['generation-1']]
        elif ' AS s\n' in query:
            rows = [[dict(id='service:a')]]
        elif ' AS e,' in query:
            label = re.search(r'AND e:(\w+) AND e.ts', query)[1]
            before = 'e.ts < $onset' in query
            rows = [[dict(__kind=label, **p), 'service:a', 'service:a']
                    for p in self.buckets.get((label, before), [])]
            key = (label, before, params['read_limit'])
            self.observation_reads[key] = self.observation_reads.get(key, 0) + 1
            if self.mutate and self.observation_reads[key] > 1 and rows:
                rows[0][0]['message'] = 'changed during validation'
        elif 'MATCH (sp:Span)' in query:
            # Keep exact tuple membership: the nested any(map ...) predicate
            # times out when full benchmark transcripts replay in P3 workers.
            assert '[sp.trace_id, sp.parent_id] IN $pairs' in query
            assert 'any(pair IN $pairs' not in query
            pairs = {tuple(p) for p in params['pairs']}
            rows = [[p] for p in self.context if (p['trace_id'], p['parent_id']) in pairs]
        else:
            rows = []
        return SimpleNamespace(result_set=rows[:params.get('read_limit', len(rows))])


def run_scope(monkeypatch, graph, raw=20, max_nodes=100, max_queries=100, trace_context=500):
    monkeypatch.setattr(p1_scope, 'remember_scope', lambda *args, **kwargs: None)
    inv = Investigation(incident_id='i', session_graph=graph.name, symptom='s',
                        anchor_service='service:a', onset_ts=1000,
                        window=InvestigationWindow(t0=0, t1=2000), budget={})
    return p1_scope._scope_investigation(
        inv, graph=graph, read_graph=graph, max_nodes=max_nodes,
        settings=ScopeSettings(raw_observations=raw, max_queries=max_queries, trace_context=trace_context))


def logs(n, before=False):
    return [dict(id=f'log:{before}:{i}', ts=999-i if before else 1000+i,
                 message='normal', z=0) for i in range(n)]


def test_empty_buckets_redistribute_with_global_cap_and_deterministic_replays(monkeypatch):
    buckets = {('LogEvent', False): logs(50), ('LogEvent', True): logs(50, True)}
    graph = ScopeGraph(buckets)
    result = run_scope(monkeypatch, graph)
    again = run_scope(monkeypatch, ScopeGraph(buckets))
    evidence = [n['id'] for n in result.nodes if n['type'] != 'Service']
    assert len(evidence) == 20 and evidence == [n['id'] for n in again.nodes if n['type'] != 'Service']
    c = result.config['completeness']
    assert c['raw_observation_rows'] == 20
    assert c['redistributed_records'] == 16
    assert c['redistribution_queries'] == 2
    assert c['initial_observation_overflow_buckets'] == c['remaining_observation_overflow_buckets'] == 2
    assert result.truncated and not result.config['trace_query_complete']
    assert len(result.config['queries']) <= 100
    reads = [(q, p) for q, p, _ in graph.calls if ' AS e,' in q]
    assert all(q.endswith('LIMIT $read_limit') and p['read_limit'] <= 21 for q, p in reads)
    assert all(count == 2 for count in graph.observation_reads.values())


def test_recovered_overflow_remains_conservative_and_admission_is_separate(monkeypatch):
    graph = ScopeGraph({('LogEvent', False): logs(5)})
    result = run_scope(monkeypatch, graph, max_nodes=3)
    c = result.config['completeness']
    assert c['initial_observation_overflow_buckets'] == 1
    assert c['remaining_observation_overflow_buckets'] == 0
    assert c['node_admission_omitted'] == result.config['omitted_records'] == 3
    assert c['source_coverage'] == 'unknown'
    assert result.truncated and not result.config['trace_query_complete']


def test_empty_buckets_are_complete_not_sampled(monkeypatch):
    result = run_scope(monkeypatch, ScopeGraph())
    c = result.config['completeness']
    assert c['raw_observation_rows'] == c['redistribution_queries'] == 0
    assert c['initial_observation_overflow_buckets'] == 0
    assert not result.truncated
    assert c['context_query_complete'] and c['source_coverage'] == 'unknown'


def test_query_budget_reserves_validation_and_does_not_refill(monkeypatch):
    graph = ScopeGraph({('LogEvent', False): logs(50)})
    result = run_scope(monkeypatch, graph, max_queries=29)
    assert result.config['completeness']['redistribution_queries'] == 0
    assert len(result.config['queries']) == 29
    assert result.truncated


def test_refill_queries_are_live_validated_before_stats_publication(monkeypatch):
    graph = ScopeGraph({('LogEvent', False): logs(5)}, mutate=True)
    with pytest.raises(ValueError, match='source changed'):
        run_scope(monkeypatch, graph)
    assert not any('SET s.sev_m' in q for q, _, _ in graph.calls)


def test_context_matches_trace_parent_pairs_not_cross_product(monkeypatch):
    parents = [dict(id='p1', ts=1000, trace_id='t1', span_id='p', duration_us=10000,
                    start_us=1000001, trace_complete=True),
               dict(id='p2', ts=1000, trace_id='t2', span_id='q', duration_us=10000,
                    start_us=1000001, trace_complete=True)]
    context = [dict(id='c', ts=1000, trace_id='t1', parent_id='p', span_id='c',
                    start_us=1000002, duration_us=1000),
               dict(id='cross', ts=1000, trace_id='t1', parent_id='q', span_id='cross',
                    duration_us=1000)]
    graph = ScopeGraph({('Span', False): parents}, context)
    result = run_scope(monkeypatch, graph)
    context_calls = [(q, p, opts) for q, p, opts in graph.calls if 'MATCH (sp:Span)' in q]
    assert len(context_calls) == 2  # Initial read and live snapshot validation.
    for query, params, options in context_calls:
        assert params['pairs'] == [['t1', 'p'], ['t2', 'q']]
        assert '[sp.trace_id, sp.parent_id] IN $pairs' in query
        assert params['read_limit'] == 501
        assert options['timeout'] == ScopeSettings().query_timeout_ms
    assert result.config['trace_context_records'] == 1
    assert result.config['trace_query_complete']
    assert result.services[0].self_time_quality == 'complete'
    assert result.services[0].self_time_ms == 10


def test_context_overflow_is_separate_from_initial_and_admission(monkeypatch):
    parent = dict(id='p', ts=1000, trace_id='t', span_id='p', duration_us=10000,
                  start_us=1000000, trace_complete=True)
    children = [dict(id=f'c{i}', ts=1000, trace_id='t', parent_id='p', span_id=f'c{i}',
                     start_us=1000000+i, duration_us=1000) for i in range(2)]
    result = run_scope(monkeypatch, ScopeGraph({('Span', False): [parent]}, children), trace_context=1)
    c = result.config['completeness']
    assert c['initial_observation_overflow_buckets'] == c['node_admission_omitted'] == 0
    assert c['context_query_overflow'] and not c['context_query_complete']
    assert not result.config['trace_query_complete']
    assert result.services[0].self_time_quality == 'partial'
    assert not result.services[0].self_time_flag
    # Legacy truncation concerns evidence admission; context has its own layer.
    assert not result.truncated


@pytest.mark.parametrize('raw', [1, 3, 9])
def test_tiny_raw_cap_never_overallocates(monkeypatch, raw):
    result = run_scope(monkeypatch, ScopeGraph({('MetricAnomaly', False): [
        dict(id=f'm{i}', ts=1000+i, metric='cpu', z=1) for i in range(20)]}), raw=raw)
    assert result.config['completeness']['raw_observation_rows'] <= raw
    assert len(result.nodes) <= raw + 1
    assert result.truncated


def trace(tid, sid, ts, parent=None, **extra):
    return dict(traceID=tid, spanID=sid, parentSpanID=parent, startTimeMillis=ts,
                serviceName='a', duration=1000, statusCode=0, operationName='op', **extra)


def test_trace_closure_is_source_relative_and_boundary_orphans_unknown():
    frame = pd.DataFrame([trace('whole', 'p', 1000), trace('whole', 'c', 1001, 'p'),
                          trace('boundary', 'outside', 2100, 'p'), trace('boundary', 'p', 1000),
                          trace('orphan', 'x', 1000, 'missing'),
                          trace('explicit', 'y', 1000, trace_complete=False)])
    # An explicit unknown/false source flag cannot be upgraded by group selection.
    frame.loc[frame.traceID != 'explicit', 'trace_complete'] = True
    selected, diag = adapter.select_trace_sample(frame, 0, 2000, 1000)
    flags = selected.groupby('traceID').trace_complete.all().to_dict()
    assert flags == {'whole': True, 'boundary': False, 'orphan': False, 'explicit': False}
    assert diag['selected_source_closed_groups'] == 1
    assert diag['selected_source_unproven_groups'] == 3
    assert diag['source_coverage'] == 'supplied-file-only-upstream-unknown'


def test_adapter_normal_logs_period_template_volume_errors_and_source_precision(monkeypatch, tmp_path):
    normal = [dict(timestamp=i, container_name='a', message='routine') for i in range(80)]
    normal += [dict(timestamp=100+i, container_name='a', message='routine') for i in range(80)]
    normal += [dict(timestamp=150, container_name='a', message='rare template')]
    errors = [dict(timestamp=120+i, container_name='a', message='error retained') for i in range(3)]
    case = dict(inject_s=100, metrics=pd.DataFrame({'time': range(200), 'a_cpu': [1.0]*200}),
                logs=pd.DataFrame(normal + errors),
                traces=pd.DataFrame([trace('t', 'p', 100001, startTime=100001987)]))
    monkeypatch.setattr(adapter, 'read_case', lambda _: case)
    captured = {}
    monkeypatch.setattr(adapter, '_write_nodes', lambda graph, label, rows: captured.setdefault(label, rows))
    monkeypatch.setattr(adapter, '_write_edges', lambda *args: None)
    result = adapter.load_case(SimpleNamespace(query=lambda *args, **kwargs: None), tmp_path,
                               topology_path=adapter.OB_TOPOLOGY)  # synthetic fixture, not a real case name
    plain = [r for r in captured['LogEvent'] if not r['is_error']]
    assert len(plain) == adapter.LOG_SAMPLE_PER_SERVICE
    assert any(r['ts'] < 100000 for r in plain) and any(r['ts'] >= 100000 for r in plain)
    assert any(r['message'] == 'rare template' for r in plain)
    assert sum(r['is_error'] for r in captured['LogEvent']) == len(errors)
    assert captured['Span'][0]['start_us'] == 100001987
    assert captured['Span'][0]['trace_complete'] is True
    assert result['onset_ts'] == case['inject_s'] * 1000
    assert result['onset_ts'] != (result['t0_ms'] + result['t1_ms']) // 2
    assert result['aggregate_diagnostics']['logs']['source_sampled']
    assert result['aggregate_diagnostics']['logs']['error_records_omitted'] == 0
    parent = dict(type='Span', id='p', service_id='service:a', properties=dict(
        id='p', ts=100001, start_us=100001987, duration_us=1000, trace_id='t', span_id='p', trace_complete=True))
    child = dict(type='Span', id='c', properties=dict(
        ts=100002, start_us=100002001, duration_us=500, trace_id='t', span_id='c', parent_id='p'))
    assert p1_scope._exclusive(parent, [child]) == (0.5, 'complete')
