"""Incident-relative bounded span context never expands admitted ownership."""
import json
from unittest.mock import Mock

import pytest

from backend.app.agent.bundle import assemble_bundle, bundle_evidence_ids, fetch_branch_bundle
from backend.app.agent.llm import estimate_tokens
from backend.app.engine import policy
from backend.app.engine.contracts import InvestigationWindow, ScopeResult, ScopeService
from backend.app.engine.scope_policy import admitted_span_pairs, remember_scope


ONSET = 100_000
WINDOW = InvestigationWindow(t0=0, t1=200_000)


def span(sid, ts=ONSET, **updates):
    return dict(id=sid, ts=ts, trace_id='trace', span_id=sid,
                status=2, duration_us=10_000) | updates


def snapshot(extra=(), *, dependency=True, parent=None, child=None):
    parent = span('parent', duration_us=20_000) if parent is None else parent
    child = span('child', ONSET + 1, parent_id='parent') if child is None else child
    nodes = [{'id': s, 'type': 'Service', 'properties': {'id': s}} for s in ('caller', 'callee', 'other')]
    edges = []
    if dependency:
        edges.append(dict(source='caller', target='callee', type='DEPENDS_ON'))
    for service, props in [('caller', parent), ('callee', child), *extra]:
        nodes.append(dict(id=props['id'], type='Span', properties=props))
        edges.append(dict(source=props['id'], target=service, type='OBSERVED_ON'))
    return ScopeResult(incident_id='bundle-pairs', anchor_service='callee', window=WINDOW,
        services=[ScopeService(id=s) for s in ('caller', 'callee', 'other')],
        nodes=nodes, edges=edges, evidence_gaps=[], n_series_in_scope=0, truncated=False)


def fetched(scope=None, *, onset=ONSET, service='callee', cap=None):
    graph = Mock()
    remember_scope(graph, scope or snapshot(), onset)
    result = fetch_branch_bundle(graph, service, WINDOW, token_cap=cap)
    graph.query.assert_not_called()
    return result


def test_authoritative_onset_replaces_earliest_fifteen_and_keeps_pre_post():
    spans = [span(f'early{i}', i) for i in range(40)]
    spans += [span('pre', ONSET - 1), span('post', ONSET + 1),
              span('normal-pre', ONSET - 2, status=0), span('normal-post', ONSET + 2, status=0)]
    a, fp = assemble_bundle('callee', [], [], [], spans=spans, onset_ts=ONSET, window=WINDOW)
    b, other_fp = assemble_bundle('callee', [], [], [], spans=list(reversed(spans)), onset_ts=ONSET, window=WINDOW)
    assert len(a['spans']) == 15
    assert {'pre', 'post', 'normal-pre', 'normal-post'} <= set(bundle_evidence_ids(a))
    assert a == b and fp == other_fp


def test_midpoint_is_used_only_without_authoritative_onset():
    spans = [span('onset', 10), span('midpoint', ONSET)]
    unknown, _ = assemble_bundle('callee', [], [], [], spans=spans, window=WINDOW)
    authoritative, _ = assemble_bundle('callee', [], [], [], spans=spans, onset_ts=10, window=WINDOW)
    assert unknown['spans'][0]['id'] == 'midpoint'
    assert authoritative['spans'][0]['id'] == 'onset'


def test_qualified_pair_is_retained_owned_and_association_only():
    result = fetched(snapshot([('other', span('unrelated')), ('caller', span('unmatched'))]))
    bundle = result['bundle']
    assert set(result['evidence_ids']) == {'parent', 'child'}
    assert bundle['span_pairs'] == [dict(dependency_source='caller', dependency_target='callee',
        evidence_refs=['parent', 'child'], association_only=True, claim_eligible=True)]
    spans = {s['id']: s for s in bundle['spans']}
    assert spans['parent']['service_id'] == 'caller'
    assert spans['parent']['evidence_role'] == 'propagation_context'
    assert spans['child']['service_id'] == 'callee'
    assert spans['child']['evidence_role'] == 'local_anomaly'
    records = {sid: dict(kind='Span', positive=True, service=s['service_id'], props=s)
               for sid, s in spans.items()}
    assert policy.propagation_support(records, dict(source='caller', target='callee')) == ['child', 'parent']


@pytest.mark.parametrize('which,updates', [
    ('parent', {'status': 0}), ('parent', {'status': 1}), ('parent', {'status': 'UNSET'}),
    ('child', {'status': 'UNSET'}), ('child', {'trace_id': 'other'}),
    ('child', {'parent_id': 'missing'}), ('child', {'parent_id': ''}),
    ('child', {'ts': ONSET - 1}), ('child', {'duration_us': 100_000}),
    ('parent', {'duration_us': 0}), ('parent', {'duration_us': 'invalid'}),
    ('child', {'ts': WINDOW.t1 + 1}), ('parent', {'ts': True}),
])
def test_unqualified_pairs_never_fetch_foreign_span(which, updates):
    parent = span('parent', duration_us=20_000)
    child = span('child', ONSET + 1, parent_id='parent')
    if which == 'parent':
        parent.update(updates)
    else:
        child.update(updates)
    bundle = fetched(snapshot(parent=parent, child=child))['bundle']
    assert not bundle.get('span_pairs')
    assert 'parent' not in bundle_evidence_ids(bundle)


def test_exact_directional_dependency_and_membership_required():
    scope = snapshot(dependency=False)
    assert not admitted_span_pairs(scope, 'callee', ONSET)
    scope.edges.append(dict(source='callee', target='caller', type='DEPENDS_ON'))
    assert not admitted_span_pairs(scope, 'callee', ONSET)
    assert not admitted_span_pairs(snapshot(), 'other', ONSET)
    assert not admitted_span_pairs(snapshot(), 'outside', ONSET)


def test_ambiguous_ownership_and_parent_identity_fail_closed():
    scope = snapshot()
    scope.edges.append(dict(source='parent', target='other', type='OBSERVED_ON'))
    assert not admitted_span_pairs(scope, 'callee', ONSET)
    duplicate = span('duplicate', span_id='parent', duration_us=20_000)
    assert not admitted_span_pairs(snapshot([('caller', duplicate)]), 'callee', ONSET)


def test_pair_retrieval_is_bounded_and_deterministic_nearest_onset():
    extras = []
    for i in range(1, 20):
        extras.extend([('caller', span(f'p{i}', ONSET + i * 100, duration_us=20_000)),
                       ('callee', span(f'c{i}', ONSET + i * 100 + 1, parent_id=f'p{i}'))])
    scope = snapshot(extras)
    pairs = admitted_span_pairs(scope, 'callee', ONSET)
    assert len(pairs) == 3
    assert admitted_span_pairs(scope, 'callee', ONSET, max_pairs=1000) == pairs
    assert [p['evidence_refs'] for p in pairs] == [['parent', 'child'], ['p1', 'c1'], ['p2', 'c2']]
    scope.nodes.reverse()
    scope.edges.reverse()
    assert admitted_span_pairs(scope, 'callee', ONSET) == pairs
    bundle = fetched(snapshot(extras))['bundle']
    assert len(bundle['spans']) <= 15 and len(bundle['span_pairs']) == 3
    assert {s['id'] for s in bundle['spans'] if s['service_id'] == 'caller'} == {'parent', 'p1', 'p2'}


def test_missing_onset_does_not_expose_pair_claim_capability():
    bundle = fetched(onset=None)['bundle']
    assert bundle['span_pairs'] and not bundle['span_pairs'][0]['claim_eligible']
    assert bundle['span_pairs'][0]['association_only']


def test_out_of_onset_range_pairs_are_context_only():
    from config import tuning
    far = ONSET + tuning.ONSET_DELTA_MS + 100
    # Use a wider admitted window rather than smuggling an out-of-window span.
    scope = snapshot(parent=span('parent', far, duration_us=20_000),
                     child=span('child', far + 1, parent_id='parent'))
    wide = InvestigationWindow(t0=0, t1=far + 100)
    scope = ScopeResult.model_validate(scope.model_dump() | {'window': wide.model_dump(), 'scope_ref': None})
    graph = Mock()
    remember_scope(graph, scope, ONSET)
    result = fetch_branch_bundle(graph, 'callee', wide)
    assert result['bundle']['span_pairs'][0]['claim_eligible'] is False


def test_trimming_cannot_leave_foreign_orphan_or_dangling_capability():
    for cap in (140, 220, 350, 1000):
        result = fetched(cap=cap)
        bundle = result['bundle']
        assert estimate_tokens(json.dumps(bundle)) <= cap
        ids = set(result['evidence_ids'])
        paired = {sid for p in bundle.get('span_pairs', []) for sid in p['evidence_refs']}
        assert paired <= ids
        assert all(s['id'] in paired for s in bundle['spans'] if s['service_id'] != 'callee')


def test_existing_metric_properties_and_generic_bundle_contract_are_preserved():
    m = dict(id='m', column='callee_cpu', metric='cpu', direction='down', z=-6, ts=ONSET)
    bundle, _ = assemble_bundle('callee', [m], [], [], spans=[span('unset', status=0)],
                                onset_ts=ONSET, window=WINDOW)
    assert bundle['metrics'] == [m]
    assert bundle['spans'][0]['status'] == 0
    assert not bundle.get('span_pairs')
    assert bundle_evidence_ids(bundle) == ['m', 'unset']
