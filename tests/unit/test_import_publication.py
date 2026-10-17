"""Pure normalization/preflight regressions for correction-safe imports."""
from unittest.mock import Mock

import pytest

from backend.app.ingest import normalize
from backend.app.ingest.publication import PUBLICATION_MARKER_ID, publish_import
from backend.app.seed import loader


@pytest.mark.parametrize('value', [2**63, str(2**63), 10**400, -1, True, float('nan')])
def test_timestamp_storage_range(value):
    with pytest.raises(ValueError):
        normalize.to_epoch_ms(value)


@pytest.mark.parametrize('label,raw,expected', [
    ('LogEvent', {'id': 'x'}, {}),
    ('LogEvent', {'id': 'x', 'level': None}, {'level': None, 'is_error': None}),
    ('LogEvent', {'id': 'x', 'message': None}, {'message': None}),
    ('LogEvent', {'id': 'x', 'message': 123}, {'message': '123'}),
    ('Span', {'id': 'x', 'status': None}, {'status': None, 'is_error': None}),
    ('Span', {'id': 'x', 'status': 2}, {'status': 'error', 'is_error': True}),
    ('MetricAnomaly', {'id': 'x', 'z': 3}, {'z': 3}),
    ('Deployment', {'id': 'x', 'ts': '2026-10-17T00:00:00Z'}, {'ts': 1792195200000}),
    ('Incident', {'id': 'x', 'start_ts': '2026-10-17T00:00:00Z'}, {'start_ts': 1792195200000}),
])
def test_partial_corrections(label, raw, expected):
    prepared = loader._prepared_node(label, raw)
    assert prepared.items() >= expected.items()
    if 'ts' not in raw:
        assert 'ts' not in prepared
    assert len(prepared['source_hash']) == 64
    assert prepared['normalizer_version'] == '3'


@pytest.mark.parametrize('props', [
    {'id': PUBLICATION_MARKER_ID}, {'id': 'x', 'value': object()},
    {'id': 'x', 'value': ('bad',)}, {'id': 'x', 'value': 2**63},
    {'id': 'x', 'message': 'x' * 65537},
])
def test_invalid_properties_fail_before_publication(props):
    graph = Mock()
    with pytest.raises(ValueError):
        loader.load_fixture(graph, {'graph_writes': [{'node': 'Service', 'props': props}]})
    graph.query.assert_not_called()


def test_query_only_adapters_fail_closed_without_writes():
    apply = Mock()
    with pytest.raises(TypeError, match='Graph handle'):
        publish_import(Mock(), apply, Mock())
    apply.assert_not_called()
