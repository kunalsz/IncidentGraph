"""Adversarial publication tests. Explicit disposable port required; never 6380."""
import json
import os
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event
from uuid import uuid4

import pytest
from falkordb import FalkorDB
from falkordb.graph import Graph
from redis.exceptions import ConnectionError, ResponseError, WatchError

from backend.app.ingest.publication import (
    PUBLICATION_MARKER_ID, PublicationConflict, PublicationOutcomeUnknown, publish_import,
)
from backend.app.seed import loader

pytestmark = pytest.mark.integration


@pytest.fixture
def db():
    port = os.environ.get('IMPORT_TEST_DISPOSABLE_PORT')
    if not port or int(port) == 6380:
        pytest.skip('set IMPORT_TEST_DISPOSABLE_PORT to an owned disposable database')
    client = FalkorDB(host='127.0.0.1', port=int(port))
    name = 'wp02-test:' + uuid4().hex
    graph = client.select_graph(name)
    try:
        yield client, graph
    finally:
        # Only this test's keys, never a shared namespace or database flush.
        for key in client.connection.scan_iter(match=name + '*'):
            client.connection.delete(key)


def add(graph, identity, value=1):
    graph.query('MERGE (n:Service {id:$id}) SET n.value=$value', params={'id': identity, 'value': value})


def ids(graph):
    return graph.query('MATCH(n:Service) RETURN n.id ORDER BY n.id').result_set


def generation(graph):
    return graph.query('MATCH(p:Artifact {id:$id}) RETURN p.source_hash',
                       params={'id': PUBLICATION_MARKER_ID}).result_set


def assert_no_stage(client, graph):
    assert list(client.connection.scan_iter(match=graph.name + ':import-stage:*')) == []
    assert all(not n.startswith(graph.name + ':import-stage:') for n in client.list_graphs())


def test_actual_copy_rename_watch_registration_and_generation(db):
    client, graph = db
    add(graph, 'old')
    with pytest.raises(ResponseError, match='destination key already exists'):
        client.execute_command('GRAPH.COPY', graph.name, graph.name)
    publish_import(graph, lambda stage: add(stage, 'new'), lambda stage: None)
    assert ids(graph) == [['new'], ['old']]
    first = generation(graph)
    assert len(first[0][0]) == 32
    publish_import(graph, lambda stage: None, lambda stage: None)
    assert generation(graph) != first
    client.execute_command('GRAPH.COPY', graph.name, graph.name + ':copy')
    assert generation(client.select_graph(graph.name + ':copy')) == generation(graph)
    assert graph.name in client.list_graphs()
    assert_no_stage(client, graph)


@pytest.mark.parametrize('point', ['write', 'verify', 'index'])
def test_failure_rolls_back_old_graph_and_indexes(db, point):
    client, graph = db
    add(graph, 'old')
    def apply(stage):
        add(stage, 'partial')
        if point == 'write':
            stage.query('THIS IS INVALID CYPHER')
        if point == 'index':
            stage.query('CREATE INDEX INVALID')
    def verify(stage):
        if point == 'verify':
            raise ValueError('verification failed')
    with pytest.raises((ResponseError, ValueError)):
        publish_import(graph, apply, verify)
    assert ids(graph) == [['old']]
    assert generation(graph) == []
    assert_no_stage(client, graph)


@pytest.mark.parametrize('existing', [False, True])
def test_nonparticipating_live_writer_conflicts_even_initial_creation(db, existing):
    client, graph = db
    if existing:
        add(graph, 'old')
    def apply(stage):
        add(stage, 'staged')
        add(graph, 'concurrent')
    with pytest.raises(PublicationConflict):
        publish_import(graph, apply, lambda stage: None)
    assert ids(graph) == ([['concurrent'], ['old']] if existing else [['concurrent']])
    assert_no_stage(client, graph)


def test_two_importers_only_one_can_publish(db):
    client, graph = db
    add(graph, 'old')
    barrier = Barrier(2)
    copied = Event()
    def run(identity):
        try:
            if identity == 'two':
                assert copied.wait(timeout=10)
            def apply(stage):
                add(stage, identity)
                copied.set()  # pinned image refuses simultaneous COPY forks
                barrier.wait(timeout=10)
            publish_import(client.select_graph(graph.name), apply, lambda stage: None)
            return 'committed'
        except PublicationConflict:
            return 'conflict'
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(run, ['one', 'two']))
    assert sorted(results) == ['committed', 'conflict']
    assert len(ids(graph)) == 2
    assert_no_stage(client, graph)


def test_readers_never_see_partial_stage(db):
    _, graph = db
    add(graph, 'old')
    ready, release = Event(), Event()
    def run():
        def apply(stage):
            add(stage, 'one')
            ready.set()
            assert release.wait(timeout=10)
            add(stage, 'two')
        publish_import(graph, apply, lambda stage: None)
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(run)
        try:
            assert ready.wait(timeout=10)
            for _ in range(20):
                assert ids(graph) == [['old']]
        finally:
            release.set()
        future.result(timeout=10)
    assert ids(graph) == [['old'], ['one'], ['two']]


def test_fixture_partial_null_forward_refs_and_ownership_rollback(db):
    client, graph = db
    loader.load_fixture(graph, {'graph_writes': [
        {'node': 'LogEvent', 'props': {'id': 'log', 'ts': '2026-10-17T00:00:00Z', 'level': 'error', 'message': 'bad'}, 'edge': ['OBSERVED_ON', 's']},
        {'node': 'Service', 'props': {'id': 's'}},
        {'node': 'Service', 'props': {'id': 'other'}},
        {'node': 'Instance', 'props': {'id': 'pod', 'service_id': 's'}},
    ]})
    loader.load_fixture(graph, {'graph_writes': [{'node': 'LogEvent', 'props': {'id': 'log', 'message': None, 'level': None}}]})
    props = graph.query('MATCH(n:LogEvent) RETURN n').result_set[0][0].properties
    assert 'message' not in props and 'level' not in props and 'is_error' not in props
    assert props['ts'] == 1792195200000
    assert props['source_hash'] and props['normalizer_version'] == '3'
    before = generation(graph)
    for write in [
        {'node': 'LogEvent', 'props': {'id': 'log'}, 'edge': ['OBSERVED_ON', 'other']},
        {'node': 'Instance', 'props': {'id': 'pod', 'service_id': 'other'}},
        {'node': 'Service', 'props': {'id': 'log'}},
        {'node': 'LogEvent', 'props': {'id': 'log'}, 'edge': ['OBSERVED_ON', 'missing']},
    ]:
        with pytest.raises(ValueError):
            loader.load_fixture(graph, {'graph_writes': [write]})
        assert generation(graph) == before
    assert_no_stage(client, graph)


def test_topology_and_force_seed_failure_keep_old(db, monkeypatch):
    client, graph = db
    topology = json.loads((loader.SEED_DIR / 'topology.json').read_text())
    loader.load_topology(graph, topology, ensure_indexes=True)
    old_ids = ids(graph)
    old_generation = generation(graph)
    monkeypatch.setattr(loader, 'graph_for', lambda name: graph)
    def broken(stage):
        stage.query('CREATE (:Service {id:"partial"})')
        raise ResponseError('index permission denied')
    monkeypatch.setattr(loader.schema, 'ensure_indexes', broken)
    with pytest.raises(ResponseError, match='permission'):
        loader.seed_demo(force=True)
    assert ids(graph) == old_ids and generation(graph) == old_generation
    assert_no_stage(client, graph)


@pytest.mark.parametrize('wrapped', [False, True])
@pytest.mark.parametrize('after_commit', [False, True])
def test_connection_loss_during_exec_is_explicit_unknown(db, monkeypatch, after_commit, wrapped):
    client, graph = db
    add(graph, 'old')
    original_pipeline = client.connection.pipeline
    def pipeline(*args, **kwargs):
        pipe = original_pipeline(*args, **kwargs)
        execute = pipe.execute
        def lost(*args, **kwargs):
            if after_commit:
                execute(*args, **kwargs)
            if wrapped:
                # redis-py emits this without explicit exception chaining.
                raise WatchError('A ConnectionError occurred while watching one or more keys')
            raise ConnectionError('reply lost')
        pipe.execute = lost
        return pipe
    monkeypatch.setattr(client.connection, 'pipeline', pipeline)
    with pytest.raises(PublicationOutcomeUnknown):
        publish_import(graph, lambda stage: add(stage, 'new'), lambda stage: None)
    assert ids(graph) == ([['new'], ['old']] if after_commit else [['old']])
    assert_no_stage(client, graph)


def test_stage_tampering_during_verification_conflicts(db):
    client, graph = db
    add(graph, 'old')
    def verify(stage):
        add(stage, 'tampered')
    with pytest.raises(PublicationConflict):
        publish_import(graph, lambda stage: add(stage, 'new'), verify)
    assert ids(graph) == [['old']]
    assert_no_stage(client, graph)
