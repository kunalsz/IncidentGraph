"""Copy-on-write publication for standalone FalkorDB (SDK 1.8).

The live key is WATCHed BEFORE GRAPH.COPY. All queries and verification operate
on a unique private graph. A transaction containing ONLY RENAME publishes it;
RENAME replaces an existing key atomically, unlike GRAPH.COPY. Any live-key
mutation (including nonparticipating writers) aborts publication, without retry
or lost updates. Readers using the original graph name see old or new for each
query, not a multi-query snapshot. Low-level merge helpers are not transactions.

Requires standalone Redis/FalkorDB with COPY, WATCH/MULTI and RENAME permission;
clusters and arbitrary query-only adapters fail closed. The pinned image may
reject concurrent COPY forks; this is a surfaced prepublication failure, with no
implicit retry. Apply/verify callbacks and staging writers must be trusted.
Stage keys are internal,
not access-control boundaries. Full graph copies require memory headroom. Process
termination can orphan a private stage; no live key is deleted during cleanup.
Connection loss during EXEC has an explicitly UNKNOWN commit outcome, never a
claim that rollback succeeded. No automatic retry is safe in that case.
"""
from uuid import uuid4

from falkordb.graph import Graph
from redis.exceptions import ConnectionError, TimeoutError, WatchError


PUBLICATION_MARKER_ID = '__incidentgraph_import_publication__'


class PublicationConflict(RuntimeError):
    """Live graph changed while the staged import was being prepared."""


class PublicationOutcomeUnknown(RuntimeError):
    """EXEC may have committed; reconcile live state before retrying."""


def publish_import(graph, apply, verify, *, replace=False):
    """Apply and verify privately; return apply's result only after publication.

    ``replace`` starts empty but still checks the old live key for conflicts.
    Cleanup errors are attached to an existing error, never mask it or turn a
    successful commit into a reported failure. Does not mutate the Graph handle.
    """
    if not isinstance(graph, Graph):
        raise TypeError('atomic imports require an installed FalkorDB Graph handle')
    connection = graph.client.connection
    if connection.__class__.__name__ == 'RedisCluster':
        raise TypeError('atomic imports require standalone FalkorDB')
    name = graph.name
    stage_name = f'{name}:import-stage:{uuid4().hex}'
    stage = Graph(graph.client, stage_name)
    failure = None
    try:
        with connection.pipeline() as transaction:
            transaction.watch(name)
            exists = transaction.exists(name)
            if exists and not replace:
                connection.execute_command('GRAPH.COPY', name, stage_name)
            else:
                # A read creates an empty graph, without a synthetic data node.
                stage.query('MATCH (n) RETURN count(n)')
            result = apply(stage)
            # Internal non-telemetry artifact: copied together with the data.
            # Readers can compare this token across reads (or copy a snapshot).
            stage.query('MERGE (p:Artifact {id: $marker_id}) SET p.source_hash = $generation',
                        params={'marker_id': PUBLICATION_MARKER_ID, 'generation': uuid4().hex})
            transaction.watch(stage_name)
            verify(stage)
            transaction.multi()
            transaction.rename(stage_name, name)
            try:
                transaction.execute()
            except WatchError as exc:
                # redis-py also wraps connection failures during WATCH as
                # WatchError; those do not prove EXEC was rejected.
                if (isinstance(exc.__cause__, (ConnectionError, TimeoutError))
                        or isinstance(exc.__context__, (ConnectionError, TimeoutError))
                        or 'occurred while watching' in str(exc)):
                    raise PublicationOutcomeUnknown(name) from exc
                raise PublicationConflict(f'concurrent mutation of {name!r}') from exc
            except (ConnectionError, TimeoutError) as exc:
                raise PublicationOutcomeUnknown(name) from exc
        graph.schema.clear()
        return result
    except BaseException as exc:
        failure = exc
        raise
    finally:
        try:
            # Redis DEL accepts missing keys and invokes module key cleanup.
            connection.delete(stage_name)
        except Exception as exc:
            if failure is not None and hasattr(failure, 'add_note'):
                failure.add_note(f'private stage cleanup failed for {stage_name}: {type(exc).__name__}')
            # Successful RENAME consumed the stage; cleanup cannot affect live.
