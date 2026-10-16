"""FalkorDB connection and graph-per-scope helpers (§4.10, IMPLEMENTATION_PLAN §4.4).

Graph namespaces:
  incidentgraph:demo           read-only public dataset (topology, teams,
                               runbooks, history, showcase incidents)
  incidentgraph:session:{uuid} per-investigation working copy (agent writes)
  incidentgraph:user:{id}      reserved
"""

import os
import uuid

from falkordb import FalkorDB

DEMO_GRAPH = "incidentgraph:demo"


def _connect() -> FalkorDB:
    host = os.environ.get("FALKORDB_HOST", "localhost")
    port = int(os.environ.get("FALKORDB_PORT", "6380"))
    return FalkorDB(host=host, port=port)


_client: FalkorDB | None = None


def client() -> FalkorDB:
    global _client
    if _client is None:
        _client = _connect()
    return _client


def graph_for(name: str):
    """Select a graph by full name (e.g. 'incidentgraph:demo')."""
    return client().select_graph(name)


def demo_graph():
    return graph_for(DEMO_GRAPH)


def session_graph(session_id: str | None = None) -> tuple[str, "object"]:
    """Return (name, graph) for a session graph, copied from the demo graph.

    Uses GRAPH.COPY so every investigation is isolated (multigraph tenant
    isolation, §4.4 of IMPLEMENTATION_PLAN) while starting from the seeded
    topology/teams/runbooks. A stale graph at the same name (rerun) is
    deleted first — sessions are disposable working copies.
    """
    sid = session_id or uuid.uuid4().hex[:12]
    name = f"incidentgraph:session:{sid}"
    try:
        client().execute_command("GRAPH.DELETE", name)
    except Exception:
        pass  # no stale graph — the normal case
    client().execute_command("GRAPH.COPY", DEMO_GRAPH, name)
    return name, graph_for(name)
