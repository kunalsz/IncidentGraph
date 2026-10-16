"""Idempotent seed loader: MERGE-on-id ingestion (§4.7 item 7).

Topology/teams/runbooks/history go to `incidentgraph:demo`; fixture
`graph_writes` go into the demo graph too (they are the showcase data judges
poke at), while investigations write only to `incidentgraph:session:*`.
"""

import json
from pathlib import Path

from backend.app.graph import schema
from backend.app.graph.client import graph_for, DEMO_GRAPH
from backend.app.ingest import normalize

SEED_DIR = Path(__file__).resolve().parent
FIXTURES_DIR = SEED_DIR / "fixtures"


def _props_str(props: dict) -> str:
    """Render a SET clause for MERGE ... ON CREATE SET from a props dict."""
    return ", ".join(f"n.{k} = ${k}" for k in props)


def merge_node(graph, label: str, props: dict) -> None:
    """MERGE on id; ON CREATE SET the remaining props (idempotent)."""
    if "id" not in props:
        raise ValueError(f"merge_node requires an 'id' prop ({label})")
    graph.query(
        f"MERGE (n:{label} {{id: $id}}) "
        f"ON CREATE SET {_props_str({k: v for k, v in props.items() if k != 'id'})}",
        params=props,
    )


def merge_edge(graph, src_id: str, rel: str, dst_id: str, props: dict | None = None) -> None:
    """MERGE a directed edge between existing nodes; SET edge props."""
    params = {"src": src_id, "dst": dst_id, **(props or {})}
    prop_set = ""
    if props:
        prop_set = " SET " + ", ".join(f"e.{k} = ${k}" for k in props)
    graph.query(
        f"MATCH (a {{id: $src}}), (b {{id: $dst}}) "
        f"MERGE (a)-[e:{rel}]->(b){prop_set}",
        params=params,
    )


def load_topology(graph, topology: dict) -> dict:
    """Load services, deps, teams, instances, runbooks, deployments, history."""
    counts = {"services": 0, "depends_on": 0, "teams": 0, "owned_by": 0,
              "instances": 0, "runbooks": 0, "runbook_params": 0,
              "applies_to": 0, "requires_parameter": 0,
              "deployments": 0, "deployed_at": 0,
              "incidents": 0, "affects": 0, "mitigated_by": 0}

    for svc in topology["services"]:
        props = dict(svc)
        props["aliases"] = json.dumps(svc.get("aliases", []))
        merge_node(graph, schema.SERVICE, props)
        counts["services"] += 1

    for dep in topology["depends_on"]:
        merge_edge(graph, dep["from"], schema.DEPENDS_ON, dep["to"],
                   {k: v for k, v in dep.items() if k not in ("from", "to")})
        counts["depends_on"] += 1

    for team in topology["teams"]:
        merge_node(graph, schema.TEAM, team)
        counts["teams"] += 1

    for svc_id, team_id in topology["owned_by"].items():
        merge_edge(graph, svc_id, schema.OWNED_BY, team_id)
        counts["owned_by"] += 1

    for inst in topology["instances"]:
        merge_node(graph, schema.INSTANCE, inst)
        counts["instances"] += 1

    for rb in topology["runbooks"]:
        params = [dict(p) for p in rb.get("params", [])]
        steps = json.dumps(rb["steps"])
        merge_node(graph, schema.RUNBOOK,
                   {"id": rb["id"], "title": rb["title"], "steps": steps,
                    "last_reviewed": rb.get("last_reviewed")})
        counts["runbooks"] += 1
        for p in params:
            merge_node(graph, schema.RUNBOOK_PARAM, p)
            counts["runbook_params"] += 1
            merge_edge(graph, rb["id"], schema.REQUIRES_PARAMETER, p["id"])
            counts["requires_parameter"] += 1
        for target in rb["applies_to"]:
            merge_edge(graph, rb["id"], schema.APPLIES_TO, target)
            counts["applies_to"] += 1

    for dep in topology["deployments"]:
        props = dict(dep)
        instance = props.pop("instance", None)
        merge_node(graph, schema.DEPLOYMENT, props)
        counts["deployments"] += 1
        if instance:
            merge_edge(graph, dep["id"], schema.DEPLOYED_AT, instance)
            counts["deployed_at"] += 1

    for inc in topology.get("past_incidents", []):
        d = inc.get("decision")
        props = {k: v for k, v in inc.items()
                 if k not in ("affects", "decision")}
        merge_node(graph, schema.INCIDENT, props)
        counts["incidents"] += 1
        merge_edge(graph, inc["id"], schema.AFFECTS, inc["affects"])
        counts["affects"] += 1
        if d:
            merge_node(graph, schema.DECISION, d)
            merge_edge(graph, inc["id"], schema.MITIGATED_BY, d["id"])
            counts["mitigated_by"] += 1

    return counts


def load_fixture(graph, fixture: dict) -> int:
    """Load a fixture's graph_writes (telemetry/change evidence) into `graph`.

    Every record passes through the normalization pipeline (§4.7) before
    touching the graph. Returns the number of nodes written.
    """
    written = 0
    for write in fixture["graph_writes"]:
        label = write["node"]
        props = dict(write["props"])
        edge = write.get("edge")

        if label == schema.ALERT:
            props = normalize.normalize_alert(props)
        elif label == schema.LOG_EVENT:
            props = normalize.normalize_log(props)
        elif label == schema.METRIC_ANOMALY:
            props = normalize.normalize_metric_anomaly(props)
        else:
            # ConfigChange/Span/...: timestamps still normalized
            if "ts" in props:
                props["ts"] = normalize.to_epoch_ms(props["ts"])

        merge_node(graph, label, props)
        written += 1
        if edge:
            merge_edge(graph, props["id"], edge[0], edge[1])
    return written


def seed_demo(force: bool = False) -> dict:
    """Seed the demo graph from topology.json (idempotent)."""
    graph = graph_for(DEMO_GRAPH)
    if force:
        from backend.app.graph.client import client
        try:
            client().execute_command("GRAPH.DELETE", DEMO_GRAPH)
        except Exception:
            pass  # graph does not exist yet — nothing to delete
        graph = graph_for(DEMO_GRAPH)
    schema.ensure_indexes(graph)
    topology = json.loads((SEED_DIR / "topology.json").read_text())
    counts = load_topology(graph, topology)
    return {"graph": DEMO_GRAPH, **counts}


def load_fixture_by_name(name: str, graph=None) -> dict:
    """Load a fixture file (name without .json) into the demo graph."""
    path = FIXTURES_DIR / f"{name}.json"
    fixture = json.loads(path.read_text())
    target = graph or graph_for(DEMO_GRAPH)
    written = load_fixture(target, fixture)
    return {"fixture": name, "nodes_written": written,
            "incident_id": fixture["incident"]["id"]}
