"""The authored topology every RCAEval case graph is built from.

The dependency map is identical for all 90 benchmark cases (verified against a
built graph: same 12 services, same 16 edges), so it is served once and drawn
once rather than derived per run.
"""
from __future__ import annotations

import copy
import json
from functools import lru_cache
from typing import Any

from . import config

TOPOLOGY_FILE = config.ROOT / "backend/app/seed/rcaeval_ob_topology.json"

# Present in the source telemetry naming, never admitted as a service node.
EXCLUDED = ("service:istio-init",)


@lru_cache(maxsize=1)
def _topology() -> dict[str, Any]:
    data = json.loads(TOPOLOGY_FILE.read_text())
    nodes = [
        {
            "id": service["id"],
            "name": service.get("name") or service["id"].removeprefix("service:"),
            "level": int(service.get("level", 0)),
            "criticality": float(service.get("criticality", 0.0)),
        }
        for service in data.get("services", [])
        if service["id"] not in EXCLUDED
    ]
    known = {node["id"] for node in nodes}
    edges = [
        {"source": edge["from"], "target": edge["to"]}
        for edge in data.get("depends_on", [])
        if edge["from"] in known and edge["to"] in known
    ]
    return {
        "nodes": nodes,
        "edges": edges,
        "excluded": list(EXCLUDED),
        "provenance": data.get("_provenance"),
        "note": f"{len(nodes)} services, {len(edges)} dependencies, built from this map.",
        "caveat": (
            "Per case, the graph's nodes are the services that emitted telemetry in that run, "
            "and its edges are these dependencies with both endpoints present. Measured over all "
            "90 cases: 85 materialise all 12 services; two lose frontend-external (and its single "
            "edge); three pick up an Istio/Envoy trace name such as PassthroughCluster. "
            "All ten cases offered on this page are among the 85, so this drawing matches every "
            "one of them exactly. A service absent from a run's graph is a coverage gap, not a "
            "missing dependency."
        ),
    }


def topology() -> dict[str, Any]:
    """A copy, so callers can never mutate the cached document."""
    return copy.deepcopy(_topology())
