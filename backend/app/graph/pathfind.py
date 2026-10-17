"""Bounded stable BFS; local abnormal-first tie ordering, not path optimality.

Trusted callers MUST pass frozen invariant adjacency. The graph-backed mode is
retained only for historical viewing and never supplies causal support.
"""
from __future__ import annotations

MAX_PATH_HOPS = 6
MAX_PATH_NODES = 60
MAX_SUPPORT_PER_EDGE = 3


def assign_roles(path, anchor, candidate):
    return [{"id": sid, "role": "candidate_cause" if sid == candidate else
             "symptom" if sid == anchor else "intermediate"} for sid in path]


def find_evidence_path(graph, from_id, to_id, window, abnormal=None,
                       approved_edges=None, approved_services=None):
    abnormal = abnormal or set()
    result = {"from_id": from_id, "to_id": to_id, "nodes": [], "edges": [],
              "found": False, "status": "not_found", "diagnostics": {}}
    allowed = set(approved_services) if approved_services is not None else None
    if allowed is not None and not {from_id, to_id} <= allowed:
        return result
    if from_id == to_id:
        result.update(nodes=assign_roles([to_id], from_id, to_id), found=True, status="found")
        return result
    adjacency = {}
    if approved_edges is not None:
        for edge in approved_edges:
            props = edge.get("properties", {})
            if (edge.get("type") != "DEPENDS_ON" or edge.get("volatile") or props.get("volatile")
                    or edge.get("invariant") is False or props.get("invariant") is False):
                continue
            if allowed is not None and not {edge["source"], edge["target"]} <= allowed:
                continue
            adjacency.setdefault(edge["source"], set()).add(edge["target"])
    frontier, visited, best, truncated = [[from_id]], {from_id}, None, False
    for depth in range(MAX_PATH_HOPS):
        next_frontier = []
        for path in frontier:
            node = path[-1]
            if approved_edges is None:
                rows = graph.query(
                    "MATCH (a:Service {id: $sid})-[r:DEPENDS_ON]->(b:Service) "
                    "WHERE coalesce(r.volatile, false) = false "
                    "AND coalesce(r.invariant, true) = true RETURN b.id, b.criticality",
                    params={"sid": node}).result_set
                neighbors = {r[0] for r in rows}
            else:
                neighbors = adjacency.get(node, set())
            for neighbor in sorted(neighbors, key=lambda sid: (sid not in abnormal, sid)):
                if neighbor in visited:
                    continue
                if len(visited) >= MAX_PATH_NODES:
                    truncated = True
                    continue
                visited.add(neighbor)
                if neighbor == to_id:
                    best = path + [neighbor]
                    break
                next_frontier.append(path + [neighbor])
            if best:
                break
        if best or not next_frontier:
            break
        frontier = next_frontier
        if depth == MAX_PATH_HOPS - 1:
            truncated = True
    result["diagnostics"] = {"visited": len(visited), "max_hops": MAX_PATH_HOPS,
                             "max_visited": MAX_PATH_NODES, "legacy_graph_read": approved_edges is None}
    if best:
        result.update(nodes=assign_roles(best, from_id, to_id), found=True, status="found",
                      edges=[{"source": a, "target": b, "type": "DEPENDS_ON", "support": []}
                             for a, b in zip(best, best[1:])])
    elif truncated:
        result["status"] = "truncated"
    return result
