"""Entity linking (§4.3 P0): alias table + fuzzy match against Service|Instance
names. RCLAgent-V2's `_normalise_name` repair pattern; unresolved anchors
clarify/abstain, never guess.

The scoring functions are pure (no graph access) so they are unit-testable;
`link_entity` fetches candidates from the graph and scores them.
"""

import re
from difflib import SequenceMatcher

from config import tuning
from backend.app.ingest.normalize import service_from_pod


def normalize_name(name: str) -> str:
    """Repair pattern: lowercase, strip non-alphanumerics, collapse spaces."""
    s = (name or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _fuzzy_threshold(query: str) -> float:
    """Short strings need a stricter fuzzy bar: one typo in a 6-char word
    (broken vs broker, ratio 0.83) is usually a different word, while in a
    longer one it is usually a typo (paymnet vs payment, ratio 0.86)."""
    return 0.85 if len(query) < 8 else tuning.LINK_FUZZY_CUTOFF


def score_match(query: str, name: str, aliases: list[str] | None = None) -> float:
    """Score a query against a node name + aliases.

    Returns 0.0 (no match) .. 1.0 (exact). Callers compare against
    LINK_FUZZY_CUTOFF; below the cutoff we abstain rather than guess.
    """
    q = normalize_name(query)
    if not q:
        return 0.0
    threshold = _fuzzy_threshold(q)
    candidates = [normalize_name(name)] + [normalize_name(a) for a in (aliases or [])]
    best = 0.0
    for cand in candidates:
        if not cand:
            continue
        if q == cand:
            best = max(best, tuning.LINK_EXACT_SCORE)
        elif q in cand or cand in q:
            best = max(best, tuning.LINK_CONTAINS_SCORE)
        else:
            ratio = SequenceMatcher(None, q, cand).ratio()
            if ratio >= threshold:
                best = max(best, ratio)
    return best


def pick_best(query: str, nodes: list[dict]) -> dict | None:
    """Pick the best-scoring node from [{id, type, name, aliases}] candidates.

    Returns None when nothing clears the fuzzy cutoff (abstain, never guess).
    Pod/instance candidates are resolved to their parent Service.
    """
    scored = []
    for node in nodes:
        s = score_match(query, node.get("name", ""), node.get("aliases"))
        if s >= tuning.LINK_FUZZY_CUTOFF:
            scored.append((s, node))
    if not scored:
        return None
    scored.sort(key=lambda t: (-t[0], t[1]["id"]))
    s, node = scored[0]
    return {"id": node["id"], "type": node["type"], "score": round(s, 4)}


def link_entity(query: str, graph) -> dict:
    """Link a free-text mention to a Service (or Instance resolved to its
    Service). Graph-facing wrapper; the returned dict is also the
    `resolve_entity` MCP tool contract (§4.5).

    Returns {"query", "matched": bool, "anchor_service": str|None,
             "nodes": [{id, type, score}], "via_instance": str|None}.
    """
    q = (query or "").strip()
    if not q:
        return {"query": q, "matched": False, "anchor_service": None,
                "nodes": [], "via_instance": None}

    # Direct pod/instance mention resolves through the single matcher.
    if ":" not in q and service_from_pod(q):
        sid = service_from_pod(q)
        row = graph.query(
            "MATCH (s:Service {id: $sid}) RETURN s.id", params={"sid": sid}
        ).result_set
        if row:
            return {"query": q, "matched": True, "anchor_service": sid,
                    "nodes": [{"id": sid, "type": "Service", "score": 1.0}],
                    "via_instance": q}

    result = graph.query(
        "MATCH (n) WHERE n:Service OR n:Instance "
        "RETURN n.id AS id, labels(n) AS labels, n.name AS name, "
        "n.aliases AS aliases, n.service_id AS service_id"
    )
    nodes = []
    for node_id, labels, name, aliases, service_id in result.result_set:
        nodes.append({
            "id": node_id,
            "type": labels[0] if labels else "Node",
            "name": name or "",
            "aliases": _parse_list(aliases),
            "service_id": service_id,
        })

    best = pick_best(q, nodes)
    if best is None:
        return {"query": q, "matched": False, "anchor_service": None,
                "nodes": [], "via_instance": None}

    node = next(n for n in nodes if n["id"] == best["id"])
    if node["type"] == "Instance" and node.get("service_id"):
        return {"query": q, "matched": True,
                "anchor_service": node["service_id"],
                "nodes": [best], "via_instance": node["id"]}
    return {"query": q, "matched": True, "anchor_service": best["id"],
            "nodes": [best], "via_instance": None}


def _parse_list(v) -> list:
    """FalkorDB returns list properties as a JSON string; parse defensively."""
    if v is None:
        return []
    if isinstance(v, list):
        return v
    import json
    try:
        parsed = json.loads(v)
        return parsed if isinstance(parsed, list) else []
    except (ValueError, TypeError):
        return []
