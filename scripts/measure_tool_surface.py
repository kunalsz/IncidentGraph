#!/usr/bin/env python
"""MCP tool-surface token budget check (Aurora's measured-budget idea).

Tier-1 schema tokens < 2K target; the MCP server registers Tier-1 tools on
Day 2 — this script asserts the surface stays under budget and fails CI
otherwise. For now it validates the planned Tier-1 tool signatures.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Tier-1 always-on graph tools (§4.5): (name, signature, description)
TIER1_TOOLS = [
    ("resolve_entity", "(name: str) -> {nodes: [{id, type, score}]}",
     "Entity linking with alias/fuzzy match"),
    ("get_neighborhood", '(service_id, hops=1, direction="both") -> {nodes, edges}',
     "Scope traversal"),
    ("get_severity", "(node_id, window_min=30) -> {vector, top_series}",
     "TORAI-style severity + gamma"),
    ("get_recent_changes", "(service_id, window_min=120) -> {deployments, config_changes}",
     "Change priors"),
    ("get_evidence", "(node_id, kinds, window_min=30, budget_tokens=2500) -> {evidence}",
     "Token-budgeted evidence bundles"),
    ("check_hypothesis", "(candidate_id, evidence_ids) -> {verdict, checks, confidence_cap}",
     "Deterministic validators"),
    ("get_evidence_path", "(from_id, to_id) -> {path, edges}",
     "The report artifact"),
    ("blast_radius", "(service_id, depth<=3) -> {dependents}",
     "Impact analysis"),
    ("get_runbook_and_owner", "(service_id) -> {runbook, team}",
     "Track 03 closure"),
    ("find_related_incidents", "(service_id, pattern, limit=5) -> {incidents}",
     "Track 02 memory"),
]

TIER1_TOKEN_BUDGET = 2000
CHARS_PER_TOKEN = 4  # 1 token ~ 4 chars (Aurora's estimate)


def measure() -> dict:
    total_chars = 0
    for name, sig, desc in TIER1_TOOLS:
        total_chars += len(name) + len(sig) + len(desc)
    tokens = total_chars // CHARS_PER_TOKEN
    return {"tools": len(TIER1_TOOLS), "tokens": tokens,
            "budget": TIER1_TOKEN_BUDGET, "pass": tokens < TIER1_TOKEN_BUDGET}


def main() -> int:
    m = measure()
    print(f"Tier-1 tools: {m['tools']}")
    print(f"Tier-1 schema tokens: {m['tokens']} (budget {m['budget']})")
    if not m["pass"]:
        print("FAIL: tool surface over budget")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
