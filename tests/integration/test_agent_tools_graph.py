"""Integration: tool surface against a seeded session graph (live FalkorDB)."""

import json
import uuid

import pytest

from backend.app.agent.llm import estimate_tokens
from backend.app.agent.tools import call_tool
from backend.app.engine.contracts import Investigation, InvestigationWindow
from backend.app.engine.p1_scope import scope_investigation
from backend.app.graph.client import client, graph_for
from backend.app.seed import loader

SERVICE = "service:payment"

# tool -> (args, required result keys)
TOOL_CASES = {
    "get_neighborhood": ({"service_id": SERVICE}, {"nodes", "edges", "n_nodes"}),
    "get_severity": ({"node_id": SERVICE}, {"node_id", "vector", "top_series"}),
    "get_recent_changes": ({"service_id": SERVICE},
                           {"service_id", "changes", "proximity_score"}),
    "blast_radius": ({"service_id": SERVICE}, {"dependents"}),
    "get_runbook_and_owner": ({"service_id": SERVICE},
                              {"service_id", "runbooks", "team"}),
    "find_related_incidents": ({"service_id": SERVICE},
                               {"service_id", "incidents"}),
    "get_evidence": ({"node_id": SERVICE, "kinds": ["metric", "log"]},
                     {"node_id", "evidence", "n_evidence"}),
    "resolve_entity": ({"name": "paymnet"}, {"matched", "anchor_service"}),
}


@pytest.fixture(scope="module")
def seeded_graph():
    name = f"incidentgraph:session:agent-tools-{uuid.uuid4().hex}"
    graph = graph_for(name)
    loader.load_topology(
        graph, json.loads((loader.SEED_DIR / "topology.json").read_text()))
    loader.load_fixture_by_name("payment_failure", graph=graph)
    # P1 populates the severity/statistics properties the tools read.
    investigation = Investigation(
        incident_id="incident:tools-test", session_graph=name,
        symptom="checkout latency up", anchor_service=SERVICE, budget={}, onset_ts=1792141200000,
        window=InvestigationWindow(t0=1792140300000, t1=1792142100000))
    scope_investigation(investigation, full_scope=True, max_nodes=100, max_series=100)
    yield graph
    client().execute_command("GRAPH.DELETE", name)


@pytest.mark.integration
def test_every_tier1_tool_returns_expected_shape(seeded_graph):
    for tool, (args, keys) in TOOL_CASES.items():
        result = call_tool(tool, seeded_graph, args)
        assert "error" not in result, f"{tool}: {result.get('error')}"
        assert keys <= set(result), f"{tool} missing {keys - set(result)}"


@pytest.mark.integration
def test_neighborhood_and_blast_radius_reach_checkout(seeded_graph):
    neighborhood = call_tool("get_neighborhood", seeded_graph,
                             {"service_id": SERVICE})
    ids = {node["id"] for node in neighborhood["nodes"]}
    assert "service:checkout" in ids  # payment's caller
    blast = call_tool("blast_radius", seeded_graph,
                      {"service_id": SERVICE, "depth": 3})
    by_depth = {dep["id"]: dep["depth"] for dep in blast["dependents"]}
    assert by_depth.get("service:checkout") == 1   # checkout calls payment
    assert by_depth.get("service:frontend") == 2   # frontend calls checkout


@pytest.mark.integration
def test_runbook_and_owner_closure(seeded_graph):
    result = call_tool("get_runbook_and_owner", seeded_graph,
                       {"service_id": SERVICE})
    runbook_ids = {runbook["id"] for runbook in result["runbooks"]}
    assert "runbook:payment-failures" in runbook_ids
    assert result["runbooks"][0]["steps"]
    assert result["team"]["oncall_handle"] == "@oncall-payments"


@pytest.mark.integration
def test_get_evidence_respects_token_budget(seeded_graph):
    result = call_tool("get_evidence", seeded_graph,
                       {"node_id": SERVICE, "kinds": ["metric", "log"],
                        "budget_tokens": 2500})
    assert "error" not in result
    assert result["n_evidence"] >= 1
    assert estimate_tokens(json.dumps(result["evidence"])) <= 2500


@pytest.mark.integration
def test_severity_vector_reflects_scope_stats(seeded_graph):
    result = call_tool("get_severity", seeded_graph, {"node_id": SERVICE})
    assert set(result["vector"]) == {"metric", "log", "trace"}
    assert any(value > 0 for value in result["vector"].values())


@pytest.mark.integration
def test_resolve_entity_links_payment_with_typo(seeded_graph):
    result = call_tool("resolve_entity", seeded_graph, {"name": "paymnet"})
    assert result["matched"] is True  # fuzzy typo repair
    assert result["anchor_service"] == SERVICE
