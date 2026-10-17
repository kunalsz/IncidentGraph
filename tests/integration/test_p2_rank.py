import json
import uuid

import pytest

from backend.app.engine.contracts import Investigation, InvestigationWindow, load_handoff
from backend.app.engine.p1_scope import scope_investigation
from backend.app.engine.p2_rank import rank_investigation
from backend.app.graph.client import client, graph_for
from backend.app.seed import loader
from scripts.eval import FIXTURES, evaluate_fixture


@pytest.mark.integration
@pytest.mark.parametrize("name", FIXTURES)
def test_fixture_top_three_gate(name):
    result = evaluate_fixture(name)
    assert result["top3_pass"]
    assert result["n_candidates"] == 7
    assert result["llm_calls"] == 0
    assert len(result["ranks"]) == len(set(result["ranks"]))


@pytest.mark.integration
def test_artifact_is_idempotent_and_scope_is_verified():
    name = f"incidentgraph:session:p2-test-{uuid.uuid4().hex}"
    graph = graph_for(name)
    try:
        loader.load_topology(graph, json.loads((loader.SEED_DIR / "topology.json").read_text()))
        loader.load_fixture_by_name("payment_failure", graph=graph)
        inv = Investigation(
            incident_id="incident:p2", session_graph=name, symptom="checkout latency up",
            anchor_service="service:checkout", budget={},
            window=InvestigationWindow(t0=1792140300000, t1=1792142100000),
        )
        scope = scope_investigation(inv)
        result = rank_investigation(inv, scope)
        rank_investigation(inv, scope)
        rows = graph.query("MATCH (a:Artifact) WHERE a.json IS NOT NULL RETURN a.json").result_set
        assert len(rows) == 1
        assert load_handoff(rows[0][0], stage="rank", incident_id=inv.incident_id,
                            scope_ref=scope.scope_ref) == result
        assert result.candidates[0].id == "service:payment"
        assert result.candidates[0].components["severity"] == 6.1
        assert "config:INC-1001/c1" in result.candidates[0].evidence
        assert any("error_logs" in candidate.rule_flags for candidate in result.candidates)
        scope.incident_id = "incident:other"
        with pytest.raises(ValueError, match="does not match"):
            rank_investigation(inv, scope)
        inv.session_graph = "incidentgraph:demo"
        with pytest.raises(ValueError, match="isolated session"):
            rank_investigation(inv, scope)
    finally:
        client().execute_command("GRAPH.DELETE", name)
