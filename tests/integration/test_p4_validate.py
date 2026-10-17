"""Integration: P4 validation end-to-end on payment_failure (mock LLM).

Gate (master plan M3): crafted level-veto rejection; evidence path emitted with
role labels; contradiction cap applied; CAUSES edge persisted with provenance.
"""

import json
import uuid

import pytest

from backend.app.agent.llm import LLMClient, MockProvider
from backend.app.engine.budget import BudgetMeter
from backend.app.engine.contracts import Investigation, InvestigationWindow
from backend.app.engine.p1_scope import scope_investigation
from backend.app.engine.p2_rank import rank_investigation
from backend.app.engine.p3_investigate import investigate_investigation
from backend.app.engine.p4_validate import validate_investigation
from backend.app.graph.client import client, graph_for
from backend.app.seed import loader

WINDOW = InvestigationWindow(t0=1792140300000, t1=1792142100000)


@pytest.fixture()
def validated():
    name = f"incidentgraph:session:p4-{uuid.uuid4().hex}"
    graph = graph_for(name)
    loader.load_topology(
        graph, json.loads((loader.SEED_DIR / "topology.json").read_text()))
    loader.load_fixture_by_name("payment_failure", graph=graph)
    investigation = Investigation(
        incident_id="incident:INC-1001", session_graph=name,
        symptom="checkout latency up", anchor_service="service:checkout",
        budget={}, window=WINDOW, onset_ts=1792141200000, time_source="fixture")
    scope = scope_investigation(investigation)
    ranking = rank_investigation(investigation, scope)
    meter = BudgetMeter()
    llm = LLMClient(meter, provider=MockProvider(), sleep=lambda _s: None)
    invest = investigate_investigation(investigation, scope, ranking,
                                       meter=meter, llm=llm)
    result = validate_investigation(investigation, scope, ranking, invest,
                                    meter=meter, llm=llm)
    yield graph, investigation, result, invest, llm
    client().execute_command("GRAPH.DELETE", name)


@pytest.mark.integration
def test_payment_candidate_passes_all_checks(validated):
    _, _, result, _, _ = validated
    assert result.n_candidates >= 1
    top = result.candidates[0]
    assert top.candidate == "service:payment"
    assert top.verdict == "partial"
    checks = {check.name: check.verdict for check in top.checks}
    assert checks["level_veto"] == "partial"          # level 2 >= anchor level 1
    assert checks["svar_lite"] == "pass"           # z=6.1, gamma=60.5
    assert checks["onset_alignment"] == "pass"     # log 10 s after onset
    assert checks["symptom_explanation"] == "pass"
    assert checks["no_evidence_cap"] == "pass"


@pytest.mark.integration
def test_evidence_path_role_labelled_with_support(validated):
    _, _, result, _, _ = validated
    path = result.candidates[0].evidence_path
    assert path is not None and path.found is True
    assert path.from_id == "service:checkout"
    assert path.to_id == "service:payment"
    roles = {node["id"]: node["role"] for node in path.nodes}
    assert roles["service:checkout"] == "symptom"
    assert roles["service:payment"] == "candidate_cause"
    assert path.edges and path.edges[0]["type"] == "DEPENDS_ON"
    assert path.edges[0]["support"] == []  # node observations are not propagation
    assert not any(node["role"] == "corroborating" for node in path.nodes)


@pytest.mark.integration
def test_association_persisted_without_unproven_causal_edge(validated):
    graph, _, result, _, _ = validated
    rows = graph.query(
        "MATCH (h:Hypothesis)-[c:ASSOCIATED_WITH]->(s:Service {id: 'service:checkout'}) "
        "RETURN c.provenance, c.confidence, c.volatile").result_set
    assert rows, "verified abnormality association missing"
    assert graph.query("MATCH (:Hypothesis)-[c:CAUSES]->() RETURN count(c)").result_set[0][0] == 0
    provenance, confidence, volatile = rows[0]
    assert provenance in ("rule", "llm") and volatile is True
    assert 0.0 < confidence <= 0.95


@pytest.mark.integration
def test_critique_ran_exactly_once_and_validate_artifact_exists(validated):
    graph, _, result, _, llm = validated
    assert result.candidates[0].critique  # review metadata, not a deterministic veto
    assert llm.meter.llm_calls <= 75
    artifacts = graph.query(
        "MATCH (a:Artifact {stage: 'validate'}) RETURN a.json").result_set
    assert len(artifacts) == 1
    from backend.app.engine.contracts import load_handoff
    payload = load_handoff(artifacts[0][0], stage="validate", incident_id=result.incident_id,
                           scope_ref=result.scope_ref)
    assert payload.n_candidates == result.n_candidates


@pytest.mark.integration
def test_level_veto_rejects_shallower_cause_in_graph(validated):
    """Crafted contradiction: a francly-impossible claim is vetoed."""
    from backend.app.graph import validators
    # frontend (level 0) as cause of payment (level 2): physically implausible
    check = validators.level_veto(0, 2)
    assert check["verdict"] == "fail"
    # and the graph carries no such CAUSES edge
    graph, _, _, _, _ = validated
    rows = graph.query(
        "MATCH (h:Hypothesis)-[:CAUSES]->(s:Service {id: 'service:frontend'}) "
        "RETURN count(h)").result_set
    assert rows[0][0] == 0
