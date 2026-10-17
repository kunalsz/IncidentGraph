"""Integration: P3 investigate end-to-end on payment_failure with the mock LLM.

Repair-v2 gate: fair preallocated slots, zero-to-cap distinct actual proposals
(no mandatory padding), evidence/sc floor and persisted HAS_EVIDENCE edges.
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
from backend.app.graph.client import client, graph_for
from backend.app.seed import loader
from config.tuning import InvestigationBudget

WINDOW = InvestigationWindow(t0=1792140300000, t1=1792142100000)


def build_scope_and_ranking(graph, name):
    investigation = Investigation(
        incident_id="incident:INT-1001", session_graph=name,
        symptom="checkout latency up", anchor_service="service:checkout",
        budget=InvestigationBudget().as_dict(), window=WINDOW)
    scope = scope_investigation(investigation)
    ranking = rank_investigation(investigation, scope)
    return investigation, scope, ranking


@pytest.fixture()
def invested():
    name = f"incidentgraph:session:p3-{uuid.uuid4().hex}"
    graph = graph_for(name)
    loader.load_topology(
        graph, json.loads((loader.SEED_DIR / "topology.json").read_text()))
    loader.load_fixture_by_name("payment_failure", graph=graph)
    investigation, scope, ranking = build_scope_and_ranking(graph, name)
    meter = BudgetMeter()
    llm = LLMClient(meter, provider=MockProvider(), sleep=lambda _s: None)
    result = investigate_investigation(investigation, scope, ranking,
                                       meter=meter, llm=llm)
    yield graph, ranking, result, meter, llm
    client().execute_command("GRAPH.DELETE", name)


@pytest.mark.integration
def test_branch_coverage_and_fair_actual_proposals(invested):
    _, ranking, result, meter, _ = invested
    assert result.n_branches == len(ranking.branch_ids)
    processed = [branch for branch in result.branches if branch.status != "unexplored"]
    assert len(processed) >= 2
    assert 0 <= result.n_hypotheses <= meter.budget.max_hypotheses
    scheduling = result.budget["p3_scheduling"]
    plan = scheduling["primary_slots"]
    assert list(plan) == ranking.branch_ids
    assert max(plan.values()) - min(plan.values()) <= 1
    primary_capacity = meter.budget.max_hypotheses - int(
        meter.budget.max_hypotheses * meter.budget.expansion_slot_fraction)
    assert sum(plan.values()) == primary_capacity
    stats = {row["branch_id"]: row for row in scheduling["branches"]}
    assert sum(row["admitted"] for row in stats.values()) == result.n_hypotheses
    for node, slots in plan.items():
        assert 0 <= stats[node]["admitted"] <= slots
        assert stats[node]["admitted"] <= stats[node]["generated"]
    # This anomalous fixture must yield real proposals, not inference padding.
    supported = [h for h in result.hypotheses if h.evidence]
    assert supported and any(h.candidate == "service:payment" for h in supported)
    assert len({(h.candidate, h.text, tuple(h.evidence)) for h in result.hypotheses}) == result.n_hypotheses
    assert meter.as_dict()["hypothesis_slots_reserved"] == 0


@pytest.mark.integration
def test_payment_branch_finds_its_local_root_cause(invested):
    _, _, result, _, _ = invested
    payment = next(branch for branch in result.branches
                   if branch.node_id == "service:payment")
    assert payment.local_root_cause == "service:payment"
    assert payment.confidence >= 0.6  # cross-modal consensus in the fixture
    assert payment.evidence_ids  # citations attached


@pytest.mark.integration
def test_hypotheses_and_evidence_persisted_with_edges(invested):
    graph, _, result, _, _ = invested
    hypothesis_count = graph.query(
        "MATCH (h:Hypothesis) RETURN count(h)").result_set[0][0]
    assert hypothesis_count == result.n_hypotheses
    evidence_count = graph.query("MATCH (e:Evidence) RETURN count(e)").result_set[0][0]
    assert evidence_count >= 1
    edges = graph.query(
        "MATCH (:Hypothesis)-[r:HAS_EVIDENCE]->(:Evidence) "
        "RETURN count(r), collect(DISTINCT r.polarity)").result_set[0]
    assert edges[0] >= 1
    assert edges[1] == ["support"]
    artifacts = graph.query(
        "MATCH (a:Artifact {stage: 'investigate'}) RETURN a.json").result_set
    assert len(artifacts) == 1
    from backend.app.engine.contracts import load_handoff
    payload = load_handoff(artifacts[0][0], stage="investigate",
                           incident_id=result.incident_id, scope_ref=result.scope_ref)
    assert payload.n_hypotheses == result.n_hypotheses


@pytest.mark.integration
def test_budget_is_metered_and_reported(invested):
    _, _, result, meter, llm = invested
    assert result.budget["llm_calls"] == meter.llm_calls
    assert meter.llm_calls <= 75
    assert result.budget["compliant"] is True
    assert result.budget["budget"]["max_llm_calls"] == 75
    # every LLM call was metered; cache served repeat prompts for free
    assert llm.stats["calls"] + llm.stats["cache_hits"] >= meter.llm_calls


@pytest.mark.integration
def test_hypothesis_scores_and_floor(invested):
    _, _, result, _, _ = invested
    assert result.hypotheses
    for hypothesis in result.hypotheses:
        assert 0.0 <= hypothesis.reward <= 1.0
        assert hypothesis.status == "partial"
        # sc floor: hypotheses without cited evidence carry sc = 0
        if not hypothesis.evidence:
            assert hypothesis.sc == 0.0


@pytest.mark.integration
def test_budget_trip_truncates_cleanly():
    name = f"incidentgraph:session:p3-trip-{uuid.uuid4().hex}"
    graph = graph_for(name)
    try:
        loader.load_topology(
            graph, json.loads((loader.SEED_DIR / "topology.json").read_text()))
        loader.load_fixture_by_name("payment_failure", graph=graph)
        investigation, scope, ranking = build_scope_and_ranking(graph, name)
        meter = BudgetMeter(InvestigationBudget(max_hypotheses=3))
        llm = LLMClient(meter, provider=MockProvider(), sleep=lambda _s: None)
        result = investigate_investigation(investigation, scope, ranking,
                                           meter=meter, llm=llm)
        assert result.truncated is True
        assert 0 <= result.n_hypotheses <= 3  # cap is a maximum, not a generation quota
        scheduling = result.budget["p3_scheduling"]
        plan = scheduling["primary_slots"]
        assert sum(plan.values()) == 3
        assert max(plan.values()) - min(plan.values()) <= 1
        assert [node for node, slots in plan.items() if slots] == ranking.branch_ids[:3]
        stats = {row["branch_id"]: row for row in scheduling["branches"]}
        assert sum(row["admitted"] for row in stats.values()) == result.n_hypotheses
        for node, slots in plan.items():
            assert stats[node]["admitted"] <= slots
            if slots == 0:
                assert stats[node]["generated"] == 0
        assert all(h.evidence and h.text for h in result.hypotheses)
        assert meter.as_dict()["hypothesis_slots_reserved"] == 0
    finally:
        client().execute_command("GRAPH.DELETE", name)
