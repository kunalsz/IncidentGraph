"""Integration: P0 -> P6 end-to-end on payment_failure with the mock LLM.

STAGE GATE (master plan M4): payment_failure produces a report with an
evidence path, typed confidence, a working machine-checked CONFIRMED gate, and
a crafted-downgrade test proving the gate is code, not prompt.
"""

import json
import re
import uuid

import pytest

from backend.app.agent.llm import LLMClient, MockProvider
from backend.app.engine.budget import BudgetMeter
from backend.app.engine.contracts import Investigation, InvestigationWindow
from backend.app.engine.p1_scope import scope_investigation
from backend.app.engine.p2_rank import rank_investigation
from backend.app.engine.p3_investigate import investigate_investigation
from backend.app.engine.p4_validate import validate_investigation
from backend.app.engine.p5_fuse import fuse_investigation
from backend.app.engine.p6_report import report_investigation
from backend.app.graph.client import client, graph_for
from backend.app.seed import loader

WINDOW = InvestigationWindow(t0=1792140300000, t1=1792142100000)


def run_pipeline(graph, name):
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
    validated = validate_investigation(investigation, scope, ranking, invest,
                                       meter=meter, llm=llm)
    fused = fuse_investigation(investigation, ranking, invest, validated)
    report = report_investigation(investigation, scope, ranking, invest,
                                  validated, fused, meter=meter, llm=llm)
    return investigation, report, meter, llm


@pytest.fixture()
def reported():
    name = f"incidentgraph:session:p6-{uuid.uuid4().hex}"
    graph = graph_for(name)
    loader.load_topology(
        graph, json.loads((loader.SEED_DIR / "topology.json").read_text()))
    loader.load_fixture_by_name("payment_failure", graph=graph)
    investigation, report, meter, llm = run_pipeline(graph, name)
    yield graph, investigation, report, meter, llm
    client().execute_command("GRAPH.DELETE", name)


@pytest.mark.integration
def test_payment_failure_likely_without_owned_parent_child_proof(reported):
    _, _, report, _, _ = reported
    assert report.verdict == "diagnosed"
    assert report.tier == "LIKELY"  # local cross-modal support is not link proof
    assert report.winner_candidate == "service:payment"
    assert report.confidence == 0.7
    assert report.confidence_rule == "single_modality"  # selected hypothesis cites metrics only


@pytest.mark.integration
def test_evidence_path_and_citations(reported):
    _, _, report, _, _ = reported
    path = report.evidence_path
    assert path is not None and path.found
    assert path.from_id == "service:checkout" and path.to_id == "service:payment"
    assert path.edges and all(not edge["support"] for edge in path.edges)
    assert report.facts and report.citations
    citations = [int(n) for n in re.findall(r"\[(\d+)\]", report.markdown)]
    assert set(citations) == set(report.citations.values())


@pytest.mark.integration
def test_operational_closure_and_next_action(reported):
    _, _, report, _, _ = reported
    steps = [step for entry in report.runbook["runbooks"]
             for step in entry["steps"]]
    assert steps == []  # historical runbooks lack explicit approval metadata
    assert report.owner["oncall_handle"] == "@oncall-payments"
    assert report.next_action and report.action_ref["kind"] == "evidence_request"


@pytest.mark.integration
def test_budget_footer_and_artifacts(reported):
    graph, _, report, meter, _ = reported
    assert report.budget["llm_calls"] == meter.llm_calls
    assert report.budget["compliant"] is True
    artifacts = graph.query(
        "MATCH (a:Artifact {stage: 'report'}) RETURN a.json, a.markdown").result_set
    assert len(artifacts) == 1
    from backend.app.engine.contracts import load_handoff
    payload = load_handoff(artifacts[0][0], stage="report", incident_id=report.incident_id,
                           scope_ref=report.scope_ref)
    assert payload.tier == "LIKELY"
    assert "LIKELY" in artifacts[0][1]


@pytest.mark.integration
def test_crafted_downgrade_when_evidence_removed():
    """The gate is code: strip the winning hypothesis's evidence from the graph
    and loss of all decisive support must abstain."""
    name = f"incidentgraph:session:p6-dg-{uuid.uuid4().hex}"
    graph = graph_for(name)
    try:
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
        validated = validate_investigation(investigation, scope, ranking, invest,
                                           meter=meter, llm=llm)
        fused = fuse_investigation(investigation, ranking, invest, validated)
        # strip every supporting Evidence node (and its links) from the graph
        graph.query(
            "MATCH (:Hypothesis)-[r:HAS_EVIDENCE]->(e:Evidence) "
            "WHERE r.polarity = 'support' DELETE r, e")
        report = report_investigation(investigation, scope, ranking, invest,
                                      validated, fused, meter=meter, llm=llm)
        assert report.tier == "INCONCLUSIVE"
        assert report.winner_candidate is None
    finally:
        client().execute_command("GRAPH.DELETE", name)
