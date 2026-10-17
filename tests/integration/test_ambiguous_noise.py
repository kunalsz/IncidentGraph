"""Integration: the abstention fixture (M5a, L12) — never-cut.

`ambiguous_noise` is an alert storm with every signal below the 3-sigma
decisive threshold and no traces. The machine-checked gate must refuse to
guess: insufficient_evidence / INCONCLUSIVE, an actionable EvidenceGap, zero
CAUSES edges, and the disproved candidates recorded as refuting evidence.
"""

import json
import uuid

import pytest

from backend.app.agent.llm import LLMClient, MockProvider
from backend.app.engine import p0_resolve
from backend.app.engine.budget import BudgetMeter
from backend.app.engine.p1_scope import scope_investigation
from backend.app.engine.p2_rank import rank_investigation
from backend.app.engine.p3_investigate import investigate_investigation
from backend.app.engine.p4_validate import validate_investigation
from backend.app.engine.p5_fuse import fuse_investigation
from backend.app.engine.p6_report import report_investigation
from backend.app.graph.client import client, graph_for
from backend.app.seed import loader

FIXTURE = "ambiguous_noise"
ALERT = "alert:INC-2001/a1"


@pytest.fixture()
def abstained():
    loader.seed_demo()
    loader.load_fixture_by_name(FIXTURE)
    fixture = json.loads((loader.FIXTURES_DIR / f"{FIXTURE}.json").read_text())
    investigation = p0_resolve.trigger_investigation(
        fixture["incident"]["symptom"], alert_id=ALERT,
        trigger_ts=fixture["incident"]["start_ts"])
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
    graph = graph_for(investigation.session_graph)
    yield investigation, report, validated, graph
    client().execute_command("GRAPH.DELETE", investigation.session_graph)


@pytest.mark.integration
def test_gate_abstains_with_evidence_gap(abstained):
    _, report, _, graph = abstained
    assert report.verdict == "insufficient_evidence"
    assert report.tier == "INCONCLUSIVE"
    assert report.confidence < 0.6
    # no leading candidate is presented on an abstention
    assert report.winner_candidate is None
    # Terminal P5 vetoes now remove every rejected candidate before P6;
    # abstention no longer depends on a rejected winner's low confidence.
    assert any("no candidate" in reason for reason in report.config["gate"]["reasons"])
    assert report.runbook == {"runbooks": []} and report.owner == {}

    gaps = report.config["gaps"]
    assert gaps and gaps[0]["missing_kind"] == "decisive_evidence"
    assert gaps[0]["suggestion"]
    # the abstention's next action is the gap's concrete query, not a runbook
    assert report.next_action == gaps[0]["suggestion"]

    rows = graph.query(
        "MATCH (g:EvidenceGap {missing_kind: 'decisive_evidence'}) "
        "RETURN count(g)").result_set
    assert rows[0][0] >= 1


@pytest.mark.integration
def test_nothing_is_confirmed_and_missing_support_is_not_disproof(abstained):
    _, _, validated, graph = abstained
    # no causal edge may exist for a case with no decisive evidence
    causes = graph.query(
        "MATCH (:Hypothesis)-[:CAUSES]->(:Service) RETURN count(*)").result_set
    assert causes[0][0] == 0
    # missing decisive evidence is a gap, not manufactured disproof
    assert validated.candidates and all(c.verdict == "partial"
                                        for c in validated.candidates)
    refutes = graph.query(
        "MATCH (e:Evidence {polarity: 'refute'}) RETURN count(e)").result_set
    assert refutes[0][0] == 0
