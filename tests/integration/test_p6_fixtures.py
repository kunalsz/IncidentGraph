"""Integration: every real-cause fixture reaches a report through the agent.

Parametrized over the three authored root-cause fixtures with the mock LLM:
each must produce a non-INCONCLUSIVE tier, name the expected service, and
retrieve the expected runbook + owner (operational closure, Track 03).
"""

import json

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
from backend.app.graph.client import client
from backend.app.seed import loader

# fixture -> (expected root service, expected runbook id, expected oncall)
FIXTURES = {
    "payment_failure": ("service:payment", "runbook:payment-failures",
                        "@oncall-payments"),
    "cart_failure": ("service:cart", "runbook:cart-store-failures",
                     "@oncall-commerce"),
    "recommendation_slowdown": ("service:recommendation",
                                "runbook:recommendation-memory",
                                "@oncall-discovery"),
}


@pytest.mark.integration
@pytest.mark.parametrize("name", list(FIXTURES))
def test_fixture_reaches_report_through_agent(name):
    expected_service, expected_runbook, expected_oncall = FIXTURES[name]
    loader.seed_demo()
    loader.load_fixture_by_name(name)
    fixture = json.loads((loader.FIXTURES_DIR / f"{name}.json").read_text())
    alert = fixture["incident"].get("anchor_alert") or next(
        (write["props"]["id"] for write in fixture["graph_writes"]
         if write["node"] == "Alert"), None)
    investigation = p0_resolve.trigger_investigation(
        fixture["incident"]["symptom"], alert_id=alert,
        trigger_ts=fixture["incident"]["start_ts"])
    try:
        scope = scope_investigation(investigation)
        ranking = rank_investigation(investigation, scope)
        meter = BudgetMeter()
        llm = LLMClient(meter, provider=MockProvider(), sleep=lambda _s: None)
        invest = investigate_investigation(investigation, scope, ranking,
                                           meter=meter, llm=llm)
        validated = validate_investigation(investigation, scope, ranking,
                                           invest, meter=meter, llm=llm)
        fused = fuse_investigation(investigation, ranking, invest, validated)
        report = report_investigation(investigation, scope, ranking, invest,
                                      validated, fused, meter=meter, llm=llm)

        assert report.tier in ("CONFIRMED", "LIKELY"), (
            f"{name}: gate returned {report.tier} "
            f"({report.config['gate']['reasons']})")
        assert report.winner_candidate == expected_service
        runbook_ids = {entry["id"] for entry in report.runbook["runbooks"]}
        assert not runbook_ids  # legacy seeded runbooks have no approval record
        assert report.action_ref["kind"] == "evidence_request"
        assert report.owner.get("oncall_handle") == expected_oncall
        assert report.evidence_path is not None and report.evidence_path.found
    finally:
        client().execute_command("GRAPH.DELETE", investigation.session_graph)
