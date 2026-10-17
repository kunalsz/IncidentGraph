"""Integration: multi-cause retention (M5b).

`glue_multi_cause` carries two independent faults (cart cache switch +
payment decline). The multi-hypothesis contract: both causes must be
retained — winner plus ranked alternatives cover {cart, payment} — and the
report may not collapse to a single-cause story.
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

FIXTURE = "glue_multi_cause"
BOTH_CAUSES = {"service:cart", "service:payment"}


@pytest.fixture()
def multi():
    loader.seed_demo()
    loader.load_fixture_by_name(FIXTURE)
    fixture = json.loads((loader.FIXTURES_DIR / f"{FIXTURE}.json").read_text())
    investigation = p0_resolve.trigger_investigation(
        fixture["incident"]["symptom"],
        alert_id=fixture["incident"]["anchor_alert"],
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
    yield investigation, fused, report
    client().execute_command("GRAPH.DELETE", investigation.session_graph)


@pytest.mark.integration
def test_both_causes_survive_fusion(multi):
    _, fused, _ = multi
    ordered = []
    for item in fused.fused:
        if item.candidate not in ordered:
            ordered.append(item.candidate)
    # the two independent causes lead the fused ranking, in either order
    assert set(ordered[:2]) == BOTH_CAUSES, ordered[:4]


@pytest.mark.integration
def test_report_keeps_both_causes(multi):
    _, _, report = multi
    assert report.tier in ("CONFIRMED", "LIKELY")
    covered = {report.winner_candidate}
    covered |= {entry["candidate"] for entry in report.alternatives}
    assert BOTH_CAUSES <= covered, covered
    # not disproved: both carry their own decisive evidence
    disproved = {entry["candidate"] for entry in report.alternatives
                 if entry.get("status") == "disproved"}
    assert not (BOTH_CAUSES & disproved)
