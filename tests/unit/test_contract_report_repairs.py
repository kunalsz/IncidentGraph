"""Adversarial v2 contract/report and bounded finalization regressions."""
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from backend.app.engine import policy
from backend.app.engine.contracts import InvestigationWindow, HypothesisRecord, ReportResult, ValidateResult, ScopeResult
from backend.app.engine.p5_fuse import fuse_investigation
from backend.app.engine.p6_report import report_investigation, _approved_closure
from backend.app.graph.pathfind import find_evidence_path
from backend.app.graph.validators import svar_lite
from backend.app.report.render import render_markdown
from tests.unit.test_p5_fuse import INV, REF, _ranking, _invest, _hypothesis, TestFuseInvestigation
from tests.unit.test_diagnosis_integrity import scope_for


@pytest.mark.parametrize("window", [{"t0": True, "t1": 2}, {"t0": -1, "t1": 2}, {"t0": 3, "t1": 2}, {"t0": "1", "t1": 2}])
def test_strict_ordered_window(window):
    with pytest.raises(ValidationError):
        InvestigationWindow(**window)


@pytest.mark.parametrize("field", ["confidence", "reward", "sc"])
def test_boolean_scores_rejected(field):
    with pytest.raises(ValidationError):
        HypothesisRecord(id="h", candidate="s", text="x", **{field: True})


def test_scope_content_edit_requires_new_fingerprint():
    scope = scope_for("service:s0")
    scope.nodes.append({"id": "outside", "type": "LogEvent"})
    with pytest.raises(ValidationError):
        ScopeResult.model_validate(scope.model_dump())


def test_provider_and_graph_free_exhausted_fallback(monkeypatch):
    graph_for = Mock(side_effect=AssertionError("no final graph I/O"))
    monkeypatch.setattr("backend.app.graph.client.graph_for", graph_for)
    llm = Mock(side_effect=AssertionError("no provider"))
    ranking, invest = _ranking([1]), _invest(_hypothesis("h1", "service:s0"))
    val = TestFuseInvestigation()._validated("service:s0")
    fused = fuse_investigation(INV, ranking, invest, val)
    report = report_investigation(INV, scope_for("service:s0"), ranking, invest, val, fused,
                                  llm=llm, final_reads_allowed=False)
    assert report.tier == "INCONCLUSIVE" and report.facts
    assert report.action_ref["kind"] == "evidence_request"
    assert report.config["persistence"] == "not_attempted"
    assert report.config["composition_payload"]["gate"] == report.config["gate"]
    graph_for.assert_not_called()
    llm.complete.assert_not_called()


def test_legacy_validation_is_viewable_but_cannot_win():
    val = TestFuseInvestigation()._validated("service:s0")
    val.schema_version = 1
    fused = fuse_investigation(INV, _ranking([1]), _invest(_hypothesis("h1", "service:s0")), val)
    assert fused.winner is None


def test_changed_proposal_cannot_reuse_validation():
    h = _hypothesis("h1", "service:s0")
    val = TestFuseInvestigation()._validated("service:s0")
    h.text = "different claim"
    assert fuse_investigation(INV, _ranking([1]), _invest(h), val).winner is None


def test_default_service_ranking_exactly_p2_no_coverage_promotion():
    ranking = _ranking([.7, .2, .1])
    val = TestFuseInvestigation()._validated("service:s2")
    h = _hypothesis("h1", "service:s2")
    fused = fuse_investigation(INV, ranking, _invest(h), val)
    duplicate = _hypothesis("h2", "service:s2")
    a = val.hypotheses["h1"].model_copy(update={"hypothesis_id": "h2", "proposal_fingerprint": policy.proposal_fingerprint(duplicate)})
    val.hypotheses["h2"] = a
    doubled = fuse_investigation(INV, ranking, _invest(h, duplicate), val)
    assert [r["candidate"] for r in fused.service_ranking] == [c.id for c in ranking.candidates]
    assert [r["candidate"] for r in doubled.service_ranking] == [c.id for c in ranking.candidates]
    assert fused.fused[0].confidence == doubled.fused[0].confidence


def test_scoped_path_uses_no_graph_read_and_reports_truncation():
    graph = Mock(side_effect=AssertionError("frozen adjacency only"))
    edges = [{"source": str(i), "target": str(i+1), "type": "DEPENDS_ON"} for i in range(8)]
    path = find_evidence_path(graph, "0", "8", INV.window, approved_edges=edges, approved_services=set(map(str, range(9))))
    assert path["status"] == "truncated" and not path["found"]
    graph.query.assert_not_called()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "8"])
def test_svar_nonfinite_statistics_never_pass(value):
    assert svar_lite([{"z": 5, "gamma": value, "direction": "up"}])["verdict"] != "pass"


def test_approved_actions_only_and_no_generated_commands():
    book = {"id": "b", "steps": [{"action": "check approved state"}]}
    assert not _approved_closure({"runbooks": [book]})[0]["runbooks"]
    book["approved"] = True
    assert _approved_closure({"runbooks": [book]})[0]["runbooks"][0]["steps"] == book["steps"]


def test_safe_markdown_and_renderer_owned_citations():
    report = ReportResult(incident_id="<img src=x onerror=alert(1)>", verdict="insufficient_evidence", tier="INCONCLUSIVE",
        facts=[{"id": "f", "text": "<script>x</script> [999](javascript:evil) `curl evil`", "evidence_ids": ["e"]}], citations={"e": 1})
    md = render_markdown(report)
    assert "<script>" not in md and "<img" not in md
    assert "[999](javascript:" not in md
    assert " [1]" in md and "\\[999\\]" in md
    report.citations["unknown"] = 2
    with pytest.raises(ValueError):
        render_markdown(report)


@pytest.mark.parametrize("mutation", ["settings", "statistics"])
def test_settings_and_derived_statistics_are_fingerprinted(mutation):
    scope = scope_for("service:s0")
    previous = scope.scope_ref.fingerprint
    data = scope.model_dump()
    if mutation == "settings":
        data["config"] = {"effective_settings": {"raw_observations": 123}}
    else:
        data["services"][0]["self_time_ms"] = 123
    with pytest.raises(ValidationError):
        ScopeResult(**data)
    data["scope_ref"] = None
    rebound = ScopeResult(**data)
    assert rebound.scope_ref.fingerprint != previous
    data = rebound.model_dump()
    data["config"]["fingerprint"] = rebound.scope_ref.fingerprint
    assert ScopeResult(**data).scope_ref == rebound.scope_ref


def test_markdown_exposes_unknown_usage_and_cooperative_deadline():
    report = ReportResult(incident_id="i", verdict="insufficient_evidence", tier="INCONCLUSIVE",
        budget={"llm_calls": 3, "provider_attempts": 3, "logical_requests": 2,
                "cache_hits": 1, "retries": 1, "tokens_total": 0,
                "tokens_charged": 4096, "tokens_committed": 512,
                "usage_complete": False, "unknown_usage_attempts": 2,
                "in_flight_attempts": 1, "strict_token_enforcement": False,
                "deadline_enforcement": "cooperative; sync HTTP inactivity timeouts, not hard total"})
    md = render_markdown(report)
    assert "Provider attempts: 3" in md and "logical requests: 2" in md
    assert "cache hits: 1" in md and "retries: 1" in md
    assert "Known tokens: 0" in md
    assert "conservative estimates for unknown usage" in md and "4096" in md
    assert "committed tokens: 512" in md
    assert "Usage complete: False" in md and "unknown usage attempts: 2" in md
    assert "in-flight attempts: 1" in md
    assert "Strict token enforcement: False" in md
    assert "cooperative; sync HTTP inactivity timeouts, not hard total" in md


def test_legacy_footer_marks_missing_accounting_unknown_and_escapes_metadata():
    report = ReportResult(incident_id="i", verdict="insufficient_evidence", tier="INCONCLUSIVE",
                          budget={"deadline_enforcement": "<script>unsafe</script> [x](javascript:evil)"})
    md = render_markdown(report)
    assert "logical requests: unknown" in md and "Usage complete: unknown" in md
    assert "unknown usage attempts: unknown" in md
    assert "<script>" not in md and "[x](javascript:" not in md


def test_optional_critique_exception_does_not_persist_provider_secrets(monkeypatch):
    from backend.app.agent.llm import LLMError
    from backend.app.engine import p4_validate as p4
    graph = Mock()
    graph.query.return_value.result_set = [[0]]
    monkeypatch.setattr("backend.app.graph.client.graph_for", lambda _: graph)
    scope = scope_for("service:s0")
    ranking = _ranking([1])
    ranking.scope_ref = scope.scope_ref
    hypothesis = _hypothesis("h1", "service:s0")
    hypothesis.scope_ref = scope.scope_ref
    invest = _invest(hypothesis)
    invest.scope_ref = scope.scope_ref
    assessment = TestFuseInvestigation()._validated("service:s0").hypotheses["h1"]
    monkeypatch.setattr(p4, "assess_hypothesis", lambda *args: assessment)
    llm = Mock()
    secret = "api_key=super-secret-provider-token"
    llm.complete.side_effect = LLMError(secret)
    result = p4.validate_investigation(INV, scope, ranking, invest, llm=llm)
    assert result.hypotheses["h1"].critique == {"reason": "critique unavailable: LLMError"}
    assert secret not in result.model_dump_json()
    assert secret not in str(graph.query.call_args_list)


def test_persisted_envelope_roundtrip_and_tamper_gate():
    import json
    from backend.app.engine.contracts import dump_handoff, load_handoff
    raw = dump_handoff("resolve", INV)
    assert load_handoff(raw, stage="resolve", incident_id="i") == INV
    for stage, incident in [("rank", "i"), ("resolve", "other")]:
        with pytest.raises(ValueError):
            load_handoff(raw, stage=stage, incident_id=incident)
    edited = json.loads(raw)
    edited["payload"]["symptom"] = "forged"
    with pytest.raises(ValueError):
        load_handoff(json.dumps(edited), stage="resolve", incident_id="i")
    with pytest.raises(ValueError):
        load_handoff('{"stage":"resolve","stage":"rank"}', stage="resolve", incident_id="i")
    with pytest.raises(ValueError):
        load_handoff(" " * (8 * 1024 * 1024 + 1), stage="resolve", incident_id="i")


def test_nested_compatibility_metadata_is_bounded_finite_and_extra_forbidden():
    for update in [{"unknown": "ignored?"}, {"budget": {"x": float("nan")}},
                   {"budget": {"x": "x" * 131073}}, {"budget": {"x": list(range(16385))}}]:
        data = INV.model_dump()
        data.update(update)
        with pytest.raises(ValueError):
            type(INV).model_validate(data)


def test_real_historical_view_and_migration_cannot_trust_legacy_results():
    from pathlib import Path
    from backend.app.engine.contracts import historical_v1_gate
    root = Path(__file__).resolve().parents[2]
    for directory in ["rcaeval-agent-20", "rcaeval-20-baseline"]:
        raw = (root / "output" / directory / "_summary.json").read_bytes()
        viewed = historical_v1_gate(raw, mode="VIEW")
        assert viewed["historical"]["n_cases"] == 20
        assert not viewed["authoritative"] and not viewed["diagnosis_eligible"]
        with pytest.raises(ValueError, match="fresh v2"):
            historical_v1_gate(raw, mode="MIGRATION")
        with pytest.raises(ValueError, match="pinned historical"):
            historical_v1_gate(raw + b" ", mode="VIEW")
    with pytest.raises(ValueError):
        historical_v1_gate(b'{"schema_version":1,"verdict":"pass"}')


def test_historical_migration_performs_fresh_checks_not_old_winner(monkeypatch):
    from pathlib import Path
    from backend.app.engine.contracts import historical_v1_gate
    from backend.app.engine import p4_validate
    raw = (Path(__file__).resolve().parents[2] / "output/rcaeval-agent-20/_summary.json").read_bytes()
    scope = scope_for("service:s0")
    inv = INV.model_copy(update={"onset_ts": 1})
    ranking = _ranking([1]).model_copy(update={"scope_ref": scope.scope_ref})
    h = _hypothesis("h1", "service:s0").model_copy(update={"scope_ref": scope.scope_ref})
    invest = _invest(h).model_copy(update={"scope_ref": scope.scope_ref})
    fresh = TestFuseInvestigation()._validated("service:s0").hypotheses["h1"].model_copy(update={"verdict": "fail"})
    check = Mock(return_value=fresh)
    monkeypatch.setattr(p4_validate, "assess_hypothesis", check)
    migrated = historical_v1_gate(raw, mode="MIGRATION", investigation=inv, scope=scope,
        ranking=ranking, invest=invest, graph=Mock())
    check.assert_called_once()
    assert migrated.hypotheses["h1"].verdict == "fail"
    assert "not_historical_recovery" in migrated.config["migration"]


@pytest.mark.parametrize("exception", [RuntimeError("api_key=secret"), TypeError("secret")])
def test_optional_critique_unexpected_errors_are_sanitized(monkeypatch, exception):
    from backend.app.engine import p4_validate as p4
    graph = Mock()
    graph.query.return_value.result_set = [[0]]
    monkeypatch.setattr("backend.app.graph.client.graph_for", lambda _: graph)
    scope = scope_for("service:s0")
    ranking = _ranking([1]).model_copy(update={"scope_ref": scope.scope_ref})
    h = _hypothesis("h1", "service:s0").model_copy(update={"scope_ref": scope.scope_ref})
    invest = _invest(h).model_copy(update={"scope_ref": scope.scope_ref})
    monkeypatch.setattr(p4, "assess_hypothesis", lambda *args: TestFuseInvestigation()._validated("service:s0").hypotheses["h1"])
    monkeypatch.setattr(p4, "LLMClient", Mock(side_effect=AssertionError("must not construct client")))
    assert p4.validate_investigation(INV, scope, ranking, invest).hypotheses["h1"].critique == {}
    llm = Mock()
    llm.complete.side_effect = exception
    result = p4.validate_investigation(INV, scope, ranking, invest, llm=llm)
    assert "secret" not in result.model_dump_json()


def test_bounded_update_is_opt_in_one_adjacent_swap_and_safety_gated():
    ranking = _ranking([.6, .3, .1])
    invest = _invest(_hypothesis("h1", "service:s2"))
    validated = TestFuseInvestigation()._validated("service:s2")
    default = fuse_investigation(INV, ranking, invest, validated)
    assert [r["candidate"] for r in default.service_ranking] == [c.id for c in ranking.candidates]
    bounded = fuse_investigation(INV, ranking, invest, validated, ranking_policy="bounded_p2_update_v1")
    assert [r["candidate"] for r in bounded.service_ranking] == ["service:s0", "service:s2", "service:s1"]
    assert all(abs(r["rank"] - r["baseline_rank"]) <= 1 for r in bounded.service_ranking)
    ranking.truncated = True
    denied = fuse_investigation(INV, ranking, invest, validated, ranking_policy="bounded_p2_update_v1")
    assert denied.service_ranking == default.service_ranking
    ranking.truncated = False
    validated.hypotheses["h1"].verdict = "fail"
    assert fuse_investigation(INV, ranking, invest, validated, ranking_policy="bounded_p2_update_v1").winner is None


def test_fixed_stable_slots_zero_missing_no_best_of_many():
    ranking = _ranking([1])
    a, z = _hypothesis("a", "service:s0"), _hypothesis("z", "service:s0", reward=1)
    val = TestFuseInvestigation()._validated("service:s0")
    base = val.hypotheses.pop("h1")
    val.hypotheses = {h.id: base.model_copy(update={"hypothesis_id": h.id,
        "proposal_fingerprint": policy.proposal_fingerprint(h)}) for h in (a, z)}
    one = fuse_investigation(INV, ranking, _invest(a), val, per_service_slots=2)
    two = fuse_investigation(INV, ranking, _invest(z, a), val, per_service_slots=2)
    expected = sum(h.h_score for h in two.fused) / 2
    assert two.service_ranking[0]["shadow_h_score"] == pytest.approx(expected)
    assert one.service_ranking[0]["shadow_h_score"] == pytest.approx(one.fused[0].h_score / 2)
    stable = fuse_investigation(INV, ranking, _invest(z, a), val, per_service_slots=1)
    assert stable.service_ranking[0]["representative"] == "a"
    val.hypotheses["a"].verdict = "fail"
    veto = fuse_investigation(INV, ranking, _invest(z, a), val, per_service_slots=1)
    assert not veto.service_ranking[0]["eligible"]


@pytest.mark.parametrize("weights,slots", [({"bad": 1}, 1), ({"deterministic": float("nan")}, 1), (None, True), (None, 0), (None, 17)])
def test_fusion_controls_reject_unbounded_or_invalid_values(weights, slots):
    with pytest.raises(ValueError):
        fuse_investigation(INV, _ranking([1]), _invest(), ValidateResult(incident_id="i", candidates=[]),
                           weights=weights, per_service_slots=slots)
