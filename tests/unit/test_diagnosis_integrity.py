"""First milestone regressions: terminal eligibility and graceful boundary rejection."""
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from backend.app.agent.llm import LLMError, LLMResponse
from backend.app.engine import p3_investigate as p3, p6_report as p6
from backend.app.engine.contracts import (
    BranchVerdict, CheckResult, HypothesisRecord, InvestResult, ScopeResult,
    ScopeService, ValidatedCandidate, ValidateResult, HypothesisValidation,
)
from backend.app.engine.p5_fuse import fuse_investigation
from backend.app.graph.validators import svar_lite
from backend.app.report.validator import gate
from tests.unit.test_p5_fuse import INV, REF, _hypothesis, _invest, _ranking
from backend.app.engine import policy


def validation(candidate, verdict="pass", coherence=1, hid="h"):
    return HypothesisValidation(candidate=candidate, verdict=verdict, hypothesis_id=hid,
        chain_coherence=coherence, evidence_scope=INV.session_graph,
        qualified_timing=True, plausible_path=True,
        evidence_roles={"local_anomaly": ["m", "l"]}, modalities=["metrics", "logs"],
        source_fingerprints={"m": "test", "l": "test"}, scope_ref=REF,
        proposal_fingerprint=policy.proposal_fingerprint(_hypothesis(hid, candidate)))


def test_rejected_point8_cannot_beat_valid_point525():
    result = fuse_investigation(
        INV, _ranking([.9, .1]),
        _invest(_hypothesis("bad", "service:s0", reward=1),
                _hypothesis("good", "service:s1", reward=.5)),
        ValidateResult(incident_id="i", scope_ref=REF, candidates=[], hypotheses={
            "bad": validation("service:s0", "fail", hid="bad"), "good": validation("service:s1", hid="good")}))
    assert {h.id: h.h_score for h in result.fused} == {"bad": .8, "good": .525}
    assert result.winner == "good"
    assert "bad" not in result.alternatives


@pytest.mark.parametrize("candidates", [[], [validation("service:s0", "fail")]])
def test_no_validation_or_all_fail_has_no_winner(candidates):
    result = fuse_investigation(INV, _ranking([1]),
                               _invest(_hypothesis("h", "service:s0")),
                               ValidateResult(incident_id="i", scope_ref=REF, candidates=candidates))
    assert result.winner is result.winner_candidate is None
    assert result.alternatives == []


@pytest.mark.parametrize("candidate", ["insufficient_evidence", "service:fake", "service:outside"])
def test_invalid_candidate_excluded_even_with_validator(candidate):
    result = fuse_investigation(INV, _ranking([1]),
                               _invest(_hypothesis("h", candidate)),
                               ValidateResult(incident_id="i", scope_ref=REF, candidates=[validation(candidate)]))
    assert result.winner is None and result.alternatives == []


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), -.1, 1.01])
def test_invalid_confidence_contract_and_gate(value):
    with pytest.raises(ValidationError):
        _hypothesis("h", "service:s0", confidence=value)
    result = gate(has_winner=True, confidence=value, coherence=1,
                  validator_all_pass=True, per_link_support=True)
    assert result["verdict"] == "insufficient_evidence"


@pytest.mark.parametrize("state", ["fail", None])
def test_gate_fail_or_missing_validation_abstains(state):
    result = gate(has_winner=True, confidence=.9, coherence=1,
                  validator_all_pass=False, validator_verdict=state,
                  per_link_support=True)
    assert result["verdict"] == "insufficient_evidence"
    assert result["tier"] == "INCONCLUSIVE"


def test_gate_explicit_partial_remains_likely():
    assert gate(has_winner=True, confidence=.9, coherence=.5,
                validator_all_pass=False, validator_verdict="partial",
                per_link_support=True)["tier"] == "LIKELY"


@pytest.mark.parametrize("text", ['{"confidence":NaN}', '{"confidence":Infinity}',
                                 '{"confidence":1e999}', '[]'])
def test_provider_rejects_nonfinite_and_nonobject(text):
    with pytest.raises(LLMError):
        LLMResponse(text, 0, 0, "test").json()


def test_directionless_metric_is_typed_partial():
    assert CheckResult(**svar_lite([{"z": 5, "gamma": 9}])).verdict == "partial"


def scope_for(*ids):
    return ScopeResult(incident_id="i", anchor_service=INV.anchor_service,
                       window=INV.window, services=[ScopeService(id=s) for s in ids],
                       nodes=[], edges=[], evidence_gaps=[], n_series_in_scope=0,
                       truncated=False)


@pytest.mark.parametrize("state", ["fail", None])
def test_rejected_or_unvalidated_report_no_action_or_composition(monkeypatch, state):
    graph = Mock()
    monkeypatch.setattr("backend.app.graph.client.graph_for", lambda _: graph)
    closure = Mock(side_effect=AssertionError("abstention must not retrieve remediation"))
    monkeypatch.setattr("backend.app.agent.tools.t_get_runbook_and_owner", closure)
    ranking = _ranking([1])
    invest = _invest(_hypothesis("h", "service:s0"))
    # Forge an old/permissive P5 result: P6 must independently enforce rejection.
    valid = ValidateResult(incident_id="i", scope_ref=REF, candidates=[validation("service:s0")])
    fused = fuse_investigation(INV, ranking, invest, valid)
    rejected = ValidateResult(incident_id="i", scope_ref=REF, candidates=(
        [validation("service:s0", state)] if state else []))
    llm = Mock()
    report = p6.report_investigation(INV, scope_for("service:s0"), ranking,
                                    invest, rejected, fused, llm=llm)
    assert report.verdict == "insufficient_evidence"
    assert report.winner_candidate is None
    assert report.runbook == {"runbooks": []} and report.owner == {}
    assert report.next_action and "missing" in report.next_action.lower()
    llm.complete.assert_not_called()


def test_expansion_failure_preserves_successful_sibling(monkeypatch):
    graph = Mock()
    monkeypatch.setattr("backend.app.graph.client.graph_for", lambda _: graph)
    scope = scope_for("service:s0", "service:s1", "service:s2")
    scope.edges = [{"source": "service:s0", "target": target, "type": "DEPENDS_ON"}
                   for target in ("service:s1", "service:s2")]
    ranking = _ranking([.8, .1, .1])
    ranking.branch_ids = ["service:s0"]
    ranking.scope_ref = scope.scope_ref
    def calls(graph, investigation, node, llm, meter, *args):
        if node == "service:s1":
            raise LLMError("optional sibling failed")
        return {"branch_id": node, "bundle": {}, "evidence_ids": [],
                "verdict": p3.NodeVerdict(node_id=node, is_abnormal=True,
                    has_error_logs=True, has_metric_anomaly=True, confidence=.8),
                "escalation": None, "hypotheses": [{
                    "candidate": {"candidate": node, "text": "test", "initial_score": 0.0, "evidence_refs": []},
                    "reward": .7, "reflection": {}, "sc": .5}]}
    def consolidate(node, *args):
        return BranchVerdict(node_id=node, local_root_cause=node,
                             confidence=.1 if node == "service:s0" else .8)
    monkeypatch.setattr(p3, "_run_branch_calls", calls)
    monkeypatch.setattr(p3, "_consolidate", consolidate)
    result = p3.investigate_investigation(INV, scope, ranking, llm=Mock())
    assert {h.candidate for h in result.hypotheses} == {"service:s0", "service:s2"}
    assert any(b.local_root_cause == "service:s2" for b in result.branches)
    assert any(e.get("status") == "partial" and "service:s1" in e.get("failed_ids", [])
               for e in result.expansions)
    assert all("branch" in e and "batches" in e for e in result.expansions)


def test_invalid_proposal_sibling_does_not_discard_valid_one(monkeypatch):
    from backend.app.agent.llm import LLMClient, MockProvider
    from backend.app.engine.budget import BudgetMeter
    class Provider(MockProvider):
        def complete(self, **kwargs):
            response = super().complete(**kwargs)
            if kwargs["task"] == "hypothesis_batch":
                payload = response.json()
                payload["hypotheses"].insert(0, {"candidate": "service:fake"})
                return LLMResponse(json.dumps(payload), 0, 0, "mock")
            return response
    monkeypatch.setattr(p3.bundle_mod, "fetch_branch_bundle", lambda *args: {
        "bundle": {"metrics": [{"id": "m", "z": 5}], "logs": []}, "evidence_ids": ["m"]})
    monkeypatch.setattr(p3, "_available_kinds", lambda *args: ["metrics"])
    meter = BudgetMeter()
    result = p3._run_branch_calls(Mock(), INV, "service:s0",
                                 LLMClient(meter, provider=Provider()), meter)
    assert result["hypotheses"]
    assert all(h["candidate"]["candidate"] == "service:s0" for h in result["hypotheses"])


@pytest.mark.parametrize("confidence", [float("nan"), 1.1])
def test_invalid_branch_confidence_degrades_without_losing_sibling(monkeypatch, confidence):
    from backend.app.agent.llm import LLMClient, MockProvider
    class Provider(MockProvider):
        def complete(self, **kwargs):
            if kwargs["task"] == "self_verify":
                payload = json.loads(kwargs["user"])
                if payload["service_id"] == "service:s0":
                    return LLMResponse(json.dumps({"node_id": "service:s0",
                        "is_abnormal": True, "has_error_logs": True,
                        "has_metric_anomaly": True, "confidence": confidence}), 0, 0, "mock")
            return super().complete(**kwargs)
    monkeypatch.setattr("backend.app.graph.client.graph_for", lambda _: Mock())
    monkeypatch.setattr(p3.bundle_mod, "fetch_branch_bundle", lambda graph, node, window: {
        "bundle": {"metrics": [{"id": "m", "metric": "cpu", "z": 5}], "logs": []},
        "evidence_ids": ["m"]})
    monkeypatch.setattr(p3, "_available_kinds", lambda *args: ["metrics"])
    scope = scope_for("service:s0", "service:s1")
    ranking = _ranking([.9, .1])
    ranking.scope_ref = scope.scope_ref
    result = p3.investigate_investigation(INV, scope,
                                         ranking,
                                         llm=LLMClient(p3.BudgetMeter(), provider=Provider()))
    assert any(b.node_id == "service:s0" and b.status == "unexplored" for b in result.branches)
    assert result.hypotheses and {h.candidate for h in result.hypotheses} == {"service:s1"}


def test_consolidation_rejects_undisclosed_candidate():
    llm = Mock()
    llm.complete.return_value = LLMResponse(json.dumps({"node_id": "service:s0",
        "local_root_cause": "service:outside", "confidence": .9}), 0, 0, "mock")
    verdict = p3.NodeVerdict(node_id="service:s0", is_abnormal=True,
                             has_error_logs=True, has_metric_anomaly=True)
    with pytest.raises(ValueError, match="undisclosed"):
        p3._consolidate("service:s0", verdict, [], [], llm)


def test_check_failure_is_terminal_even_with_permissive_aggregate():
    assessment = validation("service:s0")
    assessment.checks = [CheckResult(name="veto", verdict="fail")]
    result = fuse_investigation(INV, _ranking([1]),
        _invest(_hypothesis("h", "service:s0")),
        ValidateResult(incident_id="i", scope_ref=REF, candidates=[], hypotheses={"h": assessment}))
    assert result.winner is None and result.fused[0].status == "disproved"


def test_historical_all20_matched16_metrics_and_cost_qualification():
    fixture = json.loads((Path(__file__).resolve().parents[2] /
                          "audit/historical-all20-matched16.json").read_text())
    baseline, agent = (fixture["runs"][k] for k in ("baseline", "agent"))
    matched = {c["case"] for c in agent if not c.get("error")}
    def metrics(cases):
        ranks = [(c["service_order"].index(c["gt_service"]) + 1
                  if c["gt_service"] in c["service_order"] else float("inf")) for c in cases]
        return [sum(r <= k for r in ranks) / len(ranks) for k in (1, 3, 5)] + [
            sum(1/r for r in ranks) / len(ranks)]
    assert len(baseline) == len(agent) == 20 and len(matched) == 16
    assert metrics(baseline) == pytest.approx([.4, .65, .95, .5841666666666667])
    assert metrics(agent) == pytest.approx([.15, .5, .65, .3433333333333333])
    assert metrics([c for c in baseline if c["case"] in matched]) == pytest.approx(
        [.4375, .5625, .9375, .5947916666666666])
    assert metrics([c for c in agent if c["case"] in matched]) == pytest.approx(
        [.1875, .625, .8125, .4291666666666667])
    b = {c["case"]: c for c in baseline}
    transitions = [0, 0, 0]
    for c in agent:
        if c["case"] not in matched:
            assert c["error"] == "KeyError: 'branch'"
            continue
        old = b[c["case"]]["service_order"].index(c["gt_service"])
        new = c["service_order"].index(c["gt_service"])
        transitions[0 if new < old else 1 if new == old else 2] += 1
    assert transitions == [4, 5, 7]
    completed = [c["agent"] for c in agent if c["case"] in matched]
    assert sum(c["llm_calls"] for c in completed) == 1548
    assert sum(c["tokens_total"] for c in completed) == 4052621
    assert sum(c["truncated"] for c in completed) == 11
    assert "not an all20 cost estimate" in fixture["cost_qualification"]
