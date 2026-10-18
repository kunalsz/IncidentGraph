"""Grounded proposal selection stays typed from provider through persisted P3."""
import json
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from backend.app.agent.llm import LLMClient, LLMResponse, MockProvider, load_prompt
from backend.app.agent.responses import Proposal
from backend.app.engine import p3_investigate as p3, policy
from backend.app.engine.budget import BudgetMeter
from backend.app.engine.contracts import (
    HypothesisRecord, Investigation, InvestigationWindow, InvestResult,
    RankResult, RankedCandidate, ScopeResult, ScopeService,
    dump_handoff, load_handoff,
)
from config import tuning


SID = "service:db"
ONSET = 1_000_000
COLUMN = "db_cpu_usage_percent"


def investigation():
    return Investigation(incident_id="grounded", session_graph="incidentgraph:session:grounded",
        symptom="errors", anchor_service=SID, budget={}, onset_ts=ONSET,
        window=InvestigationWindow(t0=0, t1=3_000_000))


def metric(**updates):
    return dict(id="m", service_id=SID, column=COLUMN, metric="cpu", z=6.0,
                gamma=7.0, direction="up", ts=ONSET) | updates


def bundle(metrics=None):
    return {"service_id": SID, "metrics": [metric()] if metrics is None else metrics,
            "logs": [], "spans": [], "changes": [], "adjacency": []}


def proposal(**updates):
    return {"candidate": SID, "text": "CPU excursion; exhaustion mechanism unproven",
            "evidence_refs": ["m"], "initial_score": .4,
            "claim_kind": "metric_excursion", "claim": {"metric": COLUMN, "direction": "up"}} | updates


class SelectingProvider(MockProvider):
    def __init__(self, proposals):
        super().__init__()
        self.proposals = proposals
        self.calls = []

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs["task"] == "hypothesis_batch":
            return LLMResponse(json.dumps({"hypotheses": self.proposals}), 1, 1, "mock")
        return super().complete(**kwargs)


def run_branch(monkeypatch, proposals, *, metrics=None, slots=3, refs=None, data=None):
    data = bundle(metrics) if data is None else data
    monkeypatch.setattr(p3.bundle_mod, "fetch_branch_bundle", lambda *args: {
        "bundle": data, "evidence_ids": refs if refs is not None else
        [m["id"] for m in data["metrics"]]})
    provider = SelectingProvider(proposals)
    meter = BudgetMeter()
    result = p3._run_branch_calls(Mock(), investigation(), SID,
        LLMClient(meter, provider=provider), meter, slots)
    return result, provider


def test_provider_to_p3_to_persisted_handoff_roundtrip(monkeypatch):
    data = bundle()
    monkeypatch.setattr(p3.bundle_mod, "fetch_branch_bundle", lambda *args: {
        "bundle": data, "evidence_ids": ["m"]})
    graph = Mock()
    monkeypatch.setattr("backend.app.graph.client.graph_for", lambda _: graph)
    inv = investigation()
    scope = ScopeResult(incident_id=inv.incident_id, anchor_service=SID, window=inv.window,
        services=[ScopeService(id=SID)], nodes=[], edges=[], evidence_gaps=[],
        n_series_in_scope=1, truncated=False)
    rank = RankResult(incident_id=inv.incident_id, scope_ref=scope.scope_ref,
        candidates=[RankedCandidate(id=SID, rank=0, score=1.0, components={},
            component_ranks={}, evidence=["m"], rule_flags=[], selected=True)],
        branch_ids=[SID], indicators=[], audit=[], n_candidates=1, truncated=False, config={})
    meter = BudgetMeter()
    provider = SelectingProvider([proposal()])
    result = p3.investigate_investigation(inv, scope, rank, meter=meter,
        llm=LLMClient(meter, provider=provider))
    assert result.n_hypotheses == 1
    h = result.hypotheses[0]
    assert h.claim_kind == "metric_excursion"
    assert h.claim.metric == COLUMN and h.claim.direction == "up"
    assert h.evidence == ["m"]
    restored = load_handoff(dump_handoff("investigate", result), stage="investigate",
        incident_id=inv.incident_id, scope_ref=scope.scope_ref)
    assert restored.hypotheses[0] == h
    persisted = [c.kwargs["params"]["rows"] for c in graph.query.call_args_list
                 if "MERGE (h:Hypothesis" in c.args[0]][0][0]
    assert json.loads(persisted["claim"])["metric"] == COLUMN
    assert persisted["claim_kind"] == "metric_excursion"
    # The downstream deterministic validator consumes the explicit fields unchanged.
    records = {"m": {"role": "local_anomaly", "kind": "MetricAnomaly", "props": metric()}}
    check = policy.claim_applicability(None, h, inv, scope, records, {"edges": []})
    assert check["verdict"] == "pass"
    assert "no causal mechanism" in check["detail"]
    assert [c["task"] for c in provider.calls] == ["self_verify", "hypothesis_batch", "consolidate"]


@pytest.mark.parametrize("updates", [
    {"claim_kind": "invented"},
    {"claim_kind": None},
    {"claim": None},
    {"claim": {}},
    {"claim": "cpu up"},
    {"claim": {"metric": COLUMN}},
    {"claim": {"metric": COLUMN, "direction": "sideways"}},
    {"claim": {"metric": 1, "direction": "up"}},
    {"claim": {"metric": " ", "direction": "up"}},
    {"claim": {"metric": COLUMN, "direction": "up", "unknown": 1}},
    {"claim": {"metric": COLUMN, "direction": "up", "change_id": "change"}},
    {"claim_kind": "service_abnormality"},
    {"unexpected": "field"},
])
def test_strict_schema_rejects_bad_fields_not_generic_downgrade(updates):
    raw = proposal(**updates)
    with pytest.raises(ValidationError):
        Proposal.model_validate(raw)
    record = {k: v for k, v in raw.items() if k != "evidence_refs"}
    with pytest.raises(ValidationError):
        HypothesisRecord.model_validate({"id": "h", "evidence": raw["evidence_refs"], **record})


@pytest.mark.parametrize("kind,claim", [
    ("dependency_propagation", {"dependency_source": "caller", "dependency_target": SID}),
    ("change_association", {"change_id": "deployment:1"}),
])
def test_other_claim_kinds_share_typed_shape(kind, claim):
    typed = Proposal.model_validate(proposal(claim_kind=kind, claim=claim))
    h = HypothesisRecord(id="h", candidate=typed.candidate, text=typed.text,
        claim_kind=typed.claim_kind, claim=typed.claim)
    assert h.claim == typed.claim


@pytest.mark.parametrize("bad", [
    {"claim": {"metric": "cpu", "direction": "up"}},  # alias is not exact column
    {"claim": {"metric": COLUMN, "direction": "down"}},
    {"claim": {"metric": "other_column", "direction": "up"}},
    {"candidate": "service:neighbor"},
    {"evidence_refs": []},
    {"evidence_refs": ["unadmitted"]},
    {"evidence_refs": ["other"]},
    {"claim": {"metric": COLUMN, "direction": "up", "unknown": True}},
])
def test_bad_proposal_rejected_without_losing_good_sibling(monkeypatch, bad):
    result, _ = run_branch(monkeypatch, [proposal(**bad), proposal()],
                           metrics=[metric(), metric(id="other", column="different")])
    assert len(result["hypotheses"]) == 1
    assert result["rejected"]
    assert result["hypotheses"][0]["candidate"]["claim_kind"] == "metric_excursion"


@pytest.mark.parametrize("updates", [
    {"column": None}, {"column": ""}, {"column": " "}, {"column": "x" * 513},
    {"z": 0}, {"z": True}, {"z": float("nan")}, {"z": -6},
    {"gamma": None}, {"gamma": 0}, {"gamma": float("inf")},
    {"direction": None}, {"direction": "down"},
    {"service_id": "service:other"},
    {"ts": ONSET + tuning.ONSET_DELTA_MS + 1}, {"ts": -1}, {"ts": True},
])
def test_options_require_exact_qualified_signed_onset_evidence(updates):
    assert p3._metric_claim_options(bundle([metric(**updates)]), {"m"}, investigation(), SID) == []


def test_options_require_admitted_refs_and_authoritative_onset():
    assert not p3._metric_claim_options(bundle(), set(), investigation(), SID)
    inv = investigation().model_copy(update={"onset_ts": None})
    assert not p3._metric_claim_options(bundle(), {"m"}, inv, SID)
    assert not p3._metric_claim_options(bundle() | {"service_id": "other"}, {"m"}, investigation(), SID)


def test_options_are_bounded_sorted_deterministic_and_group_exact_series():
    metrics = [metric(id=f"m{i}", column=f"column{i:02d}") for i in range(30)]
    refs = {m["id"] for m in metrics}
    first = p3._metric_claim_options(bundle(metrics), refs, investigation(), SID)
    second = p3._metric_claim_options(bundle(list(reversed(metrics))), refs, investigation(), SID)
    assert first == second and len(first) == 15
    pair = [metric(), metric(id="onset", ts=ONSET - 1)]
    assert p3._metric_claim_options(bundle(pair), {"m", "onset"}, investigation(), SID)[0]["evidence_refs"] == ["m", "onset"]
    pair[1].update(z=-6, direction="down")
    assert not p3._metric_claim_options(bundle(pair), {"m", "onset"}, investigation(), SID)


def test_each_offered_reference_independently_passes_baseline_check():
    metrics = [metric(id="a", gamma=None), metric(id="b", gamma=7)]
    options = p3._metric_claim_options(bundle(metrics), {"a", "b"}, investigation(), SID)
    assert options[0]["evidence_refs"] == ["b"]
    assert options == p3._metric_claim_options(bundle(list(reversed(metrics))),
                                             {"a", "b"}, investigation(), SID)


def test_downward_option_and_inclusive_onset_boundary(monkeypatch):
    data = metric(z=-6, direction="down", ts=ONSET + tuning.ONSET_DELTA_MS)
    raw = proposal(claim={"metric": COLUMN, "direction": "down"})
    result, _ = run_branch(monkeypatch, [raw], metrics=[data])
    assert len(result["hypotheses"]) == 1 and not result["rejected"]


def test_unsupported_explicit_metric_not_downgraded(monkeypatch):
    result, _ = run_branch(monkeypatch, [proposal()], metrics=[metric(column=None)])
    assert not result["hypotheses"]
    assert "supported exact-column option" in result["rejected"][0]


def test_claim_options_and_prompt_obey_slots_current_candidate_and_abstention(monkeypatch):
    result, provider = run_branch(monkeypatch, [proposal(), proposal(text="second")], slots=1)
    call = next(c for c in provider.calls if c["task"] == "hypothesis_batch")
    payload = json.loads(call["user"])
    assert payload["max_hypotheses"] == 1
    assert payload["allowed_candidate_ids"] == [SID]
    assert payload["supported_claim_options"] == [{"claim_kind": "metric_excursion",
        "claim": {"metric": COLUMN, "direction": "up"}, "evidence_refs": ["m"]}]
    assert len(result["hypotheses"]) == 1
    assert any("slot allocation" in reason for reason in result["rejected"])
    prompt = load_prompt("hypothesis_batch")
    assert "0..max_hypotheses" in prompt and "no minimum" in prompt
    assert "at least 5" not in prompt and "current candidate only" in prompt
    assert "mechanism" in prompt and "supported_claim_options" in prompt
    abstained, _ = run_branch(monkeypatch, [], slots=1)
    assert not abstained["hypotheses"] and not abstained.get("partial_error")
    zero, _ = run_branch(monkeypatch, [proposal()], slots=0)
    assert not zero["hypotheses"]


def pair_bundle():
    return bundle([]) | {"spans": [{"id": "parent"}, {"id": "child"}, {"id": "other"}],
        "span_pairs": [{"dependency_source": "caller", "dependency_target": SID,
                        "evidence_refs": ["parent", "child"], "claim_eligible": True}]}


def dependency_proposal(**updates):
    return proposal(claim_kind="dependency_propagation",
        claim={"dependency_source": "caller", "dependency_target": SID},
        evidence_refs=["parent", "child"]) | updates


def test_dependency_option_and_admission_require_offered_retained_pair(monkeypatch):
    data = pair_bundle()
    result, provider = run_branch(monkeypatch, [dependency_proposal()], data=data,
                                  refs=["parent", "child", "other"])
    assert len(result["hypotheses"]) == 1
    payload = json.loads(next(c for c in provider.calls if c["task"] == "hypothesis_batch")["user"])
    assert payload["supported_claim_options"] == [{"claim_kind": "dependency_propagation",
        "claim": {"dependency_source": "caller", "dependency_target": SID},
        "evidence_refs": ["child", "parent"]}]
    assert "BOTH" in load_prompt("hypothesis_batch")
    assert "only when offered" in load_prompt("hypothesis_batch")


@pytest.mark.parametrize("patch", [
    {"claim_eligible": False}, {"claim_eligible": 1},
    {"dependency_target": "other"}, {"dependency_source": SID},
    {"dependency_source": ""}, {"evidence_refs": ["parent"]},
    {"evidence_refs": ["parent", "parent"]},
    {"evidence_refs": ["parent", "missing"]},
])
def test_invalid_or_ineligible_span_pair_offers_nothing(patch):
    data = pair_bundle()
    data["span_pairs"][0].update(patch)
    assert not p3._dependency_claim_options(data, {"parent", "child", "missing"}, SID)


def test_dependency_pair_refs_must_be_retained_and_admitted():
    data = pair_bundle()
    assert not p3._dependency_claim_options(data, {"parent"}, SID)
    data["spans"] = [{"id": "parent"}]
    assert not p3._dependency_claim_options(data, {"parent", "child"}, SID)


@pytest.mark.parametrize("bad", [
    {"evidence_refs": ["parent"]}, {"evidence_refs": ["child"]},
    {"evidence_refs": ["parent", "other"]},
    {"claim": {"dependency_source": "invented", "dependency_target": SID}},
    {"claim": {"dependency_source": "caller", "dependency_target": "other"}},
])
def test_arbitrary_dependency_pair_or_partial_citation_rejects(monkeypatch, bad):
    result, _ = run_branch(monkeypatch, [dependency_proposal(**bad), dependency_proposal()],
                           data=pair_bundle(), refs=["parent", "child", "other"])
    assert len(result["hypotheses"]) == 1
    assert "offered exact pair" in result["rejected"][0]


def test_dependency_citations_cannot_combine_halves_of_different_pairs(monkeypatch):
    data = pair_bundle()
    data["spans"].append({"id": "second_child"})
    data["span_pairs"].append(data["span_pairs"][0] | {"evidence_refs": ["other", "second_child"]})
    result, _ = run_branch(monkeypatch,
        [dependency_proposal(evidence_refs=["parent", "second_child"])],
        data=data, refs=["parent", "child", "other", "second_child"])
    assert not result["hypotheses"] and result["rejected"]


def coverage_context(monkeypatch, *, maximum=3, fraction=1.0, consumed=0):
    from config.tuning import InvestigationBudget
    ids = [f"service:{i}" for i in range(5)]
    inv = investigation()
    scope = ScopeResult(incident_id=inv.incident_id, anchor_service=SID, window=inv.window,
        services=[ScopeService(id=s) for s in ids], nodes=[], edges=[], evidence_gaps=[],
        n_series_in_scope=0, truncated=False)
    rank = RankResult(incident_id=inv.incident_id, scope_ref=scope.scope_ref,
        candidates=[RankedCandidate(id=s, rank=i, score=1.0, components={},
            component_ranks={}, evidence=[], rule_flags=[], selected=True) for i, s in enumerate(ids)],
        branch_ids=ids, indicators=[], audit=[], n_candidates=len(ids), truncated=False, config={})
    monkeypatch.setattr("backend.app.graph.client.graph_for", lambda _: Mock())
    monkeypatch.setattr(p3.bundle_mod, "fetch_branch_bundle", lambda *args: {
        "bundle": {}, "evidence_ids": []})
    meter = BudgetMeter(InvestigationBudget(max_hypotheses=maximum, expansion_slot_fraction=fraction))
    for _ in range(consumed):
        meter.new_hypothesis()
    provider = SelectingProvider([])
    return inv, scope, rank, meter, provider


@pytest.mark.parametrize("floor,expected", [(None, 0), (0, 0), (3, 3), (9, 5)])
def test_optional_primary_coverage_borrows_expansion_slots_without_budget_increase(monkeypatch, floor, expected):
    inv, scope, rank, meter, provider = coverage_context(monkeypatch, maximum=5)
    result = p3.investigate_investigation(inv, scope, rank, meter=meter,
        llm=LLMClient(meter, provider=provider), minimum_primary_coverage=floor)
    slots = result.budget["p3_scheduling"]["primary_slots"]
    assert sum(slots.values()) == expected
    assert all(slots[s] >= 1 for s in rank.branch_ids[:expected])
    assert sum(slots.values()) <= meter.budget.max_hypotheses == 5
    batches = [json.loads(c["user"])["service_id"] for c in provider.calls if c["task"] == "hypothesis_batch"]
    assert set(batches) == set(rank.branch_ids[:expected])


@pytest.mark.parametrize("maximum,consumed", [(2, 0), (3, 1)])
def test_impossible_primary_coverage_rejects_before_any_llm_call(monkeypatch, maximum, consumed):
    inv, scope, rank, meter, provider = coverage_context(monkeypatch, maximum=maximum, consumed=consumed)
    with pytest.raises(ValueError, match="coverage exceeds"):
        p3.investigate_investigation(inv, scope, rank, meter=meter,
            llm=LLMClient(meter, provider=provider), minimum_primary_coverage=3)
    assert provider.calls == [] and meter.as_dict()["hypothesis_slots_reserved"] == 0


@pytest.mark.parametrize("floor", [True, -1, 1.5, "3"])
def test_primary_coverage_floor_is_strict(floor):
    with pytest.raises(ValueError, match="minimum_primary_coverage"):
        p3.investigate_investigation(None, None, None, minimum_primary_coverage=floor)


def test_legacy_mock_generic_is_accepted_without_inferred_claim(monkeypatch):
    legacy = proposal()
    legacy.pop("claim_kind")
    legacy.pop("claim")
    result, _ = run_branch(monkeypatch, [legacy])
    admitted = result["hypotheses"][0]["candidate"]
    assert admitted["claim_kind"] == "service_abnormality" and admitted["claim"] is None
    provider = MockProvider()
    raw = provider.complete(task="hypothesis_batch", system=load_prompt("hypothesis_batch"),
        user=json.dumps({"service_id": SID, "bundle": bundle(), "max_hypotheses": 1}),
        model="mock", max_tokens=1600).json()["hypotheses"][0]
    assert Proposal.model_validate(raw).claim_kind == "service_abnormality"
