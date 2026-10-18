"""Provider/database-free regressions for audited P4/P5 scoring mismatches."""
from copy import deepcopy
from unittest.mock import Mock

import pytest

from backend.app.engine import p4_validate as p4, policy
from backend.app.engine.contracts import (
    HypothesisRecord, HypothesisValidation, Investigation, InvestigationWindow,
    InvestResult, RankedCandidate, RankResult, ScopeResult, ScopeService,
    StructuredClaim, ValidateResult,
)
from backend.app.engine.p5_fuse import fuse_investigation


@pytest.fixture(autouse=True)
def no_provider(monkeypatch):
    monkeypatch.setattr(p4, "LLMClient", Mock(side_effect=AssertionError("no providers")))


def _case(order=("s2", "s1", "s0", "s3")):
    inv = Investigation(incident_id="audit", session_graph="incidentgraph:session:audit",
        symptom="errors", anchor_service="s0", budget={}, onset_ts=10,
        window=InvestigationWindow(t0=0, t1=100))
    observations = {
        f"m:{s}": (s, "MetricAnomaly", {"id": f"m:{s}", "ts": 10,
            "metric": "resource", "z": 6, "gamma": 8, "direction": "up"})
        for s in order
    }
    for s, ts, duration, parent in (("s0", 10, 30000, None),
                                    ("s1", 11, 20000, "s0"),
                                    ("s2", 12, 1000, "s1")):
        observations[f"p:{s}"] = (s, "Span", {"id": f"p:{s}", "ts": ts,
            "status": 2, "trace_id": "trace", "span_id": s,
            "parent_id": parent, "duration_us": duration})
    scope = ScopeResult(incident_id=inv.incident_id, anchor_service="s0", window=inv.window,
        evidence_gaps=[], n_series_in_scope=0, truncated=False,
        services=[ScopeService(id=s) for s in order],
        nodes=[{"id": s, "type": "Service", "properties": {
            "level": 0 if s == "s0" else 1 if s in ("s1", "s3") else 2,
            "level_convention": "dependency_depth_v1"}} for s in order]
            + [{"id": sid, "type": kind, "properties": deepcopy(props)}
               for sid, (_, kind, props) in observations.items()],
        edges=[{"source": a, "target": b, "type": "DEPENDS_ON"}
               for a, b in (("s0", "s1"), ("s1", "s2"), ("s0", "s3"))]
            + [{"source": sid, "target": s, "type": "OBSERVED_ON"}
               for sid, (s, _, _) in observations.items()])
    ranking = RankResult(incident_id=inv.incident_id, scope_ref=scope.scope_ref,
        branch_ids=list(order), indicators=[], audit=[], n_candidates=len(order),
        truncated=False, config={},
        candidates=[RankedCandidate(id=s, rank=i, score=1 / i, components={},
            component_ranks={}, evidence=[], rule_flags=[], selected=True)
                    for i, s in enumerate(order, 1)])
    return inv, scope, ranking, observations


def _proposal(scope, hid, service, reward=.5, evidence=None, **kwargs):
    return HypothesisRecord(id=hid, candidate=service, scope_ref=scope.scope_ref,
        text="observed abnormality", reward=reward,
        evidence=evidence if evidence is not None else [f"m:{service}"], **kwargs)


def _invest(inv, scope, hypotheses):
    return InvestResult(incident_id=inv.incident_id, scope_ref=scope.scope_ref,
        branches=[], hypotheses=hypotheses, n_hypotheses=len(hypotheses))


def _validate(monkeypatch, case, hypotheses):
    inv, scope, ranking, observations = case
    by_id = {h.id: h for h in hypotheses}
    def query(statement, params=None, **kwargs):
        rows = []
        if "OBSERVED_ON" in statement:
            for sid in by_id[params["hid"]].evidence:
                if sid not in observations:
                    continue
                service, kind, props = observations[sid]
                rows.append([{"kind": kind, "ts": props["ts"], "polarity": "support"},
                    {"polarity": "support", "weight": 1}, deepcopy(props), [kind], service])
        return Mock(result_set=rows)
    graph = Mock()
    graph.query.side_effect = query
    monkeypatch.setattr("backend.app.graph.client.graph_for", lambda _: graph)
    return p4.validate_investigation(inv, scope, ranking, _invest(inv, scope, hypotheses))


@pytest.mark.parametrize("services", [("s2",), ("s2", "s1"), ("s2", "s1", "s0")])
def test_topology_and_proposal_membership_without_support_are_zero(monkeypatch, services):
    case = _case()
    inv, scope, ranking, _ = case
    hypotheses = [_proposal(scope, s, s) for s in services]
    validated = _validate(monkeypatch, case, hypotheses)
    for assessment in validated.hypotheses.values():
        assert assessment.chain_coherence == 0
        assert assessment.evidence_roles["propagation"] == []
        assert all(not e["support"] for e in assessment.evidence_path.edges)
        assert assessment.verdict != "fail"
        assert policy.evidence_strength(assessment)[2]
    fused = fuse_investigation(inv, ranking, _invest(inv, scope, hypotheses), validated)
    assert fused.winner is not None
    assert all(h.components["CC"] == 0 for h in fused.fused)


@pytest.mark.parametrize("spans,expected", [([], 0), (["p:s0", "p:s1"], 0),
    (["p:s1", "p:s2"], .5), (["p:s0", "p:s1", "p:s2"], 1)])
def test_only_contiguous_supported_callers_count(monkeypatch, spans, expected):
    case = _case()
    _, scope, _, _ = case
    h = _proposal(scope, "root", "s2", evidence=["m:s2", *spans])
    validated = _validate(monkeypatch, case,
        [h, _proposal(scope, "middle", "s1"), _proposal(scope, "anchor", "s0")])
    assert validated.hypotheses[h.id].chain_coherence == expected


@pytest.mark.parametrize("mutation", ["trace", "parent", "status", "interval", "duration", "uncited", "foreign_owner"])
def test_invalid_or_unowned_spans_cannot_supply_coherence(monkeypatch, mutation):
    case = _case()
    _, scope, _, observations = case
    child = observations["p:s2"][2]
    evidence = ["m:s2", "p:s1", "p:s2"]
    if mutation == "trace":
        child["trace_id"] = "other"
    elif mutation == "parent":
        child["parent_id"] = "other"
    elif mutation == "status":
        child["status"] = 1
    elif mutation == "interval":
        child["duration_us"] = 90000
    elif mutation == "duration":
        child["duration_us"] = 0
    elif mutation == "uncited":
        evidence.remove("p:s1")
    else:
        observations["p:s1"] = ("outside", "Span", observations["p:s1"][2])
    h = _proposal(scope, "root", "s2", evidence=evidence)
    sibling = _proposal(scope, "sibling", "s2", evidence=["m:s2", "p:s1", "p:s2"])
    validated = _validate(monkeypatch, case, [h, sibling, _proposal(scope, "middle", "s1")])
    assert validated.hypotheses[h.id].chain_coherence == 0
    assert validated.hypotheses[h.id].evidence_roles["propagation"] == []
    if mutation == "uncited":
        assert validated.hypotheses[sibling.id].chain_coherence == .5


def test_structured_propagation_semantics_preserved(monkeypatch):
    case = _case()
    inv, scope, ranking, _ = case
    h = _proposal(scope, "root", "s2", evidence=["m:s2", "p:s1", "p:s2"],
        claim_kind="dependency_propagation",
        claim=StructuredClaim(dependency_source="s1", dependency_target="s2"))
    hypotheses = [h, _proposal(scope, "middle", "s1")]
    validated = _validate(monkeypatch, case, hypotheses)
    a = validated.hypotheses[h.id]
    claim = next(c for c in a.checks if c.name == "claim_applicability")
    assert claim.verdict == "pass" and "not established" in claim.detail
    assert a.chain_coherence == .5
    assert a.evidence_path.edges[-1]["support"] == ["p:s1", "p:s2"]
    assert policy.evidence_strength(a)[2]
    fused = fuse_investigation(inv, ranking, _invest(inv, scope, hypotheses), validated)
    assert next(h for h in fused.fused if h.id == "root").components["CC"] == .5


@pytest.mark.parametrize("metric", ["mem", "socket"])
def test_resource_candidate_not_demoted_by_empty_propagation_bonus(monkeypatch, metric):
    case = _case(order=("s3", "s2", "s1", "s0"))
    inv, scope, ranking, observations = case
    observations["m:s3"][2]["metric"] = metric
    hypotheses = [_proposal(scope, s, s) for s in ("s3", "s2", "s1", "s0")]
    validated = _validate(monkeypatch, case, hypotheses)
    fused = fuse_investigation(inv, ranking, _invest(inv, scope, hypotheses), validated,
        ranking_policy="corrected_h_eligible_first")
    assert fused.winner_candidate == "s3"
    assert fused.service_ranking[0]["candidate"] == "s3"
    assert all(h.components["CC"] == 0 for h in fused.fused)


def _fusion_case(specs):
    inv, scope, ranking, _ = _case(order=("s0", "s1", "s2", "s3"))
    hypotheses, assessments = [], {}
    for hid, service, reward, verdict in specs:
        h = _proposal(scope, hid, service, reward)
        hypotheses.append(h)
        if verdict is None:
            continue
        assessments[hid] = HypothesisValidation(hypothesis_id=hid, candidate=service,
            scope_ref=scope.scope_ref, proposal_fingerprint=policy.proposal_fingerprint(h),
            evidence_scope=inv.session_graph, verdict=verdict, qualified_timing=True,
            plausible_path=True, evidence_roles={"local_anomaly": [f"m:{service}"]},
            source_fingerprints={f"m:{service}": "fingerprint"}, modalities=["metrics"])
    validated = ValidateResult(incident_id=inv.incident_id, scope_ref=scope.scope_ref,
        candidates=[], hypotheses=assessments)
    return inv, ranking, _invest(inv, scope, hypotheses), validated


REWARD_ONLY = {"deterministic": 0, "reflection": 1, "chain_coherence": 0, "validator": 0}


@pytest.mark.parametrize("verdict", ["pass", "partial", "fail", None])
@pytest.mark.parametrize("ranking_policy", ["deterministic_first_shadow", "corrected_h_eligible_first", "bounded_p2_update_v1"])
def test_later_sibling_cannot_win_or_rescue_fixed_slot(verdict, ranking_policy):
    args = _fusion_case([("a", "s0", .1, verdict), ("z", "s0", 1, "pass"),
                         ("b", "s1", .6, "pass")])
    result = fuse_investigation(*args, weights=REWARD_ONLY, ranking_policy=ranking_policy)
    assert result.winner == "b" and result.winner_candidate == "s1"
    assert result.fused[0].id == "z" and result.fused[0].h_score == 1
    assert len(result.fused) == 3
    row = next(r for r in result.service_ranking if r["candidate"] == "s0")
    assert row["representative"] == ("a" if verdict in ("pass", "partial") else None)
    assert result.alternatives == (["a"] if verdict in ("pass", "partial") else [])
    if ranking_policy == "deterministic_first_shadow":
        assert [r["candidate"] for r in result.service_ranking] == ["s0", "s1", "s2", "s3"]


@pytest.mark.parametrize("slots", [2, 3])
def test_winner_uses_service_mean_not_peak_and_matches_representative(slots):
    args = _fusion_case([("a", "s0", .1, "pass"), ("b", "s0", 1, "pass"),
                         ("c", "s1", .7, "pass"), ("d", "s1", .7, "pass")])
    result = fuse_investigation(*args, weights=REWARD_ONLY, per_service_slots=slots,
        ranking_policy="corrected_h_eligible_first")
    assert result.fused[0].id == "b"  # peak is audit-only, not the service winner
    assert result.winner == "c" and result.winner_candidate == "s1"
    assert result.service_ranking[0]["representative"] == result.winner
    assert result.service_ranking[0]["shadow_h_score"] == pytest.approx(1.4 / slots)
    assert result.service_ranking[1]["shadow_h_score"] == pytest.approx(1.1 / slots)
    assert result.alternatives == ["a"]
    # Proposal iteration order never changes slots, tie-breaking, or audit order.
    args[2].hypotheses.reverse()
    assert fuse_investigation(*args, weights=REWARD_ONLY, per_service_slots=slots,
        ranking_policy="corrected_h_eligible_first") == result


def test_all_reserved_slots_ineligible_has_no_winner_despite_valid_sibling():
    args = _fusion_case([("a", "s0", 0, "fail"), ("b", "s0", 0, None),
                         ("z", "s0", 1, "pass")])
    result = fuse_investigation(*args, per_service_slots=2)
    assert result.winner is result.winner_candidate is None
    assert result.alternatives == [] and len(result.fused) == 3
    assert all(not row["eligible"] for row in result.service_ranking)


def test_service_mean_ties_use_baseline_rank_not_peak_or_proposal_order():
    args = _fusion_case([("z", "s0", .4, "pass"), ("a", "s1", .8, "pass"),
                         ("y", "s0", .4, "pass")])
    result = fuse_investigation(*args, weights=REWARD_ONLY, per_service_slots=2,
        ranking_policy="corrected_h_eligible_first")
    assert result.winner == "y" and result.winner_candidate == "s0"
    assert result.service_ranking[0]["representative"] == "y"
    assert result.service_ranking[0]["shadow_h_score"] == result.service_ranking[1]["shadow_h_score"]
    assert result.alternatives == ["a"]
