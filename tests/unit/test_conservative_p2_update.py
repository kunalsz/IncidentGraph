"""Synthetic, provider/DB-free conservative association-update regressions."""
import pytest

from backend.app.engine import policy
from backend.app.engine.contracts import (
    CheckResult, EvidencePath, HypothesisRecord, HypothesisValidation,
    Investigation, InvestigationWindow, InvestResult, RankedCandidate,
    RankResult, ScopeRef, StructuredClaim, ValidateResult,
)
from backend.app.engine.p5_fuse import fuse_investigation


POLICY = "conservative_p2_update_v1"


def case():
    ref = ScopeRef(incident_id="i", fingerprint="a" * 64)
    inv = Investigation(incident_id="i", session_graph="incidentgraph:session:i",
        anchor_service="anchor", symptom="errors", budget={}, onset_ts=100,
        window=InvestigationWindow(t0=0, t1=200000))
    order = ["unknown", "incumbent", "challenger", "other_unknown"]
    ranking = RankResult(incident_id="i", scope_ref=ref, n_candidates=4,
        truncated=False, config={}, branch_ids=order, indicators=[], audit=[],
        candidates=[RankedCandidate(id=s, rank=i, score=1 / i, components={},
            component_ranks={}, evidence=[], rule_flags=[], selected=True)
            for i, s in enumerate(order, 1)])
    hypotheses = [HypothesisRecord(id=hid, candidate=s, scope_ref=ref,
        text="explicit metric association", claim_kind="metric_excursion",
        claim=StructuredClaim(metric="cpu", direction="up"),
        evidence=[f"{hid}:metric", f"{hid}:log"])
        for hid, s in [("a", "incumbent"), ("b", "challenger")]]
    assessments = {}
    for h in hypotheses:
        assessments[h.id] = HypothesisValidation(hypothesis_id=h.id,
            candidate=h.candidate, scope_ref=ref, claim_type=h.claim_kind,
            proposal_fingerprint=policy.proposal_fingerprint(h),
            evidence_scope=inv.session_graph, verdict="pass",
            checks=[CheckResult(name="claim_applicability", verdict="pass"),
                    CheckResult(name="onset_alignment", verdict="pass")],
            modalities=["metrics", "logs"], qualified_timing=True, plausible_path=True,
            evidence_roles={"local_anomaly": h.evidence},
            source_fingerprints={sid: "fp" for sid in h.evidence})
    invest = InvestResult(incident_id="i", scope_ref=ref, branches=[],
        hypotheses=hypotheses, n_hypotheses=2)
    validated = ValidateResult(incident_id="i", scope_ref=ref,
        hypotheses=assessments, candidates=[])
    return inv, ranking, invest, validated


def rebind(args):
    for h in args[2].hypotheses:
        a = args[3].hypotheses.get(h.id)
        if a:
            a.proposal_fingerprint = policy.proposal_fingerprint(h)
            a.claim_type = h.claim_kind
    args[2].n_hypotheses = len(args[2].hypotheses)


def run(args, **kwargs):
    default = fuse_investigation(*args, **kwargs)
    result = fuse_investigation(*args, ranking_policy=POLICY, **kwargs)
    assert result.service_ranking == default.service_ranking
    assert result.winner == default.winner
    assert result.alternatives == default.alternatives
    assert [r["candidate"] for r in result.service_ranking] == [c.id for c in args[1].candidates]
    audit = result.config["conservative_update"]
    assert audit["decision"] == "abstain" and audit["swaps"] == 0
    assert audit["max_adjacent_swaps"] == audit["max_rank_displacement"] == 1
    assert audit["semantics"] == "association_only_not_causal_superiority"
    assert audit["supported_positive_triggers"] == []
    return audit


def attempt(audit):
    return next(a for a in audit["attempts"] if a["challenger"] == "challenger")


@pytest.mark.parametrize("state", ["absent_proposal", "absent_assessment", "partial",
    "missing_path", "missing_authority", "late", "generic_fail", "provider_failure",
    "wrong_scope", "wrong_fingerprint", "candidate_summary_only"])
def test_unknown_or_failed_incumbent_never_grants_permission(state):
    args = case()
    a = args[3].hypotheses["a"]
    if state == "absent_proposal":
        args[2].hypotheses.pop(0)
        args[2].n_hypotheses -= 1
    elif state == "absent_assessment":
        del args[3].hypotheses["a"]
    elif state == "partial":
        a.verdict = "partial"
    elif state == "missing_path":
        a.plausible_path = False
        a.checks.append(CheckResult(name="scoped_path", verdict="partial"))
    elif state == "missing_authority":
        a.checks.append(CheckResult(name="level_veto", verdict="partial"))
    elif state == "late":
        a.qualified_timing = False
        a.checks.append(CheckResult(name="onset_alignment", verdict="partial"))
    elif state == "generic_fail":
        a.verdict = "fail"
        a.checks.append(CheckResult(name="level_veto", verdict="fail"))
        a.contradictions = ["ev:refute:i:a"]
    elif state == "provider_failure":
        args[2].budget["p3_execution_errors"] = {"incumbent": "LLMError"}
    elif state == "wrong_scope":
        a.scope_ref = ScopeRef(incident_id="i", fingerprint="b" * 64)
    elif state == "wrong_fingerprint":
        a.proposal_fingerprint = "stale"
    else:
        args[3].candidates = [a]
        del args[3].hypotheses["a"]
    run(args)


@pytest.mark.parametrize("phase", [1, 2])
def test_truncation_blocks_even_with_strong_challenger(phase):
    args = case()
    args[phase].truncated = True
    assert run(args)["reason"] == "incomplete_ranking_or_investigation"


@pytest.mark.parametrize("mutation", ["low_confidence", "one_modality", "partial",
    "generic_anomaly", "no_claim", "claim_partial", "contradiction", "refutation_role",
    "partial_check", "wrong_scope", "stale", "legacy"])
def test_challenger_must_have_unanimous_exact_structured_crossmodal_proof(mutation):
    args = case()
    h, a = args[2].hypotheses[1], args[3].hypotheses["b"]
    if mutation == "low_confidence":
        a.confidence_cap = .84
    elif mutation == "one_modality":
        a.modalities = ["metrics"]
    elif mutation == "partial":
        a.verdict = "partial"
    elif mutation == "generic_anomaly":
        h.claim_kind, h.claim = "service_abnormality", None
        a.checks[0].verdict = "not_applicable"
        rebind(args)
    elif mutation == "no_claim":
        h.claim = None
        rebind(args)
    elif mutation == "claim_partial":
        a.checks[0].verdict = "partial"
    elif mutation == "contradiction":
        a.contradictions = ["refute"]
    elif mutation == "refutation_role":
        a.evidence_roles["refutation"] = [h.evidence[0]]
    elif mutation == "partial_check":
        a.checks.append(CheckResult(name="level_veto", verdict="partial"))
    elif mutation == "wrong_scope":
        a.scope_ref = ScopeRef(incident_id="i", fingerprint="b" * 64)
    elif mutation == "stale":
        a.proposal_fingerprint = "stale"
    else:
        a.schema_version = 1
    assert attempt(run(args))["reason"] != "incumbent_explicit_claim_refutation_unsupported"


def propagation_case():
    args = case()
    h, a = args[2].hypotheses[1], args[3].hypotheses["b"]
    h.claim_kind = "dependency_propagation"
    h.claim = StructuredClaim(dependency_source="middle", dependency_target=h.candidate)
    h.evidence += ["p0", "p1", "p2"]
    a.source_fingerprints.update({sid: "fp" for sid in ["p0", "p1", "p2"]})
    a.evidence_roles["propagation"] = ["p0", "p1", "p2"]
    a.evidence_path = EvidencePath(from_id="anchor", to_id=h.candidate,
        found=True, status="found", nodes=[{"id": sid} for sid in ["anchor", "middle", h.candidate]],
        edges=[{"source": "anchor", "target": "middle", "type": "DEPENDS_ON", "support": ["p0", "p1"]},
               {"source": "middle", "target": h.candidate, "type": "DEPENDS_ON", "support": ["p1", "p2"]}])
    rebind(args)
    return args


@pytest.mark.parametrize("mutation", ["none", "empty", "truncated", "prefix_unsupported",
    "unowned", "wrong_anchor", "disconnected", "not_found"])
def test_propagation_requires_every_link_to_anchor(mutation):
    args = propagation_case()
    a = args[3].hypotheses["b"]
    if mutation == "none":
        a.evidence_path = None
    elif mutation == "empty":
        a.evidence_path.edges = []
    elif mutation == "truncated":
        a.evidence_path.status = "truncated"
    elif mutation == "prefix_unsupported":
        a.evidence_path.edges[0]["support"] = []
    elif mutation == "unowned":
        a.evidence_path.edges[0]["support"] = ["outside"]
    elif mutation == "wrong_anchor":
        a.evidence_path.from_id = "outside"
    elif mutation == "disconnected":
        a.evidence_path.edges[1]["source"] = "anchor"
    else:
        a.evidence_path.found = False
    assert attempt(run(args))["reason"] == "challenger_propagation_path_not_fully_supported"


def test_full_supported_path_still_requires_incumbent_refutation():
    assert attempt(run(propagation_case()))["reason"] == "incumbent_explicit_claim_refutation_unsupported"


def test_metric_claim_failure_does_not_distinguish_missing_metadata_from_contradiction():
    args = case()
    inv, _, invest, validated = args
    h = invest.hypotheses[0]
    records = {"a:metric": {"role": "local_anomaly", "kind": "MetricAnomaly",
        "props": {"metric": "cpu", "z": 6, "gamma": 8, "ts": 100}}}
    missing = policy.claim_applicability(None, h, inv, None, records, {"edges": []})
    records["a:metric"]["props"].update(direction="down", z=-6)
    opposite = policy.claim_applicability(None, h, inv, None, records, {"edges": []})
    assert missing == opposite
    assert opposite["verdict"] == "fail"
    assert opposite["detail"] == "cited series contradicts claimed direction"
    a = validated.hypotheses["a"]
    a.checks[0] = CheckResult(**opposite)
    a.verdict = "fail"
    a.contradictions = ["ev:refute:i:a"]
    assert attempt(run(args))["reason"] == "incumbent_explicit_claim_refutation_unsupported"


def test_fixed_slots_not_rescued_by_later_success_and_audit_binds_checks():
    args = case()
    h = args[2].hypotheses[1].model_copy(deep=True, update={"id": "z"})
    args[2].hypotheses.append(h)
    a = args[3].hypotheses["b"].model_copy(deep=True, update={"hypothesis_id": "z"})
    args[3].hypotheses["z"] = a
    rebind(args)
    args[3].hypotheses["b"].verdict = "fail"
    audit = run(args)
    proof = attempt(audit)["challenger_proofs"]
    assert [p["hypothesis_id"] for p in proof] == ["b"]
    assert proof[0]["assessment_id"] == "b"
    assert proof[0]["check_ids"] == ["b:0:claim_applicability", "b:1:onset_alignment"]
    assert proof[0]["proposal_fingerprint"] == policy.proposal_fingerprint(args[2].hypotheses[1])
    assert attempt(run(args, per_service_slots=2))["reason"] == "missing_fixed_slots"
    args[2].hypotheses.reverse()
    assert run(args) == audit


def test_all_incumbent_fixed_slots_must_have_exact_assessments():
    args = case()
    for original, hid in list(zip(args[2].hypotheses, ["c", "d"])):
        h = original.model_copy(deep=True, update={"id": hid})
        args[2].hypotheses.append(h)
        args[3].hypotheses[hid] = args[3].hypotheses[original.id].model_copy(
            deep=True, update={"hypothesis_id": hid})
    rebind(args)
    del args[3].hypotheses["c"]
    assert attempt(run(args, per_service_slots=2))["reason"] == "missing_or_mismatched_exact_assessment"


@pytest.mark.parametrize("mutation", ["rank_count", "hypothesis_count", "scope", "legacy"])
def test_incomplete_or_untrusted_handoffs_abstain(mutation):
    args = case()
    if mutation == "rank_count":
        args[1].n_candidates += 1
    elif mutation == "hypothesis_count":
        args[2].n_hypotheses += 1
    elif mutation == "scope":
        args[2].scope_ref = None
    else:
        args[1].schema_version = 1
    assert run(args)["reason"] in {"incomplete_ranking_or_investigation", "untrusted_or_mismatched_scope"}


def test_policy_and_audit_survive_contract_roundtrip():
    from backend.app.engine.contracts import dump_handoff, load_handoff
    result = fuse_investigation(*case(), ranking_policy=POLICY)
    restored = load_handoff(dump_handoff("fuse", result), stage="fuse", incident_id="i")
    assert restored == result


def test_historical_policy_outputs_and_defaults_remain_independent():
    args = case()
    args[2].hypotheses.pop(0)
    args[2].n_hypotheses = 1
    default = fuse_investigation(*args)
    assert default.ranking_policy == "deterministic_first_shadow"
    assert "conservative_update" not in default.config
    bounded = fuse_investigation(*args, ranking_policy="bounded_p2_update_v1")
    assert [r["candidate"] for r in bounded.service_ranking] == ["unknown", "challenger", "incumbent", "other_unknown"]
    corrected = fuse_investigation(*args, ranking_policy="corrected_h_eligible_first")
    assert corrected.service_ranking[0]["candidate"] == "challenger"
    run(args)
