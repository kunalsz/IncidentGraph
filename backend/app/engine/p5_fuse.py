"""P5 — FUSE & RANK (§4.3): zero LLM. H(h) = 0.35·D + 0.25·R + 0.20·CC + 0.20·V.

D  = rank-normalized deterministic score of the hypothesis's service (P2)
R  = reflection blend (P3 reward)
CC = chain coherence from P4 (0 / 0.5 / 1)
V  = validator verdict (1 all-pass, 0.5 partial, 0 fail).
Failed, unvalidated and invalid candidates retain audit scores but are never
winner/alternative eligible; a weighted score cannot override a veto.

Confidence is typed by derivation with hard caps (L7), never self-reported;
the multi-hypothesis contract keeps every branch with confidence >= theta as a
ranked alternative (the single-fault framing is what every baseline does — we
did not build this to collapse). Presentation chooses single-vs-multi by the
RWR mass-concentration signal (top-3 share > 0.7 => single cause).
"""

from __future__ import annotations

from config import tuning
from backend.app.engine import policy

from backend.app.engine.contracts import (FuseResult, FusedHypothesis,
                                          InvestResult, RankResult,
                                          ValidateResult)

RULE_CAPS = {
    "cross_modal_consensus": 0.85,  # evidence strength, not probability
    "single_modality": tuning.CONF_CAP_SINGLE_MODAL,        # 0.70
    "inference_only": tuning.CONF_CAP_INFERENCE_ONLY,       # 0.40
}

VERDICT_VALUE = {"pass": 1.0, "partial": 0.5, "fail": 0.0}


def rank_normalized_d(rank: int, n_candidates: int) -> float:
    """D: 1.0 for the top-ranked service, 0.0 for the last (rank-normalized)."""
    if n_candidates <= 1:
        return 1.0
    return 1.0 - (rank - 1) / (n_candidates - 1)


def fuse_score(d: float, r: float, cc: float, v: float, weights=None) -> float:
    """H(h) with the published weights (no hidden function — RCLAgent's F_synth
    flaw is not repeated)."""
    weights = dict(tuning.FUSE_WEIGHTS if weights is None else weights)
    if (set(weights) != set(tuning.FUSE_WEIGHTS) or
            any(not policy.finite_number(v) or v < 0 for v in weights.values()) or
            abs(sum(weights.values()) - 1) > 1e-9):
        raise ValueError("fusion weights must be finite nonnegative unit-sum named components")
    return (weights["deterministic"] * d + weights["reflection"] * r
            + weights["chain_coherence"] * cc + weights["validator"] * v)


def typed_confidence(rule: str, rule_cap: float | None = None,
                     extra_cap: float | None = None) -> float:
    """Confidence is TYPED: the rule assigns the ceiling, caps only lower it."""
    ceiling = RULE_CAPS.get(rule, tuning.CONF_CAP_INFERENCE_ONLY)
    caps = [ceiling] + [cap for cap in (rule_cap, extra_cap)
                        if cap is not None]
    return min(caps)


def rwr_top3_share(ranking: RankResult) -> float:
    masses = [candidate.components.get("rwr", 0.0)
              for candidate in ranking.candidates]
    total = sum(masses)
    if total <= 0:
        return 0.0
    return sum(sorted(masses, reverse=True)[:3]) / total


def _status_for(verdict: str) -> str:
    return {"pass": "validated", "partial": "partial", "fail": "disproved"}[verdict]


def fuse_investigation(investigation, ranking: RankResult, invest: InvestResult,
                       validated: ValidateResult, *, ranking_policy: str = "deterministic_first_shadow",
                       weights=None, per_service_slots=1) -> FuseResult:
    """P5 entry point: fuse every hypothesis into one ranked, typed list."""
    if any(result.incident_id != investigation.incident_id
           for result in (ranking, invest, validated)):
        raise ValueError("phase results do not match investigation")
    if isinstance(per_service_slots, bool) or not isinstance(per_service_slots, int) or not 1 <= per_service_slots <= 16:
        raise ValueError("per_service_slots must be a predeclared integer in [1,16]")
    weights = dict(tuning.FUSE_WEIGHTS if weights is None else weights)
    fuse_score(0, 0, 0, 0, weights)  # validate even on empty proposals
    if ranking_policy not in {"deterministic_first_shadow", "corrected_h_eligible_first", "bounded_p2_update_v1"}:
        raise ValueError("unknown service ranking policy")
    if len({c.id for c in ranking.candidates}) != len(ranking.candidates):
        raise ValueError("duplicate ranking candidate")
    if len({h.id for h in invest.hypotheses}) != len(invest.hypotheses):
        raise ValueError("duplicate hypothesis identity")
    trusted = all(r.schema_version == 2 for r in (investigation, ranking, invest, validated))
    trusted = trusted and validated.scope_ref is not None and ranking.scope_ref == invest.scope_ref == validated.scope_ref
    rank_by_service = {candidate.id: candidate.rank
                       for candidate in ranking.candidates}
    n_candidates = max(1, len(ranking.candidates))

    fused: list[FusedHypothesis] = []
    for hypothesis in invest.hypotheses:
        validator = validated.hypotheses.get(hypothesis.id)
        if not trusted or not policy.assessment_matches(validator, hypothesis, investigation, validated.scope_ref):
            validator = None
        d = rank_normalized_d(rank_by_service.get(hypothesis.candidate,
                                                  n_candidates), n_candidates)
        cc = validator.chain_coherence if validator else 0.0
        v = VERDICT_VALUE[validator.verdict] if validator else 0.0
        score = fuse_score(d, hypothesis.reward, cc, v, weights)
        confidence, rule, eligible = policy.evidence_strength(validator)
        fused.append(FusedHypothesis(
            id=hypothesis.id, candidate=hypothesis.candidate, rank=0,
            h_score=round(score, 6),
            components={"D": round(d, 4), "R": round(hypothesis.reward, 4),
                        "CC": cc, "V": v},
            confidence=round(confidence, 4),
            confidence_rule=rule,
            confidence_cap=validator.confidence_cap if validator else None,
            status=("invalid" if hypothesis.candidate not in rank_by_service
                    else "disproved" if validator and (validator.verdict == "fail" or any(
                        check.verdict == "fail" for check in validator.checks))
                    else (_status_for(validator.verdict) if eligible else "unsupported") if validator
                    else "unvalidated"),
            evidence=list(hypothesis.evidence), confidence_derivation={
                "policy_version": policy.VERSION, "calibrated": False,
                "semantics": "deterministic_evidence_strength", "score": confidence,
                "modalities": sorted(validator.modalities) if validator else [],
                "rule_ids": [rule], "applied_caps": [validator.confidence_cap] if validator and validator.confidence_cap is not None else []} ))

    fused.sort(key=lambda item: (-item.h_score, -item.confidence, item.id))
    for position, item in enumerate(fused, start=1):
        item.rank = position

    # Preserve rejected scores for audit, but never let weights revive a veto.
    eligible = [item for item in fused
                if item.status in ("validated", "partial")]
    winner = eligible[0] if eligible else None
    # Stable predeclared slots, not best-of-many. Missing slots contribute zero.
    # A vetoed slot is never replaced by a later successful sibling.
    slots = {service: sorted((h for h in fused if h.candidate == service), key=lambda h: h.id)[:per_service_slots]
             for service in rank_by_service}
    representatives = {}
    service_scores = {}
    for service, items in slots.items():
        service_scores[service] = sum(h.h_score for h in items if h.status in ("validated", "partial")) / per_service_slots
        representative = next((h for h in items if h.status in ("validated", "partial")), None)
        if representative:
            representatives[service] = representative
    alternatives = [item.id for item in representatives.values()
                    if item.candidate != (winner.candidate if winner else None)]
    service_ranking = []
    for c in sorted(ranking.candidates, key=lambda c: (c.rank, c.id)):
        representative = representatives.get(c.id)
        service_ranking.append({"candidate": c.id, "rank": len(service_ranking) + 1,
            "baseline_rank": c.rank, "baseline_score": c.score,
            "eligible": representative is not None,
            "representative": representative.id if representative else None,
            "shadow_h_score": service_scores[c.id] if representative else None,
            "n_hypotheses": sum(h.candidate == c.id for h in invest.hypotheses)})
    if ranking_policy == "corrected_h_eligible_first":
        service_ranking.sort(key=lambda r: (not r["eligible"], -(r["shadow_h_score"] or 0) if r["eligible"] else 0,
                                             r["baseline_rank"], r["candidate"]))
        for pos, row in enumerate(service_ranking, 1):
            row["rank"] = pos
    if ranking_policy == "bounded_p2_update_v1":
        # Frozen before evaluation: at most one adjacent swap, only a complete
        # scoped ranking and unanimous preallocated cross-modal all-pass slots.
        # This is an experimental association update, not a default promotion.
        safe = bool(trusted and not ranking.truncated and not invest.truncated)
        swapped = False
        for index in range(1, len(service_ranking)):
            row, previous = service_ranking[index], service_ranking[index - 1]
            items = slots[row["candidate"]]
            strong = (len(items) == per_service_slots and all(
                h.status == "validated" and h.confidence >= .85 and
                validated.hypotheses[h.id].verdict == "pass" and
                not validated.hypotheses[h.id].contradictions for h in items))
            if safe and strong and not previous["eligible"] and not swapped:
                service_ranking[index - 1], service_ranking[index] = row, previous
                swapped = True
        for pos, row in enumerate(service_ranking, 1):
            row["rank"] = pos
    share = rwr_top3_share(ranking)
    presentation = "single" if share > tuning.RWR_TOP3_SHARE_SINGLE else "multi"

    return FuseResult(
        incident_id=investigation.incident_id, scope_ref=validated.scope_ref, presentation=presentation,
        service_ranking=service_ranking, ranking_policy=ranking_policy,
        fused=fused, winner=winner.id if winner else None,
        winner_candidate=winner.candidate if winner else None,
        alternatives=alternatives, rwr_top3_share=round(share, 4),
        config={"weights": weights, "per_service_slots": per_service_slots,
                "service_aggregation": "stable_id_fixed_slot_mean_zero_missing",
                "bounded_update": {"version": "bounded_p2_update_v1", "max_adjacent_swaps": 1,
                    "max_rank_displacement": 1, "requires_complete": True,
                    "requires_all_slots_crossmodal_pass": True, "experimental": True},
                "alpha_reflection": tuning.ALPHA_REFLECTION,
                "rule_caps": dict(RULE_CAPS),
                "theta": tuning.CONFIDENCE_GATE_THETA,
                "multi_hypothesis": "retain all conf >= theta",
                "presentation_rule": "rwr_top3_share > 0.7 => single cause"})
