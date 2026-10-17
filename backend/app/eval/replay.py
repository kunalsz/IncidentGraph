"""Frozen offline stage replay. No provider calls or historical reconstruction."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, Field
from backend.app.eval.rcaeval_runner import atomic_json


class FrozenService(BaseModel):
    service: str
    p2_rank: int
    p2_components: dict = Field(default_factory=dict)
    hypotheses: list[dict] = Field(default_factory=list)
    branch_status: str = "not_recorded"
    eligible: bool = False
    rejection_reasons: list[str] = Field(default_factory=list)
    final_rank: int | None = None
    fallback_source: str | None = None


class FrozenCase(BaseModel):
    schema_version: int = 2
    case: str
    source_sha256: str
    provenance: dict = Field(default_factory=dict)
    baseline_order: list[str]
    services: list[FrozenService]
    stage_artifacts_available: bool = False
    kind: str = "recorded"
    limitations: list[str] = Field(default_factory=list)


def freeze_record(record: dict) -> FrozenCase:
    artifacts = record.get("artifacts", {})
    ranking = artifacts.get("ranking", {})
    baseline = record.get("baseline_order", [])
    if not baseline and ranking.get("candidates"):
        baseline = [c["id"].removeprefix("service:") for c in ranking["candidates"]]
    hypotheses = copy.deepcopy(artifacts.get("fusion", {}).get("fused", []))
    assessments = artifacts.get("validation", {}).get("hypotheses", {})
    proposals = {p.get("id"): p for p in artifacts.get("investigation", {}).get("hypotheses", [])}
    expected_ref = ranking.get("scope_ref")
    scope_ref = artifacts.get("scope", {}).get("scope_ref")
    # Eligibility is explicit frozen attribution, never inferred from a legacy
    # status label alone. Require a bound v2 deterministic assessment.
    for hypothesis in hypotheses:
        assessment = assessments.get(hypothesis.get("id"), {})
        proposal = proposals.get(hypothesis.get("id"), {})
        from backend.app.engine.contracts import HypothesisRecord
        from backend.app.engine.policy import proposal_fingerprint
        try:
            digest = proposal_fingerprint(HypothesisRecord.model_validate(proposal))
        except ValueError:
            digest = None
        hypothesis["eligible"] = bool(assessment.get("schema_version") == 2 and proposal.get("schema_version") == 2
            and assessment.get("hypothesis_id") == hypothesis.get("id")
            and assessment.get("candidate") == hypothesis.get("candidate")
            and expected_ref and assessment.get("scope_ref") == expected_ref == scope_ref == proposal.get("scope_ref")
            and assessment.get("proposal_fingerprint") == digest
            and proposal.get("candidate") == hypothesis.get("candidate")
            and assessment.get("qualified_timing") is True and assessment.get("plausible_path") is True
            and assessment.get("claim_type", "service_abnormality") == proposal.get("claim_kind")
            and assessment.get("evidence_scope") == artifacts.get("context", {}).get("session_graph")
            and bool(assessment.get("evidence_scope"))
            and set(assessment.get("source_fingerprints", {})) <= set(proposal.get("evidence", []))
            and assessment.get("verdict") in ("pass", "partial")
            and not any(check.get("verdict") == "fail" for check in assessment.get("checks", []))
            and hypothesis.get("status") in ("validated", "partial"))
    candidates = {x["id"].removeprefix("service:"): x for x in ranking.get("candidates", [])}
    branches = artifacts.get("investigation", {}).get("branches", [])
    if isinstance(branches, dict):
        branches = list(branches.values())
    services = []
    for index, name in enumerate(baseline, 1):
        hs = [copy.deepcopy(h) for h in hypotheses if h.get("candidate", "").removeprefix("service:") == name]
        eligible = any(is_eligible(h) for h in hs)
        branch = next((b for b in branches if b.get("node_id", b.get("candidate", "")).removeprefix("service:") == name), {})
        services.append(FrozenService(service=name, p2_rank=index,
            p2_components=candidates.get(name, {}).get("components", {}), hypotheses=hs,
            branch_status=branch.get("status", "not_recorded"), eligible=eligible,
            rejection_reasons=[h.get("status", "unknown") for h in hs if not is_eligible(h)],
            final_rank=(record.get("service_order", []).index(name)+1 if name in record.get("service_order", []) else None),
            fallback_source=None if eligible else "p2"))
    available = all(key in artifacts for key in ("ranking", "investigation", "validation", "fusion"))
    return FrozenCase(case=record["case"], source_sha256=hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest(),
        provenance={key:record[key] for key in ("run_id", "protocol", "data", "sample_diagnostics", "universes") if key in record},
        baseline_order=baseline, services=services, stage_artifacts_available=available,
        limitations=[] if available else ["P3/P4 structured artifacts absent: historical proposals, evidence, eligibility and fusion cannot be reconstructed."])


def is_eligible(h):
    return h.get("status") in ("validated", "partial") and h.get("eligible") is True


def rerank(frozen: FrozenCase, policy="p2", *, weights=None, slots=1):
    """Weight-only arms keep eligibility fixed. Veto is never disabled.

    Corrected diagnostic fusion uses exactly the first stable-ID eligible slot
    per service by default, not the maximum of an unequal proposal count.
    Uncovered services retain rank-normalized P2 scores (no membership bonus).
    """
    if policy in ("p2", "shadow", "negative_control"):
        return list(frozen.baseline_order)
    if not frozen.stage_artifacts_available and frozen.kind != "mock":
        raise ValueError("P3/P4 artifacts absent; cannot replay historical fusion")
    if slots <= 0:
        raise ValueError("slots must be positive")
    if policy == "legacy_membership":
        scored = [(max(h.get("h_score", 0) for h in s.hypotheses), s.p2_rank, s.service)
                  for s in frozen.services if s.hypotheses]
        lead = [name for _,_,name in sorted(scored, key=lambda x:(-x[0], x[1], x[2]))]
        return lead+[name for name in frozen.baseline_order if name not in lead]
    if policy not in ("corrected", "eligibility"):
        raise ValueError(policy)
    weights = weights or {"D": .35, "R": .25, "CC": .2, "V": .2}
    n = len(frozen.baseline_order)
    values = []
    for service in frozen.services:
        d = 1-(service.p2_rank-1)/max(1,n-1)
        hs = sorted((h for h in service.hypotheses if is_eligible(h)), key=lambda h:h.get("id", ""))[:slots]
        if policy == "eligibility":
            score = d if service.eligible else -1
        elif hs:
            score = sum(sum(weights.get(k,0)*float(h.get("components",{}).get(k, d if k == "D" else 0))
                            for k in ("D","R","CC","V")) for h in hs)/len(hs)
        else:
            score = d  # explicit P2 fallback; no has-hypothesis tail.
        values.append((score, service.p2_rank, service.service))
    return [name for _,_,name in sorted(values, key=lambda x:(-x[0],x[1],x[2]))]


def mock_artifacts(frozen):
    result = frozen.model_copy(deep=True)
    result.kind = "mock"
    result.limitations = ["Synthetic deterministic proposals: plumbing/negative-control evidence only, not model superiority."]
    for service in result.services:
        d = 1-(service.p2_rank-1)/max(1,len(result.services)-1)
        service.eligible = True
        service.branch_status = "mock_complete"
        service.hypotheses = [{"id": f"mock:{service.service}:0", "candidate": f"service:{service.service}",
            "status": "validated", "eligible": True, "components": {"D": d, "R": d, "CC": d, "V": 1},
            "h_score": .8*d+.2, "confidence_rule": "synthetic_not_probability"}]
    return result


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--policy", default="p2", choices=["p2","shadow","corrected","legacy_membership","eligibility"])
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output exists; choose a fresh artifact path")
    record = json.loads(args.input.read_text())
    frozen = FrozenCase.model_validate(record) if "source_sha256" in record else freeze_record(record)
    if args.mock:
        frozen = mock_artifacts(frozen)
    atomic_json(args.output, {"frozen": frozen.model_dump(mode="json"), "policy": args.policy,
                              "order": rerank(frozen, args.policy)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
