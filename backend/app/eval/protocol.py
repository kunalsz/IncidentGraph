"""Predeclared evaluation-only split labels; never passed to engine/provider."""
import hashlib
import json
from backend.app.eval.rcaeval_runner import PROTOCOL, parse_case_name

SPLIT_VERSION = "service-fault-holdout-v1"
HELDOUT_SERVICES = ("recommendationservice",)
HELDOUT_FAULTS = ("loss",)


def freeze_protocol(cases):
    """Freeze before any inference. Existing local90 is development, not pristine holdout.

    Unknown labels mean unseen service/fault categories relative to the declared
    development partition, NOT an assertion local historical data was never seen.
    """
    assignments = {}
    for case in sorted(set(cases)):
        service, fault, _ = parse_case_name(case)
        unseen_s, unseen_f = service in HELDOUT_SERVICES, fault in HELDOUT_FAULTS
        split = ("unknown_service_fault" if unseen_s and unseen_f else
                 "unknown_service" if unseen_s else "unknown_fault" if unseen_f else "development")
        assignments[case] = split
    protocol = {"version": SPLIT_VERSION, "engine_protocol": PROTOCOL,
        "seed": 0, "concurrency": 1, "anchor": "service:frontend",
        "runtime_boundary": "injected adapter phases; cooperative deadlines, not hard worker isolation",
        "p0": "adapter clock/opaque incident id; no product identity resolution",
        "p1": "scope-benchmark-v3-projected-repeat-read full admitted scope",
        "primary_branches": 2, "proposal_provider": "deterministic local evidence rules",
        "mock_consolidation_confidence": .1,
        "mock_confidence_purpose": "fixed low scheduling confidence exercises expansion; never evidence strength or labels",
        "hypothesis_slots": [1, 2], "F_weight_policy": "zero selected component; renormalize remaining weights to unit sum; veto unchanged", "weights": {"deterministic": .35, "reflection": .25,
            "chain_coherence": .2, "validator": .2},
        "heldout_services": list(HELDOUT_SERVICES), "heldout_faults": list(HELDOUT_FAULTS),
        "assignments": assignments, "label_input_policy": "split/answer metadata evaluator-only after inference",
        "selection_policy": "explicit requested cases; no post-result selection",
        "claim_scope": "local development mechanism test, not untouched heldout generalization",
        "promotion_gate": {"status": "blocked_pending_explicit_live_provider_authorization",
            "requirements": ["explicit provider/model and spend authorization",
                "new untouched service/fault-heldout inputs locked before inference",
                "matched P2/agent input hashes, scope and budgets",
                "full-order shadow equality; deterministic validator veto unchanged",
                "intent-to-run paired metrics, service/fault clustered uncertainty and settled cost ledgers",
                "recorded proposal/assessment/fusion stages and source/input provenance"]}}
    protocol["sha256"] = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    return protocol
