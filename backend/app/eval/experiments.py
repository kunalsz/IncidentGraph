"""Runnable frozen experiment matrix. No network, database, or paid provider.

A is recorded repaired P2. B/F/H are explicitly mock diagnostic mechanisms.
C exercises mock proposal/fusion construction but exports a deep-frozen P2
snapshot; it is not a claim that historical live P3/P4 was reconstructed.
D/E require complete recorded P3/P4 artifacts, otherwise report unavailable.
"""
import argparse
import hashlib
import json
from pathlib import Path

from backend.app.eval.comparison import compare, load_results
from backend.app.eval.rcaeval_runner import atomic_json
from backend.app.eval.replay import freeze_record, mock_artifacts, rerank


def run_experiments(source, output, baseline_source=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    results = load_results(source)
    baseline = load_results(baseline_source) if baseline_source else None
    for key, record in results.items():
        if baseline is not None:
            if key not in baseline:
                raise ValueError(f"missing matched P2 baseline: {key}")
            record["baseline_order"] = baseline[key]["service_order"]
        elif "baseline_order" not in record:
            raise ValueError("P2 baseline absent; supply --baseline, never relabel historical fused order as P2")
    frozen = {key: freeze_record(record) for key,record in results.items()}
    arms = {name: {} for name in ("A_p2", "B_mock_eligibility", "C_mock_shadow", "D_mock_corrected",
        "D_mock_legacy", "F_mock_reflection_off", "F_mock_coherence_off", "F_mock_validator_score_off", "H_mock_two_slots")}
    for name, case in frozen.items():
        mock = mock_artifacts(case)
        before = list(case.baseline_order)
        # Exercise fusion computations before the immutable shadow export.
        rerank(mock, "corrected")
        two_slots = mock.model_copy(deep=True)
        for service in two_slots.services:
            second = dict(service.hypotheses[0])
            second["id"] = second["id"].removesuffix(":0") + ":1"
            service.hypotheses.append(second)
        orders = {"A_p2": rerank(case, "p2"), "B_mock_eligibility": rerank(mock, "eligibility"),
                  "C_mock_shadow": rerank(mock, "negative_control"),
                  "D_mock_corrected": rerank(mock, "corrected"),
                  "D_mock_legacy": rerank(mock, "legacy_membership"),
                  "H_mock_two_slots": rerank(two_slots, "corrected", slots=2)}
        for label, disabled in (("reflection", "R"), ("coherence", "CC"), ("validator_score", "V")):
            weights = {"D":.35, "R":.25, "CC":.2, "V":.2}
            weights[disabled] = 0
            orders[f"F_mock_{label}_off"] = rerank(mock, "corrected", weights=weights)
        if orders["C_mock_shadow"] != before or case.baseline_order != before:
            raise AssertionError("negative control mutated P2")
        for arm,order in orders.items():
            record = baseline[name] if baseline is not None else results[name]
            arms[arm][name] = {**record, "service_order": order, "final_order": order,
                              "experiment_arm": arm, "mock": arm != "A_p2",
                              "resources": {"llm_calls": 0, "tokens": 0, "unknown_expenditure": False},
                              "agent": None}
        atomic_json(output/f"{name}.frozen.json", case.model_dump(mode="json"))
        atomic_json(output/f"{name}.mock.json", mock.model_dump(mode="json"))
    summaries = {}
    for arm, records in arms.items():
        atomic_json(output/f"{arm}.json", {"cases": list(records.values())})
        summaries[arm] = compare(arms["A_p2"], records)
    complete = sum(c.stage_artifacts_available for c in frozen.values())
    result = {"schema_version": 2, "input_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest(),
        "n_cases": len(frozen), "negative_control_full_order_equal": True,
        "recorded_complete_stage_cases": complete, "comparisons": summaries,
        "unavailable_arms": {"D_historical_fusion": "P3/P4 absent" if complete != len(frozen) else "use replay corrected/legacy explicitly",
            "E_recorded_vs_mock": "requires matched complete recorded stages and fixed input protocol",
            "G_expansion": "cannot synthesize unrecorded expanded evidence"},
        "limitations": ["Mock arms test deterministic replay/gating/plumbing, not information gain or agent superiority.",
            "C is frozen mock-shadow, not execution of missing historical P3/P4.",
            "Historical20 is development/diagnosis only; no held-out generalization claim."]}
    atomic_json(output/"_experiments.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, help="required when historical source lacks a recorded P2 order")
    args = parser.parse_args(argv)
    result = run_experiments(args.input, args.output, args.baseline)
    print(json.dumps({k:v for k,v in result.items() if k != "comparisons"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
