"""Predeclared: does the envelope-derived token cap stop the mid-run failures?

Control: the 200k cap that failed cases after scope admission grew.
Candidate: benchmark_budget() = max_llm_calls x (bundle cap + max output + framing).
Cases: the three OB substrate cases with the most admitted records (rule fixed
before running, label-blind), mock provider, shadow export, no paid calls.
Success: candidate completes every case with zero branch-level budget errors and
never exceeds the call cap; wall clock is reported as the remaining constraint.
"""
import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.app.agent.llm import LLMClient, MockProvider
from backend.app.eval import rcaeval_runner as rr
from backend.app.eval.rcaeval_runner import atomic_json

SUBSTRATE = ROOT / "output/ob-substrate-current/20261018T194111Z-54bd58f243"
CASES = 3


def budget_errors(row):
    errors = (row.get("artifacts", {}).get("investigation", {}) or {}).get("budget", {}).get("p3_execution_errors") or {}
    return {k: v for k, v in errors.items() if "budget" in str(v).lower()}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=ROOT / "output" / (
        "budget-envelope-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())))
    parser.add_argument("--cases", type=str,
                        help="explicit comma-separated cases (overrides the evidence-volume rule)")
    parser.add_argument("--case-rule-note", type=str, default=None)
    parser.add_argument("--primary-branches", type=int, default=3,
                        help="0 means no branch cap (the configuration that failed)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    scored = []
    for path in sorted(SUBSTRATE.glob("*.record.json")):
        record = json.loads(path.read_text())
        counts = record.get("counts") or {}
        scored.append((sum(counts.values()), record["case"], counts))
    scored.sort(reverse=True)
    explicit = [c.strip() for c in args.cases.split(",")] if args.cases else None
    cases = explicit or [case for _, case, _ in scored[:CASES]]
    protocol = {"version": "budget-envelope-v1", "primary_branches": args.primary_branches,
                "case_rule": args.case_rule_note or (
                    f"the {CASES} OB substrate cases with the most admitted records "
                    "(sum of written metrics, logs and spans)"),
                "cases": cases, "case_evidence": [{"case": c, "counts": n} for _, c, n in scored[:CASES]],
                "control": {"budget": rr.BENCH_BUDGET.as_dict()},
                "candidate": {"budget": rr.benchmark_budget().as_dict(),
                              "derivation": "max_llm_calls x (BUNDLE_TOKEN_CAP + 4096 + 1000)"},
                "provider": "MockProvider (deterministic, no paid calls)",
                "export_policy": "shadow", "rank_policy": rr.BENCHMARK_RANK_POLICY,
                "success": "candidate completes every case with zero branch-level budget errors; "
                           "wall clock reported as the remaining constraint",
                "source_sha256": rr.sha256(ROOT / "backend/app/eval/rcaeval_runner.py")}
    atomic_json(args.out / "protocol.json", protocol)
    results = {}
    for label, budget in (("control", rr.BENCH_BUDGET), ("candidate", rr.benchmark_budget())):
        for case in cases:
            started = time.monotonic()
            row = rr.run_case(ROOT / "data/RE2-OB" / case, args.out / label,
                              with_agent=True, budget=budget,
                              **(dict(primary_branches=None, minimum_primary_coverage=None)
                                 if args.primary_branches == 0 else
                                 dict(primary_branches=args.primary_branches,
                                      minimum_primary_coverage=args.primary_branches)),
                              export_policy="shadow",
                              llm_factory=lambda meter: LLMClient(meter, provider=MockProvider(),
                                                                  sleep=lambda _s: None))
            errors = budget_errors(row)
            entry = {"status": row["status"], "wall_s": round(time.monotonic() - started, 1),
                     "llm_calls": row["resources"].get("llm_calls"),
                     "tokens_charged": row["resources"].get("tokens_charged"),
                     "budget_errors": len(errors), "examples": list(errors.values())[:2],
                     "branches_with_errors": len(errors)}
            results.setdefault(label, {})[case] = entry
            print(label, case, entry["status"], "calls", entry["llm_calls"],
                  "charged", entry["tokens_charged"], "budget_errors", entry["budget_errors"],
                  f"{entry['wall_s']}s", flush=True)
    summary = {"protocol": protocol, "results": results,
               "control_failures": sum(1 for e in results["control"].values() if e["status"] != "ok"),
               "candidate_failures": sum(1 for e in results["candidate"].values() if e["status"] != "ok"),
               "control_budget_errors": sum(e["budget_errors"] for e in results["control"].values()),
               "candidate_budget_errors": sum(e["budget_errors"] for e in results["candidate"].values())}
    atomic_json(args.out / "results.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "protocol"}, indent=2))


if __name__ == "__main__":
    main()
