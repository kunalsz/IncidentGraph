"""Intent-to-run paired comparisons with diagnostic matched subsets and uncertainty."""
from __future__ import annotations

import collections
import json
import random
from pathlib import Path

from backend.app.eval.rcaeval_runner import atomic_json


def load_results(path):
    data = json.loads(Path(path).read_text())
    rows = data.get("cases", [data])
    names = [r["case"] for r in rows]
    if len(names) != len(set(names)):
        raise ValueError("duplicate cases cannot be paired")
    return {row["case"]: row for row in rows}


def _completed(row):
    return bool(row and not row.get("error") and row.get("status", "ok") == "ok"
                and (row.get("agent") or {}).get("outcome") != "failed")


def _order(row):
    if not _completed(row):
        return []  # Primary agent intent-to-run; fallback is a separate system outcome.
    return row.get("service_order", row.get("final_order", [])) or []


def compare(left, right, *, seed=0, draws=2000):
    keys = sorted(set(left)|set(right))
    rows = []
    for name in keys:
        a, b = left.get(name), right.get(name)
        answer = (a or b)["gt_service"]
        def rank(row):
            order = _order(row)
            return order.index(answer)+1 if answer in order else None
        ar, br = rank(a), rank(b)
        rows.append({"case": name, "service": answer, "fault": (a or b)["gt_fault"],
            "left_rank": ar, "right_rank": br,
            "left_hit1": ar == 1, "right_hit1": br == 1,
            "left_hit5": ar is not None and ar <= 5, "right_hit5": br is not None and br <= 5,
            "mrr_delta": (1/br if br else 0)-(1/ar if ar else 0),
            "matched_success": _completed(a) and _completed(b),
            "failed_or_missing": not (_completed(a) and _completed(b)),
            "truncated": any((r or {}).get("agent", {}).get("truncated", False) for r in (a,b) if (r or {}).get("agent")),
            "left_winner": next(iter(_order(a)), None), "right_winner": next(iter(_order(b)), None)})
    def stats(selected):
        n = len(selected)
        return {"n": n, "ac1_delta": sum(int(r["right_hit1"])-int(r["left_hit1"]) for r in selected)/n if n else None,
            "ac5_delta": sum(int(r["right_hit5"])-int(r["left_hit5"]) for r in selected)/n if n else None,
            "mean_mrr_delta": sum(r["mrr_delta"] for r in selected)/n if n else None,
            "top1_preserved": sum(r["left_hit1"] and r["right_hit1"] for r in selected),
            "top1_lost": sum(r["left_hit1"] and not r["right_hit1"] for r in selected),
            "top1_gained": sum(not r["left_hit1"] and r["right_hit1"] for r in selected),
            "top5_transitions": dict(collections.Counter(f"{int(r['left_hit5'])}->{int(r['right_hit5'])}" for r in selected))}
    rng = random.Random(seed)
    def interval(groups):
        if not groups:
            return None
        estimates = []
        for _ in range(draws):
            sample = [r for _ in groups for r in rng.choice(groups)]
            estimates.append(sum(r["mrr_delta"] for r in sample)/len(sample))
        estimates.sort()
        return [estimates[int(.025*(draws-1))], estimates[int(.975*(draws-1))]]
    grouped = collections.defaultdict(list)
    for row in rows:
        grouped[(row["service"],row["fault"])].append(row)
    def cost(results):
        known_calls = known_tokens = unknown = charged = unknown_attempts = 0
        for r in results.values():
            ledger = r.get("resources") or r.get("agent") or {}
            known_calls += ledger.get("llm_calls", 0)
            known_tokens += ledger.get("tokens", ledger.get("tokens_total", 0))
            charged += ledger.get("tokens_charged", ledger.get("tokens_total", 0))
            unknown_attempts += ledger.get("unknown_usage_attempts", ledger.get("usage_unknown_attempts", 0))
            unknown += bool(ledger.get("unknown_expenditure") or ledger.get("usage_complete") is False
                            or ledger.get("unknown_usage_attempts", ledger.get("usage_unknown_attempts", 0))
                            or (r.get("error") and not r.get("resources")))
        return {"schema_version": 2, "known_calls": known_calls, "known_tokens": known_tokens,
                "charged_tokens_including_unknown_reservations": charged,
                "unknown_usage_attempts": unknown_attempts, "unknown_expenditure_cases": unknown}
    return {"intent_to_run": stats(rows), "matched_success_diagnostic_only": stats([r for r in rows if r["matched_success"]]),
        "strata": {"failed_or_missing": stats([r for r in rows if r["failed_or_missing"]]),
                   "truncated": stats([r for r in rows if r["truncated"]])},
        "uncertainty": {"seed": seed, "draws": draws, "case_paired_mrr_delta_ci95": interval([[r] for r in rows]),
            "service_fault_cluster_mrr_delta_ci95": interval(list(grouped.values())),
            "qualification": "Development comparison; repeated instances correlated. Not held-out topology generalization."},
        "winner_concentration": {"left": dict(collections.Counter(r["left_winner"] for r in rows)),
                                  "right": dict(collections.Counter(r["right_winner"] for r in rows))},
        "costs": {"left": cost(left), "right": cost(right)}, "cases": rows,
        "limitations": ["Different sampler/source protocols are descriptive, not causal agent comparisons.",
                        "Historical absent P3/P4 artifacts cannot be reconstructed from final rankings."]}


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output exists; choose a fresh path")
    result = compare(load_results(args.left), load_results(args.right))
    atomic_json(args.output, result)
    print(json.dumps(result["intent_to_run"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
