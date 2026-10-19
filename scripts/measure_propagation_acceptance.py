"""How many caller/callee failure pairs does the P4 validator actually accept?

Answers, on real data and without touching the database:

  raw pairs in the traces            (same trace, parent/child, cross-service,
                                      both halves a failure status)
    -> pairs that survive sampling   (select_trace_sample, error-priority)
      -> pairs the validator accepts (policy.propagation_support, on the scoped
                                      DEPENDS_ON edge, with both spans cited)

Each accepted pair is the exact certificate P4 requires to call a dependency link
"supported", and what the claim-option builder then offers as a
`dependency_propagation` option. Reports the old rule (statusCode == 2) alongside
the widened one so the delta is visible, and the trace-severity impact.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.app.engine import policy
from backend.app.eval import rcaeval_adapter as adapter
from backend.app.eval.rcaeval_runner import atomic_json, sha256

LEGACY_RULE = lambda value: str(value).lower() in ("2", "2.0", "error", "failed", "failure")


def span_records(frame: pd.DataFrame, *, error_rule) -> dict:
    """Shape spans the way `load_case` stores them, so the validator sees real inputs."""
    # `frame["service"]` must already carry the "service:" prefix the scoped
    # edges use: propagation_support compares it with edge source/target, and
    # passing bare names made every pair look like a different service.
    records = {}
    for _, row in frame.iterrows():
        records[str(row["spanID"])] = {
            "kind": "Span", "service": str(row["service"]), "positive": True,
            "props": {"id": str(row["spanID"]), "ts": int(row["startTimeMillis"]),
                      "duration_us": int(row["duration"]) if pd.notna(row["duration"]) else 0,
                      "trace_id": str(row["traceID"]), "span_id": str(row["spanID"]),
                      "parent_id": str(row["parentSpanID"]) if pd.notna(row["parentSpanID"]) else "",
                      "status": row["statusCode"]}}
    return records


def raw_pairs(frame: pd.DataFrame, error_rule) -> int:
    """Cross-service parent/child pairs whose two halves both fail."""
    failing = frame[frame["statusCode"].map(error_rule)]
    by_id = frame.set_index("spanID")
    pairs = 0
    for _, child in failing.iterrows():
        parent_id = child.get("parentSpanID")
        if not isinstance(parent_id, str) or parent_id not in by_id.index:
            continue
        parent = by_id.loc[parent_id]
        if isinstance(parent, pd.DataFrame):
            parent = parent.iloc[0]
        if (error_rule(parent["statusCode"]) and parent["serviceName"] != child["serviceName"]
                and parent["traceID"] == child["traceID"]):
            pairs += 1
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=ROOT / "data/RE2-OB")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    topology = json.loads(adapter.OB_TOPOLOGY.read_text())
    edges = [{"source": dep["from"], "target": dep["to"], "type": "DEPENDS_ON", "properties": {}}
             for dep in topology["depends_on"]]
    per_case, totals = [], Counter()
    for case_dir in sorted(p for p in args.data.iterdir()
                           if p.is_dir() and p.name.startswith("re2ob_")):
        traces = pd.read_parquet(case_dir / "traces.parquet")
        inject_ms = int((case_dir / "inject_time.txt").read_text().strip()) * 1000
        window = traces[traces["startTimeMillis"].between(inject_ms - 600_000, inject_ms + 600_000)]
        row = {"case": case_dir.name,
               "raw_pairs_new_rule": raw_pairs(window, policy.error_status),
               "raw_pairs_old_rule": raw_pairs(window, LEGACY_RULE)}
        for label, keep in (("new", policy.error_status), ("old", LEGACY_RULE)):
            sample, _ = adapter.select_trace_sample(traces, inject_ms - 600_000,
                                                    inject_ms + 600_000, inject_ms)
            sample = sample.copy()
            sample["service"] = sample["serviceName"].astype(str).map(
                lambda x: "service:" + adapter.TRACE_SERVICE_MAP.get(x, x))
            sample = sample[sample["statusCode"].map(keep)]
            records = span_records(sample, error_rule=keep)
            pairs = sum(len(policy.propagation_support(records, edge)) // 2 for edge in edges)
            row[f"failure_spans_sampled_{label}"] = int(len(sample))
            row[f"pairs_sampled_and_accepted_{label}"] = pairs
            totals[f"failure_spans_sampled_{label}"] += int(len(sample))
            totals[f"pairs_sampled_and_accepted_{label}"] += pairs
        totals["raw_pairs_new_rule"] += row["raw_pairs_new_rule"]
        totals["raw_pairs_old_rule"] += row["raw_pairs_old_rule"]
        if row["raw_pairs_new_rule"] or row["pairs_sampled_and_accepted_new"]:
            per_case.append(row)
        print(case_dir.name, row, flush=True)
    report = {"corpus": args.data.name,
              "rule": "policy.error_status (codes 2..16 plus named failures)",
              "legacy_rule": "statusCode == 2 only",
              "totals": dict(totals),
              "cases_with_any_failure_pair": len(per_case),
              "accepted_pairs_by_case": {r["case"]: r["pairs_sampled_and_accepted_new"]
                                         for r in per_case
                                         if r["pairs_sampled_and_accepted_new"]},
              "per_case": per_case,
              "adapter_sha256": sha256(ROOT / "backend/app/eval/rcaeval_adapter.py"),
              "policy_sha256": sha256(ROOT / "backend/app/engine/policy.py")}
    args.out.mkdir(parents=True, exist_ok=True)
    atomic_json(args.out / "propagation-acceptance.json", report)
    print(json.dumps(report["totals"], indent=2))
    print("accepted pairs per case:", report["accepted_pairs_by_case"])


if __name__ == "__main__":
    main()
