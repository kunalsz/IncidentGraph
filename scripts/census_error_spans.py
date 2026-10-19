"""Do these corpora contain any error spans at all?

The P4 propagation validator needs two cited, same-trace, parent/child ERROR
spans on a scoped dependency. Every live audit so far found none, and SockShop
ships no traces at all - so either that claim family is simply unexercised on
these corpora, or the adapter is discarding errors. This counts the raw data.

Read-only, offline, no database.
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.app.eval.rcaeval_runner import atomic_json

# traces.parquet has NO error column: the adapter derives it from statusCode
# (adapter line ~273: statusCode in {2, "error", "failed", "failure"}). The
# census uses the same rule so its numbers describe what the pipeline sees.
COLUMNS = ("traceID", "spanID", "parentSpanID", "serviceName", "statusCode", "startTimeMillis")
ERROR_CODES = {"2", "2.0", "error", "failed", "failure"}


def census(case_dir: Path) -> dict | None:
    path = case_dir / "traces.parquet"
    if not path.is_file():
        return None
    frame = pd.read_parquet(path, columns=[c for c in COLUMNS])
    inject = int((case_dir / "inject_time.txt").read_text().strip()) * 1000
    window = frame[frame["startTimeMillis"].between(inject - 600_000, inject + 600_000)]
    error_flag = window["statusCode"].map(lambda v: str(v).lower() in ERROR_CODES)
    status_error = pd.to_numeric(window["statusCode"], errors="coerce") == 2
    row = {
        "case": case_dir.name, "spans_in_window": int(len(window)),
        "error_flag_spans": int(error_flag.sum()),
        "status_code_histogram": {str(k): int(v) for k, v in
                                  Counter(window["statusCode"].astype(str)).most_common()},
        "services_in_window": int(window["serviceName"].nunique()),
        "error_flag_services": sorted(window.loc[error_flag, "serviceName"].astype(str).unique().tolist()),
    }
    truth = case_dir.name[len("re2ob_"):].rsplit("_", 2)[0]
    mapped = window["serviceName"].astype(str).str.replace("frontendservice", "frontend")
    row["truth_service_has_error_span"] = bool(error_flag[mapped == truth].any())
    # propagation precondition: a caller/callee pair whose parent and child both
    # error, inside one trace
    by_id = window.assign(_err=error_flag, _svc=mapped).set_index("spanID")
    pairs = 0
    for _, child in by_id[by_id["_err"]].iterrows():
        parent_id = child.get("parentSpanID")
        if not isinstance(parent_id, str) or parent_id not in by_id.index:
            continue
        parent = by_id.loc[parent_id]
        if isinstance(parent, pd.DataFrame):
            parent = parent.iloc[0]
        if bool(parent["_err"]) and parent["_svc"] != child["_svc"] and \
                parent["traceID"] == child["traceID"]:
            pairs += 1
    row["error_parent_child_cross_service_pairs"] = pairs
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows, trace_less = [], []
    for case_dir in sorted(p for p in args.data.iterdir() if p.is_dir() and p.name.startswith("re2")):
        row = census(case_dir)
        (rows if row else trace_less).append(row or case_dir.name)
        print(case_dir.name, row["error_flag_spans"] if row else "no traces", flush=True)
    fault = lambda name: name.rsplit("_", 2)[1]
    summary = {
        "corpus": args.data.name, "cases_with_traces": len(rows),
        "cases_without_traces": len(trace_less),
        "spans_in_window_total": sum(r["spans_in_window"] for r in rows),
        "error_flag_spans_total": sum(r["error_flag_spans"] for r in rows),
        "status2_spans_total": sum(sum(int(v) for k, v in r["status_code_histogram"].items()
                                        if k in ("2", "2.0")) for r in rows),
        "cases_with_any_error_flag_span": sum(1 for r in rows if r["error_flag_spans"]),
        "cases_with_truth_service_error_span": sum(1 for r in rows if r["truth_service_has_error_span"]),
        "cases_with_cross_service_error_pair": sum(1 for r in rows if r["error_parent_child_cross_service_pairs"]),
        "by_fault": {f: {"cases": sum(1 for r in rows if fault(r["case"]) == f),
                         "cases_with_error_spans": sum(1 for r in rows
                                                       if fault(r["case"]) == f and r["error_flag_spans"])}
                     for f in sorted({fault(r["case"]) for r in rows})},
        "per_case": rows, "trace_less_cases": trace_less}
    args.out.mkdir(parents=True, exist_ok=True)
    atomic_json(args.out / "error-span-census.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("per_case", "by_fault")}, indent=2))


if __name__ == "__main__":
    main()
