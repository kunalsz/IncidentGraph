"""Where should error traces sit in the span sample? Predeclared policy comparison.

Context: `ERROR_SPAN_BUDGET = 3_000  # of which reserved for traces carrying error
spans` is documentation only - nothing reads it. The sampler's sole error
preference is a within-stratum priority, so with ~120 minute x period x service
strata, failure spans are diluted: 937 of 6,481 reach the sample and 28 of 122
caller/callee pairs survive.

Policies measured, off the same traces, no database and no model:

  A control            error traces lead within each stratum (production)
  B error_first        whole error-carrying traces first, then the same
                       stratified round robin fills the rest
  C error_reserved     like B but error traces may consume at most
                       ERROR_SPAN_BUDGET spans, so the remainder keeps the
                       latency/baseline coverage the stratified pass provides
  D error_first_12k    policy B with twice the total span budget

Reported per case and policy: failure spans admitted, caller/callee pairs the P4
propagation validator accepts, traces kept, pre/post span balance, distinct
services, and non-error spans - the last one is the cost, since those carry the
latency baseline.

Fidelity: policy A must reproduce production's selection exactly. If it does not,
no other policy result is trustworthy and the run reports the mismatch.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.app.engine import policy
from backend.app.eval import rcaeval_adapter as adapter
from backend.app.eval.rcaeval_runner import atomic_json, sha256

POLICIES = ("A_control", "B_error_first", "C_error_reserved", "D_error_first_12k")
RESERVED_ERROR_SPANS = 3_000  # arm C only: cap on error-trace spans


def prepare(traces: pd.DataFrame, t0_ms: int, t1_ms: int, inject_ms: int):
    source = traces[traces["startTimeMillis"].notna()].copy()
    window = source[source["startTimeMillis"].between(t0_ms, t1_ms)].copy()
    window["service_id"] = window["serviceName"].astype(str).map(
        lambda x: adapter.TRACE_SERVICE_MAP.get(x, x))
    window["ts"] = (window["startTimeMillis"] // 1000).astype("int64")
    window["duration_us"] = window["duration"]
    window["error"] = window["statusCode"].map(policy.error_status)
    groups = {str(tid): group for tid, group in window.groupby("traceID", sort=True)}
    error_traces = {tid for tid, group in groups.items() if bool(group["error"].any())}
    return window, groups, error_traces


def stratified_fill(window, groups, budget: int, seed: int, take_hook, inject_ms: int,
                    reserved_error_spans: int | None = None):
    """Production's selection loop, parameterised only in what it takes next.

    `take_hook` decides whether an error trace is admissible on the error-first
    pass; the stratified round robin itself is byte-for-byte the production one.
    """
    import hashlib

    def priority(tid):
        group = groups[tid]
        return (not bool(group["error"].any()),
                hashlib.sha256(f"{seed}:{tid}".encode()).hexdigest(), tid)

    strata = {}
    for tid, group in groups.items():
        if len(group) > budget:
            continue
        for row in group.itertuples():
            period = "pre" if row.startTimeMillis < inject_ms else "post"
            key = (period, int((row.startTimeMillis - inject_ms) // 60000), row.service_id)
            strata.setdefault(key, set()).add(tid)
    queues = {key: sorted(ids, key=priority) for key, ids in sorted(strata.items())}
    selected, consumed = set(), 0
    error_spans_used = 0

    def take(tid, from_error_pass=False):
        nonlocal consumed, error_spans_used
        if tid in selected or consumed + len(groups[tid]) > budget:
            return False
        if from_error_pass and reserved_error_spans is not None:
            if error_spans_used + len(groups[tid]) > reserved_error_spans:
                return False
        selected.add(tid)
        consumed += len(groups[tid])
        if bool(groups[tid]["error"].any()):
            error_spans_used += len(groups[tid])
        return True

    if take_hook is not None:
        for tid in take_hook(groups, priority):
            take(tid, from_error_pass=True)
    for period in ("pre", "post"):
        ids = {tid for key, ids in strata.items() if key[0] == period for tid in ids}
        for tid in sorted(ids, key=lambda tid: (len(groups[tid]), priority(tid))):
            if take(tid):
                break
    while any(queues.values()):
        for key in queues:
            queue = queues[key]
            while queue:
                tid = queue.pop(0)
                if take(tid):
                    break
    return window[window["traceID"].astype(str).isin(selected)].copy()


def select(traces, t0_ms, t1_ms, inject_ms, name):
    budget = 12_000 if name == "D_error_first_12k" else adapter.SPAN_BUDGET
    window, groups, error_traces = prepare(traces, t0_ms, t1_ms, inject_ms)
    if name == "A_control":
        hook = None
    elif name in ("B_error_first", "D_error_first_12k"):
        hook = lambda groups_, priority_: sorted(
            (tid for tid, g in groups_.items() if bool(g["error"].any()) and len(g) <= budget),
            key=priority_)
    elif name == "C_error_reserved":
        hook = lambda groups_, priority_: sorted(
            (tid for tid, g in groups_.items() if bool(g["error"].any()) and len(g) <= budget),
            key=priority_)
    else:
        raise ValueError(name)
    included = stratified_fill(window, groups, budget, 0, hook, inject_ms,
                               reserved_error_spans=RESERVED_ERROR_SPANS
                               if name == "C_error_reserved" else None)
    return included.sort_values(["startTimeMillis", "traceID", "spanID"]).reset_index(drop=True)


def accepted_pairs(included: pd.DataFrame, edges) -> int:
    sample = included.copy()
    sample["service"] = sample["serviceName"].astype(str).map(
        lambda x: "service:" + adapter.TRACE_SERVICE_MAP.get(x, x))
    failing = sample[sample["statusCode"].map(policy.error_status)]
    records = {str(r["spanID"]): {
        "kind": "Span", "service": r["service"], "positive": True,
        "props": {"id": str(r["spanID"]), "ts": int(r["startTimeMillis"]),
                  "duration_us": int(r["duration"]) if pd.notna(r["duration"]) else 0,
                  "trace_id": str(r["traceID"]), "span_id": str(r["spanID"]),
                  "parent_id": str(r["parentSpanID"]) if pd.notna(r["parentSpanID"]) else "",
                  "status": r["statusCode"]}} for _, r in failing.iterrows()}
    return sum(len(policy.propagation_support(records, edge)) // 2 for edge in edges)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=ROOT / "data/RE2-OB")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fidelity-json", type=Path,
                        help="earlier production measurement to check policy A against, "
                             "instead of re-sampling with production (halves the runtime)")
    args = parser.parse_args()
    topology = json.loads(adapter.OB_TOPOLOGY.read_text())
    edges = [{"source": d["from"], "target": d["to"], "type": "DEPENDS_ON", "properties": {}}
             for d in topology["depends_on"]]
    # Only cases with failure spans can differ across policies; everything else
    # is control by construction. Scope fixed here, before any result is read.
    cases = []
    for case_dir in sorted(p for p in args.data.iterdir() if p.is_dir() and p.name.startswith("re2ob_")):
        frame = pd.read_parquet(case_dir / "traces.parquet", columns=["statusCode"])
        if frame["statusCode"].map(policy.error_status).any():
            cases.append(case_dir)
    recorded = {}
    if args.fidelity_json and args.fidelity_json.is_file():
        previous = json.loads(args.fidelity_json.read_text())
        recorded = {r["case"]: r for r in previous["per_case"]}
    started = time.monotonic()
    report = {"version": "sampler-policy-v1", "policies": list(POLICIES),
              "cases": [c.name for c in cases],
              "fidelity_note": "A must equal production select_trace_sample",
              "per_case": {}, "totals": {p: {} for p in POLICIES}}
    for case_dir in cases:
        traces = pd.read_parquet(case_dir / "traces.parquet")
        inject_ms = int((case_dir / "inject_time.txt").read_text().strip()) * 1000
        t0, t1 = inject_ms - 600_000, inject_ms + 600_000
        production = None
        if case_dir.name not in recorded:
            production, _ = adapter.select_trace_sample(traces, t0, t1, inject_ms)
        report["per_case"][case_dir.name] = {}
        for name in POLICIES:
            included = select(traces, t0, t1, inject_ms, name)
            entry = {"failure_spans": int(included["statusCode"].map(policy.error_status).sum()),
                     "accepted_pairs": accepted_pairs(included, edges),
                     "traces": int(included["traceID"].nunique()),
                     "spans": int(len(included)),
                     "pre": int((included["startTimeMillis"] < inject_ms).sum()),
                     "post": int((included["startTimeMillis"] >= inject_ms).sum()),
                     "services": int(included["serviceName"].nunique()),
                     "non_error_spans": int((~included["statusCode"].map(policy.error_status)).sum())}
            if name == "A_control":
                if production is not None:
                    prod_fail = int(production["statusCode"].map(policy.error_status).sum())
                    prod_pairs = accepted_pairs(production, edges)
                else:
                    prod_fail = recorded[case_dir.name]["failure_spans_sampled_new"]
                    prod_pairs = recorded[case_dir.name]["pairs_sampled_and_accepted_new"]
                entry["production_failure_spans"] = prod_fail
                entry["production_accepted_pairs"] = prod_pairs
                entry["fidelity_ok"] = (entry["failure_spans"] == prod_fail
                                        and entry["accepted_pairs"] == prod_pairs)
            report["per_case"][case_dir.name][name] = entry
            for key, value in entry.items():
                if isinstance(value, int):
                    report["totals"][name][key] = report["totals"][name].get(key, 0) + value
        print(case_dir.name, {n: report["per_case"][case_dir.name][n]["accepted_pairs"] for n in POLICIES}, flush=True)
    report["runtime_s"] = round(time.monotonic() - started, 1)
    report["adapter_sha256"] = sha256(ROOT / "backend/app/eval/rcaeval_adapter.py")
    report["policy_sha256"] = sha256(ROOT / "backend/app/engine/policy.py")
    args.out.mkdir(parents=True, exist_ok=True)
    atomic_json(args.out / "sampler-policy-results.json", report)
    for name in POLICIES:
        t = report["totals"][name]
        print(f"{name:18s} failure_spans {t.get('failure_spans',0):5d} "
              f"accepted_pairs {t.get('accepted_pairs',0):3d} traces {t.get('traces',0):5d} "
              f"non_error_spans {t.get('non_error_spans',0):5d} services {t.get('services',0)}")
    fidelity = all(v["A_control"]["fidelity_ok"] for v in report["per_case"].values())
    print("control fidelity vs production:", fidelity)


if __name__ == "__main__":
    main()
