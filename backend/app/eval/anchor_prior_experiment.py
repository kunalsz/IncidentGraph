"""Predeclared anchor-prior repair candidates for `baseline_rrf` (versioned).

Why this exists
---------------
`rcaeval_runner` hardcodes `anchor_service="service:frontend"` because RCAEval
supplies no symptom. `random_walk` restarts at the anchor, so in RE2-OB the rwr
component elects `frontend` in 90/90 archived cases while the true service is
never its rank-1 (0/90). One of the four effectively voting components is thus a
constant vote against the answer, which is the standing explanation for
baseline_rrf's 35-40% AC@1 while severity alone reaches 78%.

These candidates repair the ANCHOR PRIOR, not the fusion: definitions below are
declared before any number is computed, and the protocol file is written to disk
before the evaluation runs. `p2_ablation.ARMS` is deliberately NOT extended —
that list is frozen for the historical experiment.

Selection discipline
--------------------
All 90 archived cases have already informed earlier decisions (sampler audit,
the original 50/40 ablation, twelve live cases), so any choice made here is
EXPLORATORY. `SELECTION_RULE` is fixed before results and only nominates a
candidate for implementation; promotion to a product default still requires
confirmation on data that had no part in choosing it.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from backend.app.eval import evaluator as ev
from backend.app.eval import p2_ablation as base
from backend.app.graph.scorers import neutral_walk
from backend.app.eval.rcaeval_runner import atomic_json, sha256
from backend.app.graph.scorers import random_walk
from config import tuning

VERSION = "anchor-prior-v2"

# Declared before evaluation. References first, then the three repair
# candidates in the order the selection rule prefers them.
CANDIDATES: dict[str, dict] = {
    "R0_anchored_entry_rwr": {
        "kind": "reference_reconstruction",
        "definition": "the prior that shipped before this repair: the personalised walk "
                      "restarting at the corpus entry service",
        "implementation": "recomputed offline from the stored scope edges, anchor and "
                          "tuning constants, because a substrate generated under the "
                          "anchor-neutral default stores neutral masses and cannot be "
                          "replayed back to the anchored prior",
        "declared": "v2 addition: a REFERENCE, not a new arm - the substrate omitted the "
                    "historical baseline, and a comparison needs its reference point"},
    "baseline_rrf": {"kind": "reference", "rwr": "unchanged",
                     "definition": "the shipped five-component fusion, replayed"},
    "severity_only": {"kind": "reference", "rwr": "unchanged",
                      "definition": "existing arm: only severity votes; upper reference"},
    "C1_cold_drop_rwr": {"kind": "repair",
                         "definition": "when the anchor is not case-derived (cold ranking), "
                                       "omit rwr from the vote entirely",
                         "implementation": "weights rwr=0; all other components 1"},
    "C2_rwr_anomalous_only": {"kind": "repair",
                              "definition": "keep the anchored walk but let it vote only for "
                                            "services that carry their own anomaly evidence",
                              "implementation": "mask rwr to 0 where stored severity <= 0"},
    "C3_anchor_neutral_rwr": {"kind": "repair",
                              "definition": "replace the anchor-personalised walk with a "
                                            "uniform-restart walk over the same scope edges",
                              "implementation": "uniform restart 1/n, phi=0.15, 20 iterations, "
                                                "dead ends stay put instead of leaking to the anchor"},
}

SELECTION_RULE = (
    "Nominate the repair candidate with the highest deployed AC@1 over all 90 archived "
    "cases. Tie-break: (1) higher pessimistic AC@1, (2) higher MRR, (3) declared candidate "
    "order. A candidate is only nominated for production implementation if it also beats "
    "baseline_rrf's deployed AC@1 with exact McNemar p < 0.01 on the paired outcomes and "
    "does not lose more cases than it wins. severity_only is an upper reference, not a "
    "candidate: choosing it here would re-decide an earlier question on cases that already "
    "answered it. No candidate may be added or reweighted after results are read."
)

# Default substrate: the original 50/40 replacement split. `--archive` accepts
# any directory of case records, so the same predeclared candidates can be
# evaluated on a regenerated substrate or on a second corpus.
ARCHIVE = {"discovery": ROOT / "output/p2-rebuilt-20261018/discovery-records",
           "holdout": ROOT / "output/p2-rebuilt-20261018/holdout-records"}


def transform(record, name):
    """(components, weights) for one candidate on one record."""
    components = base.candidate_components(record)
    weights = {component: 1.0 for component in base.COMPONENTS}
    if name == "severity_only":
        return components, {component: float(component == "severity")
                            for component in base.COMPONENTS}
    if name == "C1_cold_drop_rwr":
        weights["rwr"] = 0.0
        return components, weights
    if name == "C2_rwr_anomalous_only":
        masked = {sid: {**values, "rwr": values.get("rwr", 0.0) if values.get("severity", 0.0) > 0 else 0.0}
                  for sid, values in components.items()}
        return masked, weights
    if name == "C3_anchor_neutral_rwr":
        scope = (record.get("artifacts") or {}).get("scope") or {}
        ids = [n["id"] for n in scope.get("nodes", []) if n.get("type") == "Service"]
        # Production definition, imported rather than duplicated.
        mass = neutral_walk(ids, scope.get("edges", []))
        return ({sid: {**values, "rwr": mass.get(sid, 0.0)} for sid, values in components.items()},
                weights)
    if name == "R0_anchored_entry_rwr":
        scope = (record.get("artifacts") or {}).get("scope") or {}
        ids = [n["id"] for n in scope.get("nodes", []) if n.get("type") == "Service"]
        anchor = scope.get("anchor_service") or base.anchor_of(record)
        if anchor not in ids:
            raise ValueError(f"anchored reference needs the anchor in scope: {anchor}")
        mass = random_walk(ids, scope.get("edges", []), anchor)  # production definition
        return ({sid: {**values, "rwr": mass.get(sid, 0.0)} for sid, values in components.items()},
                weights)
    if name == "baseline_rrf":
        return components, weights
    raise ValueError(f"unknown candidate: {name}")


def replay(record, name):
    """Mirror of `p2_ablation.replay_case` with a candidate transform injected.

    Fusion, tie order, alias collapsing and tie bounds all come from the shared
    helpers so only the component preparation differs.
    """
    components, weights = transform(record, name)
    config = record["artifacts"]["ranking"].get("config") or {}
    scores = base.fuse(components, weights, anchor=base.anchor_of(record),
                       epsilon=config.get("rrf_epsilon"))
    scope = (record.get("artifacts") or {}).get("scope") or {}
    pagerank = {n["id"]: n.get("properties", {}).get("pagerank", 0) for n in scope.get("nodes", [])}
    order = base.deployed_order(scores, pagerank)
    service_scores = {}
    for sid, score in scores.items():
        for alias in base.service_names([sid]):
            service_scores[alias] = max(service_scores.get(alias, score), score)
    truth = record["gt_service"]
    truth_score = service_scores.get(truth)
    if truth_score is None:
        pessimistic = optimistic = None
    else:
        pessim = 1 + sum(1 for name_, s in service_scores.items()
                         if name_ != truth and s >= truth_score - 1e-12)
        optim = 1 + sum(1 for name_, s in service_scores.items()
                        if name_ != truth and s > truth_score + 1e-12)
        pessimistic, optimistic = pessim, optim
    return {"order": order, "names": base.service_names(order),
            "pessimistic": pessimistic, "optimistic": optimistic}


def evaluate(records, name):
    service_all = ev.Evaluator(5)
    per_class: dict[str, ev.Evaluator] = {}
    top1, pessimistic, optimistic, positions = [], [], [], []
    for record in records:
        result = replay(record, name)
        answer = ev.Node(record["gt_service"], "unknown")
        ranks = [ev.Node(n, "unknown") for n in result["names"]]
        service_all.add_case(ranks, answer, record.get("n_candidates_service"),
                             record.get("answer_in_service_universe", True))
        fault = ev.FAULT_CLASS.get(record.get("gt_fault"), "unknown")
        per_class.setdefault(fault, ev.Evaluator(5)).add_case(
            ranks, answer, record.get("n_candidates_service"),
            record.get("answer_in_service_universe", True))
        top1.append(bool(result["names"]) and result["names"][0] == record["gt_service"])
        pessimistic.append(result["pessimistic"] == 1)
        optimistic.append(result["optimistic"] == 1)
        positions.append(result["names"].index(record["gt_service"]) + 1
                         if record["gt_service"] in result["names"] else None)
    n = len(records)

    def block(evaluator):
        return {"n": evaluator.num, "ac@1": evaluator.accuracy_service(1),
                "ac@3": evaluator.accuracy_service(3), "ac@5": evaluator.accuracy_service(5),
                "mrr": evaluator.mrr_service()}
    overall = block(service_all)
    overall.update({"ac@1_ci95": base.wilson(sum(top1), n) if n else None,
                    "ac@1_pessimistic": (sum(pessimistic) / n) if n else None,
                    "ac@1_optimistic": (sum(optimistic) / n) if n else None})
    return {"overall": overall, "per_class": {k: block(v) for k, v in sorted(per_class.items())},
            "correct_at_1": top1, "positions": positions}


def run(records) -> dict:
    usable = [r for r in records if base.usable(r)]
    skipped = [r.get("case") for r in records if not base.usable(r)]
    fidelity = {name: [r["case"] for r in usable
                       if replay(r, name)["order"] != base.recorded_order(r)]
                for name in ("baseline_rrf",)}
    results = {name: evaluate(usable, name) for name in CANDIDATES}
    baseline = results["baseline_rrf"]["correct_at_1"]
    report = {"version": VERSION, "n_cases": len(usable), "skipped": skipped,
              "baseline_replay_mismatches": len(fidelity["baseline_rrf"]),
              "baseline_replay_mismatch_cases": fidelity["baseline_rrf"], "candidates": {}}
    for name, result in results.items():
        entry = {**result["overall"], "per_class": result["per_class"]}
        if name != "baseline_rrf":
            arm_only = sum(a and not b for a, b in zip(result["correct_at_1"], baseline))
            base_only = sum(b and not a for a, b in zip(result["correct_at_1"], baseline))
            entry["vs_baseline"] = {"candidate_only_correct": arm_only,
                                    "baseline_only_correct": base_only,
                                    "mcnemar_exact_p": round(base.mcnemar_exact(arm_only, base_only), 8)}
        report["candidates"][name] = entry
    return report


def render(report: dict) -> str:
    lines = [f"anchor-prior repair {report['version']}  n={report['n_cases']}  "
             f"baseline replay mismatches={report['baseline_replay_mismatches']}",
             f"{'candidate':24s} {'AC@1':>6s} {'95% CI':>15s} {'pess':>6s} {'opt':>6s} "
             f"{'AC@3':>6s} {'MRR':>6s}  vs baseline"]
    fmt = lambda x: "   n/a" if x is None else f"{x:6.1%}"
    for name, row in report["candidates"].items():
        ci = row["ac@1_ci95"] or [0, 0]
        comparison = row.get("vs_baseline")
        extra = (f"{comparison['candidate_only_correct']} win / "
                 f"{comparison['baseline_only_correct']} lose, p={comparison['mcnemar_exact_p']}"
                 if comparison else "-")
        lines.append(f"{name:24s} {fmt(row['ac@1'])} [{ci[0]:5.1%},{ci[1]:5.1%}] "
                     f"{fmt(row['ac@1_pessimistic'])} {fmt(row['ac@1_optimistic'])} "
                     f"{fmt(row['ac@3'])} {row['mrr']:6.3f}  {extra}")
    lines += ["", "All 90 archived cases are development data: selection here is exploratory "
                  "and cannot promote a product default by itself.", SELECTION_RULE]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--archive", type=Path, action="append",
                        help="record directory to evaluate (repeatable); defaults to the frozen 50/40 split")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "output" / ("anchor-prior-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())))
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    sources = ["backend/app/eval/anchor_prior_experiment.py", "backend/app/eval/p2_ablation.py",
               "backend/app/graph/scorers.py", "backend/app/engine/p2_rank.py", "config/tuning.py"]
    archives = args.archive or list(ARCHIVE.values())
    protocol = {"version": VERSION, "candidates": CANDIDATES, "selection_rule": SELECTION_RULE,
                "archive": [str(path) for path in archives],
                "evaluation_scope": "all 90 archived cases from both partitions; exploratory, not a holdout",
                "frozen_p2_ablation_arms": False,
                "source_sha256": {path: sha256(ROOT / path) for path in sources},
                "reused_scoring": "p2_ablation.fuse/deployed_order/service_names, evaluator, mcnemar_exact"}
    atomic_json(args.output / "protocol.json", protocol)  # predeclaration, before any number
    records = [record for path in archives for record in base.load_records(path)]
    report = run(records)
    report["protocol_sha256"] = sha256(args.output / "protocol.json")
    atomic_json(args.output / "results.json", report)
    (args.output / "RESULTS.md").write_text(
        f"# Anchor-prior repair candidates ({VERSION})\n\n"
        f"Predeclared in `protocol.json` before evaluation; candidates were not changed afterwards.\n\n"
        "```\n" + render(report) + "\n```\n\n"
        "Protocol file: `protocol.json` (sha256 recorded in results). "
        "Machine-readable numbers: `results.json`.\n")
    print(render(report))
    print("wrote", args.output / "results.json")
    return 1 if report["baseline_replay_mismatches"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
