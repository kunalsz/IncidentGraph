"""Offline P2 fusion ablation over recorded RCAEval case records.

No graph, no LLM, no telemetry: every `<case>.record.json` already stores the
five P2 components per candidate (`artifacts.ranking.candidates[*].components`),
so alternative fusion policies can be replayed for free with the REAL
`reciprocal_rank_fusion`. Motivation (from the 50-case agent run): the true
service had the highest `severity` in 40/50 cases but ranked first in only
16/50, i.e. rank-only fusion appears to discard a strong magnitude signal.

Protocol guards
---------------
* Arms are a frozen, declared list (`ARMS`, versioned). Do not add arms after
  looking at results on the same cases: that is tuning on the test set. Use
  `--cases-file` to evaluate on cases that were not used to choose an arm.
* Each record's declared P2 policy must reproduce its recorded ranking (legacy
  records without policy metadata use baseline_rrf). The report counts replay
  fidelity mismatches. Large mismatch means the
  record schema or fusion code drifted and no arm result should be trusted.
* Ties are explicit. `ac@1` uses the deployed tie order (P2 breaks score ties
  alphabetically); `ac@1_optimistic/_pessimistic` bound the effect of ties.
* Differences are tested pairwise against the baseline (exact McNemar), never
  eyeballed from two marginal accuracies on n~50.

Ground-truth boundary: this module reads labels from the records and is
evaluation-only, like `evaluator.py`; nothing here feeds the engine.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from backend.app.eval import evaluator as ev
from backend.app.graph.scorers import reciprocal_rank_fusion
from config import tuning

ARMS_VERSION = "p2-ablation-v1"
COMPONENTS = ("phi1", "phi3", "severity", "rwr", "deploy_proximity")
DEFAULT_ANCHOR = "service:frontend"
_ALL = {name: 1.0 for name in COMPONENTS}

# name -> (component weights, exclude the anchor's own random-walk mass)
ARMS: dict[str, dict] = {
    "baseline_rrf": {"weights": _ALL, "rwr_excludes_anchor": False},
    "severity_only": {"weights": {"severity": 1.0}, "rwr_excludes_anchor": False},
    "drop_rwr": {"weights": {**_ALL, "rwr": 0.0}, "rwr_excludes_anchor": False},
    "drop_phi1": {"weights": {**_ALL, "phi1": 0.0}, "rwr_excludes_anchor": False},
    "drop_phi3": {"weights": {**_ALL, "phi3": 0.0}, "rwr_excludes_anchor": False},
    "rwr_excl_anchor": {"weights": _ALL, "rwr_excludes_anchor": True},
    "severity_x2": {"weights": {**_ALL, "severity": 2.0}, "rwr_excludes_anchor": False},
    "severity_x3": {"weights": {**_ALL, "severity": 3.0}, "rwr_excludes_anchor": False},
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_records(run_dir, cases=None):
    """Case records with a usable P2 artifact. Falls back to `_checkpoint.json`."""
    run_dir = Path(run_dir)
    records = []
    for path in sorted(run_dir.glob("*.record.json")):
        records.append(json.loads(path.read_text()))
    if not records and (run_dir / "_checkpoint.json").is_file():
        records = list(json.loads((run_dir / "_checkpoint.json").read_text()).get("cases", []))
    if cases is not None:
        wanted = set(cases)
        records = [r for r in records if r.get("case") in wanted]
    return records


def usable(record) -> bool:
    ranking = (record.get("artifacts") or {}).get("ranking")
    return bool(record.get("status") == "ok" and ranking and ranking.get("candidates")
                and record.get("gt_service"))


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------

def fuse(components: dict[str, dict], weights: dict[str, float], *, anchor=None,
         rwr_excludes_anchor=False, epsilon=None):
    """Weighted RRF using the production rank logic. Returns {id: score}.

    Component ranks (average ties, non-positive values excluded) come from
    `scorers.reciprocal_rank_fusion`; only the weighting is applied here.
    """
    epsilon = tuning.RRF_EPSILON if epsilon is None else epsilon
    work = {sid: {name: float(c.get(name, 0.0)) for name in COMPONENTS}
            for sid, c in components.items()}
    for name in COMPONENTS:
        if weights.get(name, 0.0) == 0.0:
            for c in work.values():
                c[name] = 0.0  # excluded from that component's ranking entirely
    if rwr_excludes_anchor and anchor in work:
        work[anchor]["rwr"] = 0.0
    _, ranks = reciprocal_rank_fusion(work, epsilon)
    return {sid: sum(weights.get(name, 0.0) / (epsilon + rank)
                     for name, rank in ranks[sid].items())
            for sid in work}


def deployed_order(scores: dict[str, float], pagerank=None) -> list[str]:
    """P2's tie rule; legacy records without scoped pagerank use zero."""
    pagerank = pagerank or {}
    return sorted(scores, key=lambda sid: (-scores[sid], -pagerank.get(sid, 0), sid))


def recorded_policy(record):
    """Missing legacy metadata means baseline, never reinterpret explicit policies."""
    config = record["artifacts"]["ranking"].get("config") or {}
    policy = config.get("rank_policy", record.get("rank_policy", "baseline_rrf"))
    if record.get("rank_policy", policy) != policy:
        raise ValueError("record and ranking rank_policy disagree")
    if policy not in tuning.P2_RANK_POLICIES:
        raise ValueError(f"unknown recorded P2 rank policy: {policy}")
    return policy


def candidate_components(record) -> dict[str, dict]:
    return {c["id"]: c["components"] for c in record["artifacts"]["ranking"]["candidates"]}


def anchor_of(record) -> str:
    return ((record.get("artifacts") or {}).get("context") or {}).get("anchor_service") or DEFAULT_ANCHOR


def service_names(order):
    """Production service identities, deduplicated in deployed rank order."""
    return ev.service_rank_entities(
        [sid.removeprefix("service:") for sid in order if sid.startswith("service:")])


def replay_case(record, arm: dict) -> dict:
    components = candidate_components(record)
    config = record["artifacts"]["ranking"].get("config") or {}
    scores = fuse(components, arm["weights"], anchor=anchor_of(record),
                  rwr_excludes_anchor=arm["rwr_excludes_anchor"],
                  epsilon=config.get("rrf_epsilon"))
    scope = (record.get("artifacts") or {}).get("scope") or {}
    pagerank = {n["id"]: n.get("properties", {}).get("pagerank", 0)
                for n in scope.get("nodes", [])}
    order = deployed_order(scores, pagerank)
    # An entity's best-scoring alias determines its deployed position. Collapse
    # aliases before computing bounds too, so a service contributes only once.
    # Keep the raw order above untouched for exact replay-fidelity comparisons.
    service_scores = {}
    for sid, score in scores.items():
        for name in service_names([sid]):
            service_scores[name] = max(service_scores.get(name, score), score)
    truth = record["gt_service"]
    gt_score = service_scores.get(truth)
    if gt_score is None:
        pess = opt = None
    else:
        tol = 1e-12
        pess = 1 + sum(1 for name, s in service_scores.items() if name != truth and s >= gt_score - tol)
        opt = 1 + sum(1 for name, s in service_scores.items() if name != truth and s > gt_score + tol)
    return {"order": order, "names": service_names(order), "pessimistic": pess, "optimistic": opt}


def recorded_order(record) -> list[str]:
    cands = sorted(record["artifacts"]["ranking"]["candidates"], key=lambda c: (c["rank"], c["id"]))
    return [c["id"] for c in cands]


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    a = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round((c - a) / d, 4), round((c + a) / d, 4)]


def mcnemar_exact(arm_only: int, base_only: int) -> float:
    """Two-sided exact binomial p on the discordant pairs."""
    n = arm_only + base_only
    if n == 0:
        return 1.0
    k = min(arm_only, base_only)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def evaluate_arm(records, arm: dict) -> dict:
    ev_all = ev.Evaluator(5)
    per_class = {}
    top1, pess1, opt1, ties = [], [], [], 0
    positions = []
    for record in records:
        result = replay_case(record, arm)
        answer = ev.Node(record["gt_service"], "unknown")
        ranks = [ev.Node(n, "unknown") for n in result["names"]]
        n_cand = record.get("n_candidates_service")
        found = record.get("answer_in_service_universe", True)
        ev_all.add_case(ranks, answer, n_cand, found)
        fault = ev.FAULT_CLASS.get(record.get("gt_fault"), "unknown")
        per_class.setdefault(fault, ev.Evaluator(5)).add_case(ranks, answer, n_cand, found)
        top1.append(bool(result["names"]) and result["names"][0] == record["gt_service"])
        pess1.append(result["pessimistic"] == 1)
        opt1.append(result["optimistic"] == 1)
        if top1[-1] and result["optimistic"] == 1 and result["pessimistic"] != 1:
            ties += 1
        positions.append(result["names"].index(record["gt_service"]) + 1
                         if record["gt_service"] in result["names"] else None)
    n = len(records)

    def block(evaluator):
        return {"n": evaluator.num, "ac@1": evaluator.accuracy_service(1),
                "ac@3": evaluator.accuracy_service(3), "ac@5": evaluator.accuracy_service(5),
                "avg@5": evaluator.average_service(5), "mrr": evaluator.mrr_service()}
    overall = block(ev_all)
    overall.update({
        "ac@1_ci95": wilson(sum(top1), n),
        "ac@1_pessimistic": sum(pess1) / n if n else None,
        "ac@1_optimistic": sum(opt1) / n if n else None,
        "cases_won_only_on_tie_order": ties,
    })
    return {"overall": overall, "per_class": {k: block(v) for k, v in sorted(per_class.items())},
            "correct_at_1": top1, "positions": positions}


def run_ablation(records, arms=None) -> dict:
    arms = ARMS if arms is None else arms
    usable_records = [r for r in records if usable(r)]
    skipped = [r.get("case") for r in records if not usable(r)]
    if not usable_records:
        return {"arms_version": ARMS_VERSION, "n_cases": 0, "skipped": skipped, "arms": {}}
    policies = {r["case"]: recorded_policy(r) for r in usable_records}
    fidelity_bad = [r["case"] for r in usable_records
                    if replay_case(r, ARMS[policies[r["case"]]])["order"] != recorded_order(r)]
    baseline_bad = [name for name in fidelity_bad if policies[name] == "baseline_rrf"]
    results = {name: evaluate_arm(usable_records, arm) for name, arm in arms.items()}
    base = results.get("baseline_rrf")
    report = {}
    for name, res in results.items():
        entry = {**res["overall"], "per_class": res["per_class"]}
        if base is not None and name != "baseline_rrf":
            arm_only = sum(a and not b for a, b in zip(res["correct_at_1"], base["correct_at_1"]))
            base_only = sum(b and not a for a, b in zip(res["correct_at_1"], base["correct_at_1"]))
            entry["vs_baseline_ac@1"] = {"arm_only_correct": arm_only, "baseline_only_correct": base_only,
                                        "mcnemar_exact_p": round(mcnemar_exact(arm_only, base_only), 6)}
        report[name] = entry
    return {"arms_version": ARMS_VERSION, "n_cases": len(usable_records), "skipped": skipped,
            "replay_fidelity": {"baseline_mismatches": len(baseline_bad),
                                "recorded_policy_mismatches": len(fidelity_bad),
                                "recorded_policies": sorted(set(policies.values())), "cases": fidelity_bad},
            "arms": report}


def render(summary: dict) -> str:
    if not summary["n_cases"]:
        return f"no usable records (skipped: {len(summary['skipped'])})"
    fidelity = summary["replay_fidelity"]
    fid = fidelity.get("recorded_policy_mismatches", fidelity["baseline_mismatches"])
    lines = [f"P2 ablation {summary['arms_version']}  n={summary['n_cases']}  "
             f"recorded-policy replay mismatches={fid}"
             + ("   <-- DO NOT TRUST ARMS" if fid else ""),
             f"{'arm':18s} {'AC@1':>6s} {'95% CI':>14s} {'pess':>6s} {'opt':>6s} {'AC@3':>6s} "
             f"{'Avg@5':>6s} {'MRR':>6s}  vs baseline (arm-only/base-only, p)"]
    f = lambda x: "  n/a" if x is None else f"{x:6.1%}"
    for name, a in summary["arms"].items():
        ci = a["ac@1_ci95"]
        cmp_ = a.get("vs_baseline_ac@1")
        extra = (f"{cmp_['arm_only_correct']}/{cmp_['baseline_only_correct']}, p={cmp_['mcnemar_exact_p']}"
                 if cmp_ else "-")
        lines.append(f"{name:18s} {f(a['ac@1'])} [{ci[0]:5.1%},{ci[1]:5.1%}] {f(a['ac@1_pessimistic'])} "
                     f"{f(a['ac@1_optimistic'])} {f(a['ac@3'])} {f(a['avg@5'])} {a['mrr']:6.3f}  {extra}")
    lines.append("")
    lines.append("Arms were declared before this run. Results on the cases used to pick an arm are "
                 "exploratory; confirm on held-out cases (--cases-file).")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", help="run directory containing *.record.json (or _checkpoint.json)")
    parser.add_argument("--cases-file", help="text file, one case name per line, to restrict evaluation")
    parser.add_argument("--json", dest="json_out", help="write the full summary here")
    args = parser.parse_args(argv)
    cases = None
    if args.cases_file:
        cases = [line.strip() for line in Path(args.cases_file).read_text().splitlines() if line.strip()]
    summary = run_ablation(load_records(args.run_dir, cases))
    print(render(summary))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(summary, indent=2))
    return 1 if summary["n_cases"] == 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
