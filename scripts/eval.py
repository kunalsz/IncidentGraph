#!/usr/bin/env python
import argparse
import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.engine.contracts import Investigation, InvestigationWindow
from backend.app.engine.p0_resolve import resolve
from backend.app.engine.p1_scope import scope_investigation
from backend.app.engine.p2_rank import rank_investigation
from backend.app.graph.client import client, graph_for
from backend.app.seed import loader
from config import tuning


FIXTURES = ("payment_failure", "cart_failure", "recommendation_slowdown")


def evaluate_fixture(name):
    fixture = json.loads((loader.FIXTURES_DIR / f"{name}.json").read_text())
    topology = json.loads((loader.SEED_DIR / "topology.json").read_text())
    graph_name = f"incidentgraph:session:eval-{uuid.uuid4().hex}"
    graph = graph_for(graph_name)
    created = False
    try:
        loader.load_topology(graph, topology)
        created = True
        loader.load_fixture(graph, fixture)
        incident = fixture["incident"]
        alert = incident.get("anchor_alert") or next(
            (write["props"]["id"] for write in fixture["graph_writes"] if write["node"] == "Alert"), None)
        anchor = resolve(incident["symptom"], graph, alert_id=alert)
        if not anchor.matched:
            raise ValueError(f"unresolved fixture anchor: {name}")
        loader.merge_node(graph, "Incident", {
            "id": incident["id"], "symptom": incident["symptom"],
            "start_ts": incident["start_ts"], "status": "investigating",
        })
        loader.merge_edge(graph, incident["id"], "AFFECTS", anchor.anchor_service)
        inv = Investigation(
            incident_id=incident["id"], session_graph=graph_name, symptom=incident["symptom"],
            anchor_service=anchor.anchor_service, anchor_alert=alert, budget={},
            window=InvestigationWindow(t0=incident["start_ts"] - tuning.WINDOW_PAD_MS,
                                       t1=incident["start_ts"] + tuning.WINDOW_PAD_MS),
        )
        ranking = rank_investigation(inv, scope_investigation(inv))
        ranks = [candidate.id for candidate in ranking.candidates]
        expected = fixture["expected"]["root_cause_service"]
        rank = ranks.index(expected) + 1 if expected in ranks else None
        return {"fixture": name, "root_rank": rank, "top3_pass": rank is not None and rank <= 3,
                "n_candidates": ranking.n_candidates, "ranks": ranks,
                "mrr": 1 / rank if rank else 0.0, "llm_calls": 0,
                "truncated": ranking.truncated}
    finally:
        if created:
            client().execute_command("GRAPH.DELETE", graph_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", action="store_true")
    parser.add_argument("--rank-only", action="store_true")
    parser.add_argument("--rcaeval", type=str, default=None,
                        help="path to a fetched RE2-OB data dir (e.g. data/RE2-OB)")
    parser.add_argument("--subset", type=int, default=None,
                        help="with --rcaeval: limit to the first N indexed cases")
    parser.add_argument("--cases", type=str, default=None,
                        help="with --rcaeval: comma-separated case names")
    parser.add_argument("--report", type=str, default="output/rcaeval",
                        help="with --rcaeval: output directory")
    parser.add_argument("--with-agent", action="store_true",
                        help="reserved for explicit client-factory callers; never implicitly contacts a paid provider")
    parser.add_argument("--compare", nargs=2, metavar=("LEFT", "RIGHT"))
    parser.add_argument("--replay", type=str)
    parser.add_argument("--experiments", type=str)
    parser.add_argument("--engine-experiments", action="store_true",
                        help="execute local deterministic P3/P4/P5 arms on explicitly selected RCAEval cases")
    parser.add_argument("--recorded", type=str, help="optional explicit real-provider records for E")
    parser.add_argument("--baseline", type=str, help="matched historical P2 source for frozen experiments")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--policy", default="p2", choices=["p2", "shadow", "corrected", "legacy_membership", "eligibility"])
    args = parser.parse_args()
    if args.compare:
        from backend.app.eval.comparison import main as comparison_main
        return comparison_main([*args.compare, "--output", args.report])
    if args.replay:
        from backend.app.eval.replay import main as replay_main
        return replay_main([args.replay, "--output", args.report, "--policy", args.policy] + (["--mock"] if args.mock else []))
    if args.engine_experiments:
        if not args.rcaeval or not args.cases:
            parser.error("--engine-experiments requires --rcaeval and explicit --cases")
        from backend.app.eval.engine_experiments import main as engine_main
        return engine_main(["--data", args.rcaeval, "--output", args.report, "--cases", args.cases]
                           + (["--recorded", args.recorded] if args.recorded else []))
    if args.experiments:
        from backend.app.eval.experiments import main as experiments_main
        return experiments_main([args.experiments, "--output", args.report]
                                + (["--baseline", args.baseline] if args.baseline else []))
    if args.with_agent:
        parser.error("no implicit live providers: use frozen --replay/--experiments, or authorized run(llm_factory=...) API")
    if args.rcaeval:
        from backend.app.eval import rcaeval_runner
        cases = [c.strip() for c in args.cases.split(",")] if args.cases else None
        return rcaeval_runner.run(args.rcaeval, subset=args.subset, cases=cases,
                                  report_dir=args.report,
                                  with_agent=args.with_agent)
    if not args.rank_only:
        parser.error("only --rank-only evaluation is available until P3 through P6 are implemented")
    results = [evaluate_fixture(name) for name in FIXTURES]
    print(json.dumps({"mode": "synthetic_fixture_rank_only", "cases": results,
                      "ac_at_1": sum(r["root_rank"] == 1 for r in results) / len(results),
                      "ac_at_3": sum(r["top3_pass"] for r in results) / len(results),
                      "mrr": sum(r["mrr"] for r in results) / len(results),
                      "gate_pass": all(r["top3_pass"] for r in results)}, indent=2))
    return 0 if all(r["top3_pass"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
