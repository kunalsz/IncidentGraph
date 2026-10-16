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
    args = parser.parse_args()
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
