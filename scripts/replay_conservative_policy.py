"""Regression replay of conservative association decisions; no new inference."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.app.engine import contracts as c
from backend.app.engine.p5_fuse import fuse_investigation
from backend.app.eval.rcaeval_runner import atomic_json, provenance


def main():
    source = ROOT / "output/severity-agent-local-frozen-20261018/agent-shadow"
    out = ROOT / "output/conservative-policy-20261018"
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    for path in sorted(source.glob("*.record.json")):
        record = json.loads(path.read_text())
        h = record["handoffs"]
        context = c.load_handoff(h["resolve"], stage="resolve", incident_id=record["artifacts"]["context"]["incident_id"])
        scope = c.load_handoff(h["scope"], stage="scope", incident_id=context.incident_id)
        ranking = c.load_handoff(h["rank"], stage="rank", incident_id=context.incident_id, scope_ref=scope.scope_ref)
        invest = c.load_handoff(h["investigate"], stage="investigate", incident_id=context.incident_id, scope_ref=scope.scope_ref)
        validation = c.load_handoff(h["validate"], stage="validate", incident_id=context.incident_id, scope_ref=scope.scope_ref)
        fused = fuse_investigation(context, ranking, invest, validation, ranking_policy="conservative_p2_update_v1")
        order = [r["candidate"].removeprefix("service:") for r in fused.service_ranking]
        baseline = [candidate.id.removeprefix("service:") for candidate in sorted(ranking.candidates, key=lambda x:(x.rank,x.id))]
        assert order == baseline
        row = {"case": record["case"], "full_order_preserved": True,
               "truth_rank": order.index(record["gt_service"])+1,
               "decision": fused.config["conservative_update"],
               "evidence": "integrity-checked stored stages; no live source revalidation or new inference"}
        rows.append(row)
        atomic_json(out / f"{record['case']}.json", {**row, "fusion": fused.model_dump(mode="json")})
    assert len(rows) == 6
    summary = {"n_cases": len(rows), "n_order_changes": 0, "harmful_overrides": 0,
               "top1_correct": sum(row["truth_rank"] == 1 for row in rows),
               "rank_update_abstentions": len(rows), "real_provider_calls": 0,
               "not_new_accuracy_evidence": True, "provenance": provenance(), "cases": rows}
    atomic_json(out / "summary.json", summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ("provenance","cases")}, indent=2))


if __name__ == "__main__":
    main()
