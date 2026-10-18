"""Replay audited coherence/winner repairs on frozen records; no new inference."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.app.engine import contracts as c, p4_validate, p5_fuse
from backend.app.graph import validators
from backend.app.eval.rcaeval_runner import atomic_json, provenance

SOURCE = ROOT / "output/severity-agent-local-frozen-20261018"
OUT = ROOT / "output/harmful-override-audit-20261018"


def main():
    OUT.mkdir(exist_ok=False, parents=True)
    results = []
    for path in sorted((SOURCE / "agent-shadow").glob("*.record.json")):
        record = json.loads(path.read_text())
        a = record["artifacts"]
        context = c.Investigation.model_validate(a["context"])
        ranking = c.RankResult.model_validate(a["ranking"])
        invest = c.InvestResult.model_validate(a["investigation"])
        validated = c.ValidateResult.model_validate(a["validation"])
        before = p5_fuse.fuse_investigation(context, ranking, invest, validated,
                                           ranking_policy="corrected_h_eligible_first")
        changes = []
        allowed = {candidate.id for candidate in ranking.candidates}
        for hypothesis in invest.hypotheses:
            assessment = validated.hypotheses[hypothesis.id]
            other = {h.candidate for h in invest.hypotheses if h.candidate in allowed
                     and h.candidate != hypothesis.candidate}
            ep = assessment.evidence_path
            callers = p4_validate._callers_closure(hypothesis.candidate, ep.edges) if ep and ep.found else set()
            explained = other & callers
            old = assessment.chain_coherence
            assessment.chain_coherence = validators.chain_coherence(len(explained), len(other)) if explained else 0.0
            if old != assessment.chain_coherence:
                changes.append({"hypothesis": hypothesis.id, "old": old,
                                "new": assessment.chain_coherence})
        after = p5_fuse.fuse_investigation(context, ranking, invest, validated,
                                          ranking_policy="corrected_h_eligible_first")
        def truth_rank(fused):
            order = [r["candidate"].removeprefix("service:") for r in fused.service_ranking]
            return order.index(record["gt_service"]) + 1
        row = {"case": record["case"], "p2_rank": record["gt_service_rank"],
               "old_coherence_new_slot_policy_rank": truth_rank(before),
               "repaired_rank": truth_rank(after), "coherence_changes": changes,
               "winner": after.winner_candidate,
               "note": "stored P4 findings retained; only audited coherence recomputed; no new graph verification or proposal generation"}
        results.append(row)
        atomic_json(OUT / f"{record['case']}.replay.json", {
            **row, "validation": validated.model_dump(mode="json"),
            "fusion": after.model_dump(mode="json")})
    summary = {"replay_only": True, "source": str(SOURCE), "provenance": provenance(),
               "n_cases": len(results), "p2_top1": sum(r["p2_rank"] == 1 for r in results),
               "repaired_top1": sum(r["repaired_rank"] == 1 for r in results),
               "remaining_harmful_overrides": sum(r["p2_rank"] == 1 and r["repaired_rank"] != 1 for r in results),
               "cases": results}
    atomic_json(OUT / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
