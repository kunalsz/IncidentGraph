"""Frozen six-case local mechanism comparison; no external model calls."""
import hashlib
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.app.eval import engine_experiments as ex
from backend.app.eval import rcaeval_runner as rr

OUT = ROOT / "output/severity-agent-local-frozen-20261018"


def main():
    if os.environ.get("FALKORDB_HOST") not in ("localhost", "127.0.0.1") or os.environ.get("FALKORDB_PORT") != "6381":
        raise ValueError("Explicit disposable localhost:6381 required")
    OUT.mkdir(parents=True, exist_ok=True)
    discovery = json.loads((ROOT / "output/p2-rebuilt-20261018/protocol.json").read_text())["discovery"]
    cases = [min(c for c in discovery if c.rsplit("_", 2)[1] == fault)
             for fault in ("cpu", "mem", "disk", "socket", "delay", "loss")]
    budget = replace(rr.BENCH_BUDGET, max_hypotheses=10, expansion_slot_fraction=.8)
    protocol = {"cases": cases, "selection": "lexicographically first discovery case per fault; no performance selection",
                "rank_policy": "severity_only", "provider": "deterministic EvidenceRuleProvider mock",
                "primary_branches": 2, "budget": budget.as_dict(),
                "fusion_policy": "corrected_h_eligible_first", "independent_holdout": False,
                "source_provenance": rr.provenance()}
    rr.atomic_json(OUT / "protocol.json", protocol)
    pairs = []
    for case in cases:
        common = dict(rank_policy="severity_only", budget=budget, primary_branches=2)
        baseline_path = OUT / "p2" / f"{case}.record.json"
        agent_path = OUT / "agent-shadow" / f"{case}.record.json"
        baseline = (json.loads(baseline_path.read_text()) if baseline_path.exists() else
                    rr.run_case(ROOT / "data/RE2-OB" / case, OUT / "p2", **common))
        agent = (json.loads(agent_path.read_text()) if agent_path.exists() else
                 rr.run_case(ROOT / "data/RE2-OB" / case, OUT / "agent-shadow", with_agent=True,
                             llm_factory=ex.deterministic_client, **common))
        agent["provider_kind"] = "mock"
        rr.atomic_json(OUT / "agent-shadow" / f"{case}.record.json", agent)
        row = {"case": case, "baseline_status": baseline["status"], "agent_status": agent["status"],
               "baseline_rank": baseline["gt_service_rank"], "baseline_latency_s": baseline["timings"]["total_s"],
               "agent_latency_s": agent["timings"]["total_s"], "resources": agent["resources"],
               "cleanup": [baseline["cleanup"], agent["cleanup"]]}
        if baseline["status"] == agent["status"] == "ok":
            assert baseline["data"] == agent["data"]
            assert baseline["baseline_order"] == agent["baseline_order"]
            assert baseline["service_order"] == agent["service_order"]
            assert baseline["artifacts"]["scope"]["services"] == agent["artifacts"]["scope"]["services"]
            fused = ex.replay_fusion(agent, ranking_policy="corrected_h_eligible_first")
            corrected = ex.arm_record(agent, "severity_agent_corrected_local_mock", fused)
            rr.atomic_json(OUT / "agent-corrected" / f"{case}.record.json", corrected)
            row.update(agent_rank=corrected["gt_service_rank"], shadow_full_order_equal=True,
                       abstained=fused.winner_candidate is None,
                       execution_diagnostics=agent["artifacts"]["investigation"].get("audit", []))
        pairs.append(row)
        print(json.dumps(row), flush=True)
    completed = [r for r in pairs if "agent_rank" in r]
    summary = {"label": "local deterministic mock mechanism test, not model performance",
               "n_requested": len(cases), "n_completed_pairs": len(completed),
               "baseline_top1": sum(r["baseline_rank"] == 1 for r in pairs),
               "agent_top1": sum(r.get("agent_rank") == 1 for r in pairs),
               "correct_corrections": sum(r["baseline_rank"] != 1 and r["agent_rank"] == 1 for r in completed),
               "harmful_overrides": sum(r["baseline_rank"] == 1 and r["agent_rank"] != 1 for r in completed),
               "abstentions": sum(r["abstained"] for r in completed), "pairs": pairs,
               "source_stable": rr.provenance()["source_sha256"] == protocol["source_provenance"]["source_sha256"],
               "real_provider_calls": 0, "real_model_comparison": "blocked: no provider/model/spend configured"}
    rr.atomic_json(OUT / "summary.json", summary)
    print(json.dumps({k:v for k,v in summary.items() if k != "pairs"}, indent=2))


if __name__ == "__main__":
    main()
