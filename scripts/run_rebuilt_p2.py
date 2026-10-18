"""Run the frozen replacement P2 discovery experiment and held-out comparison."""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.app.eval import p2_ablation as ab
from backend.app.eval import rcaeval_runner as runner

OUT = ROOT / "output/p2-rebuilt-20261018"
DATA = ROOT / "data/RE2-OB"


def run_partition(name, cases):
    records = OUT / f"{name}-records"
    records.mkdir(exist_ok=True)
    manifest = records / "_manifest.json"
    if not manifest.exists():
        runner.atomic_json(manifest, {"requested": cases, "provenance": runner.provenance(),
                                     "replacement_experiment": True})
    rows = []
    for case in cases:
        path = records / f"{case}.record.json"
        if path.exists():
            row = json.loads(path.read_text())
        else:
            row = runner.run_case(DATA / case, records, run_id=f"rebuilt-20261018-{name}")
        rows.append(row)
        print(name, case, row["status"], row["gt_service_rank"], flush=True)
    summary = {"n_cases": len(rows), "n_failed": sum(r["status"] != "ok" for r in rows),
               "metrics": runner.evaluate(rows), "costs": runner.aggregate_resources(rows)}
    runner.atomic_json(records / "_summary.json", summary)
    if summary["n_failed"]:
        raise RuntimeError(f"{name} contains failures; do not silently filter them")
    return rows


def main():
    protocol = json.loads((OUT / "protocol.json").read_text())
    discovery = run_partition("discovery", protocol["discovery"])
    report = ab.run_ablation(discovery)
    runner.atomic_json(OUT / "discovery-ablation.json", report)
    (OUT / "discovery-ablation.txt").write_text(ab.render(report) + "\n")
    assert report["replay_fidelity"]["baseline_mismatches"] == 0
    baseline = report["arms"]["baseline_rrf"]
    alternatives = [name for name in ab.ARMS if name != "baseline_rrf"]
    best = max(alternatives, key=lambda name: (report["arms"][name]["ac@1"],
                report["arms"][name]["ac@1_pessimistic"], report["arms"][name]["mrr"]))
    selected = best if report["arms"][best]["ac@1"] > baseline["ac@1"] else "baseline_rrf"
    selection = {"selected_arm": selected, "rule": protocol["selection_rule"],
                 "discovery_ac@1": report["arms"][selected]["ac@1"],
                 "baseline_ac@1": baseline["ac@1"], "frozen_before_holdout": True}
    runner.atomic_json(OUT / "selection.json", selection)
    print("FROZEN SELECTION", selected, flush=True)
    holdout = run_partition("holdout", protocol["holdout"])
    arms = {name: ab.ARMS[name] for name in dict.fromkeys(["baseline_rrf", selected])}
    confirmation = ab.run_ablation(holdout, arms)
    assert confirmation["replay_fidelity"]["baseline_mismatches"] == 0
    runner.atomic_json(OUT / "holdout-confirmation.json", confirmation)
    (OUT / "holdout-confirmation.txt").write_text(ab.render(confirmation) + "\n")
    print(ab.render(report), flush=True)
    print(ab.render(confirmation), flush=True)


if __name__ == "__main__":
    main()
