"""Finish a P2 substrate run, skipping cases already recorded.

`rcaeval_runner.run` keeps a checkpoint holding every completed case result,
which for the richer post-redistribution scope is ~11 MB per case - 601 MB at 57
cases, and it is rewritten after every case, so peak usage doubles. That filled
the disk and killed the database mid-run. This resume path keeps the durable
per-case records as the single source of truth and writes only a small progress
file, then rebuilds the run summary from the records on disk.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.app.eval import rcaeval_runner as rr


def load(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--rank-policy", default=rr.BENCHMARK_RANK_POLICY)
    args = parser.parse_args()
    selected = load(args.data / "_subset.json")["selected"]
    records, failed = [], []
    for case in selected:
        path = args.report / f"{case}.record.json"
        if path.is_file():
            records.append(load(path))
            continue
        started = time.monotonic()
        row = rr.run_case(args.data / case, args.report, rank_policy=args.rank_policy)
        records.append(row)
        if row["status"] != "ok":
            failed.append({"case": case, "stage": row.get("failed_stage"), "code": row.get("error_code")})
        print(f"{case}: {row['status']} rank={row.get('gt_service_rank')} "
              f"({time.monotonic() - started:.0f}s)", flush=True)
        # Small progress file: names and statuses only, never full records.
        rr.atomic_json(args.report / "_progress.json", {
            "completed": len(records), "total": len(selected),
            "failed": failed, "last": case})
    summary = {"schema_version": 2, "protocol": rr.PROTOCOL, "mode": "p2_offline",
               "rank_policy": args.rank_policy, "n_cases": len(records),
               "n_failed": len(failed), "failed": failed,
               "status": "failed" if failed else "ok",
               "provenance": rr.provenance(),
               "metrics": rr.evaluate(records), "costs": rr.aggregate_resources(records)}
    rr.atomic_json(args.report / "_summary.json", summary)
    print(json.dumps({"n_cases": len(records), "n_failed": len(failed),
                      "ac@1": summary["metrics"]["overall"]["service"]["ac@1"]}, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
