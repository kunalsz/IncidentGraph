"""Regenerate a P2 substrate under the current scoring, keeping per-candidate components.

Free (no provider calls): every case runs P0-P2 only. Records retain the
component scores, so every ranking policy stays replayable offline afterwards.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.app.eval import rcaeval_runner as rr


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--cases-file", type=Path)
    parser.add_argument("--rank-policy", default=rr.BENCHMARK_RANK_POLICY,
                        choices=["baseline_rrf", "severity_only", "rrf_anchor_neutral"])
    args = parser.parse_args()
    cases = None
    if args.cases_file:
        cases = [line.strip() for line in args.cases_file.read_text().splitlines() if line.strip()]
        missing = [c for c in cases if not (args.data / c).is_dir()]
        if missing:
            raise SystemExit(f"missing case dirs: {missing[:3]}")
    return rr.run(args.data, cases=cases, report_dir=args.report, rank_policy=args.rank_policy)


if __name__ == "__main__":
    raise SystemExit(main())
