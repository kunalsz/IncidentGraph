#!/usr/bin/env python
"""Inject a fixture incident and run P0 (TRIGGER & RESOLVE) against it.

Usage: python scripts/demo_inject.py --fixture payment_failure [--run-p0]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.seed import loader  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", default="payment_failure")
    ap.add_argument("--run-p0", action="store_true")
    ap.add_argument("--run-p1", action="store_true")
    ap.add_argument("--run-p2", action="store_true")
    args = ap.parse_args()

    loader.seed_demo()
    result = loader.load_fixture_by_name(args.fixture)
    print("fixture loaded:", json.dumps(result))

    if args.run_p0 or args.run_p1 or args.run_p2:
        from backend.app.engine import p0_resolve
        fixture = json.loads(
            (loader.FIXTURES_DIR / f"{args.fixture}.json").read_text())
        symptom = fixture["incident"]["symptom"]
        alert_id = fixture["incident"].get("anchor_alert") or next(
            (write["props"]["id"] for write in fixture["graph_writes"]
             if write["node"] == "Alert"), None)
        inv = p0_resolve.trigger_investigation(
            symptom, alert_id=alert_id, trigger_ts=fixture["incident"]["start_ts"])
        print("P0 result:")
        print(json.dumps(json.loads(inv.model_dump_json()), indent=2))
        if args.run_p1 or args.run_p2:
            from backend.app.engine.p1_scope import scope_investigation
            scope = scope_investigation(inv)
            print("P1 result:")
            print(scope.model_dump_json(indent=2))
            if args.run_p2:
                from backend.app.engine.p2_rank import rank_investigation
                print("P2 result:")
                print(rank_investigation(inv, scope).model_dump_json(indent=2))


if __name__ == "__main__":
    main()
