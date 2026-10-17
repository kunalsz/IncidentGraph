#!/usr/bin/env python
"""Explicit fixture injection and shared P0-P6 runner. Requires --seed-demo
before mutating the demo graph; ordinary execution never reseeds existing DBs.
"""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.app.seed import loader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", default="payment_failure")
    ap.add_argument("--seed-demo", action="store_true", help="Explicitly authorize demo graph seeding and fixture injection")
    for phase in range(7):
        ap.add_argument(f"--run-p{phase}", action="store_true")
    args = ap.parse_args()
    allowed = {path.stem for path in loader.FIXTURES_DIR.glob("*.json")}
    if args.fixture not in allowed:
        ap.error("unknown fixture")
    if args.seed_demo:
        loader.seed_demo()
        print("fixture loaded:", json.dumps(loader.load_fixture_by_name(args.fixture)))
    requested = [phase for phase in range(7) if getattr(args, f"run_p{phase}")]
    if not requested:
        return
    fixture = json.loads((loader.FIXTURES_DIR / f"{args.fixture}.json").read_text())
    alert_id = fixture["incident"].get("anchor_alert") or next(
        (write["props"]["id"] for write in fixture["graph_writes"] if write["node"] == "Alert"), None)
    from backend.app.engine.runner import run_investigation
    result = run_investigation(fixture["incident"]["symptom"], alert_id=alert_id,
        trigger_ts=fixture["incident"]["start_ts"], until_phase=max(requested))
    print(result.model_dump_json(indent=2))
    if result.report:
        print(result.report.markdown)
    if result.status != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
