#!/usr/bin/env python
"""Seed the demo graph: python scripts/seed_demo.py [--force]"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.seed import loader  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="delete and recreate the demo graph")
    args = ap.parse_args()
    counts = loader.seed_demo(force=args.force)
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
