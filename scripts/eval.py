#!/usr/bin/env python
"""Evaluation harness entry point (Day 3).

Currently: verify P0 rank-only smoke (fixture top-3 gate). The full
RCAEval-adapter evaluation lands with P1/P2.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    print("eval harness: not yet wired (P1/P2 pending)")


if __name__ == "__main__":
    main()
