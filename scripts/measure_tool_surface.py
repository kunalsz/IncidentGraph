#!/usr/bin/env python
"""MCP tool-surface token budget check (Aurora's measured-budget idea).

Measures the *real* registered surface in `backend.app.agent.tools` — not a
planned-signature list — so the CI gate tracks exactly what ships. Tier-1
schema tokens must stay under 2K; the whole registered surface under 8K.
Exits non-zero on FAIL (CI-checkable).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from backend.app.agent.tools import TOOLS, surface_tokens  # noqa: E402

TIER1_TOKEN_BUDGET = 2000
ALL_TOKEN_BUDGET = 8000


def measure() -> dict:
    tier1 = surface_tokens((1,))
    meta = surface_tokens((0,))
    total = tier1 + meta + surface_tokens((2,))
    tier1_tools = [spec.name for spec in TOOLS.values() if spec.tier == 1]
    return {
        "tools": len(TOOLS),
        "tier1_tools": len(tier1_tools),
        "tier1_tokens": tier1,
        "meta_tokens": meta,
        "total_tokens": total,
        "tier1_budget": TIER1_TOKEN_BUDGET,
        "all_budget": ALL_TOKEN_BUDGET,
        "pass": tier1 < TIER1_TOKEN_BUDGET and total < ALL_TOKEN_BUDGET,
    }


def main() -> int:
    m = measure()
    print(f"Registered tools: {m['tools']} (Tier-1: {m['tier1_tools']})")
    print(f"Tier-1 schema tokens: {m['tier1_tokens']} (budget {m['tier1_budget']})")
    print(f"Meta schema tokens:   {m['meta_tokens']}")
    print(f"Total schema tokens:  {m['total_tokens']} (budget {m['all_budget']})")
    verdict = "PASS" if m["pass"] else "FAIL"
    print(f"Budget check: {verdict}")
    return 0 if m["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
