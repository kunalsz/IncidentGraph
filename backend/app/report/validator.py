"""P6 machine-checkable gate (§4.3 P6) — the differentiator, as CODE.

Prompt discipline is not an invariant (Aurora's lesson): `CONFIRMED` is
refused unless every dependency association in the evidence path carries at least one
supporting evidence node AND the deterministic validators all passed. The LLM
writes the prose; this module decides the tier. A downgrade is the product
working, not a failure (L12).
"""

from __future__ import annotations

import math

from config import tuning


def gate(*, has_winner: bool, confidence: float, coherence: float,
         validator_all_pass: bool, per_link_support: bool,
         theta: float | None = None,
         validator_verdict: str | None = None) -> dict:
    """Return {"verdict", "tier", "reasons"} per the machine-checked rules."""
    theta = tuning.CONFIDENCE_GATE_THETA if theta is None else theta
    reasons: list[str] = []

    if not has_winner:
        reasons.append("no candidate produced a hypothesis above the gate")
        return {"verdict": "insufficient_evidence", "tier": "INCONCLUSIVE",
                "reasons": reasons}

    # Legacy all-pass callers remain safe; False is ambiguous and must not
    # silently mean partial. Partial eligibility requires an explicit state.
    state = validator_verdict or ("pass" if validator_all_pass else None)
    if (state not in ("pass", "partial")
            or (state == "pass" and not validator_all_pass)):
        return {"verdict": "insufficient_evidence", "tier": "INCONCLUSIVE",
                "reasons": ["deterministic validators failed or validation is missing"]}

    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) or not 0 <= value <= 1
           for value in (confidence, coherence, theta)):
        return {"verdict": "insufficient_evidence", "tier": "INCONCLUSIVE",
                "reasons": ["invalid nonfinite or out-of-range gate score"]}

    if confidence < theta:
        reasons.append(f"winner confidence {confidence:.2f} < theta {theta}")
        return {"verdict": "insufficient_evidence", "tier": "INCONCLUSIVE",
                "reasons": reasons}

    if state == "pass" and validator_all_pass and per_link_support:
        reasons.append("every dependency association carries supporting evidence and all "
                       "validators passed")
        return {"verdict": "diagnosed", "tier": "CONFIRMED", "reasons": reasons}

    if not validator_all_pass:
        reasons.append("deterministic validators did not all pass")
    if not per_link_support:
        reasons.append("at least one dependency association lacks compatible trace supporting evidence; causality is not established")
    return {"verdict": "diagnosed", "tier": "LIKELY", "reasons": reasons}
