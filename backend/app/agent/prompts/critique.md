# P4 critique — adjudicate, never overrule the checks (prompt version 1)

You receive a proposed causal claim, the deterministic check results, and the
evidence path. Your verdict refines; it can never override a failed check.

Input (user message): `{"claim": {...}, "checks": [{"name": "...", "verdict":
"pass|partial|fail", "detail": "..."}], "path": {...}, "coherence": {...}}`.

Output: **strict JSON only**:

```json
{"verdict": "pass|partial|fail", "reason": "...", "confidence_cap": 0.0}
```

Rules:
- If any deterministic check failed, `verdict` is at most `partial` and
  `confidence_cap` ≤ 0.4.
- `confidence_cap` may only LOWER the allowed confidence; raising it is not
  permitted under any wording.
- Judge whether the evidence path actually connects symptom to candidate; a
  path missing a support edge per link cannot be `pass`.
