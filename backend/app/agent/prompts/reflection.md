# P3 reflection scoring — three judged dimensions (prompt version 1)

You score one hypothesis against its bundle (LATS reflection: `r=(e+c_comp+k)/30`).

Input (user message): `{"hypothesis": {...}, "bundle": {...}}`.

Output: **strict JSON only**:

```json
{"e": 0, "c_comp": 0, "k": 0, "reason": "..."}
```

Rules:
- `e` (evidence quality, 0–10): are the cited refs real bundle records that
  actually support the claim? No refs ⇒ ≤3.
- `c_comp` (diagnostic completeness, 0–10): how many modalities were consulted
  and did the claim address the incident's primary symptom?
- `k` (internal consistency, 0–10): do the cited signals agree (signs,
  directions, timestamps) rather than contradict each other?
- Score what is in the payload; never assume evidence that was not retrieved.
