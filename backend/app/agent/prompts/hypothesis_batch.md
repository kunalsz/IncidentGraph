# P3 hypothesis batching — breadth before depth (prompt version 1)

You receive one node's self-verify verdict plus its bundle. Produce candidate
root-cause hypotheses for the incident, **at least 5** (LATS: batching breadth
beats evidence depth at equal spend; ≥5 per call).

Input (user message): `{"service_id": "...", "self_verify": {...}, "bundle": {...}}`.

Output: **strict JSON only**:

```json
{"hypotheses": [
  {"text": "one-sentence causal claim", "candidate": "service:x",
   "evidence_refs": ["<record ids from the bundle>"], "initial_score": 0.0}
]}
```

Rules:
- Each hypothesis names ONE candidate (this service, a dependency from
  `adjacency`, or an explicit statement that evidence is insufficient).
- `evidence_refs` only from record ids present in the bundle; empty means the
  hypothesis is inference-only (score it low).
- `initial_score` in [0,1]: how well this bundle's evidence supports the claim.
- Do not repeat the same claim with cosmetic wording changes.
- Cover the space: direct resource faults, dependency propagation, change
  proximity, and at least one alternative that contradicts the leading claim.
