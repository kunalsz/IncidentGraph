# P3 grounded hypothesis batching (prompt version 2)

You receive one current candidate's self-verify verdict and admitted evidence
bundle, `allowed_candidate_ids` (the current `service_id` only),
`max_hypotheses` (allocated slots), and `supported_claim_options`.
Produce **0..max_hypotheses** distinct proposals for this current candidate only.
There is no minimum count. Abstain with `{"hypotheses": []}` when evidence is
insufficient; never invent filler to use allocated slots.

Output **strict JSON only**, with no unknown fields:

```json
{"hypotheses": [
  {"text": "exact observed metric excursion; mechanism remains unproven",
   "candidate": "<current service_id>", "evidence_refs": ["<admitted record id>"],
   "initial_score": 0.0, "claim_kind": "metric_excursion",
   "claim": {"metric": "<exact column from a supported option>", "direction": "up"}}
]}
```

Rules:
- Prefer supported metric facts when options are present: explicitly copy an
  option's `claim_kind` and `claim`, and cite its supporting `evidence_refs`.
  Options are bounded deterministic exact-column excursions qualified near onset,
  not evidence of a causal mechanism. Never substitute a generic metric alias,
  change direction, or invent a metric option.
- `candidate` must equal the current `service_id`. Adjacency is context, not
  permission to propose another candidate.
- `evidence_refs` must be record ids from the bundle. Cite the selected metric
  option's evidence; other disclosed context may also be cited.
- Generic service abnormality is permitted with `claim_kind` set to
  `service_abnormality` and `claim` set to null (these are also the legacy defaults).
  Empty citations mean inference only, not a supported metric fact.
- A `dependency_propagation` association is permitted only when offered in
  `supported_claim_options`. Copy its exact `dependency_source` and
  `dependency_target` (the current candidate), and cite BOTH evidence refs from
  that same option. Do not construct arbitrary span pairs or combine halves of
  different options. This is a same-trace error association only: it does not
  establish causal propagation, causal direction, or a causal mechanism.
- A `change_association` must use `claim` with `change_id` only, supported by
  cited evidence. Temporal proximity alone does not establish causation. All
  associations remain subject to deterministic validation; do not mix fields
  from different claim kinds.
- Mechanism prose (for example resource exhaustion or causal propagation) is a
  hypothesis, not a verified structured fact. Do not present a supported metric
  excursion as proof that a particular mechanism caused the incident.
- `initial_score` is in [0,1], never evidence of verification.
- Do not repeat the same claim with cosmetic wording changes.
