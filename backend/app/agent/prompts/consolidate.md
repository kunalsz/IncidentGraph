# P3 consolidate — leaf-to-root, frozen payloads only (prompt version 1)

You receive your node's self-verify verdict and the frozen verdicts of its
children (upstream callers' callees processed leaf-first). Produce the branch's
local root cause. You never see transcripts — only typed payloads.

Input (user message): `{"node_id": "...", "verdict": {...},
"children": [{"node_id": "...", "is_abnormal": true, "confidence": 0.0,
"hypothesis": "...", "evidence_ids": []}]}`.

Output: **strict JSON only**:

```json
{"node_id": "service:x", "local_root_cause": "service:x", "reason": "...",
 "confidence": 0.0, "confidence_rule": "inference_only",
 "evidence_ids": [], "contradictions": [], "status": "closed"}
```

Rules:
- `local_root_cause` is this node when it is abnormal with the strongest
  evidence; otherwise the strongest abnormal child; otherwise `null`
  (status `partial`), never a guess.
- A locally-persuasive node that merely forwards another's anomaly is NOT the
  root cause — propagation is not causation.
- `evidence_ids` merged from the node and the chosen child, deduplicated.
- `contradictions`: evidence ids that argue against the chosen cause.
- `confidence`: the chosen source's confidence, discounted for distance
  (child verdicts carry 0.9× weight).
