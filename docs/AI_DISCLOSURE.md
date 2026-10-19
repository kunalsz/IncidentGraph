# AI coding-assistant disclosure

Required by the hackathon rules: AI assistants are allowed, their use must be
disclosed, and the team must be able to explain the graph model, the queries and
the architecture. This file states exactly where assistants were used, what the
team owns, and which claims reviewers should treat with the stated caution.

## 1. Which assistants, and how the work was divided

Command Code / ZCode agent sessions were used as a coding and analysis
assistant throughout. The division of labour was consistent:

**Set by the team (human decisions):**

- The product framing: incident investigation on a graph, FalkorDB as the
  system of record rather than a cache, deterministic checks that veto rather
  than nudge, and "missing evidence must never be reported as normal".
- The acceptance standard: every empirical claim must come from a recorded run
  with provenance; experiment arms and selection rules are written down
  **before** results are read; a policy is not promoted on cases that chose it.
- Repeated rejection of inflated or unverifiable claims, which produced several
  of the corrections.

**Produced and Performed by the team, not the assistant:** The code for the core implementation,the graph schema and query layer, P0 to P6 engine phases, agent tool surface, running the demo, provisioning FalkorDB and cloud resources, the recorded video, the submission itself, and the final go/no-go on what is claimed publicly.

**Produced by assistants, under those constraints:** most of the implementation's tests, scorers, the RCAEval adapter, the evaluation
harnesses and their reports, and this documentation set.

## 2. What reviewers can hold the team to

- **Deterministic core with hand-checked expectations.** The scorers, validators
  and contract checks carry unit tests whose expected values were derived by
  hand from the formulas (for example `phi1 = p99/p50`, the personalized-PageRank
  update, RRF with average ties), not merely snapshotted from the code.
- **Nothing is presented as measured that was not measured.** Mock-provider runs
  are labelled mock; the six-case real-model runs are labelled exploratory
  development data, not a holdout; costs are reported as unknown where provider
  pricing was unavailable instead of being estimated.
- **Provenance.** Runs record source, configuration, prompt and input hashes, and
  predeclared protocol files; failed and partial runs are kept rather than
  overwritten.
- **The team can explain the graph model and queries** in
  [GRAPH_MODEL.md](GRAPH_MODEL.md): schema, the scoped read/write Cypher, and the
  graph algorithms with their parameters.

## 3. Explicitly not AI-derived

Deployment configuration and hosting choices, the demo script, the video, and
the decision of which track(s) to enter were made by the team. No benchmark
labels, ground truth, or provider responses were fabricated or edited; where a
run's artifacts were lost, the affected tests now skip visibly rather than being
deleted or replaced with invented numbers.

## 4. Corrections made during the work (evidence of review, not a clean-room claim)

Assistants proposed several things that the team challenged and that were then
corrected or narrowed. They are listed because they show where human review
changed the result, and because a reviewer should know the final numbers already
include these fixes:

| Original claim / state | What review found | Final state |
|---|---|---|
| A flat `z = 6` sentinel for zero-variance baselines | Metric `z` and `gamma` disagreed, so the same record could never qualify as evidence | Shared baseline-scale policy; sentinels removed |
| Log/trace severity capped at `z = 6` | Every trace-driven service saturated at exactly 6.0 while metric `z` reached hundreds, so one modality could never lead | Same scale policy as metrics; measured spread uncapped |
| "Severity-only beats fusion by 42.5 points" | The baseline was misconfigured: a hardcoded `frontend` anchor made one of four votes constant (frontend rank-1 in 90/90 cases, true service 0/90) | Predeclared anchor-prior experiment; the repair recovers ~28 of the points, and the claim was restated |
| Peak-only metric sampling | Timely onset evidence was discarded, so older cases could not be corrected under the existing timing rule | Onset sample retained; truth services with a timely option went 78/90 → 90/90 |
| A walk variant whose mass did not normalise | Intermediate implementation error | Fixed; re-running reproduced identical accuracy, confirming the fix changed scale only |
| Conservative P5 update policy | Its premise could not be verified from existing validators | Shipped as explicit, audited abstention: it reorders nothing rather than pretending to rank |
| "Anchor-neutral is the principled default for cold ranking" | A second corpus reversed the result: the anchored prior beat anchor-neutral on SockShop (43.3% vs 23.3%) though it lost badly on Boutique (24.4% vs 57.8%) | Claim retracted, benchmark default set to `severity_only`, the only policy leading on both corpora |
| Trace failures were detected by `statusCode == 2` | The column is a gRPC code space: INTERNAL (13), UNAVAILABLE (14) and DEADLINE_EXCEEDED (4) were read as success, hiding 6,481 failing spans in 19 of 90 cases | Shared `policy.error_status` covering both status spaces; 122 caller/callee failure pairs recovered, 28 accepted by the propagation validator |

## 5. Known limitations we are not hiding

- The 90 RE2-OB cases are **development data**: they informed the sampler audit,
  the policy ablation and the live runs, so numbers derived from them are not an
  independent holdout.
- The severity vectors behind the earliest ablation predate the log/trace
  scoring fix; the substrate is being regenerated rather than silently reused.
- Historical artifacts from an earlier 20-case provider run are lost (not in
  version control); the tests that asserted their aggregate numbers now skip
  with a named reason.
- The agent currently runs in **shadow** mode by default: it produces ranked
  findings but does not silently overwrite the deterministic P2 order.

## 6. Declaration

The team understands and can explain the data model, the queries in
`backend/app/graph/`, the deterministic scorers and validators, and the evidence
behind every number quoted in the README and in `docs/`. Assistant-written code
passed the same review and test gates as any other contribution, and anything
that could not be substantiated was removed, narrowed, or marked as unknown.
