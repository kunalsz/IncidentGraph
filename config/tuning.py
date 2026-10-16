"""Every tunable constant in one place — the single source of truth.

Provenance for each value is cited (vault note / paper / repo). CORAL's
"no hyperparameters published" flaw is not repeated here: any constant the
engine uses is defined in this file and echoed into report output.

See IncidentGraph.md §4.4 ("Chance/normalization constants" row) and §4.3
(budget object).
"""

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# P0 — TRIGGER & RESOLVE
# ---------------------------------------------------------------------------

# Investigation window padding around the reported/alert timestamp: [t1-Δτ, t1+Δτ].
# Provenance: master plan §4.2 time discipline / §4.3 P0. GALA's unspecified
# Δτ is documented here as exactly ±15 minutes.
WINDOW_PAD_MS = 15 * 60 * 1000

# Entity linking (P0): alias table + fuzzy match against Service|Instance
# names; unresolved anchors -> clarify/abstain, never guess (§4.3 P0).
LINK_FUZZY_CUTOFF = 0.75          # difflib similarity ratio threshold
LINK_EXACT_SCORE = 1.0
LINK_CONTAINS_SCORE = 0.9
# Pods/instances: single canonical matcher regex (§4.7 item 3; RCLAgent-V2
# shipped three divergent matchers — we ship exactly one).
POD_PATTERN = r"^(.+?)-[a-z0-9]{5,}-[a-z0-9]{4,}$"


# ---------------------------------------------------------------------------
# P1 — SCOPE
# ---------------------------------------------------------------------------

SCOPE_MAX_NODES = 40              # hard cap on scope subgraph (§4.3 P1)
SCOPE_MAX_SERIES = 12             # max metric series in scope
NSIGMA_N = 3                      # n-sigma deviation threshold (RCLAgent-V2)
NSIGMA_DELTA_MS = 60_000          # ±60s spike window for metric checks
SELF_TIME_THRESHOLD_MS = 500      # parent - Σchildren > 500ms => network suspicion
                                 # (RCLAgent-V2 dead rule, resurrected; §4.4)


# ---------------------------------------------------------------------------
# P2 — RANK INIT (deterministic oracle)
# ---------------------------------------------------------------------------

RRF_EPSILON = 60                  # RRF constant (GALA)
RWR_PHI = 0.15                    # restart probability for personalized PageRank
RWR_ITERATIONS = 20               # bounded iterations (scope <= 40 nodes)
TOP_K_CANDIDATES = 8              # candidates entering P3 as branches (§4.3 P2)
EXPANSION_BRANCHES = 6            # k = 6 (GALA Algorithm 1)
EXPANSION_BATCH_SIZE = 2          # b = 2 (GALA: random batches beat score-ordered)


# ---------------------------------------------------------------------------
# P3 — INVESTIGATE
# ---------------------------------------------------------------------------

CONFIDENCE_GATE_THETA = 0.6       # θ: expand only if ĉ < θ (GALA)
MAX_EXPANSIONS_PER_BRANCH = 6     # ≤6 expansions per branch (GALA)
MIN_HYPOTHESES_PER_EXPANSION = 5  # ≥5 hypotheses per expansion call (LATS)
MAX_AGENT_PARALLEL = 8            # concurrent branch pool cap (RCLAgent cap 32, scaled)
BUNDLE_TOKEN_CAP = 8_000          # diagnostic bundle ~8K tokens (L3)
LOG_WINDOW_MS = 60_000            # logs ±60s (RCLAgent-V2)
METRIC_SPIKE_WINDOW_MS = 600_000  # metrics ±600s spike window (RCLAgent-V2)
METRIC_BASELINE_WINDOW_MS = 1_200_000  # metrics ±1200s baseline (RCLAgent-V2)


# ---------------------------------------------------------------------------
# P4 — VALIDATE
# ---------------------------------------------------------------------------

ONSET_DELTA_MS = 5 * 60 * 1000    # evidence ts within ±5min of alert (§4.3 P4)
NO_EVIDENCE_CONFIDENCE_CAP = 0.4  # candidate with no error logs & normal metrics
                                 # gets min(ĉ, 0.4) (§4.3 P4 contradiction ledger)


# ---------------------------------------------------------------------------
# P5 — FUSE & RANK
# ---------------------------------------------------------------------------

# H(h) = 0.35·D + 0.25·R + 0.20·CC + 0.20·V (§4.3 P5)
FUSE_WEIGHTS = {"deterministic": 0.35, "reflection": 0.25,
                "chain_coherence": 0.20, "validator": 0.20}
ALPHA_REFLECTION = 0.5            # R = 0.5·r + 0.5·sc (LATS paper value; lats-rca
                                 # code used 0.7 — we default to the paper's and
                                 # re-rank offline at other α for free)
MAX_EVIDENCE_ITEMS = 10           # lats-rca evidence cap
ESCALATE_CONF_BELOW = 0.7         # escalation gate (LATS)
ESCALATE_COMPLETENESS_BELOW = 0.6
CONF_CAP_CROSS_MODAL = 0.95       # L7 confidence typing caps
CONF_CAP_SINGLE_MODAL = 0.7
CONF_CAP_INFERENCE_ONLY = 0.4
RWR_TOP3_SHARE_SINGLE = 0.7       # top-3 RWR mass share => single-cause presentation


# ---------------------------------------------------------------------------
# Budget (L11 — hard input, not aspiration)
# ---------------------------------------------------------------------------

@dataclass
class InvestigationBudget:
    max_llm_calls: int = 75
    max_tokens: int = 200_000
    max_wall_s: int = 600
    max_nodes_visited: int = 60
    max_expansions_per_branch: int = MAX_EXPANSIONS_PER_BRANCH
    max_hypotheses: int = 30

    def as_dict(self) -> dict:
        return {
            "max_llm_calls": self.max_llm_calls,
            "max_tokens": self.max_tokens,
            "max_wall_s": self.max_wall_s,
            "max_nodes_visited": self.max_nodes_visited,
            "max_expansions_per_branch": self.max_expansions_per_branch,
            "max_hypotheses": self.max_hypotheses,
        }


DEFAULT_BUDGET = InvestigationBudget()
