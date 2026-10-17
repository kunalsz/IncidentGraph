"""
Every tunable constant in one place
"""

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# P0 — TRIGGER & RESOLVE
# ---------------------------------------------------------------------------

# Investigation window padding around the reported/alert timestamp: [t1-Δτ, t1+Δτ].
WINDOW_PAD_MS = 15 * 60 * 1000

# Entity linking (P0): alias table + fuzzy match against Service|Instance
# names; unresolved anchors -> clarify/abstain, never guess.
LINK_FUZZY_CUTOFF = 0.75          # difflib similarity ratio threshold
LINK_EXACT_SCORE = 1.0
LINK_CONTAINS_SCORE = 0.9
# Pods/instances: single canonical matcher regex
POD_PATTERN = r"^(.+?)-[a-z0-9]{5,}-[a-z0-9]{4,}$"


# ---------------------------------------------------------------------------
# P1 — SCOPE
# ---------------------------------------------------------------------------

SCOPE_MAX_NODES = 40              # hard cap on scope subgraph
SCOPE_MAX_SERIES = 12             # max metric series in scope
NSIGMA_N = 3                      # n-sigma deviation threshold
NSIGMA_DELTA_MS = 60_000          # ±60s spike window for metric checks
SELF_TIME_THRESHOLD_MS = 500      # parent - Σchildren > 500ms => network suspicion


# ---------------------------------------------------------------------------
# P2 — RANK INIT (deterministic oracle)
# ---------------------------------------------------------------------------

RRF_EPSILON = 60                  # RRF constant
RWR_PHI = 0.15                    # restart probability for personalized PageRank
RWR_ITERATIONS = 20               # bounded iterations (scope <= 40 nodes)
TOP_K_CANDIDATES = 8              # candidates entering P3 as branches
EXPANSION_BRANCHES = 6            # k = 6
EXPANSION_BATCH_SIZE = 2          # b = 2 (random batches beat score-ordered)


# ---------------------------------------------------------------------------
# P3 — INVESTIGATE
# ---------------------------------------------------------------------------

CONFIDENCE_GATE_THETA = 0.6       # θ: expand only if ĉ < θ
MAX_EXPANSIONS_PER_BRANCH = 6     # ≤6 expansions per branch
MIN_HYPOTHESES_PER_EXPANSION = 5  # ≥5 hypotheses per expansion call
MAX_AGENT_PARALLEL = 8            # concurrent branch pool cap
BUNDLE_TOKEN_CAP = 8_000          # diagnostic bundle ~8K tokens 
LOG_WINDOW_MS = 60_000            # logs ±60s 
METRIC_SPIKE_WINDOW_MS = 600_000  # metrics ±600s spike window 
METRIC_BASELINE_WINDOW_MS = 1_200_000  # metrics ±1200s baseline


# ---------------------------------------------------------------------------
# P4 — VALIDATE
# ---------------------------------------------------------------------------

ONSET_DELTA_MS = 5 * 60 * 1000    # evidence ts within ±5min of alert
NO_EVIDENCE_CONFIDENCE_CAP = 0.4  # candidate with no error logs & normal metrics


# ---------------------------------------------------------------------------
# P5 — FUSE & RANK
# ---------------------------------------------------------------------------

# H(h) = 0.35·D + 0.25·R + 0.20·CC + 0.20·V
FUSE_WEIGHTS = {"deterministic": 0.35, "reflection": 0.25,
                "chain_coherence": 0.20, "validator": 0.20}
ALPHA_REFLECTION = 0.5            # R = 0.5·r + 0.5·sc 
MAX_EVIDENCE_ITEMS = 10           #evidence cap
ESCALATE_CONF_BELOW = 0.7         # escalation gate
ESCALATE_COMPLETENESS_BELOW = 0.6
CONF_CAP_CROSS_MODAL = 0.95       # confidence typing caps
CONF_CAP_SINGLE_MODAL = 0.7
CONF_CAP_INFERENCE_ONLY = 0.4
RWR_TOP3_SHARE_SINGLE = 0.7       # top-3 RWR mass share => single-cause presentation


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------

# Operational safety ceilings, not tuning defaults. Supervisor operations must
# never scale with an unrestricted user-supplied counter.
BUDGET_OPERATIONAL_CEILINGS = {
    "max_llm_calls": 10_000, "max_tokens": 100_000_000,
    "max_nodes_visited": 10_000, "max_expansions_per_branch": 1_000,
    "max_hypotheses": 10_000, "max_wall_s": 86_400,
    "finalization_headroom_s": 3_600,
}
MAX_BUDGET_BRANCHES = 1024


@dataclass(frozen=True)
class InvestigationBudget:
    max_llm_calls: int = 75
    max_tokens: int = 200_000
    max_wall_s: int = 600
    max_nodes_visited: int = 60
    max_expansions_per_branch: int = MAX_EXPANSIONS_PER_BRANCH
    max_hypotheses: int = 30

    # v2: max_llm_calls caps outbound attempts, including failures/rotations.
    finalization_headroom_s: float = 5.0
    strict_tokens: bool = False
    expansion_slot_fraction: float = 0.2

    def __post_init__(self):
        import math
        for name in ("max_llm_calls", "max_tokens", "max_nodes_visited",
                     "max_expansions_per_branch", "max_hypotheses"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in ("max_wall_s", "finalization_headroom_s"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name, ceiling in BUDGET_OPERATIONAL_CEILINGS.items():
            if getattr(self, name) > ceiling:
                raise ValueError(f"{name} exceeds operational ceiling {ceiling}")
        if type(self.strict_tokens) is not bool:
            raise ValueError("strict_tokens must be boolean")
        if (type(self.expansion_slot_fraction) not in (int, float)
                or not math.isfinite(self.expansion_slot_fraction)
                or not 0 <= self.expansion_slot_fraction <= 1):
            raise ValueError("expansion_slot_fraction must be within [0, 1]")

    def as_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


DEFAULT_BUDGET = InvestigationBudget()
