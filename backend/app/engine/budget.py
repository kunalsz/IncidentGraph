"""Synchronized v2 ledger. max_llm_calls means outbound attempts, NOT the
historical v1 successful logical-call counter. Unknown expenditure is charged
at its reservation and remains explicitly unknown; estimates are not strict
provider-token bounds. Deadline checks are cooperative, not thread termination.
"""
from __future__ import annotations

from dataclasses import dataclass
import threading
import time
import uuid
from config.tuning import InvestigationBudget


class BudgetExceeded(Exception):
    def __init__(self, resource: str, used, cap):
        self.resource, self.used, self.cap = resource, used, cap
        super().__init__(f"budget exceeded: {resource} used={used} cap={cap}")

    def __reduce__(self):
        return type(self), (self.resource, self.used, self.cap)


@dataclass(frozen=True)
class AttemptLease:
    id: str
    logical_id: str
    phase: str
    commitment: int


class BudgetMeter:
    def __init__(self, budget: InvestigationBudget | None = None, *, clock=time.monotonic):
        from dataclasses import replace
        self.budget = replace(budget) if budget is not None else InvestigationBudget()
        self._clock = clock
        self._start = clock()
        self._lock = threading.RLock()
        self.llm_calls = self.tokens_in = self.tokens_out = 0
        self.logical_requests = self.successful_responses = self.cache_hits = 0
        self.retries = self.rejected_admissions = self.unknown_usage = 0
        self.tokens_charged = 0
        self.nodes_visited = set()
        self.hypotheses = 0
        self.expansions_per_branch = {}
        self._leases = {}
        self._logical_ids = set()
        self._slots = {}
        self.violations = []
        self.strict_token_enforcement = False

    @property
    def tokens(self):
        return self.tokens_in + self.tokens_out

    def wall_s(self):
        return max(0.0, self._clock() - self._start)

    def remaining_seconds(self, phase="work"):
        reserve = min(self.budget.max_wall_s, self.budget.finalization_headroom_s)
        return max(0.0, self.budget.max_wall_s - self.wall_s()
                   - (0 if phase in ("report", "finalize") else reserve))

    def _refuse(self, resource, used, cap):
        self.rejected_admissions += 1
        raise BudgetExceeded(resource, used, cap)

    def check(self, phase="work"):
        with self._lock:
            if self.violations:
                raise BudgetExceeded("enforcement_violation", 1, 0)
            if self.tokens_charged > self.budget.max_tokens:
                raise BudgetExceeded("tokens", self.tokens_charged, self.budget.max_tokens)
            if self.remaining_seconds(phase) <= 0:
                raise BudgetExceeded("wall_s", self.wall_s(), self.budget.max_wall_s)

    def begin_logical(self):
        with self._lock:
            self.logical_requests += 1
            return uuid.uuid4().hex

    def record_cache_hit(self):
        with self._lock:
            self.cache_hits += 1

    def reserve_attempt(self, logical_id, phase, input_bound, output_limit, *, strict=False):
        if any(type(v) is not int or v < 0 for v in (input_bound, output_limit)):
            raise ValueError("token commitments must be nonnegative integers")
        with self._lock:
            self.check(phase)
            if self.budget.strict_tokens and not strict:
                raise ValueError("strict token enforcement unavailable for this tokenizer/provider")
            commitment = input_bound + output_limit
            if self.llm_calls >= self.budget.max_llm_calls:
                self._refuse("llm_calls", self.llm_calls + 1, self.budget.max_llm_calls)
            if len(self._logical_ids) >= self.budget.max_llm_calls and logical_id not in self._logical_ids:
                self._refuse("logical_attempt_ids", len(self._logical_ids) + 1, self.budget.max_llm_calls)
            committed = sum(lease.commitment for lease in self._leases.values())
            if self.tokens_charged + committed + commitment > self.budget.max_tokens:
                self._refuse("tokens", self.tokens_charged + committed + commitment,
                             self.budget.max_tokens)
            lease = AttemptLease(uuid.uuid4().hex, logical_id, phase, commitment)
            self.llm_calls += 1
            if logical_id in self._logical_ids:
                self.retries += 1
            self._logical_ids.add(logical_id)
            self._leases[lease.id] = lease
            self.strict_token_enforcement = strict and (self.llm_calls == 1 or self.strict_token_enforcement)
            return lease

    def settle_attempt(self, lease, outcome, usage=None, usage_status="unknown"):
        with self._lock:
            if self._leases.get(lease.id) != lease:
                raise ValueError("attempt lease already settled or not owned by this meter")
            del self._leases[lease.id]
            if outcome == "success":
                self.successful_responses += 1
            known = (usage_status == "known" and isinstance(usage, tuple)
                     and len(usage) == 2 and all(type(v) is int and v >= 0 for v in usage))
            if known:
                self.tokens_in += usage[0]
                self.tokens_out += usage[1]
                charged = sum(usage)
                if charged > lease.commitment:
                    self.violations.append("provider_usage_exceeded_reservation")
            else:
                self.unknown_usage += 1
                charged = lease.commitment
            self.tokens_charged += charged
            if self.tokens_charged > self.budget.max_tokens:
                self.violations.append("token_cap_exceeded")

    def reserve_llm_call(self):
        """Legacy preflight only; callers must use reserve_attempt for dispatch."""
        with self._lock:
            self.check()
            if self.llm_calls >= self.budget.max_llm_calls:
                self._refuse("llm_calls", self.llm_calls + 1, self.budget.max_llm_calls)

    def record_llm_call(self, tokens_in=0, tokens_out=0):
        """Compatibility settlement for already-performed calls (not admission)."""
        with self._lock:
            self.reserve_llm_call()
            logical = self.begin_logical()
            self.llm_calls += 1
            self.successful_responses += 1
            self.tokens_in += tokens_in
            self.tokens_out += tokens_out
            self.tokens_charged += tokens_in + tokens_out
            self.check()

    def visit(self, node_id):
        with self._lock:
            self.check()
            if node_id not in self.nodes_visited and len(self.nodes_visited) >= self.budget.max_nodes_visited:
                self._refuse("nodes_visited", len(self.nodes_visited) + 1, self.budget.max_nodes_visited)
            self.nodes_visited.add(node_id)

    def allocate_hypothesis_slots(self, branch_ids, limit=None):
        """Round-robin fair, deterministic allocation before any expensive call."""
        with self._lock:
            from config.tuning import MAX_BUDGET_BRANCHES
            if (type(branch_ids) not in (list, tuple) or len(branch_ids) > MAX_BUDGET_BRANCHES
                    or any(type(branch) is not str or len(branch) > 512 for branch in branch_ids)):
                raise ValueError("branch IDs must be a bounded sequence of short strings")
            if limit is not None and (type(limit) is not int or limit < 0
                                      or limit > self.budget.max_hypotheses):
                raise ValueError("invalid hypothesis slot limit")
            ids = list(dict.fromkeys(branch_ids))
            if len(set(self._slots).union(ids)) > MAX_BUDGET_BRANCHES:
                raise ValueError("too many active hypothesis branches")
            remaining = self.budget.max_hypotheses - self.hypotheses - sum(self._slots.values())
            remaining = min(remaining, limit) if limit is not None else remaining
            # O(branches), never O(user budget). Exact round-robin allocation.
            quotient, remainder = divmod(max(0, remaining), len(ids)) if ids else (0, 0)
            plan = {branch: quotient + int(index < remainder)
                    for index, branch in enumerate(ids)}
            for branch, count in plan.items():
                self._slots[branch] = self._slots.get(branch, 0) + count
            return plan

    def release_hypothesis_slots(self, branch):
        with self._lock:
            self._slots.pop(branch, None)

    def new_hypothesis(self, branch=None):
        with self._lock:
            if branch is not None:
                if not self._slots.get(branch):
                    self._refuse("hypothesis_slots", 1, 0)
                self._slots[branch] -= 1
            elif self.hypotheses + sum(self._slots.values()) >= self.budget.max_hypotheses:
                self._refuse("hypotheses", self.hypotheses + 1, self.budget.max_hypotheses)
            self.hypotheses += 1

    def expansion(self, branch_id):
        with self._lock:
            self.check()
            from config.tuning import MAX_BUDGET_BRANCHES
            if branch_id not in self.expansions_per_branch and len(self.expansions_per_branch) >= MAX_BUDGET_BRANCHES:
                self._refuse("expansion_branches", MAX_BUDGET_BRANCHES + 1, MAX_BUDGET_BRANCHES)
            count = self.expansions_per_branch.get(branch_id, 0) + 1
            if count > self.budget.max_expansions_per_branch:
                self._refuse(f"expansions[{branch_id}]", count, self.budget.max_expansions_per_branch)
            self.expansions_per_branch[branch_id] = count
            return count

    def as_dict(self):
        with self._lock:
            return {"semantics_version": 2, "llm_calls": self.llm_calls,
                    "provider_attempts": self.llm_calls, "logical_requests": self.logical_requests,
                    "successful_responses": self.successful_responses, "retries": self.retries,
                    "cache_hits": self.cache_hits, "rejected_admissions": self.rejected_admissions,
                    "tokens_in": self.tokens_in, "tokens_out": self.tokens_out,
                    "tokens_total": self.tokens, "tokens_charged": self.tokens_charged,
                    "unknown_usage_attempts": self.unknown_usage,
                    "usage_complete": self.unknown_usage == 0 and not self._leases,
                    "tokens_committed": sum(x.commitment for x in self._leases.values()),
                    "in_flight_attempts": len(self._leases),
                    "strict_token_enforcement": self.strict_token_enforcement,
                    "deadline_enforcement": "cooperative; sync HTTP inactivity timeouts, not hard total",
                    "enforcement_violations": list(self.violations),
                    "wall_s": round(self.wall_s(), 3), "nodes_visited": len(self.nodes_visited),
                    "hypotheses": self.hypotheses, "hypothesis_slots_reserved": sum(self._slots.values()),
                    "expansions_per_branch": dict(self.expansions_per_branch),
                    "budget": self.budget.as_dict(), "compliant": self._compliant()}

    snapshot = as_dict

    def abandon_inflight(self):
        """Only after the owned worker is killed/reaped: charge unknown attempts."""
        with self._lock:
            for lease in list(self._leases.values()):
                self.settle_attempt(lease, "cancelled", usage_status="unknown")
            self._slots.clear()

    def __getstate__(self):
        # Exceptions travel over the private worker socket, meters never do.
        raise TypeError("BudgetMeter is parent-owned and cannot be serialized")

    def _compliant(self):
        return (not self.violations and self.llm_calls <= self.budget.max_llm_calls
                and self.tokens_charged <= self.budget.max_tokens
                and self.wall_s() <= self.budget.max_wall_s)
