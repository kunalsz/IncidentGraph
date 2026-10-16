"""Budget meter (L11): calls/tokens/wall-clock caps are config, enforced,
and reported per incident. Created at P0, checked before every LLM call.
"""

import time

from config.tuning import InvestigationBudget


class BudgetExceeded(Exception):
    def __init__(self, resource: str, used: int, cap: int):
        self.resource, self.used, self.cap = resource, used, cap
        super().__init__(f"budget exceeded: {resource} used={used} cap={cap}")


class BudgetMeter:
    def __init__(self, budget: InvestigationBudget | None = None):
        self.budget = budget or InvestigationBudget()
        self.llm_calls = 0
        self.tokens_in = 0
        self.tokens_out = 0
        self.nodes_visited: set = set()
        self.hypotheses = 0
        self.expansions_per_branch: dict = {}
        self._start = time.monotonic()

    @property
    def tokens(self) -> int:
        return self.tokens_in + self.tokens_out

    def check(self) -> None:
        """Raise BudgetExceeded when any cap is breached."""
        b = self.budget
        if self.llm_calls > b.max_llm_calls:
            raise BudgetExceeded("llm_calls", self.llm_calls, b.max_llm_calls)
        if self.tokens > b.max_tokens:
            raise BudgetExceeded("tokens", self.tokens, b.max_tokens)
        if self.wall_s() > b.max_wall_s:
            raise BudgetExceeded("wall_s", int(self.wall_s()), b.max_wall_s)
        if len(self.nodes_visited) > b.max_nodes_visited:
            raise BudgetExceeded("nodes_visited", len(self.nodes_visited), b.max_nodes_visited)
        if self.hypotheses > b.max_hypotheses:
            raise BudgetExceeded("hypotheses", self.hypotheses, b.max_hypotheses)

    def record_llm_call(self, tokens_in: int = 0, tokens_out: int = 0) -> None:
        self.llm_calls += 1
        self.tokens_in += tokens_in
        self.tokens_out += tokens_out
        self.check()

    def visit(self, node_id: str) -> None:
        self.nodes_visited.add(node_id)
        self.check()

    def new_hypothesis(self) -> None:
        self.hypotheses += 1
        self.check()

    def expansion(self, branch_id: str) -> int:
        n = self.expansions_per_branch.get(branch_id, 0) + 1
        self.expansions_per_branch[branch_id] = n
        if n > self.budget.max_expansions_per_branch:
            raise BudgetExceeded(
                f"expansions[{branch_id}]", n, self.budget.max_expansions_per_branch)
        return n

    def wall_s(self) -> float:
        return time.monotonic() - self._start

    def as_dict(self) -> dict:
        return {
            "llm_calls": self.llm_calls,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "tokens_total": self.tokens,
            "wall_s": round(self.wall_s(), 2),
            "nodes_visited": len(self.nodes_visited),
            "hypotheses": self.hypotheses,
            "expansions_per_branch": dict(self.expansions_per_branch),
            "budget": self.budget.as_dict(),
            "compliant": self._compliant(),
        }

    def _compliant(self) -> bool:
        b = self.budget
        try:
            self.check()
            return True
        except BudgetExceeded as e:
            return False
