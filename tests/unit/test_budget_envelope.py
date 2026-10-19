"""The per-case token cap must not be smaller than what the call cap already allows.

Regression for a real failure: scope admission grew ~3x, bundles grew with it,
and cases began failing mid-investigation with BudgetExceeded because the token
cap (200k, authored when ~1.8k spans were admitted) could not cover the calls
the run was still permitted to make.
"""

from dataclasses import replace

import pytest

from backend.app.engine.budget import BudgetMeter
from backend.app.eval import rcaeval_runner as rr
from config import tuning


class TestEnvelopeBound:
    def test_bound_is_the_worst_case_product(self):
        assert rr.envelope_token_bound(75, bundle_token_cap=8_000,
                                       max_output_tokens=4_096, framing_tokens=1_000) == 75 * 13_096

    def test_bound_rejects_negative_inputs(self):
        with pytest.raises(ValueError):
            rr.envelope_token_bound(-1)

    def test_benchmark_budget_keeps_every_other_cap(self):
        default = rr.InvestigationBudget()
        derived = rr.benchmark_budget()
        assert replace(derived, max_tokens=default.max_tokens) == default
        assert derived.max_tokens == rr.envelope_token_bound(default.max_llm_calls)

    def test_derived_cap_covers_the_envelope_it_permits(self):
        # The anti-recurrence property: a case can always afford the calls its
        # own call cap allows, so the token cap cannot silently fail it.
        derived = rr.benchmark_budget()
        per_call = tuning.BUNDLE_TOKEN_CAP + 4_096 + 1_000
        assert derived.max_tokens >= derived.max_llm_calls * per_call

    def test_product_default_is_untouched(self):
        # only the benchmark harness derives a budget; the engine default and
        # any explicit budget a caller passes are unchanged
        assert rr.BENCH_BUDGET.max_tokens == rr.InvestigationBudget().max_tokens
        explicit = replace(rr.InvestigationBudget(), max_tokens=123)
        assert explicit.max_tokens == 123

    def test_a_case_can_admit_the_calls_the_cap_allows(self):
        # Meter-level check: admitting max_llm_calls reservations of the per-call
        # bound must not raise, which is what used to happen mid-investigation.
        budget = rr.benchmark_budget()
        meter = BudgetMeter(budget)
        logical = meter.begin_logical()
        for index in range(budget.max_llm_calls):
            lease = meter.reserve_attempt(logical, "work", 6_000, 4_000, strict=False)
            meter.settle_attempt(lease, "success", (6_000, 1_000), "known")
        assert meter.as_dict()["tokens_charged"] <= budget.max_tokens
