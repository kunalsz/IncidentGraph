"""Unit tests for the budget meter (L11)."""

import pytest

from backend.app.engine.budget import BudgetExceeded, BudgetMeter
from config.tuning import InvestigationBudget


class TestBudgetMeter:
    def test_under_budget_compliant(self):
        m = BudgetMeter()
        m.record_llm_call(tokens_in=100, tokens_out=200)
        m.visit("service:payment")
        m.new_hypothesis()
        d = m.as_dict()
        assert d["llm_calls"] == 1
        assert d["tokens_total"] == 300
        assert d["compliant"] is True

    def test_llm_call_cap(self):
        m = BudgetMeter(InvestigationBudget(max_llm_calls=2))
        m.record_llm_call()
        m.record_llm_call()
        with pytest.raises(BudgetExceeded, match="llm_calls"):
            m.record_llm_call()

    def test_token_cap(self):
        m = BudgetMeter(InvestigationBudget(max_tokens=1000))
        with pytest.raises(BudgetExceeded, match="tokens"):
            m.record_llm_call(tokens_in=600, tokens_out=500)
        assert not m.as_dict()["compliant"]

    def test_hypothesis_cap(self):
        m = BudgetMeter(InvestigationBudget(max_hypotheses=1))
        m.new_hypothesis()
        with pytest.raises(BudgetExceeded, match="hypotheses"):
            m.new_hypothesis()

    def test_expansion_cap_per_branch(self):
        m = BudgetMeter(InvestigationBudget(max_expansions_per_branch=2))
        assert m.expansion("h-1") == 1
        assert m.expansion("h-1") == 2
        with pytest.raises(BudgetExceeded, match=r"expansions\[h-1\]"):
            m.expansion("h-1")
        # Other branches are not affected
        assert m.expansion("h-2") == 1

    def test_wall_s_never_negative(self):
        assert BudgetMeter().wall_s() >= 0.0


class TestBudgetDataclass:
    def test_as_dict_roundtrip(self):
        b = InvestigationBudget().as_dict()
        assert b["max_llm_calls"] == 75
        assert b["max_tokens"] == 200_000
        assert b["max_wall_s"] == 600
