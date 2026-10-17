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

    def test_refusals_do_not_corrupt_meter(self):
        """A refused operation must leave the meter at the cap, compliant —
        otherwise later phases (consolidate, report) would all fail."""
        m = BudgetMeter(InvestigationBudget(max_hypotheses=1, max_llm_calls=1))
        m.new_hypothesis()
        with pytest.raises(BudgetExceeded):
            m.new_hypothesis()
        assert m.hypotheses == 1  # refusal did not increment
        m.record_llm_call()  # the one allowed call
        with pytest.raises(BudgetExceeded):
            m.reserve_llm_call()  # a second call would exceed -> refused
        assert m.llm_calls == 1  # still exactly at the cap
        assert m.as_dict()["compliant"] is True

    def test_visit_cap_refuses_without_adding(self):
        m = BudgetMeter(InvestigationBudget(max_nodes_visited=1))
        m.visit("service:a")
        with pytest.raises(BudgetExceeded, match="nodes_visited"):
            m.visit("service:b")
        m.visit("service:a")  # revisits are free
        assert len(m.nodes_visited) == 1

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
