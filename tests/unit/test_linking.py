"""Unit tests for entity linking (pure scoring functions)."""

import pytest

from backend.app.engine import linking
from config import tuning


NODES = [
    {"id": "service:payment", "type": "Service", "name": "payment",
     "aliases": ["payments", "payment service", "pay", "billing"]},
    {"id": "service:checkout", "type": "Service", "name": "checkout",
     "aliases": ["checkout service", "orders", "order service"]},
    {"id": "service:cart", "type": "Service", "name": "cart",
     "aliases": ["carts", "cart service", "basket"]},
    {"id": "service:recommendation", "type": "Service", "name": "recommendation",
     "aliases": ["recs", "recommendations", "recommendation service"]},
    {"id": "instance:payment-7d9f4-x2v4b", "type": "Instance",
     "name": "payment-7d9f4-x2v4b", "aliases": [], "service_id": "service:payment"},
]


class TestScoreMatch:
    def test_exact_name(self):
        assert linking.score_match("payment", "payment") == tuning.LINK_EXACT_SCORE

    def test_exact_alias(self):
        assert linking.score_match("payments", "payment",
                                   ["payments"]) == tuning.LINK_EXACT_SCORE

    def test_contains(self):
        s = linking.score_match("pay", "payment")
        assert s == tuning.LINK_CONTAINS_SCORE

    def test_fuzzy_typo(self):
        # "paymnet" vs "payment": SequenceMatcher ratio >= 0.85 (long words)
        assert linking.score_match("paymnet", "payment") >= 0.85

    def test_short_word_typo_is_not_a_match(self):
        # "broken" vs kafka alias "broker": ratio 0.83 — a different word,
        # not a typo; must not clear the length-dependent threshold.
        assert linking.score_match("broken", "kafka", ["broker"]) == 0.0

    def test_no_match(self):
        assert linking.score_match("kubernetes api", "payment") < tuning.LINK_FUZZY_CUTOFF

    def test_case_and_punct_normalized(self):
        assert linking.score_match("Payment-Service!", "payment",
                                   ["payment service"]) == tuning.LINK_EXACT_SCORE


class TestPickBest:
    def test_picks_service_over_instance(self):
        best = linking.pick_best("payment", NODES)
        assert best["id"] == "service:payment"
        assert best["type"] == "Service"

    def test_abstains_below_cutoff(self):
        assert linking.pick_best("zzz totally unrelated", NODES) is None

    def test_tie_breaks_on_id(self):
        a = {"id": "service:a", "type": "Service", "name": "cart", "aliases": []}
        b = {"id": "service:b", "type": "Service", "name": "cart", "aliases": []}
        assert linking.pick_best("cart", [a, b])["id"] == "service:a"


class TestNormalizeName:
    def test_repairs_case_and_symbols(self):
        assert linking.normalize_name("Payment-Service!") == "payment service"

    def test_collapses_whitespace(self):
        assert linking.normalize_name("  cart   store ") == "cart store"
