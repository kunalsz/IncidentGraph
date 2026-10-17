"""Unit tests for P3 policy helpers (pure functions, no graph/LLM)."""

import pytest

from backend.app.engine.p3_investigate import (_stable_seed, combined_reward,
                                               expansion_batches,
                                               hypothesis_signature,
                                               reflection_reward,
                                               signature_self_consistency)


class TestReflectionReward:
    def test_hand_computed(self):
        # (8 + 7 + 9) / 30
        assert reflection_reward({"e": 8, "c_comp": 7, "k": 9}) == pytest.approx(0.8)

    def test_bounds(self):
        assert reflection_reward({"e": 10, "c_comp": 10, "k": 10}) == 1.0
        assert reflection_reward({}) == 0.0
        assert reflection_reward({"e": 50, "c_comp": 50, "k": 50}) == 1.0
        assert reflection_reward({"e": -5, "c_comp": 0, "k": 0}) == 0.0


class TestCombinedReward:
    def test_alpha_blend(self):
        reflection = {"e": 9, "c_comp": 9, "k": 9}  # r = 0.9
        assert combined_reward(reflection, sc=0.5) == pytest.approx(0.7)
        assert combined_reward(reflection, sc=1.0, alpha=0.0) == pytest.approx(1.0)
        assert combined_reward(reflection, sc=0.0, alpha=1.0) == pytest.approx(0.9)


class TestSignature:
    def test_signature_sorts_and_dedups_labels(self):
        assert hypothesis_signature(["metrics", "logs", "metrics"]) == (
            "bundle:fetch", ("logs", "metrics"))

    def test_self_consistency_majority(self):
        a = hypothesis_signature(["metrics"])
        b = hypothesis_signature(["logs"])
        scores = signature_self_consistency([a, a, b], [True, True, True])
        assert scores == pytest.approx([2 / 3, 2 / 3, 1 / 3])

    def test_no_graph_read_floor(self):
        a = hypothesis_signature(["metrics"])
        scores = signature_self_consistency([a, a], [True, False])
        assert scores[1] == 0.0  # sentinel loophole closed

    def test_empty(self):
        assert signature_self_consistency([], []) == []


class TestExpansionBatches:
    def test_batch_size_and_cap(self):
        neighbors = [f"service:s{i}" for i in range(20)]
        batches = expansion_batches(neighbors, batch_size=2, max_batches=6)
        assert len(batches) == 6
        assert all(len(batch) <= 2 for batch in batches)

    def test_seeded_determinism(self):
        neighbors = [f"service:s{i}" for i in range(10)]
        assert expansion_batches(neighbors, seed=7) == expansion_batches(neighbors, seed=7)

    def test_small_neighborhood_fully_covered(self):
        neighbors = ["service:a", "service:b", "service:c"]
        batches = expansion_batches(neighbors, batch_size=2, max_batches=6)
        covered = [node for batch in batches for node in batch]
        assert sorted(covered) == sorted(neighbors)


class TestStableSeed:
    def test_deterministic_and_varied(self):
        assert _stable_seed("service:payment") == _stable_seed("service:payment")
        assert _stable_seed("service:payment") != _stable_seed("service:cart")
