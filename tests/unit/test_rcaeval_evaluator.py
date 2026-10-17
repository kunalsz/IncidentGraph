"""Unit tests for the RCAEval evaluator port (hand-computed expectations).

Ported semantics: RCAEval/benchmark/evaluation.py + main.py rank parsing.
Every expected value here is computed by hand, not by the code under test.
"""

import pytest

from backend.app.eval import evaluator as ev


class TestNode:
    def test_value_equality_and_hash(self):
        assert ev.Node("checkoutservice", "cpu") == ev.Node("checkoutservice", "cpu")
        assert ev.Node("checkoutservice", "cpu") != ev.Node("checkoutservice", "mem")
        assert hash(ev.Node("a", "b")) == hash(ev.Node("a", "b"))

    def test_asdict(self):
        assert ev.Node("x", "y").asdict() == {"entity": "x", "metric": "y"}


class TestRankParsing:
    def test_service_entities_strip_db_and_dedup(self):
        ranks = ["checkoutservice_cpu", "checkoutservice_mem", "carts-db_cpu",
                 "checkoutservice_cpu", "frontend"]
        # first token, -db stripped, dedup preserving first occurrence
        assert ev.service_rank_entities(ranks) == ["checkoutservice", "carts", "frontend"]

    def test_fine_ranks_two_tokens(self):
        assert ev.fine_ranks(["checkoutservice_cpu", "frontend", "a_b_c"]) == [
            ev.Node("checkoutservice", "cpu"),
            ev.Node("a", "b_c"),
        ]

    def test_service_ranks_nodes(self):
        assert ev.service_ranks(["x_cpu", "x_mem", "y"]) == [
            ev.Node("x", "unknown"), ev.Node("y", "unknown")]

    def test_answer_metric_mapping(self):
        # main.py: delay/loss answer latency, disk answers diskio (RE2+)
        assert ev.answer_node("emailservice", "delay") == ev.Node("emailservice", "latency")
        assert ev.answer_node("emailservice", "loss") == ev.Node("emailservice", "latency")
        assert ev.answer_node("emailservice", "disk") == ev.Node("emailservice", "diskio")
        assert ev.answer_node("checkoutservice", "cpu") == ev.Node("checkoutservice", "cpu")


class TestEvaluator:
    def _case(self, ranking, answer_service, answer_metric="unknown"):
        return ev.service_ranks(ranking), ev.Node(answer_service, "unknown")

    def test_accuracy_and_average_single_case(self):
        evaluator = ev.Evaluator(max_k=5)
        ranks = ev.service_ranks(["a_x", "b_x", "c_x", "d_x", "e_x"])
        evaluator.add_case(ranks=ranks, answer=ev.Node("b", "unknown"), n_candidates=5)
        assert evaluator.accuracy_service(1) == 0.0
        assert evaluator.accuracy_service(2) == 1.0
        assert evaluator.accuracy_service(3) == 1.0
        # Avg@3 = (0 + 1 + 1) / 3
        assert evaluator.average_service(3) == pytest.approx(2 / 3)

    def test_two_cases(self):
        evaluator = ev.Evaluator(max_k=5)
        evaluator.add_case(ranks=ev.service_ranks(["a_x", "b_x"]),
                           answer=ev.Node("a", "unknown"), n_candidates=2)
        evaluator.add_case(ranks=ev.service_ranks(["a_x", "b_x"]),
                           answer=ev.Node("b", "unknown"), n_candidates=2)
        assert evaluator.accuracy_service(1) == 0.5
        assert evaluator.accuracy_service(2) == 1.0
        # Avg@2 = (0.5 + 1.0) / 2
        assert evaluator.average_service(2) == pytest.approx(0.75)

    def test_chance_and_lift(self):
        evaluator = ev.Evaluator(max_k=5)
        # chance AC@k for n=10 is min(k,10)/10
        evaluator.add_case(ranks=ev.service_ranks(["a_x", "b_x"]),
                           answer=ev.Node("a", "unknown"), n_candidates=10)
        assert evaluator.chance_accuracy(1) == pytest.approx(0.1)
        assert evaluator.chance_accuracy(5) == pytest.approx(0.5)
        # chance Avg@5 = (0.1+0.2+0.3+0.4+0.5)/5 = 0.3
        assert evaluator.chance_average(5) == pytest.approx(0.3)
        # Avg@5 = (1+1+1+1+1)/5 = 1.0
        assert evaluator.lift(5) == pytest.approx(0.7)

    def test_chance_none_without_candidate_counts(self):
        evaluator = ev.Evaluator(max_k=5)
        evaluator.add_case(ranks=ev.service_ranks(["a_x"]), answer=ev.Node("a", "unknown"))
        assert evaluator.n_missing_candidates == 1
        assert evaluator.chance_accuracy(5) is None
        assert evaluator.lift(5) is None

    def test_mrr_service(self):
        evaluator = ev.Evaluator(max_k=5)
        evaluator.add_case(ranks=ev.service_ranks(["a_x", "b_x"]),
                           answer=ev.Node("a", "unknown"), n_candidates=2)
        evaluator.add_case(ranks=ev.service_ranks(["a_x", "b_x", "c_x"]),
                           answer=ev.Node("c", "unknown"), n_candidates=3)
        # (1 + 1/3) / 2
        assert evaluator.mrr_service() == pytest.approx((1 + 1 / 3) / 2)

    def test_fine_grained_answer(self):
        evaluator = ev.Evaluator(max_k=5)
        ranks = ev.fine_ranks(["a_mem", "b_latency", "b_cpu"])
        evaluator.add_case(ranks=ranks, answer=ev.Node("b", "cpu"), n_candidates=3)
        assert evaluator.accuracy(1) == 0.0
        assert evaluator.accuracy(2) == 0.0  # b_latency != (b, cpu)
        assert evaluator.accuracy(3) == 1.0

    def test_retrieval_rerank_identity(self):
        evaluator = ev.Evaluator(max_k=5)
        evaluator.add_case(ranks=ev.service_ranks(["a_x", "b_x", "c_x", "d_x"]),
                           answer=ev.Node("c", "unknown"), n_candidates=4)
        # accuracy(3) = retrieval(3) * rerank(3, 3)
        assert evaluator.accuracy_service(3) == pytest.approx(
            evaluator.retrieval_service(3) * evaluator.rerank_service(3, 3))


class TestCandidateCounting:
    def test_counts_drop_time_and_constants(self):
        columns = ["time", "a_cpu", "a_mem", "b_cpu", "c-db_cpu"]
        rows = [
            [0, 1.0, 5.0, 2.0, 9.0],
            [1, 2.0, 5.0, 3.0, 9.0],
        ]
        # a_mem and c-db_cpu are constant (first-row reference) -> dropped
        counts = ev.count_candidates(columns, rows)
        assert counts == {"service": 2, "metric": 2}

    def test_nan_vs_value_counts_as_non_constant(self):
        # pandas semantics: NaN != first row -> column survives
        columns = ["time", "a_cpu"]
        rows = [[0, float("nan")], [1, float("nan")]]
        assert ev.count_candidates(columns, rows) == {"service": 1, "metric": 1}

    def test_db_suffix_stripped_for_service_count(self):
        columns = ["time", "x-db_cpu", "x-db_mem"]
        rows = [[0, 1.0, 1.0], [1, 2.0, 3.0]]
        counts = ev.count_candidates(columns, rows)
        assert counts["service"] == 1
        assert counts["metric"] == 2
