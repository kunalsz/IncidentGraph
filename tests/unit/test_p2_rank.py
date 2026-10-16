import pytest

from backend.app.engine.contracts import InvestigationWindow, ScopeResult, ScopeService
from backend.app.engine.p2_rank import rank_candidates
from backend.app.graph.scorers import indicator_scores, latency_scores, random_walk, reciprocal_rank_fusion
from config import tuning


def edge(source, target):
    return {"source": source, "target": target, "type": "DEPENDS_ON", "properties": {}}


def observation(sid, node_id, kind="Span", **props):
    return {"service_id": sid, "id": node_id, "type": kind, "properties": props}


def test_latency_formulas_and_linear_percentiles():
    rows = [observation("a", "s1", duration_us=10_000),
            observation("a", "s2", duration_us=30_000),
            observation("b", "s3", duration_us=5_000)]
    scores = latency_scores(["a", "b", "c"], rows, [edge("a", "b")])
    assert scores["a"]["phi1"] == pytest.approx(29.8 / 20)
    assert scores["a"]["phi3"] == 30
    assert scores["b"]["phi3"] == 5
    assert scores["c"]["phi1"] == 0
    assert scores["c"]["phi3"] == 0


def test_latency_deduplicates_spans_and_clamps_negative_impact():
    row = observation("a", "s", duration_us=10_000)
    scores = latency_scores(["a", "b"], [row, row, observation("b", "b1", duration_us=20_000)],
                            [edge("a", "b")])
    assert scores["a"]["latency_samples"] == 1
    assert scores["a"]["phi3"] == 0


def test_walk_moves_from_symptom_to_dependency_and_conserves_mass():
    mass = random_walk(["a", "b", "c"], [edge("a", "b"), edge("a", "b")], "a",
                       restart=0.15, iterations=1)
    assert mass == pytest.approx({"a": 0.15, "b": 0.85, "c": 0})
    assert sum(random_walk(["a", "b", "c"], [edge("a", "b")], "a").values()) == pytest.approx(1)
    assert random_walk(["a"], [], "a") == {"a": 1}


def test_rrf_uses_average_ties_and_does_not_score_missing_signals():
    components = {sid: {"phi1": 0, "phi3": 0, "severity": severity,
                        "rwr": 0, "deploy_proximity": 0}
                  for sid, severity in (("a", 4), ("b", 4), ("c", 0))}
    scores, ranks = reciprocal_rank_fusion(components)
    assert scores["a"] == scores["b"] == pytest.approx(1 / 61.5)
    assert ranks["a"]["severity"] == 1.5
    assert scores["c"] == 0
    assert ranks["c"] == {}


def test_gamma_aggregates_by_indicator_and_skips_zero_iqr():
    rows = [observation("a", "m1", "MetricAnomaly", metric="cpu", value=18, baseline_median=10, baseline_iqr=2),
            observation("a", "m2", "MetricAnomaly", metric="cpu", gamma=6),
            observation("a", "m3", "MetricAnomaly", metric="memory", value=18, baseline_median=10, baseline_iqr=0)]
    assert indicator_scores(rows) == [{"service_id": "a", "indicator": "cpu", "gamma": 6,
                                      "evidence": ["m1", "m2"]}]


def test_no_telemetry_does_not_invent_anomaly_scores():
    scope = ScopeResult(
        incident_id="i", anchor_service="a", window=InvestigationWindow(t0=0, t1=2),
        services=[ScopeService(id="a")],
        nodes=[{"id": "a", "type": "Service", "properties": {}}],
        edges=[], evidence_gaps=[], n_series_in_scope=0, truncated=False,
    )
    candidate = rank_candidates(scope).candidates[0]
    assert candidate.components == {"phi1": 0, "phi3": 0, "severity": 0,
                                    "rwr": 1, "deploy_proximity": 0}
    assert candidate.rule_flags == []
    assert candidate.evidence == []


def test_rule_flags_survive_top_k_and_order_is_stable(monkeypatch):
    monkeypatch.setattr(tuning, "TOP_K_CANDIDATES", 1)
    scope = ScopeResult(
        incident_id="i", anchor_service="a", window=InvestigationWindow(t0=0, t1=2),
        services=[ScopeService(id="a", sev_m=5), ScopeService(id="b", self_time_flag=True)],
        nodes=[{"id": "a", "type": "Service", "properties": {}},
               {"id": "b", "type": "Service", "properties": {}}],
        edges=[], evidence_gaps=[], n_series_in_scope=0, truncated=False,
    )
    result = rank_candidates(scope)
    assert result.candidates[0].id == "a"
    assert result.branch_ids == ["a", "b"]
    assert result.candidates[1].rule_flags == ["self_time"]
    assert result == rank_candidates(scope)
    assert result.n_candidates == 2
