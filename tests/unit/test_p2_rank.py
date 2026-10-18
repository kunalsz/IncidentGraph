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


def test_explicit_policies_preserve_diagnostics_and_default(monkeypatch):
    from backend.app.engine.contracts import dump_handoff, load_handoff
    from backend.app.eval import p2_ablation as ab

    monkeypatch.setattr(tuning, "TOP_K_CANDIDATES", 1)
    scope = ScopeResult(
        incident_id="policies", anchor_service="service:frontend", window={"t0": 0, "t1": 2},
        services=[ScopeService(id="service:frontend", sev_m=1.0),
                  ScopeService(id="service:root", sev_m=40.0),
                  ScopeService(id="service:flagged", self_time_flag=True)],
        nodes=[{"id": "service:" + n, "type": "Service", "properties": {}}
               for n in ("frontend", "root", "flagged")],
        edges=[], evidence_gaps=[], n_series_in_scope=0, truncated=False)
    baseline = rank_candidates(scope)
    assert baseline == rank_candidates(scope, rank_policy="baseline_rrf")
    severity = rank_candidates(scope, rank_policy="severity_only")
    assert baseline.candidates[0].id == "service:frontend"
    assert severity.candidates[0].id == "service:root"
    assert severity.branch_ids == ["service:root", "service:flagged"]
    by_id = {c.id: c for c in baseline.candidates}
    expected = ab.fuse({c.id: c.components for c in baseline.candidates}, {"severity": 1.0})
    for c in severity.candidates:
        assert c.score == pytest.approx(expected[c.id])
        assert c.components == by_id[c.id].components
        assert c.component_ranks == by_id[c.id].component_ranks
        assert c.evidence == by_id[c.id].evidence
        assert c.rule_flags == by_id[c.id].rule_flags
    assert severity.config["rank_policy"] == "severity_only"
    assert severity.config["component_weights"] == {
        "phi1": 0.0, "phi3": 0.0, "severity": 1.0, "rwr": 0.0, "deploy_proximity": 0.0}
    assert severity.scope_ref == baseline.scope_ref == scope.scope_ref
    assert severity.indicators == baseline.indicators
    assert load_handoff(dump_handoff("rank", severity), stage="rank",
                        incident_id=scope.incident_id, scope_ref=scope.scope_ref) == severity
    with pytest.raises(ValueError, match="unknown P2 rank policy"):
        rank_candidates(scope, rank_policy="typo")


@pytest.mark.parametrize("policy", ["baseline_rrf", "severity_only"])
def test_rank_investigation_forwards_and_persists_policy(monkeypatch, policy):
    from types import SimpleNamespace
    from backend.app.engine import p2_rank
    from backend.app.engine.contracts import Investigation, load_handoff

    scope = ScopeResult(incident_id="forward", anchor_service="a", window={"t0": 0, "t1": 2},
        services=[ScopeService(id="a")],
        nodes=[{"id": "a", "type": "Service", "properties": {}}],
        edges=[], evidence_gaps=[], n_series_in_scope=0, truncated=False)
    inv = Investigation(incident_id="forward", anchor_service="a", window=scope.window,
                        session_graph="incidentgraph:session:forward", symptom="test", budget={})
    writes = []
    monkeypatch.setattr(p2_rank, "graph_for", lambda n: SimpleNamespace(
        query=lambda query, params: writes.append(params)))
    result = p2_rank.rank_investigation(inv, scope, rank_policy=policy)
    assert result.config["rank_policy"] == policy
    assert load_handoff(writes[0]["payload"], stage="rank", incident_id=inv.incident_id,
                        scope_ref=scope.scope_ref) == result


@pytest.mark.parametrize("value", [0.0, 5.0])
def test_severity_policy_ties_keep_pagerank_then_id(value):
    scope = ScopeResult(
        incident_id="ties", anchor_service="a", window={"t0": 0, "t1": 2},
        services=[ScopeService(id=s, sev_m=value) for s in ("a", "b", "c")],
        nodes=[{"id": s, "type": "Service", "properties": {"pagerank": float(s == "c")}}
               for s in ("a", "b", "c")], edges=[], evidence_gaps=[],
        n_series_in_scope=0, truncated=False)
    result = rank_candidates(scope, rank_policy="severity_only")
    assert [c.id for c in result.candidates] == ["c", "a", "b"]
    assert all(c.score == (1 / 62 if value else 0.0) for c in result.candidates)
