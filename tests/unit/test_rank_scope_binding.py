import pytest

from backend.app.engine.contracts import InvestigationWindow, ScopeResult, ScopeService
from backend.app.engine.p2_rank import rank_candidates


@pytest.mark.parametrize("rank_policy", ["baseline_rrf", "severity_only"])
def test_rank_result_binds_admitted_scope_and_ignores_unattached_metric(rank_policy):
    scope = ScopeResult(
        incident_id="rank-scope", anchor_service="service:a",
        window=InvestigationWindow(t0=0, t1=2000),
        services=[ScopeService(id="service:a")],
        nodes=[{"id": "service:a", "type": "Service", "properties": {}},
               {"id": "outside", "type": "MetricAnomaly",
                "properties": {"metric": "cpu", "z": 99, "ts": 1000}}],
        edges=[], evidence_gaps=[], n_series_in_scope=0, truncated=False,
    )
    result = rank_candidates(scope, rank_policy=rank_policy)
    assert result.scope_ref == scope.scope_ref
    assert result.indicators == []
    assert result.candidates[0].evidence == []
    assert result.candidates[0].components["severity"] == 0
