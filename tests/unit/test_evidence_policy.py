"""Finite evidence-strength rules, independent of model confidence metadata."""
import pytest
from backend.app.engine import policy
from backend.app.engine.contracts import HypothesisValidation, CheckResult, ValidateResult
from backend.app.engine.p5_fuse import fuse_investigation
from tests.unit.test_p5_fuse import INV, REF, _ranking, _invest, _hypothesis


def assessment(**updates):
    args = dict(hypothesis_id="h", candidate="service:s0", verdict="pass",
        evidence_scope=INV.session_graph, evidence_roles={"local_anomaly": ["m", "l"]},
        modalities=["metrics", "logs"], source_fingerprints={"m": "fp", "l": "fp"},
        qualified_timing=True, plausible_path=True, scope_ref=REF,
        proposal_fingerprint=policy.proposal_fingerprint(_hypothesis("h", "service:s0")))
    args.update(updates)
    return HypothesisValidation(**args)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), True, "5"])
def test_nonfinite_or_non_numeric_cannot_be_decisive(value):
    assert policy.modality("MetricAnomaly", {"z": value, "gamma": 8}) is None


@pytest.mark.parametrize("status,expected", [(0, False), (1, False), (2, True),
    ("ok", False), ("error", True), ("ERROR", True), ("unset", False), (True, False)])
def test_span_status_normalization(status, expected):
    assert bool(policy.error_status(status)) is expected


@pytest.mark.parametrize("updates,score,eligible", [({}, .85, True),
    ({"modalities": ["metrics"]}, .7, True),
    ({"modalities": []}, .4, False),
    ({"qualified_timing": False}, .4, False),
    ({"verdict": "partial"}, .7, True),
    ({"plausible_path": False}, .85, False),
    ({"verdict": "fail"}, 0, False),
    ({"confidence_cap": .4}, .4, False),
    ({"checks": [CheckResult(name="veto", verdict="fail")]}, 0, False)])
def test_finite_policy_table(updates, score, eligible):
    result = policy.evidence_strength(assessment(**updates))
    assert result[0] == score and result[2] is eligible


@pytest.mark.parametrize("confidence,rule", [(1, "cross_modal_consensus"), (.01, "inference_only"), (.9, "unknown")])
def test_model_assertions_do_not_change_one_modality_class(confidence, rule):
    a = assessment(modalities=["metrics"])
    h = _hypothesis("h", "service:s0", confidence=confidence, rule=rule)
    result = fuse_investigation(INV, _ranking([1]), _invest(h),
        ValidateResult(incident_id="i", scope_ref=REF, candidates=[], hypotheses={"h": a}))
    assert result.fused[0].confidence == .7
    assert result.fused[0].confidence_rule == "single_modality"


@pytest.mark.parametrize("updates", [{"hypothesis_id": "other"}, {"candidate": "service:s1"},
    {"evidence_scope": "other"}, {"source_fingerprints": {"not-cited": "fp"}}])
def test_exact_assessment_identity_required(updates):
    a = assessment(**updates)
    result = fuse_investigation(INV, _ranking([1]), _invest(_hypothesis("h", "service:s0")),
        ValidateResult(incident_id="i", scope_ref=REF, candidates=[], hypotheses={"h": a}))
    assert result.winner is None


def test_candidate_summary_never_substitutes_for_hypothesis_assessment():
    a = assessment()
    result = fuse_investigation(INV, _ranking([1]), _invest(_hypothesis("h", "service:s0")),
        ValidateResult(incident_id="i", scope_ref=REF, candidates=[a]))
    assert result.winner is None


def test_trace_derived_metric_is_not_independent_metric_modality():
    assert policy.modality("MetricAnomaly", {"z": 5, "gamma": 8, "modality": "trace"}) == "traces"


@pytest.mark.parametrize('reverse', [False, True])
def test_live_owner_conflict_is_independent_of_row_order(reverse):
    from types import SimpleNamespace
    from unittest.mock import Mock
    h = _hypothesis('h', 'service:s0').model_copy(update={'evidence': ['m']})
    scope = SimpleNamespace(services=[SimpleNamespace(id='service:s0')], nodes=[{'id': 'm'}],
        edges=[{'source': 'm', 'target': 'service:s0', 'type': 'OBSERVED_ON'}])
    wrapper = {'kind': 'MetricAnomaly', 'ts': INV.window.t0}
    props = {'id': 'm', 'ts': INV.window.t0, 'z': 5, 'gamma': 7}
    rows = [[wrapper, {'polarity': 'support', 'weight': 1}, props, ['MetricAnomaly'], owner]
            for owner in ['service:s0', 'outside']]
    graph = Mock()
    graph.query.return_value.result_set = list(reversed(rows)) if reverse else rows
    assert not policy.owned_evidence(graph, h, INV, scope)


def test_structured_metric_exact_identity_and_direction_required():
    from backend.app.engine.contracts import StructuredClaim
    h = _hypothesis("h", "service:s0").model_copy(update={"claim_kind": "metric_excursion",
        "claim": StructuredClaim(metric="cpu", direction="up")})
    records = {"m": {"role": "local_anomaly", "kind": "MetricAnomaly",
        "props": {"metric": "cpu", "z": 6, "gamma": 7, "direction": "up", "ts": INV.window.t0}}}
    inv = INV.model_copy(update={"onset_ts": INV.window.t0})
    assert policy.claim_applicability(None, h, inv, None, records, {"edges": []})["verdict"] == "pass"
    records["m"]["props"]["metric"] = "latency"
    assert policy.claim_applicability(None, h, inv, None, records, {"edges": []})["verdict"] == "partial"
    records["m"]["props"].update(metric="cpu", direction="down")
    assert policy.claim_applicability(None, h, inv, None, records, {"edges": []})["verdict"] == "fail"


def test_structured_claim_cannot_use_generic_abnormality_as_eligibility():
    a = assessment(claim_type="metric_excursion")
    assert not policy.evidence_strength(a)[2]
    a.checks = [CheckResult(name="claim_applicability", verdict="partial")]
    assert not policy.evidence_strength(a)[2]
    a.checks = [CheckResult(name="claim_applicability", verdict="pass")]
    assert policy.evidence_strength(a)[2]


def test_parent_child_error_association_not_causal_inference():
    from backend.app.engine.contracts import StructuredClaim
    h = _hypothesis("h", "service:s0").model_copy(update={"claim_kind": "dependency_propagation",
        "claim": StructuredClaim(dependency_source="caller", dependency_target="service:s0")})
    edge = {"source": "caller", "target": "service:s0"}
    records = {"p": {"kind": "Span", "positive": True, "service": "caller",
        "props": {"status": 2, "trace_id": "t", "span_id": "p", "ts": 1, "duration_us": 10000}},
        "c": {"kind": "Span", "positive": True, "service": "service:s0",
        "props": {"status": 2, "trace_id": "t", "span_id": "c", "parent_id": "p", "ts": 2, "duration_us": 1000}}}
    inv = INV.model_copy(update={"onset_ts": 1})
    result = policy.claim_applicability(None, h, inv, None, records, {"edges": [edge]})
    assert result["verdict"] == "pass" and "not established" in result["detail"]
    records["p"]["props"]["status"] = 1
    assert not policy.propagation_support(records, edge)
    assert policy.claim_applicability(None, h, INV, None, records, {"edges": [edge]})["verdict"] == "partial"


def test_live_cited_change_association_requires_target_timestamp_and_membership():
    from types import SimpleNamespace
    from unittest.mock import Mock
    from backend.app.engine.contracts import StructuredClaim
    h = _hypothesis("h", "service:s0").model_copy(update={"claim_kind": "change_association",
        "evidence": ["change"], "claim": StructuredClaim(change_id="change")})
    inv = INV.model_copy(update={"onset_ts": INV.window.t1})
    scope = SimpleNamespace(nodes=[{"id": "change"}], edges=[{"source": "change", "target": "service:s0", "type": "CHANGED"}])
    ts = INV.window.t0
    wrapper = SimpleNamespace(properties={"ts": ts, "kind": "ConfigChange"}, labels=["Evidence"])
    source = SimpleNamespace(properties={"id": "change", "ts": ts}, labels=["ConfigChange"])
    row = [wrapper, {"polarity": "support", "weight": 1}, source, ["ConfigChange"], "service:s0"]
    graph = Mock()
    graph.query.return_value.result_set = [row]
    records = {}
    result = policy.claim_applicability(graph, h, inv, scope, records, {"edges": []})
    assert result["verdict"] == "pass" and "not causation" in result["detail"]
    assert records["change"]["role"] == "change_association"
    args, kwargs = graph.query.call_args
    assert "RETURN e, properties(r)" not in args[0]
    assert "toJSON(c.metric)" in args[0]
    assert kwargs["timeout"] == 2000 and kwargs["params"]["read_limit"] == 3
    assert kwargs["params"]["field_chars"] == 4096
    row[-1] = "other"
    assert policy.claim_applicability(graph, h, inv, scope, {}, {"edges": []})["verdict"] == "partial"
    row[-1] = "service:s0"
    source.properties["ts"] = inv.onset_ts + 1
    assert policy.claim_applicability(graph, h, inv, scope, {}, {"edges": []})["verdict"] == "partial"
