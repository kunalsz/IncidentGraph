"""Severity 0 must be distinguishable between 'not measured' and 'measured, normal'."""
import math
from types import SimpleNamespace

import pytest

from backend.app.agent import tools
from backend.app.engine.contracts import InvestigationWindow, ScopeResult, ScopeService
from backend.app.engine.p1_scope import service_statistics
from backend.app.engine.scope_policy import scored_modalities


def record(kind, node_id, sid="service:a", **props):
    return {"id": node_id, "type": kind, "service_id": sid,
            "properties": {"id": node_id, "ts": 1000, **props}}


def test_no_records_means_modality_is_omitted():
    assert scored_modalities([]) == {}
    assert scored_modalities([record("Alert", "a1")]) == {}


def test_records_without_z_are_present_but_unscored():
    records = [record("Span", "s1", duration_us=5), record("LogEvent", "l1", is_error=True)]
    assert scored_modalities(records) == {"trace": False, "log": False}


def test_one_scorable_record_marks_the_modality_scored():
    records = [record("Span", "s1"), record("Span", "s2", z=0.0),
               record("LogEvent", "l1", z=4)]
    assert scored_modalities(records) == {"trace": True, "log": True}


@pytest.mark.parametrize("bad", [None, True, "5", math.nan, math.inf])
def test_unusable_z_values_do_not_count(bad):
    assert scored_modalities([record("Span", "s1", z=bad)]) == {"trace": False}


def test_trace_kind_counts_as_trace_modality():
    assert scored_modalities([record("Trace", "t1", z=2)]) == {"trace": True}


def test_metric_scored_from_baseline_but_not_from_zero_variance():
    ok = record("MetricAnomaly", "m1", value=18, baseline_mean=10, baseline_std=2)
    flat = record("MetricAnomaly", "m2", value=11, baseline_mean=10, baseline_std=0)
    assert scored_modalities([ok]) == {"metric": True}
    assert scored_modalities([flat]) == {"metric": False}


def test_filters_by_service():
    records = [record("Span", "s1", sid="service:a", z=3), record("Span", "s2", sid="service:b")]
    assert scored_modalities(records, "service:a") == {"trace": True}
    assert scored_modalities(records, "service:b") == {"trace": False}


@pytest.mark.parametrize("records", [
    [record("Span", "s1", duration_us=10)],
    [record("LogEvent", "l1", is_error=True)],
    [record("MetricAnomaly", "m1", value=11, baseline_mean=10, baseline_std=0)],
    [record("Span", "s1", z=math.nan), record("LogEvent", "l1", z=None)],
])
def test_unscored_implies_zero_severity_in_service_statistics(records):
    """Keeps scored_modalities in lock-step with the z rules of service_statistics."""
    scored = scored_modalities(records)
    stat = service_statistics("service:a", records, [], 1000)
    severity = {"metric": stat.sev_m, "log": stat.sev_l, "trace": stat.sev_tc}
    for modality, flag in scored.items():
        if flag is False:
            assert severity[modality] == 0


@pytest.mark.parametrize("records", [
    [record("Span", "s1", z=2.5)],
    [record("LogEvent", "l1", z=-3)],
    [record("MetricAnomaly", "m1", value=18, baseline_mean=10, baseline_std=2)],
])
def test_scored_implies_nonzero_severity_for_nonzero_z(records):
    scored = scored_modalities(records)
    stat = service_statistics("service:a", records, [], 1000)
    severity = {"metric": stat.sev_m, "log": stat.sev_l, "trace": stat.sev_tc}
    assert all(scored.values()) and all(severity[m] > 0 for m in scored)


def _context():
    window = InvestigationWindow(t0=0, t1=2000)
    nodes = [{"id": "service:a", "type": "Service", "properties": {"id": "service:a"}},
             {"id": "m1", "type": "MetricAnomaly",
              "properties": {"id": "m1", "ts": 1000, "metric": "cpu", "z": 6.0}},
             {"id": "s1", "type": "Span", "properties": {"id": "s1", "ts": 1000, "duration_us": 9}},
             {"id": "l1", "type": "LogEvent", "properties": {"id": "l1", "ts": 1000, "is_error": True}}]
    edges = [{"source": n, "target": "service:a", "type": "OBSERVED_ON", "properties": {}}
             for n in ("m1", "s1", "l1")]
    scope = ScopeResult(incident_id="i", anchor_service="service:a", window=window,
                        services=[ScopeService(id="service:a", sev_m=6.0)], nodes=nodes,
                        edges=edges, evidence_gaps=[], n_series_in_scope=1, truncated=False)
    investigation = SimpleNamespace(incident_id="i", window=window)
    return tools.DiagnosticContext(investigation, scope, 1000)


def test_get_severity_reports_which_modalities_were_scored():
    result = tools._contextual_tool("get_severity", SimpleNamespace(), _context(),
                                    {"node_id": "service:a"})
    assert result["vector"] == {"metric": 6.0, "log": 0.0, "trace": 0.0}
    assert result["scored"] == {"metric": True, "log": False, "trace": False}


def test_get_severity_marks_absent_modalities_as_none():
    context = _context()
    scope = context.scope.model_copy(update={"edges": [e for e in context.scope.edges
                                                       if e["source"] == "m1"]})
    context = tools.DiagnosticContext(context.investigation, scope, 1000)
    result = tools._contextual_tool("get_severity", SimpleNamespace(), context,
                                    {"node_id": "service:a"})
    assert result["scored"] == {"metric": True, "log": None, "trace": None}
