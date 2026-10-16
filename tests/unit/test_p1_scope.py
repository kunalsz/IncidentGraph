import math

import pytest

from backend.app.engine.p1_scope import service_statistics


def record(kind, node_id, sid="service:a", **props):
    return {"id": node_id, "type": kind, "service_id": sid,
            "properties": {"id": node_id, "ts": 1000, **props}}


def test_absent_modalities_are_zero():
    stat = service_statistics("service:a", [], [], 1000)
    assert (stat.sev_m, stat.sev_l, stat.sev_tc) == (0, 0, 0)
    assert stat.hours_since_change is None
    assert stat.deploy_proximity == 0


def test_severity_and_sigma_window():
    observations = [record("MetricAnomaly", "m1", z=-4),
                    record("MetricAnomaly", "m2", z=9, ts=62_000),
                    record("LogEvent", "l1", z=-2),
                    record("Span", "s1", z=3)]
    stat = service_statistics("service:a", observations, [], 1000)
    assert (stat.sev_m, stat.sev_l, stat.sev_tc) == (9, 2, 3)
    assert stat.nsigma_evidence == ["m1"]


def test_sigma_from_baseline_and_strict_threshold():
    observations = [record("MetricAnomaly", "m1", value=18, baseline_mean=10, baseline_std=2),
                    record("MetricAnomaly", "m2", z=3)]
    stat = service_statistics("service:a", observations, [], 1000)
    assert stat.sev_m == 4
    assert stat.nsigma_evidence == ["m1"]


def test_self_time_joins_trace_and_direct_parent_only():
    observations = [
        record("Span", "p", trace_id="t1", span_id="p", duration_us=1_000_000),
        record("Span", "c", sid="service:b", trace_id="t1", span_id="c", parent_id="p", duration_us=200_000),
        record("Span", "g", sid="service:b", trace_id="t1", span_id="g", parent_id="c", duration_us=100_000),
        record("Span", "other", sid="service:b", trace_id="t2", parent_id="p", duration_us=900_000),
    ]
    stat = service_statistics("service:a", observations, [], 1000)
    assert stat.self_time_ms == 800
    assert stat.self_time_flag


def test_logs_without_z_do_not_invent_severity():
    stat = service_statistics("service:a", [record("LogEvent", "l1", is_error=True)], [], 1000)
    assert stat.sev_l == 0


def test_zero_variance_baseline_flags_changed_value_without_infinite_score():
    observations = [record("MetricAnomaly", "m1", value=11, baseline_mean=10, baseline_std=0)]
    stat = service_statistics("service:a", observations, [], 1000)
    assert stat.nsigma_evidence == ["m1"]
    assert stat.sev_m == 0


def test_change_prior_excludes_future_changes():
    changes = [record("Deployment", "d1", ts=0),
               record("ConfigChange", "d2", ts=86_400_001)]
    stat = service_statistics("service:a", [], changes, 86_400_000)
    assert stat.hours_since_change == 24
    assert stat.deploy_proximity == pytest.approx(math.exp(-1))
