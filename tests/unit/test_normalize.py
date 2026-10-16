"""Unit tests for the normalization pipeline (§4.7)."""

import pytest

from backend.app.ingest import normalize


class TestToEpochMs:
    def test_int_passthrough(self):
        assert normalize.to_epoch_ms(1792141200000) == 1792141200000

    def test_float_truncates_to_int(self):
        assert normalize.to_epoch_ms(1792141200.5) == 1792141200

    def test_iso_z(self):
        assert normalize.to_epoch_ms("2026-10-16T09:00:00Z") == 1792141200000

    def test_iso_offset(self):
        assert normalize.to_epoch_ms("2026-10-16T14:30:00+05:30") == 1792141200000

    def test_naive_treated_as_utc(self):
        assert normalize.to_epoch_ms("2026-10-16T09:00:00") == 1792141200000

    def test_numeric_string(self):
        assert normalize.to_epoch_ms("1792141200000") == 1792141200000

    def test_rejects_garbage(self):
        with pytest.raises(ValueError):
            normalize.to_epoch_ms("not-a-date")
        with pytest.raises(ValueError):
            normalize.to_epoch_ms(-5)


class TestSeverity:
    def test_canonical_levels(self):
        assert normalize.canonical_severity("critical") == "critical"
        assert normalize.canonical_severity("ERROR") == "major"
        assert normalize.canonical_severity("warn") == "warning"
        assert normalize.canonical_severity("info") == "info"

    def test_none_maps_to_info(self):
        assert normalize.canonical_severity(None) == "info"

    def test_unknown_maps_to_info_and_logs(self):
        assert normalize.canonical_severity("catastrophic") == "info"


class TestPodMatcher:
    def test_canonical_pod(self):
        assert normalize.service_from_pod(
            "payment-7d9f4-x2v4b") == "service:payment"

    def test_non_pod_name(self):
        assert normalize.service_from_pod("payment") is None
        assert normalize.service_from_pod("a-b") is None


class TestLogs:
    def test_stacktrace_aggregation(self):
        lines = [
            "payment service unreachable: connection refused",
            "    at oteldemo.PaymentService.charge(PaymentService.java:112)",
            "    at io.grpc.stub.ServerCalls.startCall(ServerCalls.java:95)",
            "new unrelated event",
        ]
        out = normalize.aggregate_stack_traces(lines)
        assert len(out) == 2
        assert "at oteldemo" in out[0]
        assert out[1] == "new unrelated event"

    def test_template_masks_numbers_and_ips(self):
        t = normalize.template_for("dial tcp 10.42.1.17:50051 refused after 3 tries")
        assert "<num>" in t and "<ip>" in t

    def test_normalize_log_flags_errors(self):
        out = normalize.normalize_log({
            "id": "log:t1", "ts": "2026-10-16T09:00:20Z", "level": "error",
            "message": "boom",
        })
        assert out["ts"] == 1792141220000
        assert out["level"] == "major"
        assert out["is_error"] is True
        assert out["template_id"] == "boom"


class TestMetricAnomaly:
    def test_normalizes_ts(self):
        out = normalize.normalize_metric_anomaly({
            "id": "metric:t1", "ts": "2026-10-16T09:01:00Z", "metric": "m",
        })
        assert out["ts"] == 1792141260000
