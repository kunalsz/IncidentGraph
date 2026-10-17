"""Unit tests for the bundle assembler (pure function, no graph)."""

import json

from backend.app.agent.bundle import (MAX_LOG_MESSAGE_CHARS, assemble_bundle,
                                      bundle_evidence_ids)
from backend.app.agent.llm import estimate_tokens


def metric(i, z):
    return {"id": f"metric:{i}", "metric": "cpu", "z": z, "direction": "up",
            "gamma": 1.0, "ts": 1000 + i}


def log(i, is_error, template="t1", message="m"):
    return {"id": f"log:{i}", "ts": 1000 + i, "level": "major" if is_error else "info",
            "template_id": template, "message": message, "is_error": is_error,
            "z": 4.0 if is_error else 0.0}


def adjacency(name, criticality):
    return {"service": name, "name": name, "direction": "calls",
            "criticality": criticality, "p99_ms": 100, "calls_per_min": 10}


class TestAssemble:
    def test_metrics_sorted_by_abs_z_and_capped(self):
        metrics = [metric(i, z=float(i)) for i in range(20)]
        bundle, _ = assemble_bundle("service:x", metrics, [], [])
        assert len(bundle["metrics"]) == 15
        zs = [abs(m["z"]) for m in bundle["metrics"]]
        assert zs == sorted(zs, reverse=True)
        assert zs[0] == 19.0

    def test_logs_errors_first_dedup_by_template(self):
        logs = [
            log(1, False, message="ok"),
            log(2, True, template="boom"),
            log(3, True, template="boom"),   # duplicate template
            log(4, True, template="crash"),
            log(5, False, message="ok2"),
        ]
        bundle, _ = assemble_bundle("service:x", [], logs, [])
        kept = bundle["logs"]
        error_templates = [entry["template_id"] for entry in kept if entry["is_error"]]
        assert error_templates == ["boom", "crash"]  # dedup, order preserved
        assert all(not entry["is_error"] for entry in kept[len(error_templates):])
        assert sum(1 for entry in kept if not entry["is_error"]) <= 5

    def test_log_message_truncated(self):
        long_message = "x" * 500
        bundle, _ = assemble_bundle("s", [], [log(1, True, message=long_message)], [])
        kept = bundle["logs"][0]["message"]
        assert len(kept) == MAX_LOG_MESSAGE_CHARS + 1  # + ellipsis
        assert kept.endswith("…")

    def test_adjacency_sorted_and_capped(self):
        nodes = [adjacency(f"service:s{i}", criticality=i / 20) for i in range(20)]
        bundle, _ = assemble_bundle("s", [], [], nodes)
        assert len(bundle["adjacency"]) == 12
        criticalities = [n["criticality"] for n in bundle["adjacency"]]
        assert criticalities == sorted(criticalities, reverse=True)

    def test_token_cap_trims(self):
        metrics = [metric(i, z=float(i)) for i in range(15)]
        logs = [log(i, True, template=f"t{i}", message="y" * 180) for i in range(30)]
        bundle, _ = assemble_bundle("s", metrics, logs, [], token_cap=200)
        assert estimate_tokens(json.dumps(bundle)) <= 200
        assert len(bundle["metrics"]) >= 1  # top anomalies survive

    def test_irreducible_bundle_rejected(self):
        import pytest
        with pytest.raises(ValueError, match='minimal bundle'):
            assemble_bundle('s', [metric(0, 5.0)], [log(0, True)], [], token_cap=1)

    def test_fingerprint_stable_and_sensitive(self):
        _, fp1 = assemble_bundle("s", [metric(1, 2.0)], [log(1, True)], [])
        _, fp2 = assemble_bundle("s", [metric(1, 2.0)], [log(1, True)], [])
        _, fp3 = assemble_bundle("s", [metric(1, 3.0)], [log(1, True)], [])
        assert fp1 == fp2
        assert fp1 != fp3

    def test_evidence_ids_collected_sorted(self):
        bundle, _ = assemble_bundle(
            "s", [metric(1, 2.0)], [log(9, True)], [],
            changes=[{"id": "config:1", "kind": "ConfigChange", "ts": 1}])
        assert bundle_evidence_ids(bundle) == ["config:1", "log:9", "metric:1"]

    def test_fingerprint_inputs_report_totals(self):
        bundle, _ = assemble_bundle("s", [metric(i, 1.0) for i in range(20)], [], [])
        assert bundle["fingerprint_inputs"]["n_metrics_total"] == 20
        assert len(bundle["metrics"]) == 15
