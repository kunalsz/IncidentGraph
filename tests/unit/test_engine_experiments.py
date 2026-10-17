import json
import pytest
from backend.app.eval.protocol import freeze_protocol
from backend.app.eval.engine_experiments import replay_fusion, matched_recorded_comparison, run_engine_experiments, EvidenceRuleProvider


def test_split_is_frozen_explicit_and_evaluator_only():
    cases = ["re2ob_checkoutservice_cpu_1", "re2ob_checkoutservice_loss_1",
             "re2ob_recommendationservice_cpu_1", "re2ob_recommendationservice_loss_1"]
    protocol = freeze_protocol(cases)
    assert protocol == freeze_protocol(list(reversed(cases)))
    assert list(protocol["assignments"].values()) == ["development", "unknown_fault", "unknown_service", "unknown_service_fault"]
    assert protocol["promotion_gate"]["status"].startswith("blocked")
    assert "not untouched" in protocol["claim_scope"]


def test_failed_agent_outcome_never_enters_matched_success():
    from backend.app.eval.comparison import compare
    baseline = {"case":"re2ob_checkoutservice_cpu_1", "gt_service":"checkoutservice",
        "gt_fault":"cpu", "status":"ok", "service_order":["checkoutservice"]}
    failed = {**baseline, "agent":{"outcome":"failed"}}
    result = compare({"a":baseline}, {"a":failed}, draws=20)
    assert result["intent_to_run"]["top1_lost"] == 1
    assert result["matched_success_diagnostic_only"]["n"] == 0
    assert result["strata"]["failed_or_missing"]["n"] == 1
    assert result["cases"][0]["right_rank"] is None


def test_mock_rule_low_scheduling_confidence_is_fixed_not_labels():
    provider = EvidenceRuleProvider()
    payload = {"node_id":"service:a", "verdict":{"is_abnormal":True,
        "confidence":.9,"hypothesis":"anomaly","evidence_ids":[]}, "children":[]}
    response = provider.complete(task="consolidate",system="",user=json.dumps(payload),model="mock",max_tokens=100)
    assert response.json()["confidence"] == .1
    assert response.usage_status == "estimated"


def test_recorded_replay_never_fabricates_missing_stages():
    with pytest.raises(ValueError, match="complete typed"):
        replay_fusion({"artifacts": {"ranking": {}}})
    assert matched_recorded_comparison({}, None)["status"] == "unavailable"


def test_isolated_db_required_before_output_creation(tmp_path, monkeypatch):
    monkeypatch.setenv("FALKORDB_HOST", "localhost")
    monkeypatch.setenv("FALKORDB_PORT", "6380")
    with pytest.raises(ValueError, match="never 6380"):
        run_engine_experiments(tmp_path, tmp_path/"out", ["re2ob_checkoutservice_cpu_1"])
    assert not (tmp_path/"out").exists()


def test_real_record_provenance_is_not_inferred_from_agent_flag(tmp_path):
    source = tmp_path/"real.json"
    row = {"case":"re2ob_checkoutservice_cpu_1", "service_order":[], "agent":{"outcome":"completed"}}
    source.write_text(json.dumps({"cases":[row]}))
    result = matched_recorded_comparison({row["case"]:row}, source)
    assert result["status"] == "unavailable"
    assert "provenance" in result["excluded"][row["case"]]
