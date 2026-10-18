"""Offline protocol checks: all HTTP is MockTransport, no provider requests."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import threading

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("grounded_shadow_six", ROOT / "scripts/run_grounded_shadow_six.py")
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)


def provider(handler):
    stop, receipts = threading.Event(), []
    p = g.RecordedProvider("DO-NOT-EXPOSE", "https://example.invalid/v1", "exact", stop,
                           receipts, lambda _: None, [g.CASES[0]], transport=httpx.MockTransport(handler))
    return p, stop, receipts


def call(p):
    return p.complete(task="unknown", system="s", user="{}", model="exact", max_tokens=30)


def envelope(**changes):
    return {"model": "exact", "usage": {"prompt_tokens": 4, "completion_tokens": 3},
            "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}], **changes}


def test_exact_raw_model_known_usage_and_no_secret_receipt():
    def handler(request):
        assert request.headers["Authorization"] == "Bearer DO-NOT-EXPOSE"
        return httpx.Response(200, json=envelope())
    p, stop, receipts = provider(handler)
    response = call(p)
    assert response.model == "exact" and response.usage_status == "known"
    assert not stop.is_set()
    assert receipts[0]["tokens_in"] == 4 and receipts[0]["http_status"] == 200
    assert "DO-NOT-EXPOSE" not in json.dumps(receipts)
    assert "system" not in receipts[0] and "user" not in receipts[0]


@pytest.mark.parametrize("changes", [{"model": None}, {"model": "other"}, {"model": ""},
    {"usage": {}}, {"usage": {"prompt_tokens": True, "completion_tokens": 2}},
    {"usage": {"prompt_tokens": -1, "completion_tokens": 2}}])
def test_unknown_usage_or_inexact_model_stops(changes):
    p, stop, receipts = provider(lambda _: httpx.Response(200, json=envelope(**changes)))
    call(p)
    assert stop.is_set()
    with pytest.raises(g.LLMError):
        call(p)
    assert len(receipts) == 1


def test_absent_returned_model_is_not_filled_in():
    body = envelope()
    del body["model"]
    p, stop, receipts = provider(lambda _: httpx.Response(200, json=body))
    assert call(p).model is None
    assert stop.is_set() and receipts[0]["returned_model"] is None


@pytest.mark.parametrize("status", [400, 401, 408, 409, 425, 500, 503])
def test_non429_errors_stop_without_retries(status):
    from backend.app.engine.budget import BudgetMeter
    p, stop, receipts = provider(lambda _: httpx.Response(status, text="DO-NOT-EXPOSE"))
    client = g.StoppingClient(BudgetMeter(g.BUDGET), provider=p, model="exact", model_cheap="exact", retries=3)
    with pytest.raises(g.LLMError):
        client.complete("unknown", "s", "{}", max_tokens=30)
    assert stop.is_set() and len(receipts) == 1
    assert "DO-NOT-EXPOSE" not in json.dumps(receipts)


def test_metered429_retries_settled_through_ledger():
    from backend.app.engine.budget import BudgetMeter
    attempts = []
    def handler(_):
        attempts.append(1)
        return httpx.Response(429) if len(attempts) == 1 else httpx.Response(200, json=envelope())
    p, stop, receipts = provider(handler)
    meter = BudgetMeter(g.BUDGET)
    client = g.StoppingClient(meter, provider=p, model="exact", model_cheap="exact", retries=3,
                              rate_limit_backoff=0)
    client.complete("unknown", "s", "{}", max_tokens=30)
    assert not stop.is_set() and len(receipts) == 2
    assert receipts[0]["metered_429_exception"] and receipts[0]["usage_status"] == "unknown"
    assert meter.as_dict()["unknown_usage_attempts"] == 1


def test_schema_error_stops_further_calls():
    from backend.app.engine.budget import BudgetMeter
    p, stop, receipts = provider(lambda _: httpx.Response(200, json=envelope()))
    client = g.StoppingClient(BudgetMeter(g.BUDGET), provider=p, model="exact", model_cheap="exact", retries=3)
    with pytest.raises(g.LLMError):
        client.complete("self_verify", "s", "{}", max_tokens=30)
    assert stop.is_set() and len(receipts) == 1


def record(case="re2ob_checkoutservice_cpu_1"):
    ids = ["service:frontend", "service:cartservice", "service:currencyservice", "service:checkoutservice"]
    return {"case": case, "gt_service": "checkoutservice", "gt_service_rank": 4, "status": "ok",
        "data": {"sha256": {"metrics.parquet": "inputhash"}}, "scope_profile": {"fp": "same"},
        "service_order": [s[8:] for s in ids], "ranks": [s[8:] for s in ids],
        "resources": {"llm_calls": 0}, "timings": {}, "cleanup": "deleted",
        "artifacts": {"ranking": {"candidates": [{"id": sid, "rank": i + 1} for i, sid in enumerate(ids)],
                                    "branch_ids": ids[:3]},
          "scope": {"services": [{"id": sid, "self_time_quality": "partial"} for sid in ids],
                    "nodes": [{"id": "span", "type": "Span"}], "config": {"completeness": {"source_coverage": "unknown"}}},
          "investigation": {"hypotheses": [{"id": "h", "claim_kind": "dependency_propagation"}],
              "budget": {"p3_scheduling": {"primary_slots": dict.fromkeys(ids[:3], 1),
                  "branches": [{"generated": 2, "admitted": 1, "rejected": ["bad"]}]}}},
          "validation": {"hypotheses": {"h": {"claim_type": "dependency_propagation", "verdict": "partial",
              "evidence_roles": {"dependency": ["span"]}}}}}}


def test_full_control_comparison_and_actual_truth_branch_rank():
    a, b = record(), record()
    result = g.compare_control(a, b)
    assert result["full_p2_order_equal"] and result["service_statistics_equal"]
    diagnostic = result["shadow"]
    assert diagnostic["truth_p2_position"] == 4 and not diagnostic["truth_selected"]
    assert not diagnostic["truth_top3_guaranteed"]
    assert diagnostic["coverage_floor_satisfied"]
    assert diagnostic["claim_type_counts"] == {"dependency_propagation": 1}
    assert diagnostic["validity_counts"] == {"partial": 1}
    assert diagnostic["validations"][0]["trace_roles"] == {"dependency": ["span"]}
    b["artifacts"]["ranking"]["candidates"].reverse()
    b["artifacts"]["scope"]["services"][0]["self_time_quality"] = "complete"
    b["data"]["sha256"]["metrics.parquet"] = "changed"
    result = g.compare_control(a, b)
    assert not result["full_p2_order_equal"]
    assert not result["service_statistics_equal"] and not result["same_inputs"]


def test_freeze_covers_prompts_schemas_actual_budget_and_inputs():
    frozen = g.freeze("exact", "https://example.invalid/v1", g.OLD_RUN)
    assert frozen["budget"] == frozen["provenance"]["budget"] == g.BUDGET.as_dict()
    assert frozen["budget"]["max_hypotheses"] == 10
    assert frozen["budget"]["expansion_slot_fraction"] == .7
    assert frozen["primary_branches"] == frozen["minimum_primary_coverage"] == 3
    assert frozen["retries"] == 3 and frozen["parallelism"] == 1 and not frozen["fallback"]
    assert any(p.endswith("hypothesis_batch.md") for p in frozen["provenance"]["config_hashes"])
    assert "backend/app/agent/responses.py" in frozen["provenance"]["source_files"]
    assert "individual_proposal" in frozen["task_schemas"]
    assert set(frozen["data_sha256"]) == set(g.CASES)
    assert "not a causal ablation" in frozen["historical_comparison_claim"]
    fp = frozen.pop("freeze_sha256")
    assert g.digest(frozen) == fp


def test_redaction_nested_and_overlapping_secrets():
    assert g.redact({"error": ["token-long token"]}, ["token", "token-long"]) == {
        "error": ["[REDACTED] [REDACTED]"]}


def test_main_freeze_then_control_and_shadow_forwarding_no_paid_calls(tmp_path, monkeypatch):
    monkeypatch.setattr(g, "ROOT", tmp_path)
    for k, v in {"LLM_PROVIDER": "openai", "LLM_MODEL": "exact", "LLM_BASE_URL": "https://example.invalid/v1",
                 "LLM_API_KEY": "DO-NOT-EXPOSE"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(g, "freeze", lambda model, base, old, cases=None:
                        {"old_run": str(tmp_path / "old"), "cases": list(cases or g.CASES),
                         "freeze_sha256": "frozen"})
    monkeypatch.setattr(g.OpenAICompatibleProvider, "complete", lambda *a, **k: pytest.fail("no provider calls permitted"))
    calls = []
    def run_case(path, out, **kwargs):
        calls.append(kwargs)
        return record(path.name)
    monkeypatch.setattr(g.rr, "run_case", run_case)
    monkeypatch.setattr(g.ex, "replay_fusion", lambda row, **kw: "fused")
    reused = []
    def arm(row, arm, fused, *, reused=False):
        assert reused is True and fused == "fused"
        return {**row, "stage_reuse": reused}
    monkeypatch.setattr(g.ex, "arm_record", arm)
    assert g.main(["--freeze-only"]) == 0
    assert not calls
    protocol = next((tmp_path / "output").glob("grounded-shadow-six-*/protocol.json"))
    assert g.main(["--frozen-protocol", str(protocol)]) == 0
    assert len(calls) == 12
    for i, kwargs in enumerate(calls):
        assert kwargs["with_agent"] == bool(i % 2)
        assert kwargs["primary_branches"] == kwargs["minimum_primary_coverage"] == 3
        assert kwargs["budget"] == g.BUDGET and kwargs["export_policy"] == "shadow"
        assert ("llm_factory" in kwargs) == bool(i % 2)
    summary = json.loads((protocol.parent / "summary.json").read_text())
    assert summary["provider_attempts"] == 0 and not summary["stop_flag"]
    assert len(list((protocol.parent / "diagnostic-only").glob("*.json"))) == 6
    with pytest.raises(FileExistsError):
        g.main(["--frozen-protocol", str(protocol)])
