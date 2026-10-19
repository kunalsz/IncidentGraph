"""Real local case plus deterministic mock provider: factory/ledger/shadow integration.

Not a model-quality or superiority test. Named benchmark profile, not default P0.
"""
from pathlib import Path

import pytest

from backend.app.agent.llm import LLMClient, MockProvider
from backend.app.eval.rcaeval_runner import benchmark_budget, run_case

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "RE2-OB"
CASE = "re2ob_checkoutservice_cpu_1"


@pytest.mark.integration
def test_agent_mode_factory_completes_and_exports_exact_p2_shadow(tmp_path):
    case_dir = DATA_DIR / CASE
    if not case_dir.exists():
        pytest.skip(f"dataset absent: {case_dir}")
    meters = []
    def factory(meter):
        meters.append(meter)
        return LLMClient(meter, provider=MockProvider(), sleep=lambda _s: None)
    # This test asserts the shadow/export contract, not the budget policy, and it
    # deliberately sets no branch cap - so every candidate is investigated. That
    # is exactly the configuration where the fixed 200k token cap pinned and left
    # 7 of 12 branches dead on BudgetExceeded (measured: output/replication-2026-10-19/
    # DIAGNOSTICS.md). Use the harness's coherent envelope instead of an ad-hoc
    # number, so the test exercises the real default.
    result = run_case(case_dir, tmp_path, with_agent=True,
                      budget=benchmark_budget(), llm_factory=factory)
    assert result["status"] == "ok", (result.get("failed_stage"),result.get("error"),result.get("phase_events"))
    assert len(meters) == 1
    assert result["agent"]["outcome"] == "completed"
    assert result["resources"]["llm_calls"] > 0
    assert result["resources"]["llm_calls"] == meters[0].llm_calls
    assert result["agent"]["export_policy"] == "p2"
    assert result["service_order"] == result["baseline_order"]
    assert all(stage in result["artifacts"] for stage in ("ranking","investigation","validation","fusion"))
    assert any(event["phase"] == "P5" and event["status"] == "completed" for event in result["phase_events"])
    assert result["cleanup"] == "deleted"
