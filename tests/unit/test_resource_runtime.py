"""Offline v2 verification: historical llm_calls counted successful logical
responses; v2 max_llm_calls and llm_calls count ALL dispatched attempts,
including failed retries and key rotations. Known tokens != charged unknowns.
"""
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import httpx
import pytest
from backend.app.agent.llm import (LLMClient, LLMError, LLMProvider, LLMResponse,
    MockProvider, OpenAIProvider, TransientLLMError)
from backend.app.engine.budget import BudgetMeter, BudgetExceeded
from config.tuning import InvestigationBudget


def test_one_attempt_cap_is_atomic():
    meter = BudgetMeter(InvestigationBudget(max_llm_calls=1))
    gate = threading.Barrier(8)
    def reserve(_):
        gate.wait()
        try:
            return meter.reserve_attempt("logical", "work", 20, 10)
        except BudgetExceeded:
            return None
    with ThreadPoolExecutor(8) as pool:
        leases = [x for x in pool.map(reserve, range(8)) if x]
    assert len(leases) == 1
    assert meter.as_dict()["tokens_committed"] == 30
    meter.settle_attempt(leases[0], "error")
    d = meter.as_dict()
    assert d["provider_attempts"] == 1 and d["unknown_usage_attempts"] == 1
    assert d["tokens_total"] == 0 and d["tokens_charged"] == 30
    assert not d["usage_complete"]
    with pytest.raises(ValueError):
        meter.settle_attempt(leases[0], "success", (1, 1), "known")


def test_tokens_inflight_commitments_block_oversubscription():
    meter = BudgetMeter(InvestigationBudget(max_tokens=100))
    lease = meter.reserve_attempt("a", "work", 50, 30)
    with pytest.raises(BudgetExceeded):
        meter.reserve_attempt("b", "work", 20, 10)
    meter.settle_attempt(lease, "success", (10, 10), "known")
    assert meter.reserve_attempt("b", "work", 50, 30)


def test_usage_overrun_is_visible_and_blocks_dispatch():
    meter = BudgetMeter()
    lease = meter.reserve_attempt("a", "work", 1, 1)
    meter.settle_attempt(lease, "success", (50, 10), "known")
    assert meter.as_dict()["enforcement_violations"]
    with pytest.raises(BudgetExceeded):
        meter.reserve_attempt("b", "work", 1, 1)


def test_six_failed_attempts_charge_all_unknown_usage():
    class Fail(LLMProvider):
        name = "fail"
        def complete(self, **kwargs):
            raise TransientLLMError("temporary")
    meter = BudgetMeter()
    client = LLMClient(meter, provider=Fail(), retries=5, sleep=lambda _: None)
    with pytest.raises(TransientLLMError):
        client.complete("test", "s", "u", max_tokens=5)
    d = meter.as_dict()
    assert d["provider_attempts"] == 6 and d["logical_requests"] == 1
    assert d["retries"] == 5 and d["unknown_usage_attempts"] == 6
    assert d["tokens_charged"] > 0 and not d["strict_token_enforcement"]


def test_retry_after_cannot_exceed_remaining_deadline():
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(429, headers={"Retry-After": "1000"}, json={})
    meter = BudgetMeter(InvestigationBudget(max_wall_s=10, finalization_headroom_s=1))
    client = LLMClient(meter, provider=OpenAIProvider("secret", transport=httpx.MockTransport(handler)),
                       sleep=lambda _: pytest.fail("must not sleep"))
    with pytest.raises(BudgetExceeded):
        client.complete("test", "s", "u", max_tokens=5)
    assert len(calls) == 1


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_permanent_http_errors_never_retry(status):
    seen = []
    def handler(request):
        seen.append(request)
        return httpx.Response(status, json={})
    meter = BudgetMeter()
    client = LLMClient(meter, provider=OpenAIProvider("topsecret", transport=httpx.MockTransport(handler)),
                       sleep=lambda _: pytest.fail("must not sleep"))
    with pytest.raises(LLMError) as exc:
        client.complete("test", "s", "u")
    assert "topsecret" not in str(exc.value)
    assert len(seen) == meter.llm_calls == 1


def test_missing_provider_usage_never_means_zero_cost():
    provider = OpenAIProvider("key", transport=httpx.MockTransport(lambda _: httpx.Response(200, json={
        "choices": [{"message": {"content": "ok"}}]})))
    meter = BudgetMeter()
    response = LLMClient(meter, provider=provider).complete("test", "s", "u", max_tokens=5)
    assert response.usage_status == "unknown"
    d = meter.as_dict()
    assert d["tokens_total"] == 0 and d["tokens_charged"] > 0 and not d["usage_complete"]


def test_single_flight_cache_and_output_limit_identity():
    class Slow(LLMProvider):
        name = "slow"
        calls = 0
        def complete(self, **kwargs):
            self.calls += 1
            time.sleep(.01)
            return LLMResponse("ok", 1, 1, "model")
    provider, meter = Slow(), BudgetMeter()
    client = LLMClient(meter, provider=provider)
    with ThreadPoolExecutor(8) as pool:
        responses = list(pool.map(lambda _: client.complete("test", "s", "u", max_tokens=10), range(8)))
    assert provider.calls == 1 and sum(r.cached for r in responses) == 7
    client.complete("test", "s", "u", max_tokens=11)
    assert provider.calls == 2 and meter.logical_requests == 9


def test_invalid_task_schema_is_not_cached_or_retried():
    provider = MockProvider({"self_verify": '{"node_id":"x","unexpected":true}'})
    meter = BudgetMeter()
    client = LLMClient(meter, provider=provider)
    for _ in range(2):
        with pytest.raises(LLMError):
            client.complete("self_verify", "s", "{}")
    assert meter.llm_calls == 2 and client.stats["retries"] == 0


def test_strict_token_mode_rejected_without_compatible_tokenizer():
    meter = BudgetMeter(InvestigationBudget(strict_tokens=True))
    with pytest.raises(ValueError, match="token"):
        LLMClient(meter, provider=MockProvider())


def test_finalization_headroom_and_cooperative_claim():
    now = [0.0]
    meter = BudgetMeter(InvestigationBudget(max_wall_s=10, finalization_headroom_s=5), clock=lambda: now[0])
    now[0] = 6
    with pytest.raises(BudgetExceeded):
        meter.check()
    meter.check("report")
    assert meter.remaining_seconds("report") == 4
    assert "not hard" in meter.as_dict()["deadline_enforcement"]


def test_fair_slots_reserved_before_work_and_released():
    meter = BudgetMeter(InvestigationBudget(max_hypotheses=5))
    plan = meter.allocate_hypothesis_slots(["a", "b", "c"])
    assert plan == {"a": 2, "b": 2, "c": 1}
    meter.new_hypothesis("c")
    with pytest.raises(BudgetExceeded):
        meter.new_hypothesis("c")
    meter.release_hypothesis_slots("a")
    assert meter.allocate_hypothesis_slots(["x"]) == {"x": 2}


def test_runner_failure_retains_state_without_graph_finalization():
    from backend.app.engine.runner import run_investigation
    meter = BudgetMeter()
    def p0(symptom, **kwargs):
        assert kwargs["meter"] is meter
        return SimpleNamespace(incident_id="incident:test", budget={})
    def p1(inv):
        return SimpleNamespace(incident_id=inv.incident_id)
    def p2(*args):
        raise BudgetExceeded("tokens", 2, 1)
    def forbidden(*args, **kwargs):
        pytest.fail("must not invoke later graph/provider phases")
    result = run_investigation("symptom", meter=meter,
        phase_functions=(p0, p1, p2, forbidden, forbidden, forbidden, forbidden))
    assert result.status == "degraded" and result.failed_stage == "P2"
    assert result.scope is not None and result.report.tier == "INCONCLUSIVE"
    assert result.report.verdict == "insufficient_evidence"
    assert result.report.runbook == {} and result.report.winner_candidate is None


def test_runner_p0_exception_also_has_report():
    from backend.app.engine.runner import run_investigation
    def fail(*args, **kwargs):
        raise RuntimeError("provider key must not be serialized")
    result = run_investigation("symptom", phase_functions=(fail,) * 7)
    assert result.report.tier == "INCONCLUSIVE" and result.failed_stage == "P0"
    assert "provider key" not in result.model_dump_json()


def test_p3_fair_slots_shared_expansion_and_partial_siblings(monkeypatch):
    from backend.app.engine import p3_investigate as p3
    from backend.app.engine.contracts import (Investigation, InvestigationWindow, ScopeResult,
        ScopeService, RankResult, RankedCandidate)
    from backend.app.graph import client as graph_client
    ids = ["service:a", "service:b", "service:x", "service:y"]
    window = InvestigationWindow(t0=0, t1=10)
    scope = ScopeResult(incident_id="i", anchor_service=ids[0], window=window,
        services=[ScopeService(id=x) for x in ids], nodes=[],
        edges=[{"type": "DEPENDS_ON", "source": p, "target": c}
               for p in ids[:2] for c in ids[2:]], evidence_gaps=[], n_series_in_scope=0, truncated=False)
    ranking = RankResult(incident_id="i", scope_ref=scope.scope_ref,
        candidates=[RankedCandidate(id=x, rank=i+1, score=1., components={},
            component_ranks={}, evidence=[], rule_flags=[], selected=i < 2) for i, x in enumerate(ids)],
        branch_ids=ids[:2], indicators=[], audit=[], n_candidates=4, truncated=False, config={})
    inv = Investigation(incident_id="i", session_graph="incidentgraph:session:test", symptom="s",
        anchor_service=ids[0], window=window, budget={})
    class Graph:
        def query(self, *args, **kwargs):
            return SimpleNamespace(result_set=[])
    monkeypatch.setattr(graph_client, "graph_for", lambda _: Graph())
    def bundle(graph, node, window):
        return {"bundle": {"service_id": node, "metrics": [{"id": "m:"+node,
            "metric": "cpu", "z": 0.2, "direction": "up"}], "logs": [], "adjacency": []},
            "evidence_ids": ["m:"+node]}
    monkeypatch.setattr(p3.bundle_mod, "fetch_branch_bundle", bundle)
    calls = []
    class Provider(MockProvider):
        def complete(self, **kwargs):
            payload = json.loads(kwargs["user"])
            calls.append((kwargs["task"], payload))
            if kwargs["task"] == "hypothesis_batch" and payload.get("service_id") == "service:y":
                raise LLMError("schema failure")
            return super().complete(**kwargs)
    meter = BudgetMeter(InvestigationBudget(max_hypotheses=10, expansion_slot_fraction=.4))
    result = p3.investigate_investigation(inv, scope, ranking, meter=meter,
        llm=LLMClient(meter, provider=Provider(), sleep=lambda _: None))
    assert result.budget["p3_scheduling"]["primary_slots"] == {"service:a": 3, "service:b": 3}
    batches = [p for task, p in calls if task == "hypothesis_batch"]
    assert len([p for p in batches if p["service_id"] == "service:x"]) == 1
    assert len([p for p in batches if p["service_id"] == "service:y"]) == 1
    assert {h.candidate for h in result.hypotheses} == {"service:a", "service:b", "service:x"}
    assert result.scope_ref == scope.scope_ref
    assert all(h.scope_ref == scope.scope_ref for h in result.hypotheses)
    assert meter.as_dict()["hypothesis_slots_reserved"] == 0
    assert len(result.expansions) == 2  # shared neighbor retains both parents
    assert all("service:x" in e["completed_ids"] for e in result.expansions)
    assert not any(task in ("reflection", "escalate") for task, _ in calls)


def test_runner_cancellation_does_not_start_later_phases():
    from backend.app.engine.runner import run_investigation
    def cancel(*args, **kwargs):
        raise KeyboardInterrupt()
    result = run_investigation("symptom", phase_functions=(cancel,) * 7)
    assert result.status == "cancelled" and result.report.tier == "INCONCLUSIVE"
