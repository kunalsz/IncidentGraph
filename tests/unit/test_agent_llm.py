"""Unit tests for the LLM layer: mock contracts, client metering, cache,
retries, and real-provider parsing via httpx.MockTransport (no network)."""

import json

import httpx
import pytest

from backend.app.agent import llm
from backend.app.agent.llm import (AnthropicProvider, LLMClient, LLMError,
                                   LLMUnavailable, LLMProvider, LLMResponse,
                                   MockProvider, NoneProvider, OpenAIProvider, TransientLLMError,
                                   estimate_tokens, provider_from_env)
from backend.app.engine.budget import BudgetExceeded, BudgetMeter
from backend.app.engine.contracts import BranchVerdict, NodeVerdict
from config.tuning import InvestigationBudget


def bundle_payload(service_id="service:payment", z=12.0, with_error=True):
    metrics = [
        {"id": f"metric:{service_id}/{i}", "metric": f"m{i}", "z": z / (i + 1),
         "direction": "up", "gamma": 5.0}
        for i in range(3)
    ]
    logs = [{"id": "log:1", "is_error": True, "template_id": "t1", "message": "boom"}] \
        if with_error else []
    return {"service_id": service_id,
            "bundle": {"service_id": service_id, "metrics": metrics, "logs": logs,
                       "adjacency": [{"service": "service:checkout", "criticality": 0.9}]}}


class TestEstimate:
    def test_estimate_tokens_min_one(self):
        assert estimate_tokens("") == 1
        assert estimate_tokens("x" * 400) == 100


class TestMockProvider:
    def test_self_verify_matches_contract(self):
        provider = MockProvider()
        response = provider.complete(
            task="self_verify", system="s", user=json.dumps(bundle_payload()),
            model="mock", max_tokens=500)
        verdict = NodeVerdict(**response.json())
        assert verdict.is_abnormal is True
        assert verdict.has_metric_anomaly is True
        assert verdict.has_error_logs is True
        assert verdict.confidence_rule == "cross_modal_consensus"
        assert verdict.key_symptoms

    def test_self_verify_healthy_node(self):
        provider = MockProvider()
        payload = {"service_id": "service:cart", "bundle": {
            "service_id": "service:cart", "metrics": [{"id": "m", "metric": "cpu", "z": 0.4}],
            "logs": [], "adjacency": []}}
        verdict = NodeVerdict(**provider.complete(
            task="self_verify", system="s", user=json.dumps(payload),
            model="mock", max_tokens=500).json())
        assert verdict.is_abnormal is False
        assert verdict.confidence_rule == "inference_only"

    def test_hypothesis_batch_floor_and_determinism(self):
        provider = MockProvider()
        payload = bundle_payload()
        first = provider.complete(task="hypothesis_batch", system="s",
                                  user=json.dumps(payload), model="m",
                                  max_tokens=800).json()
        second = provider.complete(task="hypothesis_batch", system="s",
                                   user=json.dumps(payload), model="m",
                                   max_tokens=800).json()
        hypotheses = first["hypotheses"]
        assert len(hypotheses) == 4  # v2: actual proposals only, no padding
        for hypothesis in hypotheses:
            assert set(hypothesis) == {"text", "candidate", "evidence_refs",
                                       "initial_score"}
        assert first == second  # deterministic

    def test_consolidate_abnormal_node_wins(self):
        provider = MockProvider()
        payload = {
            "node_id": "service:payment",
            "verdict": {"node_id": "service:payment", "is_abnormal": True,
                        "confidence": 0.8, "confidence_rule": "cross_modal_consensus",
                        "hypothesis": "cpu spike", "evidence_ids": ["e1"]},
            "children": [{"node_id": "service:checkout", "confidence": 0.9,
                          "confidence_rule": "single_modality", "evidence_ids": ["e2"]}],
        }
        verdict = BranchVerdict(**provider.complete(
            task="consolidate", system="s", user=json.dumps(payload),
            model="m", max_tokens=400).json())
        assert verdict.local_root_cause == "service:payment"
        assert verdict.evidence_ids == ["e1"]

    def test_consolidate_child_wins_with_discount(self):
        provider = MockProvider()
        payload = {
            "node_id": "service:frontend",
            "verdict": {"node_id": "service:frontend", "is_abnormal": False,
                        "confidence": 0.2, "confidence_rule": "inference_only",
                        "evidence_ids": []},
            "children": [{"node_id": "service:payment", "confidence": 0.8,
                          "confidence_rule": "cross_modal_consensus",
                          "evidence_ids": ["e9"]}],
        }
        verdict = BranchVerdict(**provider.complete(
            task="consolidate", system="s", user=json.dumps(payload),
            model="m", max_tokens=400).json())
        assert verdict.local_root_cause == "service:payment"
        assert verdict.confidence == pytest.approx(0.72)  # 0.8 * 0.9
        assert verdict.status == "closed"

    def test_consolidate_no_signal_is_partial(self):
        provider = MockProvider()
        payload = {"node_id": "service:x",
                   "verdict": {"node_id": "service:x", "is_abnormal": False},
                   "children": []}
        verdict = BranchVerdict(**provider.complete(
            task="consolidate", system="s", user=json.dumps(payload),
            model="m", max_tokens=400).json())
        assert verdict.local_root_cause is None
        assert verdict.status == "partial"

    def test_reflection_scores_refs(self):
        provider = MockProvider()
        hypothesis = {"evidence_refs": ["metric:service:payment/0"]}
        payload = {"hypothesis": hypothesis, "bundle": bundle_payload()["bundle"]}
        scores = provider.complete(task="reflection", system="s",
                                   user=json.dumps(payload), model="m",
                                   max_tokens=300).json()
        assert 0 <= scores["e"] <= 10 and 0 <= scores["c_comp"] <= 10
        assert 0 <= scores["k"] <= 10
        # unknown refs score no better than no refs
        payload2 = {"hypothesis": {"evidence_refs": ["bogus"]},
                    "bundle": bundle_payload()["bundle"]}
        scores2 = provider.complete(task="reflection", system="s",
                                    user=json.dumps(payload2), model="m",
                                    max_tokens=300).json()
        assert scores2["e"] <= scores["e"]

    def test_overrides_and_unknown_task(self):
        assert MockProvider({"self_verify": '{"node_id": "x"}'}).complete(
            task="self_verify", system="s", user="{}", model="m",
            max_tokens=10).text == '{"node_id": "x"}'
        stub = MockProvider().complete(task="mystery", system="s", user="{}",
                                       model="m", max_tokens=10).json()
        assert stub["task"] == "mystery"


class TestResponseParsing:
    def test_code_fence_stripped(self):
        response = LLMResponse('```json\n{"a": 1}\n```', 1, 1, "m")
        assert response.json() == {"a": 1}

    def test_json_salvaged_from_surrounding_prose(self):
        response = LLMResponse('Sure! Here is the JSON: {"a": 1} Hope that helps.',
                               1, 1, "m")
        with pytest.raises(LLMError):
            response.json()  # v2 rejects surrounding prose rather than salvaging

    def test_invalid_json_raises(self):
        with pytest.raises(LLMError):
            LLMResponse("not json", 1, 1, "m").json()


class TestProviderFromEnv:
    def test_none_default(self, monkeypatch):
        # explicit "none" (an unset var may now be filled from a local .env)
        monkeypatch.setenv("LLM_PROVIDER", "none")
        assert provider_from_env().name == "none"
        with pytest.raises(LLMUnavailable):
            NoneProvider().complete(task="t", system="s", user="u",
                                    model="m", max_tokens=1)

    def test_mock(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "mock")
        assert provider_from_env().name == "mock"

    def test_openai_without_key(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        with pytest.raises(LLMUnavailable):
            provider_from_env()

    def test_openai_with_key(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        assert provider_from_env().name == "openai"

    def test_unknown_provider(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "wat")
        with pytest.raises(LLMUnavailable):
            provider_from_env()


class TestLLMClient:
    def client(self, meter=None, provider=None, **kwargs):
        return LLMClient(meter or BudgetMeter(), provider=provider or MockProvider(),
                         sleep=lambda _s: None, **kwargs)

    def test_meters_and_caches(self):
        meter = BudgetMeter()
        client = self.client(meter)
        user = json.dumps(bundle_payload())
        first = client.complete("self_verify", "sys", user)
        assert meter.llm_calls == 1
        assert meter.tokens_charged > 0
        assert first.usage_status == "estimated"  # mock counts are not known provider usage
        assert meter.tokens == 0
        second = client.complete("self_verify", "sys", user)
        assert second.cached is True
        assert meter.llm_calls == 1  # cache hits are free
        assert client.stats["cache_hits"] == 1

    def test_budget_cap_raises(self):
        meter = BudgetMeter(InvestigationBudget(max_llm_calls=1))
        client = self.client(meter)
        client.complete("self_verify", "sys", json.dumps(bundle_payload("a")))
        with pytest.raises(BudgetExceeded):
            client.complete("self_verify", "sys", json.dumps(bundle_payload("b")))

    def test_retries_then_success(self):
        class Flaky(LLMProvider):
            name = "flaky"

            def __init__(self):
                self.attempts = 0

            def complete(self, **kwargs):
                self.attempts += 1
                if self.attempts < 3:
                    raise TransientLLMError("boom")
                return LLMResponse("{}", 1, 1, "m")

        provider = Flaky()
        client = self.client(provider=provider)
        response = client.complete("t", "s", "u")
        assert response.text == "{}"
        assert client.stats["retries"] == 2
        assert provider.attempts == 3

    def test_retries_exhausted(self):
        class AlwaysFail(LLMProvider):
            name = "fail"

            def complete(self, **kwargs):
                raise LLMError("down")

        client = self.client(provider=AlwaysFail())
        with pytest.raises(LLMError):
            client.complete("t", "s", "u")
        assert client.stats["failures"] == 1

    def test_model_tiers(self):
        client = self.client(model="big", model_cheap="small")
        assert client.model_for("default") == "big"
        assert client.model_for("cheap") == "small"


class TestRealProviderParsing:
    def test_openai_parses_usage(self):
        def handler(request):
            assert request.headers["authorization"] == "Bearer sk-test"
            return httpx.Response(200, json={
                "choices": [{"message": {"content": "hello"}}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 3}})

        provider = OpenAIProvider("sk-test", transport=httpx.MockTransport(handler))
        response = provider.complete(task="t", system="s", user="u",
                                     model="gpt-x", max_tokens=10)
        assert (response.text, response.tokens_in, response.tokens_out) == ("hello", 7, 3)

    def test_openai_http_error_raises_llmerror(self):
        provider = OpenAIProvider("sk-test", transport=httpx.MockTransport(
            lambda request: httpx.Response(500, json={"error": "boom"})))
        with pytest.raises(LLMError):
            provider.complete(task="t", system="s", user="u", model="m", max_tokens=1)

    def test_anthropic_parses_usage(self):
        def handler(request):
            assert request.headers["x-api-key"] == "sk-ant"
            return httpx.Response(200, json={
                "content": [{"text": "yo"}],
                "usage": {"input_tokens": 5, "output_tokens": 2}})

        provider = AnthropicProvider("sk-ant", transport=httpx.MockTransport(handler))
        response = provider.complete(task="t", system="s", user="u",
                                     model="claude-x", max_tokens=10)
        assert (response.text, response.tokens_in, response.tokens_out) == ("yo", 5, 2)


@pytest.mark.parametrize("provider_name,stop", [("openai", "length"), ("anthropic", "max_tokens")])
@pytest.mark.parametrize("text", ['{"node_id":"service:x"}', '{"node_id":'])
def test_output_limit_stop_is_rejected_and_charged(provider_name, stop, text):
    def handler(request):
        if provider_name == "openai":
            body = {"choices": [{"message": {"content": text}, "finish_reason": stop}],
                    "usage": {"prompt_tokens": 7, "completion_tokens": 3}}
        else:
            body = {"content": [{"text": text}], "stop_reason": stop,
                    "usage": {"input_tokens": 7, "output_tokens": 3}}
        return httpx.Response(200, json=body)
    transport = httpx.MockTransport(handler)
    provider = (OpenAIProvider("test", transport=transport) if provider_name == "openai"
                else AnthropicProvider("test", transport=transport))
    meter = BudgetMeter()
    client = LLMClient(meter, provider=provider, retries=2)
    for _ in range(2):
        with pytest.raises(llm.LLMTruncatedError, match="truncated"):
            client.complete("consolidate", "s", "u", max_tokens=10)
    assert meter.llm_calls == 2  # neither retried nor cached
    assert meter.tokens == 20 and client.stats["failures"] == 2
    assert meter.as_dict()["tokens_committed"] == 0


class TestPromptFiles:
    def test_all_prompts_load_and_declare_version(self):
        for name in ("self_verify", "hypothesis_batch", "consolidate",
                     "reflection", "critique", "report"):
            text = llm.load_prompt(name)
            assert "version 1" in text.lower() or "version 2" in text.lower()
            assert "json" in text.lower()
