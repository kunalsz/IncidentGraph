"""Unit tests: multi-key rotation, 429 failover, the OpenAI-compatible
gateway provider, rate-limit backoff, and the .env loader — all offline via
httpx.MockTransport / tmp files."""

import os

import httpx
import pytest

from backend.app.agent.llm import (KeyRotator, LLMClient, LLMError,
                                   LLMProvider, LLMUnavailable, MockProvider,
                                   OpenAIProvider, RateLimitedError, _env_keys,
                                   provider_from_env)
from backend.app.engine.budget import BudgetMeter
from backend.app.envfile import load_env_file

BASE = "https://gateway.test/v1"


def ok_response(text="ok", tokens_in=3, tokens_out=2):
    return httpx.Response(200, json={
        "choices": [{"message": {"content": text}}],
        "usage": {"prompt_tokens": tokens_in, "completion_tokens": tokens_out}})


class TestKeyRotator:
    def test_round_robin_order(self):
        rotator = KeyRotator(["k1", "k2", "k3"])
        assert rotator.order() == ["k1", "k2", "k3"]
        assert rotator.order() == ["k2", "k3", "k1"]
        assert rotator.order() == ["k3", "k1", "k2"]
        assert len(rotator) == 3

    def test_empty_keys_raise(self):
        with pytest.raises(LLMUnavailable):
            KeyRotator([])
        with pytest.raises(LLMUnavailable):
            KeyRotator(["", " "])

    def test_filters_blank_entries(self):
        assert len(KeyRotator(["k1", "", "k2"])) == 2


class TestEnvKeys:
    def test_comma_separated_with_whitespace(self, monkeypatch):
        monkeypatch.setenv("LLM_API_KEYS", " a , b ,, c ")
        assert _env_keys("LLM_API_KEYS") == ["a", "b", "c"]

    def test_first_non_empty_name_wins(self, monkeypatch):
        monkeypatch.delenv("LLM_API_KEYS", raising=False)
        monkeypatch.setenv("LLM_API_KEY", "single")
        assert _env_keys("LLM_API_KEYS", "LLM_API_KEY") == ["single"]

    def test_absent_returns_empty(self, monkeypatch):
        monkeypatch.delenv("NOPE_API_KEYS", raising=False)
        assert _env_keys("NOPE_API_KEYS") == []


class TestGatewayFailover:
    def test_429_falls_over_to_next_key(self):
        seen = []

        def handler(request):
            seen.append(request.headers["authorization"])
            if len(seen) == 1:
                return httpx.Response(429, json={"error": "rate limited"})
            return ok_response("pong")

        provider = OpenAIProvider(["k1", "k2"],
                                  transport=httpx.MockTransport(handler),
                                  base_url=BASE)
        meter = BudgetMeter()
        response = LLMClient(meter, provider=provider, sleep=lambda _: None).complete(
            "t", "s", "u", max_tokens=5)
        assert meter.llm_calls == 2  # v2 rotations are metered attempts
        assert response.text == "pong"
        assert seen == ["Bearer k1", "Bearer k2"]

    def test_rotation_advances_between_calls(self):
        seen = []

        def handler(request):
            seen.append(request.headers["authorization"])
            return ok_response()

        provider = OpenAIProvider(["k1", "k2"],
                                  transport=httpx.MockTransport(handler),
                                  base_url=BASE)
        provider.complete(task="t", system="s", user="u", model="m", max_tokens=5)
        provider.complete(task="t", system="s", user="u", model="m", max_tokens=5)
        assert seen == ["Bearer k1", "Bearer k2"]

    def test_all_keys_rate_limited_raises_typed_error(self):
        calls = []

        def handler(request):
            calls.append(request)
            return httpx.Response(429, json={})

        provider = OpenAIProvider(["k1", "k2"],
                                  transport=httpx.MockTransport(handler),
                                  base_url=BASE)
        with pytest.raises(RateLimitedError, match="rate-limited"):
            provider.complete(task="t", system="s", user="u", model="m", max_tokens=5)
        assert len(calls) == 1  # v2 provider methods dispatch exactly once

    def test_non_429_error_does_not_failover(self):
        calls = []

        def handler(request):
            calls.append(request)
            return httpx.Response(500, json={"error": "boom"})

        provider = OpenAIProvider(["k1", "k2"],
                                  transport=httpx.MockTransport(handler),
                                  base_url=BASE)
        with pytest.raises(LLMError, match="call failed"):
            provider.complete(task="t", system="s", user="u", model="m", max_tokens=5)
        assert len(calls) == 1  # a different key cannot fix a server error

    def test_requests_go_to_the_configured_base_url(self):
        seen = {}

        def handler(request):
            seen["url"] = str(request.url)
            return ok_response()

        provider = OpenAIProvider(["k"], transport=httpx.MockTransport(handler),
                                  base_url=BASE)
        provider.complete(task="t", system="s", user="u", model="m", max_tokens=5)
        assert seen["url"] == f"{BASE}/chat/completions"


class TestProviderSelection:
    def test_openai_compatible_from_env(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("LLM_API_KEY", "rv_test_key")
        monkeypatch.setenv("LLM_BASE_URL", "https://api.river.ai/v1")
        provider = provider_from_env()
        assert provider.name == "openai"
        assert provider.base_url == "https://api.river.ai/v1"

    def test_single_key_becomes_one_key_rotator(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("LLM_API_KEY", "rv_test_key")
        assert len(provider_from_env()._rotator) == 1

    def test_openai_without_key_raises(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        for name in ("LLM_API_KEYS", "LLM_API_KEY", "OPENAI_API_KEYS",
                     "OPENAI_API_KEY"):
            monkeypatch.delenv(name, raising=False)
        with pytest.raises(LLMUnavailable):
            provider_from_env()


class TestClientModelDefaults:
    def test_provider_default_used_when_env_blank(self, monkeypatch):
        monkeypatch.setenv("LLM_MODEL", "")
        monkeypatch.delenv("LLM_MODEL_CHEAP", raising=False)
        client = LLMClient(BudgetMeter(), provider=OpenAIProvider(["k"]))
        assert client.model == "gpt-4o-mini"  # generic default
        mock = LLMClient(BudgetMeter(), provider=MockProvider())
        assert mock.model  # falls back to the generic default as well

    def test_env_model_wins(self, monkeypatch):
        monkeypatch.setenv("LLM_MODEL", "deepseek-ai/DeepSeek-V4.1-Flash")
        client = LLMClient(BudgetMeter(), provider=OpenAIProvider(["k"]))
        assert client.model == "deepseek-ai/DeepSeek-V4.1-Flash"


class TestEnvFile:
    def test_loads_and_never_overrides(self, tmp_path, monkeypatch):
        env = tmp_path / ".env"
        env.write_text("# comment\nFOO_TEST_KEY=abc\nBAR_TEST_KEY='x y'\n\nQUOTED=\"q\"\n")
        monkeypatch.delenv("FOO_TEST_KEY", raising=False)
        monkeypatch.delenv("QUOTED", raising=False)
        monkeypatch.setenv("BAR_TEST_KEY", "already")
        try:
            load_env_file(str(env))
            assert os.environ["FOO_TEST_KEY"] == "abc"
            assert os.environ["BAR_TEST_KEY"] == "already"  # not overridden
            assert os.environ["QUOTED"] == "q"
        finally:
            os.environ.pop("FOO_TEST_KEY", None)
            os.environ.pop("QUOTED", None)


class TestRateLimitBackoff:
    def test_long_backoff_and_rate_limit_stat(self):
        class AlwaysLimited(LLMProvider):
            name = "limited"

            def complete(self, **kwargs):
                raise RateLimitedError("busy")

        sleeps = []
        client = LLMClient(BudgetMeter(), provider=AlwaysLimited(), retries=2,
                           sleep=sleeps.append, rate_limit_backoff=5.0)
        with pytest.raises(RateLimitedError):
            client.complete("t", "s", "u")
        assert sleeps == [5.0, 10.0]  # waits out the window, then doubles
        assert client.stats["rate_limited"] == 3

    def test_plain_error_keeps_short_backoff(self):
        class Broken(LLMProvider):
            name = "broken"

            def complete(self, **kwargs):
                raise LLMError("boom")

        sleeps = []
        client = LLMClient(BudgetMeter(), provider=Broken(), retries=2,
                           sleep=sleeps.append)
        with pytest.raises(LLMError):
            client.complete("t", "s", "u")
        assert sleeps == []  # v2 unclassified/schema/auth errors are not retryable


class TestPoolSizing:
    def test_real_providers_get_gentle_pool(self, monkeypatch):
        from types import SimpleNamespace
        from backend.app.engine.p3_investigate import _max_parallel
        from config import tuning

        monkeypatch.delenv("P3_MAX_PARALLEL", raising=False)
        mock_llm = SimpleNamespace(provider=SimpleNamespace(name="mock"))
        real_llm = SimpleNamespace(provider=SimpleNamespace(name="openai"))
        assert _max_parallel(mock_llm) == tuning.MAX_AGENT_PARALLEL
        assert _max_parallel(real_llm) == 3

    def test_env_override_wins(self, monkeypatch):
        from types import SimpleNamespace
        from backend.app.engine.p3_investigate import _max_parallel

        monkeypatch.setenv("P3_MAX_PARALLEL", "1")
        real_llm = SimpleNamespace(provider=SimpleNamespace(name="openai"))
        assert _max_parallel(real_llm) == 1
