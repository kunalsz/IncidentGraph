"""LLM provider abstraction for the P3-P6 agent loop (§4.3, L11).

Providers: `none` (explicitly unconfigured), `mock` (deterministic rule-based
generators implementing the same JSON output contracts as the prompts — the
default for tests, CI and offline demos), `openai` and `anthropic` (httpx,
no vendor SDK). Every call passes through `LLMClient`, which enforces the
BudgetMeter, caches by prompt fingerprint, and retries with backoff.

Mock token counts are synthetic character estimates. Real usage is known only
when both provider counts are present; missing usage is charged at reservation.
No compatible provider tokenizer is installed, so strict token mode is rejected.
"""

from __future__ import annotations

import hashlib
import json
import math
import inspect
import random
import email.utils
import os
import re
import threading
import time
from abc import ABC, abstractmethod

import httpx

from backend.app.engine.budget import BudgetMeter
from backend.app.envfile import load_env_file

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")


class LLMError(Exception):
    """Provider call failed after retries, or the response was unusable."""


class LLMTruncatedError(LLMError):
    """Provider stopped at its output limit; even parseable JSON is incomplete."""


class LLMUnavailable(LLMError):
    """No provider is configured (LLM_PROVIDER=none) or the task needs one."""


class TransientLLMError(LLMError):
    """Explicitly retryable failure."""


class RateLimitedError(TransientLLMError):
    """One outbound attempt was rate-limited."""
    def __init__(self, message="provider rate-limited", retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


def estimate_tokens(text: str) -> int:
    """Display-only character estimate, never an admission/tokenizer bound."""
    return max(1, len(text) // 4)


def load_prompt(name: str) -> str:
    """Load a versioned prompt file (agents/README surfaces these to judges)."""
    with open(os.path.join(PROMPTS_DIR, f"{name}.md"), encoding="utf-8") as fh:
        return fh.read()


class LLMResponse:
    def __init__(self, text: str, tokens_in: int, tokens_out: int, model: str,
                 cached: bool = False, usage_status: str = "known",
                 finish_reason: str | None = None):
        self.finish_reason = finish_reason
        self.usage_status = usage_status
        self.text = text
        self.tokens_in = tokens_in
        self.tokens_out = tokens_out
        self.model = model
        self.cached = cached

    def json(self) -> dict:
        """Strict finite object parsing; only a complete code fence is allowed."""
        if self.finish_reason in ("length", "max_tokens", "model_length"):
            raise LLMTruncatedError(f"response truncated: {self.finish_reason}")
        text = self.text.strip()
        fence = re.match(r"^```(?:json)?\s*(.*?)```\s*$", text, re.DOTALL)
        if fence:
            text = fence.group(1)
        def finite_float(value):
            result = float(value)
            if not math.isfinite(result):
                raise ValueError("nonfinite JSON number")
            return result

        def reject_constant(value):
            raise ValueError(f"nonfinite JSON constant: {value}")

        def parse(value):
            result = json.loads(value, parse_float=finite_float,
                                parse_constant=reject_constant)
            if not isinstance(result, dict):
                raise ValueError("response must be a JSON object")
            return result

        try:
            return parse(text)
        except (ValueError, TypeError) as exc:
            raise LLMError(f"invalid JSON response: {exc}") from exc



class LLMProvider(ABC):
    name = "base"

    @abstractmethod
    def complete(self, *, task: str, system: str, user: str,
                 model: str, max_tokens: int) -> LLMResponse:
        ...


class NoneProvider(LLMProvider):
    """Explicitly unconfigured: P3-P6 calls fail loudly, P0-P2 stay zero-LLM."""
    name = "none"

    def complete(self, **kwargs) -> LLMResponse:
        raise LLMUnavailable(
            "LLM_PROVIDER is 'none'; set LLM_PROVIDER=mock|openai|anthropic "
            "(with a key) to run the agent phases")


# ---------------------------------------------------------------------------
# Mock provider: deterministic generators implementing the prompt contracts
# ---------------------------------------------------------------------------

def _symptom_line(metric: dict) -> str:
    z = metric.get("z", 0.0)
    return f"{metric.get('metric', '?')} z={z:.1f} ({metric.get('direction', '?')})"


def _mock_self_verify(payload: dict) -> dict:
    bundle = payload.get("bundle", {})
    service_id = payload.get("service_id", bundle.get("service_id", ""))
    metrics = bundle.get("metrics", [])
    logs = bundle.get("logs", [])
    anomalies = [m for m in metrics if abs(m.get("z", 0.0)) >= 3.0]
    errors = [line for line in logs if line.get("is_error")]
    has_metric = bool(anomalies)
    has_logs = bool(errors)
    has_traces = bool(bundle.get("traces")) or any(
        m.get("modality") == "trace" for m in metrics)
    rule = ("cross_modal_consensus" if has_metric and has_logs
            else "single_modality" if has_metric or has_logs
            else "inference_only")
    confidence = {"cross_modal_consensus": 0.85, "single_modality": 0.6,
                  "inference_only": 0.35}[rule]
    strongest = max((abs(m.get("z", 0.0)) for m in metrics), default=0.0)
    if has_metric and strongest >= 10:
        confidence = min(confidence + 0.1, 0.95)
    symptoms = [_symptom_line(m) for m in
                sorted(anomalies, key=lambda m: -abs(m.get("z", 0.0)))[:3]]
    if errors:
        symptoms.append(f"{len(errors)} error logs"
                        f" ({errors[0].get('template_id', 'template')})")
    hypothesis = (f"{service_id}: {symptoms[0]}" if symptoms
                  else f"{service_id}: no significant anomaly in window")
    modalities = int(has_metric) + int(has_logs) + int(has_traces)
    return {
        "node_id": service_id, "is_abnormal": has_metric or has_logs,
        "has_error_logs": has_logs, "has_metric_anomaly": has_metric,
        "key_symptoms": symptoms[:3], "hypothesis": hypothesis,
        "confidence": confidence, "confidence_rule": rule,
        "completeness": round(modalities / 3, 2),
    }


def _mock_hypothesis_batch(payload: dict) -> dict:
    bundle = payload.get("bundle", {})
    service_id = payload.get("service_id", bundle.get("service_id", ""))
    metrics = sorted(bundle.get("metrics", []),
                     key=lambda m: -abs(m.get("z", 0.0)))
    hypotheses = []
    for metric in metrics[:5]:
        z = metric.get("z", 0.0)
        hypotheses.append({
            "text": f"{metric.get('metric')} anomaly on {service_id} "
                    f"(z={z:.1f}, {metric.get('direction', '?')}) explains the incident",
            "candidate": service_id,
            "evidence_refs": [metric["id"]] if metric.get("id") else [],
            "initial_score": min(1.0, abs(z) / 20),
        })
    errors = [line for line in bundle.get("logs", []) if line.get("is_error")]
    if errors:
        hypotheses.append({
            "text": f"error template '{errors[0].get('template_id')}' on "
                    f"{service_id} ({len(errors)} occurrences) marks the failure",
            "candidate": service_id,
            "evidence_refs": [line["id"] for line in errors[:3] if line.get("id")],
            "initial_score": min(1.0, 0.3 + len(errors) / 50),
        })
    for neighbor in bundle.get("adjacency", [])[:3]:
        hypotheses.append({
            "text": f"anomaly originates in {neighbor.get('service')} and "
                    f"propagates to {service_id}",
            "candidate": neighbor.get("service", ""),
            "evidence_refs": [],
            "initial_score": 0.2 if not neighbor.get("evidence_ids") else 0.4,
        })
    limit = max(0, int(payload.get("max_hypotheses", 8)))
    hypotheses = [h for h in hypotheses if h["candidate"] == service_id]
    return {"hypotheses": hypotheses[:limit]}


def _mock_consolidate(payload: dict) -> dict:
    verdict = payload.get("verdict", {})
    children = payload.get("children", [])
    node_id = payload.get("node_id", verdict.get("node_id", ""))
    best_child = max(children, key=lambda c: c.get("confidence", 0.0), default=None)
    if verdict.get("is_abnormal"):
        chosen, confidence = node_id, verdict.get("confidence", 0.0)
        rule = verdict.get("confidence_rule", "inference_only")
        evidence = list(verdict.get("evidence_ids", []))
        reason = f"{node_id} is abnormal in-window: {verdict.get('hypothesis', '')}"
        status = "closed"
    elif best_child and best_child.get("confidence", 0.0) >= 0.35:
        chosen = best_child.get("node_id")
        confidence = round(0.9 * best_child.get("confidence", 0.0), 3)
        rule = best_child.get("confidence_rule", "inference_only")
        evidence = list(best_child.get("evidence_ids", []))
        reason = f"{node_id} looks normal; strongest abnormal child is {chosen}"
        status = "closed"
    else:
        chosen, confidence, rule, evidence = None, 0.0, "inference_only", []
        reason = f"{node_id} and its children show no decisive anomaly"
        status = "partial"
    return {
        "node_id": node_id, "local_root_cause": chosen, "reason": reason,
        "confidence": confidence, "confidence_rule": rule,
        "evidence_ids": sorted(set(evidence)), "contradictions": [],
        "status": status,
    }


def _mock_reflection(payload: dict) -> dict:
    bundle = payload.get("bundle", {})
    hypothesis = payload.get("hypothesis", {})
    metrics = bundle.get("metrics", [])
    anomalies = [m for m in metrics if abs(m.get("z", 0.0)) >= 3.0]
    refs = hypothesis.get("evidence_refs") or hypothesis.get("evidence") or []
    known = {m.get("id") for m in metrics} | {
        line.get("id") for line in bundle.get("logs", [])}
    valid_refs = [r for r in refs if r in known]
    e = 3 if not refs else min(10, 4 + 2 * len(valid_refs))
    modalities = int(bool(metrics)) + int(bool(bundle.get("logs"))) + int(
        bool(bundle.get("traces")) or any(m.get("modality") == "trace" for m in metrics))
    c_comp = round(10 * modalities / 3)
    directions = {m.get("direction") for m in anomalies}
    k = 8 if len(directions) <= 1 else 6
    return {"e": e, "c_comp": c_comp, "k": k,
            "reason": f"{len(valid_refs)}/{len(refs)} refs verified; "
                      f"{modalities} modalities; {len(directions)} directions"}


def _mock_escalate(payload: dict) -> dict:
    """One-hypothesis escalation hop over the frozen handoff (escalate.md)."""
    confidence = float(payload.get("confidence", 0.0))
    completeness = float(payload.get("completeness", 0.0))
    available = payload.get("available_kinds") or []
    missing = payload.get("missing_kind", "another modality")
    if available:
        confidence = min(0.9, confidence + 0.15)
        completeness = min(1.0, completeness + 0.33)
        summary = (f"{payload.get('summary', '')} — complementary {missing} "
                   f"evidence found in {', '.join(sorted(available))}").strip()
    else:
        summary = (f"{payload.get('summary', '')} — no {missing} telemetry "
                   f"available for this window").strip()
    return {
        "summary": summary, "confidence": round(confidence, 3),
        "completeness": round(completeness, 3),
        "evidence_count": int(payload.get("evidence_count", 0)),
        "escalate": False,
    }


def _mock_critique(payload: dict) -> dict:
    """Adjudicate, never overrule the checks (critique.md)."""
    checks = payload.get("checks", [])
    verdicts = {check.get("verdict") for check in checks}
    path = payload.get("path", {})
    if "fail" in verdicts:
        verdict, cap = "fail", 0.4
    elif "partial" in verdicts:
        verdict, cap = "partial", None
    elif not path.get("found"):
        verdict, cap = "partial", None  # a claim without a path cannot pass
    else:
        verdict, cap = "pass", None
    return {"verdict": verdict,
            "reason": f"{len(checks)} deterministic checks: {sorted(verdicts)}; "
                      f"path_found={bool(path.get('found'))}",
            "confidence_cap": cap}


def _mock_report(payload: dict) -> dict:
    """Three paragraphs; every [n] binds to an evidence-path entry (report.md)."""
    path = payload.get("evidence_path") or {}
    nodes = list(path.get("nodes") or [])
    number = {node["id"]: index + 1 for index, node in enumerate(nodes)}
    verdict = payload.get("verdict") or {}
    candidate = verdict.get("candidate") or payload.get("winner_candidate") or "unknown"
    symptom = payload.get("symptom") or "degraded service"
    window = payload.get("window") or {}
    budget = payload.get("budget") or {}
    alternatives = payload.get("alternatives") or []
    ruled_out = payload.get("ruled_out") or []
    not_checked = payload.get("not_checked") or []
    chain_ids = [node["id"] for node in nodes if node.get("role") != "corroborating"]
    chain = " -> ".join(f"{node_id} [{number[node_id]}]" for node_id in chain_ids)
    corroborating = sum(1 for node in nodes if node.get("role") == "corroborating")
    candidate_cite = (f" [{number[candidate]}]" if candidate in number else "")

    first = (f"Between {window.get('t0')} and {window.get('t1')} the platform "
             f"reported: {symptom}.")
    second = (f"The leading root cause is {candidate}{candidate_cite}. "
              f"Evidence path: {chain or 'none'} "
              f"({corroborating} corroborating evidence node(s), "
              f"validator confidence {verdict.get('confidence', 0):.2f} "
              f"via {verdict.get('confidence_rule', 'inference_only')}).")
    third = (f"Impact and timeline: {len(alternatives)} alternative hypothesis(es) "
             f"retained above the gate, {len(ruled_out)} ruled out, "
             f"{len(not_checked)} not checked; investigation cost "
             f"{budget.get('llm_calls', 0)} LLM calls / "
             f"{budget.get('tokens_total', 0)} tokens.")

    runbook = payload.get("runbook") or {}
    steps = [step for entry in (runbook.get("runbooks") or [])
             for step in (entry.get("steps") or [])]
    if steps:
        first_step = steps[0]
        next_action = (f"{first_step.get('action', '')} "
                       f"(validate: {first_step.get('validate_cmd', 'n/a')})")
    else:
        next_action = "Collect the missing evidence named in the gaps, then re-run."
    unanswered = [f"not checked: {item.get('node_id') or item.get('id') or item}"
                  for item in not_checked]
    return {"summary": [first, second, third], "next_action": next_action,
            "unanswered_questions": unanswered}


_MOCK_GENERATORS = {
    "self_verify": _mock_self_verify,
    "hypothesis_batch": _mock_hypothesis_batch,
    "consolidate": _mock_consolidate,
    "reflection": _mock_reflection,
    "escalate": _mock_escalate,
    "critique": _mock_critique,
    "report": _mock_report,
}


class MockProvider(LLMProvider):
    """Deterministic rule-based provider: valid typed outputs, zero network.

    Implements the same output contracts as the prompt files, so the whole
    P3-P6 pipeline runs, demos and tests without an API key. `overrides`
    (task -> raw response text) forces exact outputs for crafted test cases.
    """

    name = "mock"

    def __init__(self, overrides: dict[str, str] | None = None):
        self.overrides = overrides or {}

    def complete(self, *, task: str, system: str, user: str,
                 model: str, max_tokens: int) -> LLMResponse:
        tokens_in = estimate_tokens(system + user)
        if task in self.overrides:
            text = self.overrides[task]
        else:
            generator = _MOCK_GENERATORS.get(task)
            if generator is None:
                # Unknown task: echo a typed stub the caller can detect.
                text = json.dumps({"task": task, "note": "mock provider stub"})
            else:
                payload = json.loads(user)
                text = json.dumps(generator(payload), sort_keys=True)
        return LLMResponse(text, tokens_in, estimate_tokens(text), model, usage_status="estimated")


# ---------------------------------------------------------------------------
# Real providers (httpx; no vendor SDKs). Multi-key round-robin with 429
# failover: each call starts at the next key in rotation, and a rate-limited
# key is retried with the following one before any error surfaces. Key
# material is only ever sent in headers — never into URLs or log lines.
# ---------------------------------------------------------------------------

class _KeyRateLimited(RateLimitedError):
    """This key is rate-limited; try the next one in the rotation."""


def _env_keys(*names: str) -> list[str]:
    """First non-empty env var among `names`, comma-separated -> key list."""
    for name in names:
        raw = os.environ.get(name)
        if raw:
            return [key.strip() for key in raw.split(",") if key.strip()]
    return []


class KeyRotator:
    """Round-robin rotation with failover order (thread-safe)."""

    def __init__(self, keys: list[str]):
        keys = [key.strip() for key in keys if key and key.strip()]
        if not keys:
            raise LLMUnavailable("no API keys configured")
        self._keys = list(keys)
        self._index = 0
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._keys)

    def order(self) -> list[str]:
        """Keys starting at the next rotation slot (advances round-robin)."""
        with self._lock:
            start = self._index % len(self._keys)
            self._index += 1
        return self._keys[start:] + self._keys[:start]


class _MultiKeyProvider(LLMProvider):
    """Shared failover loop: try keys in rotation order, one attempt each."""

    def _attempt(self, key: str, *, task: str, system: str, user: str,
                 model: str, max_tokens: int) -> LLMResponse:
        raise NotImplementedError

    def complete(self, *, task: str, system: str, user: str,
                 model: str, max_tokens: int, timeout: float = 90) -> LLMResponse:
        # Exactly one outbound attempt, including one key selection.
        return self._attempt(self._rotator.order()[0], task=task, system=system,
                             user=user, model=model, max_tokens=max_tokens, timeout=timeout)



def _raise_provider_status(response, name):
    status = response.status_code
    if status == 429:
        raw = response.headers.get("retry-after")
        try:
            delay = max(0.0, float(raw)) if raw else None
        except ValueError:
            try:
                delay = max(0.0, email.utils.parsedate_to_datetime(raw).timestamp() - time.time())
            except (ValueError, TypeError, OverflowError):
                delay = None
        raise RateLimitedError(f"{name} rate-limited", retry_after=delay)
    if status in (408, 409, 425) or status >= 500:
        raise TransientLLMError(f"{name} call failed: transient HTTP {status}")
    if status >= 400:
        raise LLMError(f"{name} request rejected: HTTP {status}")


class OpenAICompatibleProvider(_MultiKeyProvider):
    """Any OpenAI chat-completions endpoint (OpenAI, River AI, gateways...).

    `base_url` comes from the constructor, else `LLM_BASE_URL`, else the
    OpenAI default — this is the one live-provider shape the project uses.
    """

    name = "openai"
    BASE_URL = "https://api.openai.com/v1"
    DEFAULT_MODEL = "gpt-4o-mini"

    def __init__(self, keys: str | list[str],
                 transport: httpx.BaseTransport | None = None,
                 base_url: str | None = None):
        self._rotator = KeyRotator(keys if isinstance(keys, list) else [keys])
        self.transport = transport
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL")
                         or self.BASE_URL).rstrip("/")
        from urllib.parse import urlsplit
        url = urlsplit(self.base_url)
        if (url.scheme not in ("http", "https") or not url.hostname
                or url.username or url.password or url.query or url.fragment):
            raise ValueError("invalid provider base URL (http(s), no credentials/query/fragment)")

    def _attempt(self, key: str, *, task: str, system: str, user: str,
                 model: str, max_tokens: int, timeout: float = 90) -> LLMResponse:
        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
        }
        url = f"{self.base_url}/chat/completions"
        try:
            with httpx.Client(transport=self.transport, timeout=timeout) as client:
                response = client.post(
                    url, json=payload,
                    headers={"Authorization": f"Bearer {key}"})
                _raise_provider_status(response, self.name)
                body = response.json()
        except httpx.TransportError as exc:
            raise TransientLLMError(f"{self.name} transport failure") from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"{self.name} request failed") from exc
        except ValueError as exc:
            raise LLMError(f"{self.name} returned non-JSON: {exc}") from exc
        try:
            usage = body.get("usage") or {}
            ti, to = usage.get("prompt_tokens"), usage.get("completion_tokens")
            known = all(type(v) is int and v >= 0 for v in (ti, to))
            text = body["choices"][0]["message"]["content"]
            if not isinstance(text, str):
                raise ValueError("content must be text")
            return LLMResponse(text, ti if known else 0, to if known else 0,
                               body.get("model") or model,
                               usage_status="known" if known else "unknown",
                               finish_reason=body["choices"][0].get("finish_reason"))
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise LLMError("malformed provider envelope") from exc



OpenAIProvider = OpenAICompatibleProvider


class AnthropicProvider(LLMProvider):
    name = "anthropic"
    URL = "https://api.anthropic.com/v1/messages"
    DEFAULT_MODEL = "claude-3-5-haiku-latest"

    def __init__(self, api_key: str, transport: httpx.BaseTransport | None = None):
        self.api_key = api_key
        self.transport = transport

    def complete(self, *, task: str, system: str, user: str,
                 model: str, max_tokens: int, timeout: float = 90) -> LLMResponse:
        payload = {
            "model": model, "max_tokens": max_tokens,
            "system": system, "messages": [{"role": "user", "content": user}],
        }
        try:
            with httpx.Client(transport=self.transport, timeout=timeout) as client:
                response = client.post(
                    self.URL, json=payload,
                    headers={"x-api-key": self.api_key,
                             "anthropic-version": "2023-06-01"})
                _raise_provider_status(response, self.name)
                body = response.json()
        except httpx.TransportError as exc:
            raise TransientLLMError("anthropic transport failure") from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise LLMError("anthropic invalid response") from exc
        try:
            usage = body.get("usage") or {}
            ti, to = usage.get("input_tokens"), usage.get("output_tokens")
            known = all(type(v) is int and v >= 0 for v in (ti, to))
            text = body["content"][0]["text"]
            if not isinstance(text, str):
                raise ValueError("content must be text")
            return LLMResponse(text, ti if known else 0, to if known else 0,
                               body.get("model") or model,
                               usage_status="known" if known else "unknown",
                               finish_reason=body.get("stop_reason"))
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise LLMError("malformed provider envelope") from exc



def provider_from_env() -> LLMProvider:
    """Build the provider named by LLM_PROVIDER (default: none).

    `openai` is the generic OpenAI-compatible shape: point it at any gateway
    with `LLM_BASE_URL` (e.g. https://api.river.ai/v1).
    """
    load_env_file()
    name = os.environ.get("LLM_PROVIDER", "none").strip().lower()
    if name in ("", "none"):
        return NoneProvider()
    if name == "mock":
        return MockProvider()
    if name in ("openai", "openai-compatible"):
        keys = _env_keys("LLM_API_KEYS", "LLM_API_KEY", "OPENAI_API_KEYS",
                         "OPENAI_API_KEY")
        if not keys:
            raise LLMUnavailable("LLM_PROVIDER=openai but no LLM_API_KEY set")
        return OpenAIProvider(keys)
    if name == "anthropic":
        keys = _env_keys("ANTHROPIC_API_KEYS", "ANTHROPIC_API_KEY", "LLM_API_KEYS",
                         "LLM_API_KEY")
        if not keys:
            raise LLMUnavailable("LLM_PROVIDER=anthropic but no ANTHROPIC_API_KEY(S) set")
        return AnthropicProvider(keys[0])
    raise LLMUnavailable(f"unknown LLM_PROVIDER {name!r}")


# ---------------------------------------------------------------------------
# Client: budget + cache + retries
# ---------------------------------------------------------------------------

class LLMClient:
    """One classified retry loop; cache single-flight is scoped to this client.

    Sync transports enforce inactivity timeouts only. Unknown provider tokenizers
    explicitly disable strict accounting. No thread timeout is called hard.
    """
    def __init__(self, meter: BudgetMeter, provider: LLMProvider | None = None,
                 model: str | None = None, model_cheap: str | None = None,
                 retries: int | None = None, sleep=time.sleep,
                 rate_limit_backoff: float | None = None, jitter=lambda: 0.0):
        self.meter = meter
        self.provider = provider if provider is not None else provider_from_env()
        self.model = model or os.environ.get("LLM_MODEL") or getattr(self.provider, "DEFAULT_MODEL", "gpt-4o-mini")
        self.model_cheap = model_cheap or os.environ.get("LLM_MODEL_CHEAP") or self.model
        self.retries = retries if retries is not None else int(os.environ.get("LLM_MAX_RETRIES", "2"))
        if type(self.retries) is not int or self.retries < 0:
            raise ValueError("retries must be nonnegative")
        self._sleep, self._jitter = sleep, jitter
        self.rate_limit_backoff = rate_limit_backoff if rate_limit_backoff is not None else float(os.environ.get("LLM_RATE_LIMIT_BACKOFF_S", "5"))
        if (type(self.rate_limit_backoff) not in (int, float)
                or not math.isfinite(self.rate_limit_backoff) or self.rate_limit_backoff < 0):
            raise ValueError("invalid rate limit backoff")
        if any(not isinstance(value, str) or not value.strip()
               for value in (self.model, self.model_cheap)):
            raise ValueError("provider model names must be nonempty strings")
        self.capabilities = {"provider_input_token_bound": False,
            "provider_output_token_cap_verified": False,
            "provider_usage_complete_guaranteed": False,
            "local_process_cancellation": os.name == "posix"}
        if meter.budget.strict_tokens:
            raise ValueError("strict token mode unsupported: requires compatible input tokenizer, "
                             "verified provider output cap, and complete accounting")
        import dataclasses
        self.effective_config = {"provider": self.provider.name, "model": self.model,
            "model_cheap": self.model_cheap, "retries": self.retries,
            "rate_limit_backoff_s": self.rate_limit_backoff,
            "budget": dataclasses.asdict(meter.budget), "capabilities": self.capabilities}
        self.config_fingerprint = hashlib.sha256(json.dumps(self.effective_config,
            sort_keys=True).encode()).hexdigest()
        self._cache = {}
        self._condition = threading.Condition(threading.RLock())
        self._inflight = set()
        self._cooldown_until = 0.0
        self.stats = {"calls": 0, "cache_hits": 0, "retries": 0, "failures": 0, "rate_limited": 0}

    def model_for(self, tier):
        return self.model_cheap if tier == "cheap" else self.model

    def _wait(self, delay, phase):
        from backend.app.engine.budget import BudgetExceeded
        if delay >= self.meter.remaining_seconds(phase):
            raise BudgetExceeded("retry_deadline", delay, self.meter.remaining_seconds(phase))
        if delay > 0:
            self._sleep(delay)
        self.meter.check(phase)

    def complete(self, task, system, user, *, tier="default", max_tokens=1500):
        from backend.app.agent.responses import SCHEMA_VERSION, TASK_SCHEMAS, validate_task
        if type(max_tokens) is not int or max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        phase = "report" if task == "report" else "work"
        logical = self.meter.begin_logical()
        model = self.model_for(tier)
        fingerprint = hashlib.sha256(json.dumps(
            [self.provider.name, model, task, system, user, max_tokens, SCHEMA_VERSION],
            ensure_ascii=False).encode()).hexdigest()
        with self._condition:
            while fingerprint in self._inflight:
                self.meter.check(phase)
                self._condition.wait(timeout=min(0.05, self.meter.remaining_seconds(phase)))
            cached = self._cache.get(fingerprint)
            if cached is not None:
                self.meter.record_cache_hit()
                self.stats["cache_hits"] += 1
                return LLMResponse(cached.text, cached.tokens_in, cached.tokens_out,
                                   cached.model, cached=True, usage_status=cached.usage_status,
                                   finish_reason=cached.finish_reason)
            self._inflight.add(fingerprint)
        try:
            for attempt in range(self.retries + 1):
                # UTF-8 bytes + framing allowance is a conservative planning
                # commitment, NOT a proven provider tokenizer/billing bound.
                self._wait(max(0.0, self._cooldown_until - time.monotonic()), phase)
                lease = self.meter.reserve_attempt(logical, phase,
                    len(system.encode()) + len(user.encode()) + 256, max_tokens, strict=False)
                if attempt:
                    self.stats["retries"] += 1
                try:
                    kwargs = dict(task=task, system=system, user=user, model=model, max_tokens=max_tokens)
                    if "timeout" in inspect.signature(self.provider.complete).parameters:
                        kwargs["timeout"] = max(0.001, self.meter.remaining_seconds(phase))
                    response = self.provider.complete(**kwargs)
                except BaseException as exc:
                    self.meter.settle_attempt(lease, "error", usage_status="unknown")
                    if isinstance(exc, RateLimitedError):
                        self.stats["rate_limited"] += 1
                    if not isinstance(exc, TransientLLMError) or attempt >= self.retries:
                        raise
                    base = self.rate_limit_backoff if isinstance(exc, RateLimitedError) else 0.5
                    delay = base * 2 ** attempt + max(0.0, min(1.0, self._jitter()))
                    if isinstance(exc, RateLimitedError) and exc.retry_after is not None:
                        delay = max(delay, exc.retry_after)
                    if isinstance(exc, RateLimitedError):
                        self._cooldown_until = time.monotonic() + delay
                    self._wait(delay, phase)
                    # The injected test sleep may not advance the wall clock.
                    self._cooldown_until = 0.0
                    continue
                self.meter.settle_attempt(lease, "success",
                    (response.tokens_in, response.tokens_out), response.usage_status)
                try:
                    if response.finish_reason in ("length", "max_tokens", "model_length"):
                        raise LLMTruncatedError(f"{task} response truncated: {response.finish_reason}")
                    if task in TASK_SCHEMAS:
                        validate_task(task, response.json())
                except (ValueError, TypeError) as exc:
                    raise LLMError(f"invalid {task} response schema") from exc
                self.meter.check(phase)
                with self._condition:
                    self._cache[fingerprint] = LLMResponse(response.text, response.tokens_in,
                        response.tokens_out, response.model, usage_status=response.usage_status,
                        finish_reason=response.finish_reason)
                    self.stats["calls"] += 1
                return response
        except BaseException:
            with self._condition:
                self.stats["failures"] += 1
            raise
        finally:
            with self._condition:
                self._inflight.discard(fingerprint)
                self._condition.notify_all()
