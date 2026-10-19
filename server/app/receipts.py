"""Per-call receipts.

Wraps a provider so each model call becomes an observable event for the UI:
which task ran, how long it took, what it cost, and whether it failed. The
wrapper is a pure passthrough — it changes no argument and no return value — so
the engine behaves exactly as it does without it.
"""
from __future__ import annotations

import inspect
import logging
import time
from typing import Any, Callable

logger = logging.getLogger(__name__)


class ReportingProvider:
    """Provider proxy that reports each call. Signature-compatible with the real
    providers (the client inspects ``complete`` for a ``timeout`` parameter)."""

    def __init__(self, inner: Any, on_call: Callable[[dict[str, Any]], None]) -> None:
        # object.__setattr__ keeps __getattr__ delegation from recursing.
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_on_call", on_call)
        object.__setattr__(
            self, "_forwards_timeout",
            "timeout" in inspect.signature(inner.complete).parameters,
        )

    def __getattr__(self, item: str) -> Any:
        return getattr(object.__getattribute__(self, "_inner"), item)

    def complete(self, *, task: str, system: str, user: str, model: str,
                 max_tokens: int, timeout: float | None = None,
                 **kwargs: Any) -> Any:
        inner = object.__getattribute__(self, "_inner")
        call_kwargs: dict[str, Any] = dict(task=task, system=system, user=user,
                                           model=model, max_tokens=max_tokens, **kwargs)
        if timeout is not None and object.__getattribute__(self, "_forwards_timeout"):
            call_kwargs["timeout"] = timeout
        started = time.monotonic()
        response = None
        status = "ok"
        try:
            response = inner.complete(**call_kwargs)
            return response
        except BaseException:
            status = "error"
            raise
        finally:
            try:
                object.__getattribute__(self, "_on_call")({
                    "task": task,
                    "ms": int((time.monotonic() - started) * 1000),
                    "tokens_in": getattr(response, "tokens_in", None),
                    "tokens_out": getattr(response, "tokens_out", None),
                    "model": getattr(response, "model", None) or model,
                    "usage": getattr(response, "usage_status", None),
                    "finish": getattr(response, "finish_reason", None),
                    "status": status,
                })
            except Exception:
                # A receipt must never affect the call it describes.
                logger.debug("could not record a call receipt", exc_info=True)


def reporting_provider(inner: Any, on_call: Callable[[dict[str, Any]], None]) -> Any:
    """Wrap when possible; fall back to the raw provider if that ever fails."""
    try:
        return ReportingProvider(inner, on_call)
    except Exception:
        return inner
