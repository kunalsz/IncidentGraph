import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parent
for p in (str(ROOT), str(ROOT / "backend")):
    if p not in sys.path:
        sys.path.insert(0, p)


@pytest.fixture(autouse=True)
def _forbid_real_http(monkeypatch):
    """No test may make a real network call — ever.

    `httpx.MockTransport` (used by the provider tests) overrides
    `handle_request` on its own instance, so it is unaffected; anything using
    a default transport (a REAL call) hits this guard and fails loudly.
    FalkorDB traffic uses the redis socket, not httpx, and is allowed.
    """
    def guard(self, request):
        raise AssertionError(
            f"test attempted a real HTTP call: {request.method} {request.url} "
            "— use MockTransport or MockProvider instead")
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", guard)
