"""No network/DB: slow graph and streaming mock provider run in owned processes."""
import json
import multiprocessing
import os
import signal
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import httpx
import pytest

from backend.app.agent.llm import LLMClient, MockProvider, OpenAIProvider
from backend.app.engine.budget import BudgetMeter
from backend.app.engine.runner import run_investigation
from config.tuning import InvestigationBudget

pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX worker boundary")


def phases(p0=None, p1=None, p3=None):
    return (p0 or (lambda *a, **k: SimpleNamespace(incident_id="i", budget={})),
        p1 or (lambda *a, **k: SimpleNamespace(incident_id="i")),
        lambda *a, **k: SimpleNamespace(incident_id="i"),
        p3 or (lambda *a, **k: None), lambda *a, **k: None,
        lambda *a, **k: None, lambda *a, **k: None)


def isolated_injection_test(fn):
    """Exercise closure-based fork injection in a clean process, never skip it.

    The default configured runtime uses spawn and has no native-thread constraint.
    A whole test-suite interpreter can already contain NumPy/native pool threads.
    """
    import functools
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        from backend.app.engine.deadline import require_isolation_support
        try:
            require_isolation_support()
        except RuntimeError:
            env = dict(os.environ, INCIDENTGRAPH_CLEAN_WORKER_TEST="1")
            assert not os.environ.get("INCIDENTGRAPH_CLEAN_WORKER_TEST"), "clean subprocess unsafe"
            completed = subprocess.run([sys.executable, "-m", "pytest", "-q",
                __file__ + "::" + fn.__name__], env=env, capture_output=True, text=True, timeout=30)
            assert completed.returncode == 0, completed.stdout + completed.stderr
            return
        return fn(*args, **kwargs)
    return wrapped


def budget():
    return InvestigationBudget(max_wall_s=.4, finalization_headroom_s=.1)


def assert_stopped(result, elapsed):
    assert elapsed < 2  # cleanup/finalization tolerance, not RT scheduler claim
    assert result.status == "degraded"
    assert result.report.tier == "INCONCLUSIVE"
    assert result.budget["local_worker_reaped"]
    assert "hard_local_process_group" in result.budget["deadline_enforcement"]
    assert not any(p.name == "incidentgraph-worker" for p in multiprocessing.active_children())


@isolated_injection_test
def test_default_entrypoint_kills_slow_graph_and_retains_p0(monkeypatch, tmp_path):
    from backend.app.engine import p0_resolve, p1_scope
    marker = tmp_path / "late"
    monkeypatch.setattr(p0_resolve, "trigger_investigation", phases()[0])
    def slow_graph(*args):
        time.sleep(3)
        marker.write_text("must never happen")
    monkeypatch.setattr(p1_scope, "scope_investigation", slow_graph)
    start = time.monotonic()
    result = run_investigation("s", provider=MockProvider(), budget=budget())
    assert_stopped(result, time.monotonic() - start)
    assert result.investigation.incident_id == "i" and result.failed_stage == "P1"
    assert result.scope is None and not marker.exists()
    assert result.errors[-1]["code"] == "hard_deadline"


@isolated_injection_test
def test_trickle_provider_cannot_defeat_total_deadline(tmp_path):
    marker = tmp_path / "trickle"
    class Trickle(httpx.SyncByteStream):
        def __iter__(self):
            for _ in range(100):
                time.sleep(.03)
                with marker.open("a") as f:
                    f.write("x")
                yield b" "
    provider = OpenAIProvider("not-a-real-key", transport=httpx.MockTransport(
        lambda request: httpx.Response(200, stream=Trickle())))
    meter = BudgetMeter(budget())
    client = LLMClient(meter, provider=provider)
    def p3(*args, **kwargs):
        kwargs["llm"].complete("test", "s", "u", max_tokens=5)
    start = time.monotonic()
    result = run_investigation("s", meter=meter, llm=client,
        phase_functions=phases(p3=p3), hard_deadline=True)
    assert_stopped(result, time.monotonic() - start)
    assert result.ranking is not None and result.failed_stage == "P3"
    d = meter.as_dict()
    assert d["provider_attempts"] == 1 and d["in_flight_attempts"] == 0
    assert d["unknown_usage_attempts"] == 1 and d["tokens_charged"] > 0
    assert not d["usage_complete"]
    size = marker.stat().st_size
    time.sleep(.1)
    assert marker.stat().st_size == size


@isolated_injection_test
def test_process_group_includes_spawned_dependencies(tmp_path):
    pidfile = tmp_path / "descendant.pid"
    marker = tmp_path / "descendant.late"
    code = ("import os,time,pathlib; pathlib.Path(" + repr(str(pidfile))
            + ").write_text(str(os.getpid())); time.sleep(4); pathlib.Path("
            + repr(str(marker)) + ").write_text('late')")
    def p1(*args):
        subprocess.Popen([sys.executable, "-c", code])
        time.sleep(3)
    result = run_investigation("s", provider=MockProvider(), budget=budget(),
        phase_functions=phases(p1=p1), hard_deadline=True)
    assert result.budget["local_worker_reaped"] and not marker.exists()
    assert pidfile.exists()
    pid = int(pidfile.read_text())
    # Reparented descendants can briefly be zombies; they must not execute work.
    status = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)],
                            capture_output=True, text=True).stdout.strip()
    assert not status or status.startswith("Z")


@isolated_injection_test
def test_worker_success_preserves_parent_owned_meter_and_state():
    meter = BudgetMeter()
    def p0(*args, **kwargs):
        kwargs["meter"].visit("service:a")
        return SimpleNamespace(incident_id="i", budget={})
    result = run_investigation("s", provider=MockProvider(), meter=meter,
        phase_functions=phases(p0=p0), hard_deadline=True, until_phase=2)
    assert result.status == "completed" and result.ranking is not None
    assert meter.as_dict()["nodes_visited"] == 1
    assert [e["status"] for e in result.events] == ["started", "completed"] * 3
    assert result.runtime["config_fingerprint"]


@isolated_injection_test
def test_worker_budget_exceptions_round_trip():
    meter = BudgetMeter(InvestigationBudget(max_llm_calls=0))
    def p3(*args, **kwargs):
        kwargs["llm"].complete("test", "s", "u", max_tokens=5)
    result = run_investigation("s", meter=meter, provider=MockProvider(),
        phase_functions=phases(p3=p3), hard_deadline=True)
    assert result.errors[-1]["code"] == "budget_exhausted"
    assert meter.llm_calls == 0


def test_hard_mode_fails_closed_for_threaded_caller():
    errors = []
    def run():
        try:
            run_investigation("s", provider=MockProvider())
        except RuntimeError as exc:
            errors.append(str(exc))
    t = threading.Thread(target=run)
    t.start()
    t.join()
    assert errors and "single-threaded" in errors[0]


@isolated_injection_test
def test_event_sink_is_cancellable(tmp_path):
    marker = tmp_path / "sink"
    def sink(event):
        time.sleep(4)
        marker.write_text("late")
    start = time.monotonic()
    result = run_investigation("s", budget=budget(), provider=MockProvider(),
        phase_functions=phases(), hard_deadline=True, event_sink=sink)
    assert_stopped(result, time.monotonic() - start)
    assert not marker.exists() and result.failed_stage == "P0"


def test_invalid_provider_startup_precedes_graph_work(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "does-not-exist")
    result = run_investigation("s", phase_functions=phases(
        p0=lambda *a, **k: pytest.fail("graph must not start")))
    assert result.failed_stage == "startup"
    assert result.errors[0]["code"] == "invalid_runtime_configuration"
    assert "does-not-exist" not in result.model_dump_json()


@pytest.mark.parametrize("kwargs", [{"strict_tokens": "false"},
    {"max_wall_s": "10"}, {"expansion_slot_fraction": True},
    {"expansion_slot_fraction": float("nan")}])
def test_invalid_budget_startup(kwargs):
    with pytest.raises(ValueError):
        InvestigationBudget(**kwargs)


@pytest.mark.parametrize("url", ["ftp://host", "https://user:pass@host/v1",
    "https://host/v1?api_key=secret", "https://host/v1#secret"])
def test_provider_url_validation(url):
    with pytest.raises(ValueError):
        OpenAIProvider("key", base_url=url)


def test_standard_config_spawn_supports_threaded_caller(monkeypatch):
    # No graph/provider I/O: worker killed during startup under a tiny budget.
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    results = []
    def run():
        results.append(run_investigation("s", budget=InvestigationBudget(
            max_wall_s=.05, finalization_headroom_s=0)))
    thread = threading.Thread(target=run)
    thread.start()
    thread.join(timeout=5)
    assert not thread.is_alive() and len(results) == 1
    assert results[0].budget["local_worker_reaped"]
    assert results[0].errors[-1]["code"] == "hard_deadline"


@pytest.mark.parametrize("name", ["max_llm_calls", "max_tokens", "max_nodes_visited",
    "max_expansions_per_branch", "max_hypotheses", "max_wall_s", "finalization_headroom_s"])
def test_huge_budget_rejected_before_supervisor_dispatch(name):
    with pytest.raises(ValueError, match="operational ceiling"):
        InvestigationBudget(**{name: 10**15})


def test_slot_allocation_is_branch_linear_even_if_invalid_budget_bypassed():
    meter = BudgetMeter()
    # Defense in depth: bypass immutable startup contract to reproduce the review
    # adversary. A cap-sized loop here would hang the test process for years.
    object.__setattr__(meter.budget, "max_hypotheses", 10**15)
    start = time.monotonic()
    result = meter.allocate_hypothesis_slots(["a", "b", "c"])
    assert time.monotonic() - start < .1
    assert result == {"a": 333333333333334, "b": 333333333333333, "c": 333333333333333}


@pytest.mark.parametrize("name,args,kwargs", [
    ("allocate_hypothesis_slots", (["a"] * 1025,), {}),
    ("allocate_hypothesis_slots", (["a"],), {"limit": 10**15}),
    ("visit", ("x" * 513,), {}),
    ("check", (), {"unsupported": "work"}),
    ("visit", ([None] * 1024,), {}),
    ("as_dict", (), {str(i): 0 for i in range(7)}),
])
def test_parent_rpc_rejects_unbounded_or_invalid_arguments(name, args, kwargs):
    from backend.app.engine.worker import _validate_meter_request
    with pytest.raises((ValueError, TypeError)):
        _validate_meter_request(name, args, kwargs)


def test_capabilities_do_not_claim_strict_provider_tokens():
    client = LLMClient(BudgetMeter(), provider=MockProvider())
    assert not client.capabilities["provider_input_token_bound"]
    assert not client.capabilities["provider_output_token_cap_verified"]
    assert "key" not in json.dumps(client.effective_config)
