"""Parent-supervised synchronous runner. Private IPC carries trusted Python state."""
from __future__ import annotations

import multiprocessing
import os
import socket

from backend.app.engine.deadline import (WorkerMeter, kill_group, receive_frame,
                                         require_isolation_support, send_frame)
from backend.app.engine.state import RunState

# Deliberately explicit: a worker cannot ask the supervisor to execute arbitrary IO.
METER_METHODS = frozenset(("as_dict", "snapshot", "check", "remaining_seconds", "wall_s",
    "begin_logical", "record_cache_hit", "reserve_attempt", "settle_attempt",
    "reserve_llm_call", "record_llm_call", "visit", "allocate_hypothesis_slots",
    "release_hypothesis_slots", "new_hypothesis", "expansion"))
ENFORCEMENT = "hard_local_process_group; remote execution cancellation not guaranteed"


def _validate_meter_request(name, args, kwargs):
    """Bound supervisor work BEFORE invoking any meter method.

    Private trusted-worker IPC is not a public deserialization protocol, but an
    accidentally oversized phase request must not monopolize the supervisor.
    """
    from backend.app.engine.budget import AttemptLease, BudgetMeter
    import inspect
    if name not in METER_METHODS or type(args) is not tuple or len(args) > 6:
        raise ValueError("invalid worker meter request")
    if type(kwargs) is not dict or len(kwargs) > 6:
        raise ValueError("invalid worker meter keyword arguments")
    remaining = [2100]
    def bounded(value, depth=0):
        remaining[0] -= 1
        if remaining[0] < 0 or depth > 4:
            raise ValueError("meter argument nesting/count exceeds bound")
        if value is None or type(value) is bool:
            return
        if type(value) is str and len(value) <= 512:
            return
        if type(value) is int and 0 <= value <= 100_000_000:
            return
        if type(value) in (list, tuple) and len(value) <= 1024:
            for item in value:
                bounded(item, depth + 1)
            return
        if type(value) is AttemptLease:
            for item in (value.id, value.logical_id, value.phase, value.commitment):
                bounded(item, depth + 1)
            return
        raise ValueError("unsupported or oversized meter argument")
    for value in args:
        bounded(value)
    for key, value in kwargs.items():
        bounded(key)
        bounded(value)
    bound = inspect.signature(getattr(BudgetMeter, name)).bind(None, *args, **kwargs)
    for key, value in bound.arguments.items():
        if key in ("phase", "node_id", "branch_id", "logical_id", "outcome", "usage_status"):
            if type(value) is not str:
                raise ValueError("meter identity/status arguments must be strings")
        if key == "branch" and value is not None and type(value) is not str:
            raise ValueError("meter branch must be a string")
        if key == "branch_ids" and (type(value) not in (list, tuple)
                                      or any(type(item) is not str for item in value)):
            raise ValueError("meter branches must be strings")


def _runtime_snapshot(llm):
    result = {"config_fingerprint": llm.config_fingerprint,
              "effective_config": llm.effective_config}
    if getattr(llm, "runtime_settings", None) is not None:
        result.update(startup_settings=dict(llm.runtime_settings.redacted_snapshot),
                      startup_settings_fingerprint=llm.runtime_settings.fingerprint)
    return result


def _child(sock, parent_socket, budget, options):
    parent_socket.close()
    os.setsid()
    # Never reuse inherited live graph sockets/pools after fork.
    from backend.app.graph import client as graph_client
    graph_client._client = None
    meter = WorkerMeter(sock, budget)
    options["meter"] = meter
    descriptor = options.pop("_llm_descriptor", None)
    if descriptor is not None:
        from backend.app.agent.llm import (LLMClient, NoneProvider, MockProvider,
                                           OpenAIProvider, AnthropicProvider)
        kind, config, settings = descriptor
        if kind == "none":
            provider = NoneProvider()
        elif kind == "mock":
            provider = MockProvider()
        elif kind == "openai":
            provider = OpenAIProvider(config.pop("keys"), base_url=config.pop("base_url"))
        elif kind == "anthropic":
            provider = AnthropicProvider(config.pop("key"))
        else:
            raise ValueError("unsupported spawn provider")
        options["llm"] = LLMClient(meter, provider=provider, **config)
        options["llm"].runtime_settings = settings
    elif options.get("llm") is not None:
        options["llm"].meter = meter
    from backend.app.engine.runner import _run_cooperative
    options["_state_sink"] = lambda state: meter.publish("state", state)
    try:
        state = _run_cooperative(**options)
        meter.publish("done", state)
    finally:
        sock.close()


def run_isolated(options, meter, *, spawn=False):
    from backend.app.engine.runner import _fallback
    if spawn:
        if os.name != "posix":
            raise RuntimeError("hard worker deadline requires POSIX process groups")
        options = dict(options)
        llm = options.pop("llm")
        config = dict(model=llm.model, model_cheap=llm.model_cheap,
                      retries=llm.retries, rate_limit_backoff=llm.rate_limit_backoff)
        if llm.provider.name == "openai":
            config.update(keys=list(llm.provider._rotator._keys), base_url=llm.provider.base_url)
        elif llm.provider.name == "anthropic":
            config["key"] = llm.provider.api_key
        options["_llm_descriptor"] = (llm.provider.name, config, llm.runtime_settings)
    else:
        require_isolation_support()
    if meter.as_dict()["in_flight_attempts"]:
        raise ValueError("isolated runner requires an idle, exclusively owned meter")
    state = RunState()
    state.runtime = _runtime_snapshot(llm if spawn else options["llm"])
    parent, child = socket.socketpair()
    process = multiprocessing.get_context("spawn" if spawn else "fork").Process(target=_child,
        args=(child, parent, meter.budget, options), name="incidentgraph-worker")
    done = False
    reason = "worker_failed"
    started = False
    try:
        process.start()
        started = True
        child.close()
        while True:
            # Work stops at the earlier deadline. P6 may use reserved headroom.
            phase = "report" if state.events and state.events[-1]["phase"] == "P6" else "work"
            remaining = meter.remaining_seconds(phase)
            if remaining <= 0:
                reason = "hard_deadline"
                break
            parent.settimeout(remaining)
            message = receive_frame(parent, lambda: meter.remaining_seconds(phase))
            if message[0] == "meter":
                _, name, args, kwargs = message
                try:
                    _validate_meter_request(name, args, kwargs)
                    result = getattr(meter, name)(*args, **kwargs)
                    reply = (True, result)
                except Exception as exc:
                    reply = (False, exc)
                # Replies are small; sending also has the same absolute bound.
                parent.settimeout(max(0.000001, meter.remaining_seconds(phase)))
                send_frame(parent, reply)
            elif message[0] in ("state", "done"):
                state = message[1]
                if not isinstance(state, RunState):
                    raise ValueError("invalid worker state")
                parent.settimeout(max(0.000001, meter.remaining_seconds(phase)))
                send_frame(parent, (True, None))
                if message[0] == "done":
                    done = True
                    break
            else:
                raise ValueError("invalid worker message")
    except socket.timeout:
        reason = "hard_deadline"
    except KeyboardInterrupt:
        reason = "cancelled"
    except (EOFError, OSError, ValueError):
        reason = "worker_failed"
    finally:
        if started:
            # Also kill descendants after normal completion; no orphan work.
            kill_group(process)
            meter.abandon_inflight()
        parent.close()
        child.close()
        if started:
            process.close()
    if not done:
        state.status = "cancelled" if reason == "cancelled" else "degraded"
        last = state.events[-1] if state.events else {"phase": "P0"}
        state.failed_stage = last["phase"]
        state.errors.append({"stage": state.failed_stage, "code": reason})
        state.events.append({"sequence": len(state.events), "phase": state.failed_stage,
                             "status": state.status})
        state.report = _fallback(state, meter)
    state.budget = meter.as_dict()
    state.budget["deadline_enforcement"] = ENFORCEMENT
    state.budget["local_worker_reaped"] = True
    if state.report is not None:
        state.report.budget = state.budget
        state.report.config["deadline_enforcement"] = ENFORCEMENT
    return state
