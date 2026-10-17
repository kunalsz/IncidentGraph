"""Shared P0-P6 runner with retained-state, provider-free finalization.

The standard configured entrypoint spawns synchronous dependencies in a killable
POSIX process group. Explicit injected phase functions retain the legacy
cooperative facade unless hard_deadline=True; hard arbitrary-object injection
uses fork and requires a native-single-threaded caller. No thread timeout is
called termination. Remote accepted work cannot be forcibly revoked.
"""
from __future__ import annotations
import inspect
from backend.app.engine.budget import BudgetMeter, BudgetExceeded
from backend.app.engine.state import RunState
from backend.app.engine.contracts import ReportResult


def _fallback(state, meter):
    """No graph/provider calls, even for rendering or persistence."""
    incident = getattr(state.investigation, "incident_id", f"unresolved:{state.run_id}")
    retained = [name for name in ("investigation", "scope", "ranking", "invest", "validated", "fused")
                if getattr(state, name) is not None]
    summary = ["The investigation could not establish a complete trusted diagnosis.",
               f"Stopped at {state.failed_stage or 'deadline'}; retained phases: {', '.join(retained) or 'none'}."]
    action = "Collect missing evidence and review the retained investigation before retrying."
    report = ReportResult(incident_id=incident, verdict="insufficient_evidence", tier="INCONCLUSIVE",
        summary=summary, next_action=action, budget=meter.as_dict(),
        not_checked=[{"stage": state.failed_stage or "remaining_phases", "reason": "execution incomplete"}],
        config={"fallback": "retained-state-v2", "persistence": "not_attempted",
                "deadline_enforcement": "cooperative_not_hard"})
    report.markdown = "# Investigation — INCONCLUSIVE\n\n" + "\n\n".join(summary) + "\n\nNext action: " + action
    return report


def run_investigation(symptom, *, alert_id=None, trigger_ts=None, budget=None,
                      meter=None, provider=None, llm=None, until_phase=6,
                      event_sink=None, phase_functions=None, hard_deadline=None):
    """Default: hard local worker boundary; injected phases default cooperative.

    Event sinks run inside the worker so a blocked sink cannot defeat cancellation.
    In hard mode observable caller side effects require IPC/external sinks, not
    mutable in-process lists. Only completed phase snapshots are retained.
    """
    if type(until_phase) is not int or not 0 <= until_phase <= 6:
        raise ValueError("until_phase must be 0..6")
    if hard_deadline is not None and type(hard_deadline) is not bool:
        raise ValueError("hard_deadline must be boolean")
    if llm is not None and meter is not None and llm.meter is not meter:
        raise ValueError("runner and provider must share the same meter")
    if budget is not None and meter is not None and budget != meter.budget:
        raise ValueError("budget and meter disagree")
    meter = meter or (llm.meter if llm is not None else BudgetMeter(budget))
    standard_spawn = provider is None and llm is None and phase_functions is None and event_sink is None
    # Validate provider configuration before graph mutations. NoneProvider is
    # permitted for deterministic phases and does not imply provider availability.
    from backend.app.agent.llm import LLMClient
    try:
        from backend.app.envfile import load_env_file
        from backend.app.settings import validate_startup_settings
        import os
        load_env_file()
        settings = validate_startup_settings()
        legacy_parallel = os.environ.get("P3_MAX_PARALLEL")
        if legacy_parallel is not None and int(legacy_parallel) != settings.parallelism:
            raise ValueError("P3_MAX_PARALLEL conflicts with frozen runtime parallelism")
        llm = llm or LLMClient(meter, provider=provider)
        llm.runtime_settings = settings
    except Exception as exc:
        state = RunState(status="degraded", failed_stage="startup")
        state.errors.append({"stage": "startup", "code": "invalid_runtime_configuration",
                             "error_type": type(exc).__name__})
        state.report = _fallback(state, meter)
        state.budget = meter.as_dict()
        state.report.budget = state.budget
        return state
    options = dict(symptom=symptom, alert_id=alert_id, trigger_ts=trigger_ts,
                   llm=llm, until_phase=until_phase, event_sink=event_sink,
                   phase_functions=phase_functions)
    isolated = phase_functions is None if hard_deadline is None else hard_deadline
    if isolated:
        from backend.app.engine.worker import run_isolated
        return run_isolated(options, meter, spawn=standard_spawn)
    return _run_cooperative(meter=meter, **options)


def _run_cooperative(symptom, *, alert_id=None, trigger_ts=None, budget=None,
                      meter=None, provider=None, llm=None, until_phase=6,
                      event_sink=None, phase_functions=None, _state_sink=None):
    """Run synchronously; injectable phases support offline fault tests.

    Returns RunState on exhaustion, ordinary errors and KeyboardInterrupt. A
    cancelled run never dispatches further provider or graph finalization work.
    No full hard latency guarantee is made for uncooperative dependencies.
    """
    if type(until_phase) is not int or not 0 <= until_phase <= 6:
        raise ValueError("until_phase must be 0..6")
    if llm is not None and meter is not None and llm.meter is not meter:
        raise ValueError("runner and provider must share the same meter")
    meter = meter or (llm.meter if llm is not None else BudgetMeter(budget))
    state = RunState()
    if llm is not None:
        state.runtime = {"config_fingerprint": llm.config_fingerprint,
                         "effective_config": llm.effective_config}
        if getattr(llm, "runtime_settings", None) is not None:
            state.runtime["startup_settings"] = dict(llm.runtime_settings.redacted_snapshot)
            state.runtime["startup_settings_fingerprint"] = llm.runtime_settings.fingerprint
    from backend.app.engine.p0_resolve import trigger_investigation
    from backend.app.engine.p1_scope import scope_investigation
    from backend.app.engine.p2_rank import rank_investigation
    from backend.app.engine.p3_investigate import investigate_investigation
    from backend.app.engine.p4_validate import validate_investigation
    from backend.app.engine.p5_fuse import fuse_investigation
    from backend.app.engine.p6_report import report_investigation
    phases = phase_functions or (trigger_investigation, scope_investigation, rank_investigation,
        investigate_investigation, validate_investigation, fuse_investigation, report_investigation)
    attrs = ("investigation", "scope", "ranking", "invest", "validated", "fused", "report")

    def emit(index, status):
        event = {"sequence": len(state.events), "phase": f"P{index}", "status": status}
        state.events.append(event)
        if _state_sink:
            _state_sink(state)
        if event_sink:
            try:
                event_sink(dict(event))
            except Exception:
                state.errors.append({"stage": "event_sink", "code": "delivery_failed"})

    index = 0
    try:
        for index in range(until_phase + 1):
            meter.check("report" if index == 6 else "work")
            emit(index, "started")
            fn = phases[index]
            if index == 0:
                kwargs = dict(alert_id=alert_id, trigger_ts=trigger_ts)
                params = inspect.signature(fn).parameters
                if "meter" in params or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
                    kwargs["meter"] = meter
                value = fn(symptom, **kwargs)
                value.budget = meter.as_dict()
            elif index == 1:
                value = fn(state.investigation)
            elif index == 2:
                value = fn(state.investigation, state.scope)
            elif index == 6:
                # Reporting is deterministic; never initialize/configure a
                # provider solely to finalize an investigation.
                value = fn(state.investigation, state.scope, state.ranking,
                           state.invest, state.validated, state.fused,
                           meter=meter, llm=llm)
            elif index in (3, 4):
                if llm is None:
                    from backend.app.agent.llm import LLMClient
                    llm = LLMClient(meter, provider=provider)
                args = [state.investigation, state.scope, state.ranking]
                if index == 4:
                    args.append(state.invest)
                value = fn(*args, meter=meter, llm=llm)
            else:
                value = fn(state.investigation, state.ranking, state.invest, state.validated)
            setattr(state, attrs[index], value)
            emit(index, "completed")
        state.status = "completed"
    except (Exception, KeyboardInterrupt) as exc:
        state.failed_stage = f"P{index}"
        state.status = "cancelled" if isinstance(exc, KeyboardInterrupt) else "degraded"
        code = "budget_exhausted" if isinstance(exc, BudgetExceeded) else type(exc).__name__
        # Do not serialize provider exception text (it may contain credentials).
        state.errors.append({"stage": state.failed_stage, "code": code})
        emit(index, state.status)
        if all(getattr(state, name) is not None for name in attrs[:6]):
            try:
                state.report = report_investigation(state.investigation, state.scope,
                    state.ranking, state.invest, state.validated, state.fused,
                    meter=meter, llm=None, final_reads_allowed=False)
            except Exception:
                state.report = _fallback(state, meter)
        else:
            state.report = _fallback(state, meter)
    state.budget = meter.as_dict()
    if state.report is not None:
        state.report.budget = state.budget
    return state
