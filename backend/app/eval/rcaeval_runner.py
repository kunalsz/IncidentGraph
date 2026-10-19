"""Versioned offline full-scope/frontend-anchor RCAEval protocol (not default P0-P6).

Each invocation creates a new run directory. P2 is the default export even
when investigation runs in shadow. Live clients must be explicitly supplied.
Ground truth is used only after inference, for scoring and universe membership.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import time
import uuid
from dataclasses import replace
from pathlib import Path

from backend.app.engine.budget import BudgetMeter
from backend.app.engine.contracts import Investigation, InvestigationWindow
from backend.app.engine.p1_scope import scope_investigation
from backend.app.engine.p2_rank import rank_investigation, validate_rank_policy
from config import tuning
from backend.app.eval import evaluator as ev
from backend.app.eval import rcaeval_adapter as adapter
from backend.app.graph.client import client, graph_for
from config.tuning import InvestigationBudget

# Corpora share one layout: <suite>_<service>_<fault>_<instance>. Adding a
# suite must not require touching the engine, which is why the prefix is a
# tuple and topology selection is by prefix (see the adapter).
SUITE_PREFIXES = ("re2ob_", "re2ss_", "re2tt_")
CASE_PREFIX = SUITE_PREFIXES[0]  # retained for older callers
# Corpus entry services differ by name but play the same role. This is only a
# placeholder for cold ranking: it must exist in scope (the anchored walk
# requires it, and P4 compares levels against it) and it must be the same node a
# real incident's symptom would be reported against, so the anchored-prior
# comparison stays comparable across corpora.
ENTRY_SERVICE_NAMES = ("frontend", "front-end", "frontendservice")
DEFAULT_ANCHOR = "service:frontend"  # OB's entry service

# RCAEval supplies no symptom, so this harness fixes the anchor to an entry
# service and ranks every service cold. Which prior to use there is decided by
# two corpora, not one:
#   Online Boutique (90): anchored entry-service prior 24.4% AC@1, anchor-neutral
#     walk 57.8%, drop the walk 51.1%, severity only 78.9%
#   SockShop (60): anchored 43.3%, anchor-neutral 23.3%, drop the walk 85.0%,
#     severity only 85.0%
# The topology prior's direction depends on graph shape (SockShop's uniform
# walk concentrates mass on leaf stores that are never injected; Boutique's
# fans out from one entry), so neither walk variant transfers. severity_only
# leads on BOTH corpora and is the only policy that does; the benchmark path
# therefore defaults to it. The product default stays baseline_rrf with the
# personalised walk, because a real incident's anchor is genuine signal and the
# P4/P5 gates exist to stop a lone spike winning on its own. The earlier
# "anchor-neutral is the principled cold-ranking default" claim was a
# single-corpus conclusion and is retracted (see output/replication-*/RESULTS.md).
BENCHMARK_RANK_POLICY = "severity_only"
MAX_K = 5
PROTOCOL = "re2ob-offline-fullscope-frontend-retained-traces-v2"
# NOTE (known imbalance, 2026-10-19): these caps were authored when P1 admitted
# ~1.8k spans and ~2.1k observations per case. Unused-quota redistribution and
# onset sampling now admit ~6k spans and ~4k more observations, so the agent
# path spends its 200k token budget on the largest cases (12 services x several
# calls) and later branches fail with BudgetExceeded. Rank-only runs are
# unaffected. Fixing this means sizing the budget from admitted evidence (or
# bounding bundle volume), which is a cost-policy change needing its own
# predeclared comparison; tracked, not silently bumped.
def envelope_token_bound(max_llm_calls: int, bundle_token_cap: int = tuning.BUNDLE_TOKEN_CAP,
                         max_output_tokens: int = 4_096, framing_tokens: int = 1_000) -> int:
    """Tokens the allowed number of calls can consume, per call, worst case.

    The per-case token cap exists to bound cost, but it must not be *smaller*
    than what the call and visit caps already permit: when it is, the run stops
    mid-investigation and whole cases fail with BudgetExceeded. Consumption per
    call is bounded by the bundle cap (input), the largest requested output and
    a framing allowance (prompt + schema), so the envelope is simply
    `max_llm_calls x per-call bound`. Calls and wall clock remain the real
    constraints; this is the safety net that makes them coherent.
    """
    if min(max_llm_calls, bundle_token_cap, max_output_tokens, framing_tokens) < 0:
        raise ValueError("envelope inputs must be nonnegative")
    return max_llm_calls * (bundle_token_cap + max_output_tokens + framing_tokens)


def benchmark_budget(budget: InvestigationBudget | None = None) -> InvestigationBudget:
    """The benchmark harness budget: same caps, coherent token envelope."""
    budget = budget or InvestigationBudget()
    return replace(budget, max_tokens=envelope_token_bound(budget.max_llm_calls))


BENCH_BUDGET = InvestigationBudget()
CLASS_ORDER = ("cpu", "mem", "io", "socket", "delay", "loss")
ROOT = Path(__file__).resolve().parents[3]


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    try:
        with tmp.open("w") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def provenance():
    def git(*args):
        try:
            return subprocess.check_output(["git", "-C", str(ROOT), *args], stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return None
    # Hash source bytes including untracked repair files, not credential/env files.
    sources = sorted(p for base in ("backend/app", "config", "scripts") for p in (ROOT/base).rglob("*.py"))
    source_hashes = {str(p.relative_to(ROOT)): sha256(p) for p in sources}
    configs = {str(p.relative_to(ROOT)): sha256(p) for base in ("config", "backend/app/agent/prompts")
               for p in (ROOT/base).rglob("*") if p.is_file() and "__pycache__" not in str(p)}
    configs[str(adapter.OB_TOPOLOGY.relative_to(ROOT))] = sha256(adapter.OB_TOPOLOGY)
    deps = {}
    for name in ("pandas", "pyarrow", "falkordb", "redis", "pydantic", "httpx"):
        try:
            deps[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            deps[name] = None
    return {"source_revision": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain")),
            "source_sha256": hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest(),
            "source_files": source_hashes, "config_hashes": configs, "dependencies": deps,
            "python": platform.python_version(), "protocol": PROTOCOL, "schema_version": 2,
            "sampler_version": adapter.SAMPLER_VERSION, "export_version": "validated-p2-shadow-v2",
            "evaluator_version": "named-universes-v2", "seed": 0, "concurrency": 1,
            "cache_mode": "none", "provider": {"requested_model": None, "returned_model": None,
            "tokenizer": "not_used_offline"}, "budget": benchmark_budget().as_dict(),
            "budget_basis": "caps bounded by max_llm_calls x (bundle cap + max output + framing); "
                            "calls and wall clock are the real constraints",
            "db_image": os.environ.get("EVAL_DB_IMAGE", "unknown"), "db_version": "unknown"}


def default_anchor(entities) -> str:
    """Entry service when present, else the first entity deterministically."""
    names = set(entities)
    for name in ENTRY_SERVICE_NAMES:
        if name in names:
            return f"service:{name}"
    return f"service:{sorted(names)[0]}"


def suite_of(case):
    """Suite prefix of a case directory name, or None when unrecognised."""
    return next((prefix for prefix in SUITE_PREFIXES if case.startswith(prefix)), None)


def parse_case_name(case):
    if not re.fullmatch(r"re2(?:ob|ss|tt)_[A-Za-z0-9-]+_(cpu|mem|disk|socket|delay|loss)_[0-9]+", case):
        raise ValueError(f"unexpected case name: {case}")
    prefix = suite_of(case)
    return tuple(case[len(prefix):].rsplit("_", 2))


def discover_cases(data_dir, subset=None, cases=None):
    data_dir = Path(data_dir)
    if subset is not None and subset <= 0:
        raise ValueError("subset must be positive")
    if cases is not None:
        return [data_dir / name for name in cases]
    if subset is not None:
        index = json.loads((data_dir / "_subset.json").read_text())["selected"]
        # Missing requested directories must reach run_case and count as failures.
        return [data_dir / name for name in index[:subset]]
    return sorted(path for path in data_dir.iterdir()
                  if path.is_dir() and suite_of(path.name) is not None)


def export_ranks(ranking, fused=None, policy="p2"):
    """Valid P2 services only; service-block gamma order, not metric reranking.

    `eligible_fusion` is an explicit diagnostic policy; default `p2`/`shadow`
    never rewards mere hypothesis membership. Rejected/unknown statuses cannot
    lead; hypothesis-bearing candidates must also belong to the P2 universe.
    """
    if policy not in ("p2", "shadow", "eligible_fusion"):
        raise ValueError(f"unknown export policy: {policy}")
    order = list(dict.fromkeys(c.id for c in ranking.candidates
                             if re.fullmatch(r"service:[A-Za-z0-9-]+", c.id)))
    if policy == "eligible_fusion" and fused is not None:
        eligible = []
        for item in fused.fused:
            status = getattr(item, "status", "unvalidated")
            explicit = getattr(item, "eligible", None)
            if (item.candidate in order and status in ("validated", "confirmed", "partial")
                    and explicit is not False and item.candidate not in eligible):
                eligible.append(item.candidate)
        order = eligible + [sid for sid in order if sid not in eligible]
    by_service = {}
    for item in ranking.indicators:
        indicator = str(item.get("indicator", ""))
        indicator = "latency" if indicator == "latency-90" else indicator
        if item.get("service_id") not in order or not re.fullmatch(r"[A-Za-z0-9-]+", indicator) or indicator == "latency-50":
            continue
        by_service.setdefault(item["service_id"], []).append({**item, "indicator": indicator})
    ranks = []
    for sid in order:
        name = sid.removeprefix("service:")
        entries = sorted(by_service.get(sid, []), key=lambda x: (-float(x.get("gamma", 0)), x["indicator"]))
        ranks.extend(f"{name}_{item['indicator']}" for item in entries)
        if not entries:
            ranks.append(name)  # service-only; fine_ranks deliberately ignores this.
    return list(dict.fromkeys(ranks))


def _dump(value):
    return value.model_dump(mode="json") if hasattr(value, "model_dump") else value


def _data_provenance(case_dir):
    names = ("metrics.parquet", "logs.parquet", "traces.parquet", "inject_time.txt")
    index = case_dir.parent / "_subset.json"
    meta = json.loads(index.read_text()) if index.exists() else {}
    return {"sha256": {name: sha256(case_dir/name) for name in names if (case_dir/name).is_file()},
            "missing_modalities": [name for name in names if not (case_dir/name).exists()],
            "dataset_revision": meta.get("revision", "historical_unpinned"),
            "selection_sha256": sha256(index) if index.exists() else None}


def _failed_case_result(case_dir, exc):
    try:
        service, fault, instance = parse_case_name(case_dir.name)
    except ValueError:
        service, fault, instance = "invalid", "cpu", "unknown"
    result = {"schema_version": 2, "protocol": PROTOCOL, "case": case_dir.name,
              "status": "failed", "failed_stage": "load", "error_code": type(exc).__name__,
              "error": type(exc).__name__, "gt_service": service, "gt_fault": fault,
              "gt_instance": instance, "gt_metric": ev.FAULT_ANSWER_METRIC[fault],
              "ranks": [], "service_order": [], "gt_service_rank": None, "top5": [],
              "baseline_order": [], "final_order": [], "ranking_source": "failed",
              "n_candidates_service": None, "n_candidates_metric": None,
              "answer_in_service_universe": True, "answer_in_metric_universe": True,
              "counts": {}, "timings": {}, "agent": None, "artifacts": {},
              "resources": {"llm_calls": 0, "tokens": 0, "unknown_expenditure": False},
              "cleanup": "not_created", "persistence": "pending"}
    # Failures retain pre-existing universes when telemetry can be read; they
    # do not assert that the answer left the universe because ranks are empty.
    try:
        data = adapter.read_case(case_dir)
        frame = adapter.window_frames(data["metrics"], data["inject_s"])["frame"]
        from types import SimpleNamespace
        services = {entity for _,entity,_ in adapter.metric_columns(frame)}
        if data["logs"] is not None:
            services |= set(data["logs"]["container_name"].astype(str))
        if data["traces"] is not None:
            services |= {adapter.TRACE_SERVICE_MAP.get(s,s) for s in data["traces"]["serviceName"].astype(str)}
        ranking = SimpleNamespace(candidates=[SimpleNamespace(id="service:"+s) for s in services])
        exports = [f"{entity}_{family}" for _,entity,family in adapter.metric_columns(frame)]
        universes = ev.candidate_universes(frame, ranking, exports)
        _set_universes(result, universes)
    except Exception:
        pass
    return result


def _set_universes(result, universes):
    result["universes"] = universes
    for kind in ("service", "metric"):
        result[f"n_candidates_{kind}"] = universes[kind]["count"]
        answer = result["gt_service"] if kind == "service" else f"{result['gt_service']}_{result['gt_metric']}"
        result[f"answer_in_{kind}_universe"] = answer in universes[kind]["identities"]


def run_case(case_dir, report_dir, with_agent=False, llm=None, *, llm_factory=None,
             run_id=None, export_policy="p2", graph_factory=None, db_factory=None,
             budget=None, fusion_options=None, primary_branches=None,
             minimum_primary_coverage=None, rank_policy=BENCHMARK_RANK_POLICY,
             anchor_service: str | None = None):
    case_dir, report_dir = Path(case_dir), Path(report_dir)
    run_id = run_id or uuid.uuid4().hex
    graph_name = f"incidentgraph:session:eval-{run_id}-{uuid.uuid4().hex}"
    result = _failed_case_result(case_dir, RuntimeError("pending"))
    result.update(run_id=run_id, session_graph=graph_name, rank_policy=rank_policy,
                  rank_policy_version=tuning.P2_RANK_POLICY_VERSION)
    meter = BudgetMeter(budget or benchmark_budget())
    created, db, start, stage = False, None, time.monotonic(), "load"
    try:
        validate_rank_policy(rank_policy)
        result["data"] = _data_provenance(case_dir)
        db = (db_factory or client)()
        graph = (graph_factory or graph_for)(graph_name)
        created = True  # DELETE even after a partial adapter write.
        loaded = adapter.load_case(graph, case_dir)
        result.update(counts=loaded["counts"], sample_diagnostics=loaded["sample_diagnostics"],
                      aggregate_diagnostics=loaded["aggregate_diagnostics"])
        # The anchor is a cold-ranking placeholder here (no symptom exists), but
        # it must exist in scope: `random_walk` requires it, and P4 compares
        # levels against it. Default to the corpus entry service, else the first
        # entity deterministically, and record which was used.
        anchor = anchor_service or default_anchor(loaded["entities"])
        result["anchor_service"] = anchor
        investigation = Investigation(incident_id=loaded["incident_id"], session_graph=graph_name,
            symptom="benchmark case (cold ranking)", anchor_service=anchor, budget={},
            window=InvestigationWindow(t0=loaded["t0_ms"], t1=loaded["t1_ms"]),
            onset_ts=loaded.get("onset_ts"))
        stage = "p1_p2"
        from backend.app.engine.runner import run_investigation
        from backend.app.engine.p3_investigate import investigate_investigation
        from backend.app.engine.p4_validate import validate_investigation
        from backend.app.engine.p5_fuse import fuse_investigation
        local_llm = None
        if with_agent:
            if llm is not None or llm_factory is None:
                raise ValueError("agent mode requires a fresh explicit llm_factory(meter), not a shared client")
            local_llm = llm_factory(meter)
        def adapter_p0(symptom, **kwargs):
            return investigation
        def benchmark_p1(inv):
            from dataclasses import asdict
            from backend.app.settings import SCOPE_BENCHMARK_SETTINGS
            result["scope_profile"] = {**asdict(SCOPE_BENCHMARK_SETTINGS), "fingerprint": SCOPE_BENCHMARK_SETTINGS.fingerprint}
            return scope_investigation(inv, full_scope=True, max_nodes=10_000, max_series=10_000,
                                       settings=SCOPE_BENCHMARK_SETTINGS)
        def benchmark_p2(inv, scoped):
            ranked = rank_investigation(inv, scoped, rank_policy=rank_policy,
                                        case_anchor=False)
            if primary_branches is not None:
                ranked.branch_ids = ranked.branch_ids[:primary_branches]
            return ranked
        def benchmark_p3(inv, scoped, ranked, meter=None, llm=None):
            options = ({"minimum_primary_coverage": minimum_primary_coverage}
                       if minimum_primary_coverage is not None else {})
            return investigate_investigation(inv, scoped, ranked, meter=meter, llm=llm, **options)
        def benchmark_p5(inv, ranked, proposed, checked):
            return fuse_investigation(inv, ranked, proposed, checked, **(fusion_options or {}))
        state = run_investigation("benchmark case (cold ranking)", meter=meter, llm=local_llm,
            until_phase=5 if with_agent else 2,
            phase_functions=(adapter_p0, benchmark_p1, benchmark_p2, benchmark_p3,
                             validate_investigation, benchmark_p5, None))
        result["phase_events"] = state.events
        result["artifacts"] = {name: _dump(getattr(state, attr)) for name, attr in
            (("context", "investigation"), ("scope", "scope"), ("ranking", "ranking"), ("investigation", "invest"),
             ("validation", "validated"), ("fusion", "fused")) if getattr(state, attr) is not None}
        from backend.app.engine.contracts import dump_handoff
        result["handoffs"] = {stage: dump_handoff(stage, getattr(state, attr)) for stage, attr in
            (("resolve", "investigation"), ("scope", "scope"), ("rank", "ranking"),
             ("investigate", "invest"), ("validate", "validated"), ("fuse", "fused"))
            if getattr(state, attr) is not None}
        result["runtime_boundary"] = "injected adapter phases; cooperative deadlines, not hard isolation"
        if state.status != "completed":
            stage = state.failed_stage
            raise RuntimeError("shared_runner_incomplete")
        scope, ranking = state.scope, state.ranking
        # Historical field name: this is the selected pre-P5 P2 order, not
        # necessarily baseline_rrf. rank_policy identifies the effective arm.
        baseline = export_ranks(ranking)
        result["baseline_order"] = ev.service_rank_entities(baseline)
        _set_universes(result, ev.candidate_universes(loaded["window_frame"], ranking, baseline))
        fused = state.fused
        if with_agent:
            stage = "agent"
            invest, validated = state.invest, state.validated
            result["agent"] = {"winner": fused.winner_candidate, "truncated": invest.truncated,
                               "outcome": "completed", "export_policy": export_policy}
        stage = "export"
        ranks = export_ranks(ranking, fused, policy=export_policy)
        service_order = ev.service_rank_entities(ranks)
        result.update(status="ok", failed_stage=None, error_code=None, ranks=ranks, service_order=service_order,
                      final_order=service_order, top5=ranks[:5], ranking_source=export_policy,
                      gt_service_rank=service_order.index(result["gt_service"])+1 if result["gt_service"] in service_order else None)
        result.pop("error", None)
    except Exception as exc:
        result.update(status="failed", failed_stage=stage, error_code=type(exc).__name__, error=type(exc).__name__,
                      ranks=[], service_order=[], final_order=[], top5=[], gt_service_rank=None)
        if with_agent:
            result["agent"] = {"outcome": "failed", "truncated": True}
    finally:
        result["resources"] = meter.as_dict()
        result["resources"]["unknown_expenditure"] = bool(with_agent and result["status"] == "failed")
        result["timings"]["total_s"] = round(time.monotonic()-start, 6)
        if created:
            try:
                db.execute_command("GRAPH.DELETE", graph_name)
                result["cleanup"] = "deleted"
            except Exception:
                result["cleanup"] = "failed"
                result["status"] = "degraded" if result["status"] == "ok" else result["status"]
        result["persistence"] = "atomic"
        atomic_json(report_dir/f"{case_dir.name}.json", {"0": result["ranks"]})
        atomic_json(report_dir/f"{case_dir.name}.record.json", result)
    return result


def evaluate(case_results):
    s_all, f_all = ev.Evaluator(MAX_K), ev.Evaluator(MAX_K)
    classes = {name: (ev.Evaluator(MAX_K), ev.Evaluator(MAX_K)) for name in CLASS_ORDER}
    for result in case_results:
        answer_service = ev.Node(result["gt_service"], "unknown")
        answer_fine = ev.answer_node(result["gt_service"], result["gt_fault"])
        for s_eval, f_eval in ((s_all, f_all), classes[ev.FAULT_CLASS[result["gt_fault"]]]):
            s_eval.add_case(ev.service_ranks(result["ranks"]), answer_service,
                result.get("n_candidates_service"), result.get("answer_in_service_universe", True))
            f_eval.add_case(ev.fine_ranks(result["ranks"]), answer_fine,
                result.get("n_candidates_metric"), result.get("answer_in_metric_universe", True))
    def block(s, f):
        def metrics(e, service=False):
            accuracy = e.accuracy_service if service else e.accuracy
            average = e.average_service if service else e.average
            chance = e.chance_average(5)
            result = {**{f"ac@{k}": accuracy(k) for k in (1, 3, 5)}, "avg@5": average(5),
                "chance_ac@5": e.chance_accuracy(5), "chance_avg@5": chance,
                "lift_avg@5": average(5)-chance if average(5) is not None and chance is not None else None,
                "universe_coverage": sum(e._answer_in_candidates)/e.num if e.num and not e.n_missing_candidates else None,
                "answer_in_universe": list(e._answer_in_candidates) if not e.n_missing_candidates else None,
                "n_unknown_universes": e.n_missing_candidates}
            if service:
                result.update(mrr=e.mrr_service(), **{"retrieval@5": e.retrieval_service(5),
                    "rerank@1": e.rerank_service(1, 5)})
            return result
        return {"service": metrics(s, True), "fine": metrics(f)}
    return {"overall": block(s_all, f_all), "per_class": {name: block(*pair) for name,pair in classes.items()}}


def aggregate_resources(results):
    """Known settled usage is distinct from charged unknown reservations."""
    ledgers = [r.get("resources") or r.get("agent") or {} for r in results]
    return {"schema_version": 2,
        "known_llm_calls": sum(row.get("llm_calls", 0) for row in ledgers),
        "known_tokens": sum(row.get("tokens_total", row.get("tokens", 0)) for row in ledgers),
        "charged_tokens_including_unknown_reservations": sum(row.get("tokens_charged", row.get("tokens_total", 0)) for row in ledgers),
        "unknown_usage_attempts": sum(row.get("unknown_usage_attempts", row.get("usage_unknown_attempts", 0)) for row in ledgers),
        "unknown_expenditure_cases": sum(bool(row.get("unknown_expenditure") or row.get("usage_complete") is False
            or row.get("unknown_usage_attempts", row.get("usage_unknown_attempts", 0))
            or (r.get("error") and not row)) for r,row in zip(results,ledgers))}


def run(data_dir, subset=None, cases=None, report_dir="output/rcaeval-v2", with_agent=False,
        llm=None, *, llm_factory=None, export_policy="p2", rank_policy=BENCHMARK_RANK_POLICY):
    validate_rank_policy(rank_policy)
    data_dir, root = Path(data_dir), Path(report_dir)
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:10]
    out = root/run_id
    out.mkdir(parents=True, exist_ok=False)
    case_dirs = discover_cases(data_dir, subset, cases)
    if not case_dirs:
        atomic_json(out/"_summary.json", {"run_id": run_id, "n_cases": 0, "n_failed": 0, "status": "empty",
                                         "rank_policy": rank_policy, "rank_policy_version": tuning.P2_RANK_POLICY_VERSION})
        return 1
    prov = {**provenance(), "rank_policy": rank_policy,
            "rank_policy_version": tuning.P2_RANK_POLICY_VERSION}
    try:
        prov["db_version"] = client().info().get("redis_version", "unknown")
    except Exception:
        pass
    atomic_json(out/"_manifest.json", {"run_id": run_id, "requested": [p.name for p in case_dirs], "provenance": prov})
    results, seen = [], set()
    for path in case_dirs:
        try:
            if path.name in seen:
                raise ValueError("duplicate requested case")
            parse_case_name(path.name)
            seen.add(path.name)
            result = run_case(path, out, with_agent=with_agent, llm=llm,
                llm_factory=llm_factory, run_id=run_id, export_policy=export_policy,
                rank_policy=rank_policy)
        except Exception as exc:
            result = _failed_case_result(path, exc)
            result.update(run_id=run_id, failed_stage="dispatch", persistence="atomic",
                          rank_policy=rank_policy, rank_policy_version=tuning.P2_RANK_POLICY_VERSION)
            # Duplicate records get a distinct request slot, never overwrite a prior result.
            suffix = f".request-{len(results)}" if path.name in [r["case"] for r in results] else ""
            safe_name = path.name if re.fullmatch(r"[A-Za-z0-9_-]+", path.name) else f"invalid-{len(results)}"
            atomic_json(out/f"{safe_name}{suffix}.json", {"0": []})
            atomic_json(out/f"{safe_name}{suffix}.record.json", result)
        results.append(result)
        print(f"{path.name}: {result['status']} rank={result['gt_service_rank']}", flush=True)
        # Names and statuses only. The full results already live in each
        # case's .record.json, and duplicating them here costs ~11 MB per case
        # at the current scope size - rewritten after every case, so peak usage
        # doubles. That filled the disk and killed the database mid-run, which
        # is why this file is now a progress marker, not a second store.
        atomic_json(out/"_checkpoint.json", {"run_id": run_id, "completed": len(results),
            "cases": [{"case": row["case"], "status": row["status"],
                       "gt_service_rank": row.get("gt_service_rank")} for row in results]})
    failed = sum(r["status"] != "ok" for r in results)
    summary = {"schema_version": 2, "run_id": run_id, "protocol": PROTOCOL,
        "mode": "agent_shadow" if with_agent else "p2_offline", "n_cases": len(results),
        "rank_policy": rank_policy, "rank_policy_version": tuning.P2_RANK_POLICY_VERSION,
        "n_failed": failed, "status": "failed" if failed else "ok", "provenance": prov,
        "metrics": evaluate(results), "cases": results,
        "costs": aggregate_resources(results)}
    atomic_json(out/"_summary.json", summary)
    print(json.dumps(summary["metrics"]["overall"], indent=2))
    print(f"wrote {out/'_summary.json'}")
    return 1 if failed else 0
