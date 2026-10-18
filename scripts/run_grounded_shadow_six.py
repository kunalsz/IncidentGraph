"""Frozen, serial grounded six-case shadow protocol. Import/tests never call a provider.

First use --freeze-only, then --frozen-protocol OUTPUT/protocol.json. A freeze
is single-use; source, inputs, settings and actual budget must still match.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.app.agent.llm import (LLMClient, LLMError, LLMResponse,
    OpenAICompatibleProvider, RateLimitedError, _raise_provider_status)
from backend.app.agent.responses import SCHEMA_VERSION, TASK_SCHEMAS, Proposal
from backend.app.eval import rcaeval_runner as rr
from backend.app.eval import engine_experiments as ex
from backend.app.settings import SCOPE_BENCHMARK_SETTINGS

CASES = ["re2ob_checkoutservice_cpu_1", "re2ob_checkoutservice_mem_2",
         "re2ob_checkoutservice_disk_3", "re2ob_checkoutservice_socket_3",
         "re2ob_checkoutservice_delay_1", "re2ob_currencyservice_loss_3"]
OLD_RUN = ROOT / "output/live-shadow-six-20261018T131755Z"
BUDGET = replace(rr.BENCH_BUDGET, max_hypotheses=10, expansion_slot_fraction=.7)
COMMON = dict(primary_branches=3, minimum_primary_coverage=3,
              rank_policy="severity_only", export_policy="shadow")
CONFOUNDING = ("Historical comparison only, confounded by changed scope, prompt and coverage "
               "(and expansion allocation); not a causal ablation or independent holdout.")


def read_cases(path):
    from backend.app.eval.rcaeval_runner import parse_case_name
    names = [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    if not names or len(names) != len(set(names)):
        raise ValueError("case list must be nonempty and unique")
    for name in names:
        parse_case_name(name)
        if not (ROOT / "data/RE2-OB" / name).is_dir():
            raise ValueError(f"case not available locally: {name}")
    return names


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def redact(value, secrets):
    """Scrub persisted free text, including stage errors; never persist raw exceptions."""
    if isinstance(value, str):
        for secret in sorted(set(s for s in secrets if s), key=len, reverse=True):
            value = value.replace(secret, "[REDACTED]")
        return value
    if isinstance(value, dict):
        return {redact(k, secrets): redact(v, secrets) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v, secrets) for v in value]
    return value


def historical_freeze(old_run):
    paths = [old_run / "protocol.json"] + [old_run / "records" / f"{c}.record.json" for c in CASES]
    return {str(p.resolve()): rr.sha256(p) if p.is_file() else None for p in paths}


def freeze(model, base, old_run, cases=None):
    url = urlsplit(base)
    if (url.scheme not in ("http", "https") or not url.hostname or url.username
            or url.password or url.query or url.fragment):
        raise ValueError("invalid provider endpoint; credentials/query/fragment forbidden")
    prov = rr.provenance()
    prov["budget"] = BUDGET.as_dict()  # never retain the default budget as effective
    schemas = {name: schema.model_json_schema() for name, schema in TASK_SCHEMAS.items()}
    schemas["individual_proposal"] = Proposal.model_json_schema()
    result = {"protocol": "grounded-shadow-six-v1", "cases": list(cases or CASES),
        "exploratory_not_independent_holdout": True, "provider": "openai-compatible",
        "endpoint_origin": f"{url.scheme}://{url.hostname}" + (f":{url.port}" if url.port else ""),
        "endpoint_sha256": hashlib.sha256(base.encode()).hexdigest(),
        "requested_model": model, "model_cheap": model, "parallelism": 1,
        "fallback": False, "retries": 3, "max_attempts_per_logical_call": 4,
        "retry_policy": "classified LLMClient; only metered HTTP 429 may continue; all other errors stop",
        "rate_limit_backoff_s": 15, **COMMON, "budget": BUDGET.as_dict(),
        "monetary_limit": None,
        "authorization": "prior explicit user monetary-limit waiver; per-case resource caps retained",
        "accounting_exception": "HTTP 429 unknown usage charged through reservation ledger; never asserted zero cost",
        "stop_on_unknown_usage_or_returned_model_mismatch": True,
        "branch_policy": "P2 severity order only; record actual selected ranks; never force truth into top3",
        "control": "fresh P0-P2 per case, same inputs/scope/settings/budget, no paid calls",
        "database": {"host": "127.0.0.1", "port": 6381},
        "scope_profile": {**asdict(SCOPE_BENCHMARK_SETTINGS), "fingerprint": SCOPE_BENCHMARK_SETTINGS.fingerprint},
        "provenance": prov,
        "validation_source_sha256": rr.sha256(ROOT / "tests/test_grounded_live_protocol.py"),
        "task_schema_version": SCHEMA_VERSION,
        "task_schemas": schemas, "task_schema_sha256": digest(schemas),
        "data_sha256": {c: rr._data_provenance(ROOT / "data/RE2-OB" / c) for c in CASES},
        "old_run": str(old_run.resolve()), "old_files_sha256": historical_freeze(old_run),
        "historical_comparison_claim": CONFOUNDING}
    result["freeze_sha256"] = digest(result)
    return result


class RecordedProvider(OpenAICompatibleProvider):
    """Strict raw-envelope accounting: missing model is NOT requested-model fallback."""
    def __init__(self, key, base, model, stop, receipts, persist, active_case, transport=None):
        super().__init__(key, base_url=base, transport=transport)
        self.expected_model, self.stop = model, stop
        self.receipts, self.persist, self.active_case = receipts, persist, active_case
        self.lock = threading.Lock()

    def _attempt(self, key, *, task, system, user, model, max_tokens, timeout=90):
        import httpx
        payload = {"model": model, "max_tokens": max_tokens,
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        with httpx.Client(transport=self.transport, timeout=timeout) as client:
            response = client.post(f"{self.base_url}/chat/completions", json=payload,
                                   headers={"Authorization": f"Bearer {key}"})
            self.current_receipt["http_status"] = response.status_code
            _raise_provider_status(response, self.name)
            body = response.json()
        returned = body.get("model")
        usage = body.get("usage") or {}
        ti, to = usage.get("prompt_tokens"), usage.get("completion_tokens")
        known = all(type(v) is int and v >= 0 for v in (ti, to))
        self.current_receipt.update(returned_model=returned if isinstance(returned, str) else None,
            usage_status="known" if known else "unknown",
            tokens_in=ti if known else None, tokens_out=to if known else None)
        text = body["choices"][0]["message"]["content"]
        finish = body["choices"][0].get("finish_reason")
        if not isinstance(text, str):
            raise LLMError("invalid provider content")
        self.current_receipt["finish_reason"] = finish if isinstance(finish, str) else None
        if not known or returned != self.expected_model:
            self.stop.set()
        return LLMResponse(text, ti if known else 0, to if known else 0,
                           returned, usage_status="known" if known else "unknown", finish_reason=finish)

    def complete(self, *, task, system, user, model, max_tokens, timeout=90):
        if self.stop.is_set():
            raise LLMError("experiment stopped")
        start = time.monotonic()
        receipt = {"task": task, "case": self.active_case[0], "requested_model": model,
                   "max_output_tokens": max_tokens, "usage_status": "unknown",
                   "returned_model": None, "system_sha256": digest(system), "user_sha256": digest(user)}
        self.current_receipt = receipt  # serial provider calls are part of frozen protocol
        try:
            response = super().complete(task=task, system=system, user=user, model=model,
                                        max_tokens=max_tokens, timeout=timeout)
            receipt["status"] = "response"
            return response
        except BaseException as exc:
            receipt.update(status="error", error_type=type(exc).__name__)
            if isinstance(exc, RateLimitedError) and receipt.get("http_status") == 429:
                receipt["metered_429_exception"] = True
                raise
            self.stop.set()
            # Do not allow transient transport/HTTP errors to trigger retries.
            raise LLMError("provider attempt failed; experiment stopped") from None
        finally:
            receipt["latency_s"] = round(time.monotonic() - start, 3)
            with self.lock:
                self.receipts.append(receipt)
                self.persist(self.receipts)


class StoppingClient(LLMClient):
    def complete(self, *args, **kwargs):
        try:
            response = super().complete(*args, **kwargs)
            if self.provider.stop.is_set():
                raise LLMError("accounting/model integrity failure")
            return response
        except BaseException:
            # Includes schema/JSON errors and exhausted metered-429 retries.
            self.provider.stop.set()
            raise LLMError("client call failed; experiment stopped") from None


def case_diagnostics(row):
    a = row.get("artifacts", {})
    scope, ranking = a.get("scope", {}), a.get("ranking", {})
    inv, val = a.get("investigation", {}), a.get("validation", {})
    candidates = ranking.get("candidates", [])
    ids = [c["id"] for c in candidates]
    selected = ranking.get("branch_ids", [])
    truth = "service:" + row.get("gt_service", "")
    scheduling = inv.get("budget", {}).get("p3_scheduling", {})
    slots = scheduling.get("primary_slots", {})
    hypotheses = inv.get("hypotheses", [])
    assessments = val.get("hypotheses", {})
    spans = {n["id"] for n in scope.get("nodes", []) if n.get("type") == "Span"}
    validations = [{"hypothesis_id": hid, "claim_type": v.get("claim_type"),
        "verdict": v.get("verdict"), "qualified_timing": v.get("qualified_timing"),
        "plausible_path": v.get("plausible_path"), "modalities": v.get("modalities", []),
        "evidence_roles": v.get("evidence_roles", {}),
        "trace_roles": {role: [ref for ref in refs if ref in spans]
                        for role, refs in v.get("evidence_roles", {}).items()}}
        for hid, v in assessments.items()]
    branches = scheduling.get("branches", [])
    return {"p2_order": ids, "service_statistics": scope.get("services", []),
        "scope_profile": row.get("scope_profile"), "scope_config": scope.get("config", {}),
        "scope_truncated": scope.get("truncated"), "scope_evidence_gaps": scope.get("evidence_gaps", []),
        "scope_layered_completeness": scope.get("config", {}).get("completeness"),
        "selected_branches": [{"id": sid, "p2_position": ids.index(sid) + 1 if sid in ids else None,
                                "reported_rank": next((c["rank"] for c in candidates if c["id"] == sid), None)}
                               for sid in selected],
        "truth_p2_position": ids.index(truth) + 1 if truth in ids else None,
        "truth_selected": truth in selected, "truth_top3_guaranteed": False,
        "primary_slots": slots, "coverage_floor": 3,
        "coverage_floor_satisfied": all(slots.get(sid, 0) >= 1 for sid in selected[:3]) if selected and slots else None,
        "scheduling": scheduling,
        "claim_type_counts": dict(Counter(h.get("claim_kind", "unknown") for h in hypotheses)),
        "validity_counts": dict(Counter(v.get("verdict", "unknown") for v in assessments.values())),
        "validity_by_claim_type": {kind: dict(Counter(v.get("verdict", "unknown") for v in assessments.values()
            if v.get("claim_type") == kind)) for kind in sorted({v.get("claim_type", "unknown") for v in assessments.values()})},
        "generated_count": sum(b.get("generated", 0) for b in branches),
        "admitted_count": len(hypotheses), "rejected_count": sum(len(b.get("rejected", [])) for b in branches),
        "unassessed_count": sum(h["id"] not in assessments for h in hypotheses),
        "validations": validations}


def compare_control(control, shadow):
    left, right = case_diagnostics(control), case_diagnostics(shadow)
    return {"control_status": control["status"], "shadow_status": shadow["status"],
        "input_hashes": {"control": control.get("data"), "shadow": shadow.get("data")},
        "same_inputs": control.get("data") == shadow.get("data"),
        "full_p2_order_equal": left["p2_order"] == right["p2_order"],
        "full_export_order_equal": control.get("ranks") == shadow.get("ranks"),
        "service_statistics_equal": left["service_statistics"] == right["service_statistics"],
        "scope_profile_equal": left["scope_profile"] == right["scope_profile"],
        "control": left, "shadow": right}


def old_comparison(row, old_run):
    path = old_run / "records" / f"{row['case']}.record.json"
    if not path.is_file():
        return {"status": "unavailable", "path": str(path), "claim": CONFOUNDING}
    old = json.loads(path.read_text())
    return {"status": "available", "path": str(path), "sha256": rr.sha256(path),
        "claim": CONFOUNDING, "causal_ablation": False, "same_input_hashes": old.get("data") == row.get("data"),
        "old": case_diagnostics(old), "new": case_diagnostics(row),
        "old_gt_service_rank": old.get("gt_service_rank"), "new_gt_service_rank": row.get("gt_service_rank")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze-only", action="store_true")
    mode.add_argument("--frozen-protocol", type=Path)
    parser.add_argument("--old-run", type=Path, default=OLD_RUN)
    parser.add_argument("--cases-file", type=Path,
                        help="text file, one case name per line (freeze-only); stored in the protocol")
    args = parser.parse_args(argv)
    from backend.app.envfile import load_env_file
    load_env_file(str(ROOT / ".env"))
    model, base, key = (os.environ.get(k, "").strip() for k in ("LLM_MODEL", "LLM_BASE_URL", "LLM_API_KEY"))
    if os.environ.get("LLM_PROVIDER") not in ("openai", "openai-compatible") or not all((model, base, key)):
        raise ValueError("explicit OpenAI-compatible provider, endpoint, model and key required")
    os.environ.update(FALKORDB_HOST="127.0.0.1", FALKORDB_PORT="6381", P3_MAX_PARALLEL="1", INCIDENTGRAPH_PARALLELISM="1")
    secrets = [key, base] + [v for k, v in os.environ.items() if any(s in k.upper() for s in ("API_KEY", "TOKEN", "PASSWORD", "SECRET"))]
    def save(path, value):
        rr.atomic_json(path, redact(value, secrets))
    if args.freeze_only:
        out = ROOT / "output" / ("grounded-shadow-six-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
        out.mkdir(parents=True, exist_ok=False)
        cases = (read_cases(args.cases_file) if args.cases_file else None)
        protocol = freeze(model, base, args.old_run, cases)
        # Protocol consists solely of curated fields/hashes, not environment values.
        rr.atomic_json(out / "protocol.json", protocol)
        print("FROZEN", out / "protocol.json", "SHA256", protocol["freeze_sha256"], flush=True)
        return 0
    protocol = json.loads(args.frozen_protocol.read_text())
    old_run = Path(protocol["old_run"])
    cases = protocol["cases"]
    if cases != list(CASES) and not all((ROOT / "data/RE2-OB" / c).is_dir() for c in cases):
        raise ValueError("frozen case list is not available under data/RE2-OB")
    if freeze(model, base, old_run, cases) != protocol:
        raise ValueError("source/input/config/provider freeze mismatch; create a new freeze")
    out = args.frozen_protocol.resolve().parent
    # Atomic exclusive marker prevents concurrent or repeated expenditure.
    with (out / "execution-started.json").open("x") as handle:
        json.dump({"freeze_sha256": protocol["freeze_sha256"]}, handle)
    stop, receipts, rows, active_case = threading.Event(), [], [], [None]
    def factory(meter):
        provider = RecordedProvider(key, base, model, stop, receipts,
            lambda value: save(out / "provider-receipts.json", value), active_case)
        return StoppingClient(meter, provider=provider, model=model, model_cheap=model,
                              retries=3, rate_limit_backoff=15)
    failure = None
    save(out / "provider-receipts.json", receipts)
    try:
        for case in cases:
            if stop.is_set():
                break
            # Verify before each case, not merely once at launch.
            if freeze(model, base, old_run, cases) != protocol:
                raise ValueError("freeze changed during execution")
            active_case[0] = case
            print("START", case, flush=True)
            control = rr.run_case(ROOT / "data/RE2-OB" / case, out / "control", with_agent=False,
                                  budget=BUDGET, **COMMON)
            save(out / "control" / f"{case}.record.json", control)
            if control["status"] != "ok" or control["resources"].get("llm_calls", 0) != 0:
                raise ValueError("fresh zero-call control failed")
            row = rr.run_case(ROOT / "data/RE2-OB" / case, out / "records", with_agent=True,
                              llm_factory=factory, budget=BUDGET, **COMMON)
            row.update(provider_kind="authorized_live", requested_model=model, monetary_cap_waived=True,
                       experiment_protocol_sha256=protocol["freeze_sha256"])
            save(out / "records" / f"{case}.record.json", row)
            comparison = compare_control(control, row)
            diagnostic = None
            if row["status"] == "ok" and not stop.is_set():
                fused = ex.replay_fusion(row, ranking_policy="corrected_h_eligible_first")
                diagnostic = ex.arm_record(row, "diagnostic_only_corrected_real", fused, reused=True)
                save(out / "diagnostic-only" / f"{case}.record.json", diagnostic)
            save(out / "comparisons" / f"{case}.json", {"within_version": comparison,
                "historical": old_comparison(row, old_run), "diagnostic_stage_reuse": True})
            rows.append({"case": case, "status": row["status"], "p2_rank": row.get("gt_service_rank"),
                "diagnostic_rank": diagnostic.get("gt_service_rank") if diagnostic else None,
                "resources": row["resources"], "timings": row["timings"], "cleanup": row["cleanup"]})
            save(out / "progress.json", {"cases": rows, "stop_flag": stop.is_set()})
            if row["status"] != "ok" or not all(comparison[k] for k in
                    ("same_inputs", "full_p2_order_equal", "full_export_order_equal", "service_statistics_equal", "scope_profile_equal")):
                raise ValueError("shadow/control integrity failure")
            if case_diagnostics(row)["coverage_floor_satisfied"] is not True:
                raise ValueError("primary coverage allocation floor not verified")
    except BaseException as exc:
        stop.set()
        failure = type(exc).__name__  # no exception messages/provider bodies/secrets
    finally:
        try:
            stable = freeze(model, base, old_run, cases) == protocol
        except Exception:
            stable = False
        save(out / "summary.json", {"requested_cases": cases, "completed_cases": rows,
            "unexecuted_cases": cases[len(rows):], "stop_flag": stop.is_set(), "failure_type": failure,
            "source_input_config_stable": stable, "provider_attempts": len(receipts),
            "known_input_tokens": sum(r.get("tokens_in") or 0 for r in receipts),
            "known_output_tokens": sum(r.get("tokens_out") or 0 for r in receipts),
            "unknown_usage_attempts": sum(r["usage_status"] != "known" for r in receipts),
            "historical_comparison_claim": CONFOUNDING, "dollar_cost": None,
            "cost_reason": "pricing unavailable; prior monetary-cap waiver, resource caps remain"})
    print("OUTPUT", out, flush=True)
    return int(stop.is_set() or not stable or len(rows) != len(CASES))


if __name__ == "__main__":
    raise SystemExit(main())
