"""Explicit authorized six-case real-provider shadow experiment; no model fallback."""
import hashlib
import json
import os
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.app.envfile import load_env_file
load_env_file(str(ROOT / ".env"))
# Never use the demo database for this experiment.
os.environ["FALKORDB_HOST"] = "127.0.0.1"
os.environ["FALKORDB_PORT"] = "6381"
os.environ["P3_MAX_PARALLEL"] = "1"
os.environ["INCIDENTGRAPH_PARALLELISM"] = "1"
from backend.app.agent.llm import LLMClient, LLMError, OpenAICompatibleProvider, RateLimitedError
from backend.app.eval import rcaeval_runner as rr
from backend.app.eval import engine_experiments as ex

CASES = ["re2ob_checkoutservice_cpu_1", "re2ob_checkoutservice_mem_2",
         "re2ob_checkoutservice_disk_3", "re2ob_checkoutservice_socket_3",
         "re2ob_checkoutservice_delay_1", "re2ob_currencyservice_loss_3"]
OUT = ROOT / "output" / ("live-shadow-six-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))


def main():
    model = os.environ.get("LLM_MODEL", "").strip()
    base = os.environ.get("LLM_BASE_URL", "").strip()
    key = os.environ.get("LLM_API_KEY", "").strip()
    if os.environ.get("LLM_PROVIDER") not in ("openai", "openai-compatible") or not all((model, base, key)):
        raise ValueError("Explicit OpenAI-compatible endpoint, model, and key required")
    OUT.mkdir(parents=True, exist_ok=False)
    budget = replace(rr.BENCH_BUDGET, max_hypotheses=10, expansion_slot_fraction=.8)
    stop = threading.Event()
    lock = threading.Lock()
    receipts = []
    protocol = {"cases": CASES, "exploratory_not_independent_holdout": True,
                "provider": "openai-compatible", "endpoint": base, "requested_model": model,
                "model_cheap": model, "retries": 3, "primary_branches": 2,
                "parallelism": 1, "rate_limit_backoff_s": 15,
                "accounting_exception": "HTTP 429 rejections recorded as unknown and retried through ledger; other unknown outcomes stop run",
                "rank_policy": "severity_only", "export_policy": "shadow",
                "budget": budget.as_dict(), "monetary_limit": None,
                "authorization": "user explicitly waived monetary limit; retain per-case resource limits",
                "stop_on_unknown_usage_or_returned_model_mismatch": True,
                "provenance": rr.provenance(),
                "data_sha256": {case: rr._data_provenance(ROOT / "data/RE2-OB" / case) for case in CASES}}
    rr.atomic_json(OUT / "protocol.json", protocol)

    class RecordedProvider(OpenAICompatibleProvider):
        def complete(self, **kwargs):
            if stop.is_set():
                raise LLMError("experiment stopped after unknown usage or model mismatch")
            start = time.monotonic()
            receipt = {"task": kwargs["task"], "requested_model": kwargs["model"],
                       "max_output_tokens": kwargs["max_tokens"], "case": active_case[0]}
            try:
                response = super().complete(**kwargs)
                receipt.update(returned_model=response.model, usage_status=response.usage_status,
                               tokens_in=response.tokens_in, tokens_out=response.tokens_out,
                               finish_reason=response.finish_reason, status="response")
                if response.usage_status != "known" or response.model != model:
                    stop.set()
                return response
            except BaseException as exc:
                receipt.update(status="error", error_type=type(exc).__name__, usage_status="unknown")
                if not isinstance(exc, RateLimitedError):
                    stop.set()
                raise
            finally:
                receipt["latency_s"] = round(time.monotonic() - start, 3)
                with lock:
                    receipts.append(receipt)
                    rr.atomic_json(OUT / "provider-receipts.json", receipts)

    active_case = [None]
    def factory(meter):
        return LLMClient(meter, provider=RecordedProvider(key, base_url=base),
                         model=model, model_cheap=model, retries=3, rate_limit_backoff=15)

    rows = []
    for case in CASES:
        if stop.is_set():
            break
        active_case[0] = case
        print("START", case, flush=True)
        row = rr.run_case(ROOT / "data/RE2-OB" / case, OUT / "records", with_agent=True,
                          llm_factory=factory, budget=budget, primary_branches=2,
                          rank_policy="severity_only", export_policy="shadow")
        row.update(provider_kind="authorized_live", requested_model=model,
                   endpoint=base, monetary_cap_waived=True)
        rr.atomic_json(OUT / "records" / f"{case}.record.json", row)
        diagnostic = None
        if row["status"] == "ok":
            assert row["service_order"] == row["baseline_order"]
            fused = ex.replay_fusion(row, ranking_policy="corrected_h_eligible_first")
            diagnostic = ex.arm_record(row, "diagnostic_only_corrected_real", fused)
            rr.atomic_json(OUT / "diagnostic-only" / f"{case}.record.json", diagnostic)
        rows.append({"case": case, "status": row["status"], "p2_rank": row["gt_service_rank"],
                     "diagnostic_rank": diagnostic["gt_service_rank"] if diagnostic else None,
                     "truncated": (row.get("agent") or {}).get("truncated"),
                     "resources": row["resources"], "timings": row["timings"], "cleanup": row["cleanup"]})
        rr.atomic_json(OUT / "progress.json", {"cases": rows, "stop_flag": stop.is_set()})
        print("DONE", case, row["status"], "rank", row["gt_service_rank"],
              "diagnostic", rows[-1]["diagnostic_rank"], flush=True)
    summary = {"requested_cases": CASES, "completed_cases": rows,
               "unexecuted_cases": CASES[len(rows):], "stopped_for_accounting_or_model": stop.is_set(),
               "source_stable": rr.provenance()["source_sha256"] == protocol["provenance"]["source_sha256"],
               "provider_attempts": len(receipts),
               "known_input_tokens": sum(r.get("tokens_in", 0) for r in receipts),
               "known_output_tokens": sum(r.get("tokens_out", 0) for r in receipts),
               "unknown_usage_attempts": sum(r.get("usage_status") != "known" for r in receipts),
               "dollar_cost": None, "cost_reason": "provider pricing unavailable; no monetary cap requested"}
    rr.atomic_json(OUT / "summary.json", summary)
    print("OUTPUT", OUT, flush=True)


if __name__ == "__main__":
    main()
