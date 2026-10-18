"""Offline old-vs-new log/trace severity impact; mirrors load_case calls exactly.

No DB/model calls. Old policy is reimplemented verbatim from the retired code
(min(z, 6.0), flat 6.0 sentinel); new policy is the current adapter.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.app.eval import rcaeval_adapter as adapter
from backend.app.eval.rcaeval_runner import atomic_json, sha256

DATA = ROOT / "data/RE2-OB"
OUT = ROOT / "output/minute-scale-audit"
LEGACY_CAP = 6.0
CASES = json.loads((ROOT / "output/metric-sampling-audit/sampling-90-summary.json").read_text())["per_case"]
LIVE12 = ["re2ob_checkoutservice_cpu_1", "re2ob_checkoutservice_mem_2", "re2ob_checkoutservice_disk_3",
          "re2ob_checkoutservice_socket_3", "re2ob_checkoutservice_delay_1", "re2ob_currencyservice_loss_3",
          "re2ob_checkoutservice_cpu_3", "re2ob_currencyservice_delay_3", "re2ob_currencyservice_disk_2",
          "re2ob_checkoutservice_loss_1", "re2ob_recommendationservice_mem_2", "re2ob_checkoutservice_loss_3"]


def legacy_minute(events, t0_s, t1_s, inject_s, latency=False, services=None):
    """Verbatim retired scoring, reusing current binning."""
    if events is None:
        return {}, {"status": "missing_modality"}
    first = math.ceil((t0_s - inject_s) / 60)
    last = math.floor((t1_s + 1 - inject_s) / 60) - 1
    bins = list(range(first, last + 1))
    pre_bins = [b for b in bins if b < 0]
    post_bins = [b for b in bins if b >= 0]
    frame = events[events["ts"].between(t0_s, t1_s)].copy()
    frame["_bin"] = ((frame["ts"] - inject_s) // 60).astype(int)
    scores = {}
    names = sorted(set(services or []) | set(frame["service_id"].astype(str)))
    for service in names:
        group = frame[frame["service_id"] == service]
        observed = (group.groupby("_bin")["duration_us"].quantile(.95) if latency
                    else group.groupby("_bin").size())
        values = observed.to_dict()
        pre = [float(values[b]) for b in pre_bins if b in values] if latency else [float(values.get(b, 0)) for b in pre_bins]
        post = [float(values[b]) for b in post_bins if b in values] if latency else [float(values.get(b, 0)) for b in post_bins]
        if len(pre) < 2 or not post:
            continue
        mean = sum(pre) / len(pre)
        std = math.sqrt(sum((v - mean) ** 2 for v in pre) / (len(pre) - 1))
        peak = max(post)
        scores[service] = min((peak - mean) / std, LEGACY_CAP) if std else (LEGACY_CAP if peak > mean else 0.0)
    return scores, {}


def modalities(case):
    """(sev_logs_old, sev_logs_new, sev_traces_old, sev_traces_new) per service."""
    import pandas as pd
    case_dir = DATA / case
    data = adapter.read_case(case_dir)
    slices = adapter.window_frames(data["metrics"], data["inject_s"])
    t0_s, t1_s, inject_s = int(slices["t0_ms"] / 1000), int(slices["t1_ms"] / 1000), data["inject_s"]
    old_l, new_l, old_t, new_t = {}, {}, {}, {}
    if data["logs"] is not None and not data["logs"].empty:
        logs = data["logs"].rename(columns={"timestamp": "ts", "container_name": "service_id"})
        logs["service_id"] = logs["service_id"].astype(str)
        logs["is_error"] = logs["message"].astype(str).str.contains(adapter._ERROR_RE)
        logs = logs[(logs["ts"] >= t0_s) & (logs["ts"] <= t1_s)]
        errors = logs[logs["is_error"]]
        services = logs["service_id"].unique().tolist()
        old_l, _ = legacy_minute(errors, t0_s, t1_s, inject_s, services=services)
        new_l = adapter.minute_z(errors, t0_s, t1_s, inject_s, services=services)
    if data["traces"] is not None and not data["traces"].empty:
        inject_ms = inject_s * 1000
        included, _ = adapter.select_trace_sample(data["traces"], slices["t0_ms"], slices["t1_ms"], inject_ms)
        services = included["service_id"].unique().tolist()
        span_errors = included[included["error"]][["ts", "service_id"]]
        old_lat, _ = legacy_minute(included, t0_s, t1_s, inject_s, latency=True, services=services)
        new_lat = adapter.latency_z(included, t0_s, t1_s, inject_s)
        old_err, _ = legacy_minute(span_errors, t0_s, t1_s, inject_s, services=services)
        new_err = adapter.minute_z(span_errors, t0_s, t1_s, inject_s, services=services)
        for service in services:
            old_t[service] = max(old_lat.get(service, 0.0), old_err.get(service, 0.0))
            new_t[service] = max(new_lat.get(service, 0.0), new_err.get(service, 0.0))
    return old_l, new_l, old_t, new_t


def main():
    started = time.monotonic()
    report = {"label": "offline old-vs-new log/trace severity; no DB/model",
              "retired_policy": "min((peak-mean)/std, 6.0); flat 6.0 when std == 0 and peak > mean",
              "cases": [], "services_changed": {}, "ordering_changed": [], "truth_position": []}
    for case in LIVE12:
        old_l, new_l, old_t, new_t = modalities(case)
        services = sorted(set(old_l) | set(new_l) | set(old_t) | set(new_t))
        per_service = {}
        for sid in services:
            o = max(abs(old_l.get(sid, 0.0)), abs(old_t.get(sid, 0.0)))
            n = max(abs(new_l.get(sid, 0.0)), abs(new_t.get(sid, 0.0)))
            per_service[sid] = {"old_logs": old_l.get(sid), "new_logs": new_l.get(sid),
                                "old_traces": old_t.get(sid), "new_traces": new_t.get(sid)}
            if abs(n - o) > 1e-9:
                report["services_changed"][sid] = report["services_changed"].get(sid, 0) + 1
        report["cases"].append({"case": case, "services": len(services),
                                "changed": sum(1 for v in per_service.values()
                                               if abs(max(abs(v["new_logs"] or 0), abs(v["new_traces"] or 0))
                                                      - max(abs(v["old_logs"] or 0), abs(v["old_traces"] or 0))) > 1e-9),
                                "detail": per_service})
        print(case, "changed", report["cases"][-1]["changed"], flush=True)
    report["runtime_s"] = round(time.monotonic() - started, 1)
    report["adapter_sha256"] = sha256(ROOT / "backend/app/eval/rcaeval_adapter.py")
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(OUT / "live12-impact.json", report)
    print("changed services across 12 cases:", sum(c["changed"] for c in report["cases"]),
          "of", sum(c["services"] for c in report["cases"]), "runtime", report["runtime_s"])


if __name__ == "__main__":
    main()
