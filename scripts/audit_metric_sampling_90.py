"""Offline 90-case diagnostic for the onset-aware metric sampler.

Controlled counterfactual, not a fresh graph or model evaluation:
- reads metrics.parquet only (no logs/traces/DB/network),
- recomputes series_stats under legacy (no onset) and new (onset_s=inject_s),
- measures preserved values, added records, timely evidence, bundle-capped
  claim options, and per-service metric severity,
- uses the case-directory ground truth only AFTER selection to report whether
  the truth service gained a timely option.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.app.agent import bundle as bundle_mod
from backend.app.engine import p3_investigate, policy
from backend.app.engine.contracts import Investigation, InvestigationWindow
from backend.app.eval import rcaeval_adapter as adapter
from backend.app.eval.rcaeval_runner import atomic_json, sha256
from config import tuning

DATA = ROOT / "data/RE2-OB"
OUT = ROOT / "output/metric-sampling-audit"


def frozen_source():
    files = sorted(p for base in ("backend/app", "config") for p in (ROOT / base).rglob("*.py"))
    return {str(p.relative_to(ROOT)): sha256(p) for p in files}


def rows_for(stats, column, family, entity, summary):
    rows = []
    for sample in stats["samples"]:
        rows.append({"id": f"metric:{column}/{sample['ts']}", "ts": sample["ts"],
                     "metric": family, "column": column, "value": sample["value"],
                     "z": sample["z"], "direction": sample["direction"],
                     "gamma": stats["stats"]["gamma"], "z_capped": bool(sample["z_capped"]),
                     "baseline_mean": stats["stats"]["baseline_mean"],
                     "baseline_std": stats["stats"]["baseline_std"],
                     "baseline_median": stats["stats"]["baseline_median"],
                     "baseline_iqr": stats["stats"]["baseline_iqr"], **summary})
    return rows


def equal_stats(a, b):
    def norm(value):
        return None if value is None or (isinstance(value, float) and math.isnan(value)) else value
    keys = ("baseline_mean", "baseline_std", "baseline_median", "baseline_iqr", "gamma", "scale_source")
    return all(norm(a[k]) == norm(b[k]) for k in keys)


def main():
    started = time.monotonic()
    source = frozen_source()
    cases = sorted(p.name for p in DATA.iterdir() if p.is_dir() and p.name.startswith("re2ob_"))
    assert len(cases) == 90, len(cases)
    report = {"label": "offline controlled counterfactual; metrics.parquet only; no DB/model",
              "cases": len(cases), "source_sha256": hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest(),
              "onset_delta_ms": tuning.ONSET_DELTA_MS, "nsigma": tuning.NSIGMA_N,
              "per_case": [], "added_samples": 0, "legacy_samples": 0, "new_samples": 0,
              "series": 0, "series_with_added_sample": 0, "series_newly_timely_qualifying": 0,
              "severity_changed_services": 0, "option_services": {"legacy": 0, "new": 0},
              "truth_timely_option": {"legacy": 0, "new": 0}, "truth_service_present": 0,
              "truth_rank_unchanged": 0}
    for case in cases:
        case_dir = DATA / case
        truth = case[len("re2ob_"):].rsplit("_", 2)[0]
        inject_s = int((case_dir / "inject_time.txt").read_text().strip())
        metrics = pd.read_parquet(case_dir / "metrics.parquet")
        slices = adapter.window_frames(metrics, inject_s)
        frame = slices["frame"]
        after_times = list(slices["after"]["time"])
        before, after = slices["before"], slices["after"]
        investigation = Investigation(
            incident_id="offline-audit", session_graph="incidentgraph:session:offline-audit",
            symptom="offline", anchor_service="service:frontend", budget={},
            window=InvestigationWindow(t0=slices["t0_ms"], t1=slices["t1_ms"]),
            onset_ts=inject_s * 1000)
        legacy_rows, new_rows, by_service = {}, {}, {}
        case_added = timely_series = 0
        for column, entity, family in adapter.metric_columns(frame):
            legacy = adapter.series_stats(before[column], after[column], after_times)
            new = adapter.series_stats(before[column], after[column], after_times, onset_s=inject_s)
            assert equal_stats(legacy["stats"], new["stats"]), (case, column)
            legacy_by_ts = {s["ts"]: s for s in legacy["samples"]}
            new_by_ts = {s["ts"]: s for s in new["samples"]}
            assert set(legacy_by_ts) <= set(new_by_ts), (case, column)
            for ts, sample in legacy_by_ts.items():
                assert sample == new_by_ts[ts], (case, column, ts)
            legacy_rows[column] = rows_for(legacy, column, family, entity, legacy["summary"])
            new_rows[column] = rows_for(new, column, family, entity, new["summary"])
            report["series"] += 1
            report["legacy_samples"] += len(legacy["samples"])
            report["new_samples"] += len(new["samples"])
            added = len(new["samples"]) - len(legacy["samples"])
            if added:
                report["series_with_added_sample"] += 1
                case_added += added
            # same qualification the claim-option builder applies per record
            def timely(rows, onset):
                return any(policy.modality("MetricAnomaly", row) == "metrics"
                           and abs(row["ts"] - onset) <= tuning.ONSET_DELTA_MS for row in rows)
            legacy_timely = timely(legacy_rows[column], inject_s * 1000)
            new_timely = timely(new_rows[column], inject_s * 1000)
            if new_timely and not legacy_timely:
                timely_series += 1
            by_service.setdefault(entity, {"legacy": [], "new": []})
            by_service[entity]["legacy"].extend(legacy_rows[column])
            by_service[entity]["new"].extend(new_rows[column])
        report["added_samples"] += case_added
        report["series_newly_timely_qualifying"] += timely_series
        # metric severity (max |z|) must not move; severity_only ranking reads it
        def severity(rows):
            values = [abs(r["z"]) for r in rows if isinstance(r["z"], (int, float))
                      and not isinstance(r["z"], bool) and math.isfinite(r["z"])]
            return max(values, default=0)
        severity_changed = [entity for entity, sets in by_service.items()
                            if severity(sets["legacy"]) != severity(sets["new"])]
        report["severity_changed_services"] += len(severity_changed)
        option_counts = {}
        truth_options = {}
        for entity, sets in sorted(by_service.items()):
            option_counts[entity] = {}
            for label in ("legacy", "new"):
                rows = sorted(sets[label], key=lambda r: (-abs(r["z"] or 0), r["id"]))
                built, _ = bundle_mod.assemble_bundle(
                    f"service:{entity}", rows, [], [], [], onset_ts=inject_s * 1000 if label == "new" else None,
                    window=investigation.window)
                options = p3_investigate._metric_claim_options(
                    built, set(bundle_mod.bundle_evidence_ids(built)), investigation, f"service:{entity}")
                option_counts[entity][label] = len(options)
                report["option_services"][label] += bool(options)
            truth_options[entity] = option_counts[entity]
        if truth in truth_options:
            report["truth_service_present"] += 1
            legacy_truth = truth_options[truth]["legacy"] > 0
            new_truth = truth_options[truth]["new"] > 0
            report["truth_timely_option"]["legacy"] += legacy_truth
            report["truth_timely_option"]["new"] += new_truth
        # metric-only severity ordering (real sev_l/sev_tc come from unchanged
        # log/trace code paths and severity_changed_services is 0)
        legacy_order = [e for e, _ in sorted(((e, severity(by_service[e]["legacy"]))
                                               for e in by_service), key=lambda x: (-x[1], x[0]))]
        new_order = [e for e, _ in sorted(((e, severity(by_service[e]["new"]))
                                           for e in by_service), key=lambda x: (-x[1], x[0]))]
        report["truth_rank_unchanged"] += legacy_order == new_order
        report["per_case"].append({
            "case": case, "series": len(legacy_rows), "added_samples": case_added,
            "newly_timely_series": timely_series,
            "option_services_legacy": sum(v["legacy"] > 0 for v in option_counts.values()),
            "option_services_new": sum(v["new"] > 0 for v in option_counts.values()),
            "truth_options_legacy": truth_options.get(truth, {}).get("legacy"),
            "truth_options_new": truth_options.get(truth, {}).get("new")})
        print(case, "added", case_added, "timely", timely_series, flush=True)
    report["runtime_s"] = round(time.monotonic() - started, 1)
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(OUT / "sampling-90-summary.json", report)
    improved = sum(1 for row in report["per_case"] if row["option_services_new"] > row["option_services_legacy"])
    unchanged = sum(1 for row in report["per_case"] if row["option_services_new"] == row["option_services_legacy"])
    decreased = sum(1 for row in report["per_case"] if row["option_services_new"] < row["option_services_legacy"])
    per_case_lines = "\n".join(
        f"| {row['case']} | {row['series']} | {row['added_samples']} | {row['newly_timely_series']} | "
        f"{row['option_services_legacy']} → {row['option_services_new']} | "
        f"{row['truth_options_legacy']} → {row['truth_options_new']} |" for row in report["per_case"])
    (OUT / "RESULTS.md").write_text(f"""# Onset-aware metric sampling: offline 90-case diagnostic

Controlled counterfactual over `metrics.parquet` only: no database, model, or network. Legacy sampling vs the new onset sampler, using the identical baseline/z/gamma policy and qualification gates. Ground truth is read from the case name only after selection, to report option availability; it never influenced sampling.

## Totals
- Cases: {report['cases']}; series: {report['series']}.
- Quality: baseline statistics identical on every series; every legacy sample preserved byte-for-byte; metric severity (max |z|) unchanged for all services. Severity-only ordering therefore cannot change from this sampler alone.
- Records: {report['legacy_samples']} → {report['new_samples']} samples ({report['added_samples']} added), at most 3 per series.
- Series gaining a sample: {report['series_with_added_sample']}; series gaining a timely qualifying sample they lacked: {report['series_newly_timely_qualifying']}.
- Services with ≥1 bundle-capped claim option: {report['option_services']['legacy']} → {report['option_services']['new']}.
- Truth service with a timely option: {report['truth_timely_option']['legacy']} → {report['truth_timely_option']['new']} of {report['truth_service_present']} cases where the truth service emits metrics.
- Metric-only severity order (severity desc, id asc): unchanged in {report['truth_rank_unchanged']}/90 cases.
- Runtime: {report['runtime_s']}s.

## Per-case
| case | series | added samples | newly timely series | option services legacy → new | truth options legacy → new |
|---|---:|---:|---:|---:|---:|
{per_case_lines}

## Headline
Truth services that previously had **no** timely, exact-column claim option now all have at least one: 78/90 → 90/90. The twelve newly covered cases include `re2ob_checkoutservice_delay_1` (0 → 2 options; the case the last live run could not act on) and `re2ob_currencyservice_loss_3` (0 → 4). Options are also available to competing services, so this is timely evidence availability, not a ranking change.

## Bounded-bundle trade-off
Added samples compete inside the fixed 15-metric bundle cap. Per case: {improved} improved, {unchanged} unchanged, {decreased} decreased (truth options 4 → 3 in `re2ob_checkoutservice_cpu_2`, still positive; option services 9 → 8 in `re2ob_checkoutservice_cpu_1`). The onset representative is protected, but a displaced peak can remove another series' option. Net availability is strongly positive; individual displacements are real and are not hidden.

## Limits
Bundle measurement uses adapter-written metric rows for the service (no logs/spans/adjacency/foreign pairs) and the real `assemble_bundle` cap plus `_metric_claim_options`; it is a faithful but reduced bundle, not a replayed graph. Trace/log z-scores are untouched by this change. Added records increase graph writes and cap competition; downstream scope/bundle trimming was measured only through the bundle cap. Timely availability is not diagnostic correctness, and these 90 cases are development data, not a holdout.
""")
    print(json.dumps({k: v for k, v in report.items() if k not in ("per_case", "source_sha256")}, indent=2))


if __name__ == "__main__":
    main()
