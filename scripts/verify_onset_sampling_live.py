"""Real-graph verification of the onset sampler: adapter -> projection -> bundle -> options.

No model calls. Requires an isolated localhost FalkorDB (never 6380).
"""
import json
import os
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if os.environ.get("FALKORDB_HOST") not in ("127.0.0.1", "localhost") or os.environ.get("FALKORDB_PORT") != "6381":
    raise SystemExit("explicit isolated localhost:6381 required")
from backend.app.agent import bundle as bundle_mod
from backend.app.engine import p3_investigate, policy
from backend.app.engine.contracts import Investigation, InvestigationWindow
from backend.app.engine.p1_scope import scope_investigation
from backend.app.eval import rcaeval_adapter as adapter
from backend.app.eval.rcaeval_runner import atomic_json, sha256
from backend.app.graph.client import client, graph_for
from backend.app.settings import SCOPE_BENCHMARK_SETTINGS
from config import tuning

CASES = {  # truth options legacy -> new from the offline 90-case audit
    "re2ob_checkoutservice_delay_1": (0, 2),
    "re2ob_currencyservice_loss_3": (0, 4),
    "re2ob_checkoutservice_cpu_1": (2, 3),
}
OUT = ROOT / "output/metric-sampling-audit"
SUMMARY_FIELDS = ("onset_clock_available", "onset_valid_count", "onset_qualified_count",
                  "onset_longest_same_direction_run", "onset_qualified_directions",
                  "onset_representative_ts", "onset_representative_direction",
                  "post_deviation_directions")


def main():
    results = {}
    for case, (legacy, expected) in CASES.items():
        truth = case[len("re2ob_"):].rsplit("_", 2)[0]
        graph_name = f"incidentgraph:session:onset-verify-{uuid.uuid4().hex}"
        graph = graph_for(graph_name)
        try:
            loaded = adapter.load_case(graph, ROOT / "data/RE2-OB" / case)
            inject_s = int((ROOT / "data/RE2-OB" / case / "inject_time.txt").read_text().strip())
            assert loaded["onset_ts"] == inject_s * 1000, (loaded["onset_ts"], inject_s)
            investigation = Investigation(
                incident_id=loaded["incident_id"], session_graph=graph_name,
                symptom="verification", anchor_service="service:frontend", budget={},
                window=InvestigationWindow(t0=loaded["t0_ms"], t1=loaded["t1_ms"]),
                onset_ts=loaded["onset_ts"])
            scope = scope_investigation(investigation, full_scope=True, max_nodes=10_000,
                                        max_series=10_000, settings=SCOPE_BENCHMARK_SETTINGS)
            # 1. projection must carry the new summary fields, uncompromised
            raw = {row[0]: dict(zip(SUMMARY_FIELDS, row[1:])) for row in graph.query(
                "MATCH (m:MetricAnomaly) RETURN m.id, " + ", ".join(f"m.{f}" for f in SUMMARY_FIELDS)).result_set}
            projected = [n for n in scope.nodes if n["type"] == "MetricAnomaly"]
            inconsistent = []
            for node in projected:
                # Scope intentionally drops None-valued keys; every non-null
                # source value must survive with identical value and type.
                expected_fields = {k: v for k, v in raw[node["id"]].items() if v is not None}
                actual = {k: v for k, v in node["properties"].items() if k in SUMMARY_FIELDS}
                if actual != expected_fields:
                    inconsistent.append({"id": node["id"], "expected": expected_fields, "actual": actual})
            assert not inconsistent, f"summary field loss/mutation in scope: {inconsistent[:2]}"
            onset_rows = [n for n in projected
                          if n["properties"].get("onset_representative_ts") is not None]
            assert onset_rows, "no projected record kept an onset representative"
            assert len(onset_rows) == sum(1 for v in raw.values() if v["onset_representative_ts"] is not None)
            # 2. real bundle + claim options for the truth service
            fetched = bundle_mod.fetch_branch_bundle(graph, f"service:{truth}", investigation.window)
            options = p3_investigate._metric_claim_options(
                fetched["bundle"], set(fetched["evidence_ids"]), investigation, f"service:{truth}")
            def ref_ts(ref):
                return int(ref.rsplit("/", 1)[1])
            onset_options = [o for o in options
                             if o["claim"]["metric"].startswith(truth + "_")
                             and any(abs(ref_ts(r) - loaded["onset_ts"]) <= tuning.ONSET_DELTA_MS
                                     for r in o["evidence_refs"])]
            results[case] = {
                "truth_service": truth, "offline_truth_options": {"legacy": legacy, "new": expected},
                "graph_metric_records": len(raw),
                "projected_metric_records": len(projected),
                "records_with_summary_fields": sum(
                    1 for n in projected if n["properties"].get("onset_representative_ts") is not None),
                "records_with_onset_representative": len(onset_rows),
                "rejected_projection_rows": sum(bool(n["properties"].__contains__("__rejected"))
                                                for n in scope.nodes),
                "bundle_metrics": len(fetched["bundle"].get("metrics", [])),
                "truth_options_live": len(options),
                "truth_onset_timely_option": bool(onset_options),
                "option_metrics": [o["claim"]["metric"] for o in options],
                "source_sha256": sha256(ROOT / "backend/app/eval/rcaeval_adapter.py")}
            print(case, results[case]["truth_options_live"], "onset", results[case]["truth_onset_timely_option"],
                  flush=True)
        finally:
            client().execute_command("GRAPH.DELETE", graph_name)
    atomic_json(OUT / "live-path-verification.json", results)
    for case, row in results.items():
        assert row["truth_options_live"] > 0, case
        assert row["truth_onset_timely_option"], case
        assert row["rejected_projection_rows"] == 0, case
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
