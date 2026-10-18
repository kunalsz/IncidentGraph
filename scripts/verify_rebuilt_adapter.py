"""Exercise repaired adapter evidence against the live database without an LLM."""
import json
import sys
import uuid
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.app.engine import policy, p4_validate
from backend.app.eval import rcaeval_adapter as adapter
from backend.app.graph.client import client, graph_for
from backend.app.graph import validators


def main():
    graph_name = f"incidentgraph:session:adapter-check-{uuid.uuid4().hex}"
    graph = graph_for(graph_name)
    before, after = 600, 600
    frame = pd.DataFrame({"time": list(range(1000, 2200)),
        "frontend_cpu": [0.0] * 1200,
        "currencyservice_cpu": [0.0] * before + [50.0] * after,
        "emailservice_cpu": [0.0] * before + [float("nan")] * 599 + [50.0],
        "checkoutservice_cpu": [float("nan")] * before + [50.0] * after})
    case = {"metrics": frame, "inject_s": 1600, "logs": None, "traces": None}
    try:
        with patch.object(adapter, "read_case", return_value=case):
            adapter.load_case(graph, ROOT / "data/RE2-OB/re2ob_currencyservice_cpu_1")
        rows = graph.query("MATCH (m:MetricAnomaly)-[:OBSERVED_ON]->(s:Service) "
            "RETURN s.id, m.z, m.gamma").result_set
        by_service = {}
        for sid, z, gamma in rows:
            by_service.setdefault(sid, []).append({"z": z, "gamma": gamma})
        assert any(policy.modality("MetricAnomaly", props) == "metrics"
                   for props in by_service["service:currencyservice"])
        assert all(props["z"] == 0 and props["gamma"] == 0
                   for props in by_service["service:emailservice"])
        assert all(props["z"] is None and props["gamma"] is None
                   and policy.modality("MetricAnomaly", props) is None
                   for props in by_service["service:checkoutservice"])
        services = graph.query("MATCH (s:Service) RETURN s.id, s.level, s.level_convention").result_set
        scope = type("Scope", (), {"nodes": [{"id": sid, "type": "Service",
            "properties": {"level": level, "level_convention": convention}}
            for sid, level, convention in services]})()
        levels = p4_validate._service_levels(scope)
        assert validators.level_veto(levels["service:currencyservice"], levels["service:frontend"])["verdict"] == "pass"
        report = {"persistent_change_qualifies": True, "sparse_blip_unqualified": True,
                  "missing_baseline_unscored": True, "p4_level_veto": "pass",
                  "stored_metrics": by_service, "authoritative_levels": levels,
                  "llm_calls": 0}
    finally:
        client().execute_command("GRAPH.DELETE", graph_name)
    out = ROOT / "output/p2-rebuilt-20261018/adapter-live-verification.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
