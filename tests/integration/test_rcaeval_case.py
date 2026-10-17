"""Integration: one real RCAEval RE2-OB case end-to-end (adapter -> P1 -> P2).

Skips cleanly when the dataset (data/RE2-OB) is absent — fetch it with
`python scripts/fetch_rcaeval.py --subset 20`.
"""

from pathlib import Path

import pytest

from backend.app.eval import rcaeval_adapter as adapter
from backend.app.eval.rcaeval_runner import export_ranks, parse_case_name, run_case
from backend.app.graph.client import graph_for

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "RE2-OB"
CASE = "re2ob_checkoutservice_cpu_1"


@pytest.mark.integration
def test_single_case_ranks_ground_truth_high(tmp_path):
    case_dir = DATA_DIR / CASE
    if not case_dir.exists():
        pytest.skip(f"dataset absent: {case_dir} (run scripts/fetch_rcaeval.py)")

    result = run_case(case_dir, tmp_path)

    assert result["gt_service"] == "checkoutservice"
    assert result["gt_fault"] == "cpu"
    # the injected service must land high on its own cpu fault
    assert result["gt_service_rank"] is not None and result["gt_service_rank"] <= 3
    # rank strings follow RCAEval's "<service>_<metric>" contract
    assert all("_" in rank or rank.isidentifier() for rank in result["ranks"])
    assert result["n_candidates_service"] >= 10
    assert result["n_candidates_metric"] >= 50
    # adapter wrote real telemetry volume
    assert result["counts"]["spans"] > 1_000
    assert result["counts"]["metrics"] > 50
    # graph is cleaned up after the case
    graph = graph_for(f"incidentgraph:session:rcaeval-{adapter.run_key(case_dir)}")
    rows = graph.query("MATCH (s:Service) RETURN count(s)").result_set
    assert rows[0][0] == 0
