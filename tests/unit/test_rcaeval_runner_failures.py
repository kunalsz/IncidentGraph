"""Unit: a crashing case must never kill a benchmark run (no graph needed).

Regression for the live failure where a single-key 429 burst aborted a
20-case agent run mid-expansion. The failed case is recorded as a miss with
honest candidate counts; the run continues.
"""

import json

from backend.app.eval import rcaeval_runner as runner


def test_crashing_case_is_recorded_not_fatal(tmp_path, monkeypatch):
    data = tmp_path / "RE2-OB"
    (data / "re2ob_checkoutservice_cpu_1").mkdir(parents=True)

    def boom(case_dir, report_dir, with_agent=False, llm=None, **kwargs):
        raise RuntimeError("provider exploded")

    monkeypatch.setattr(runner, "run_case", boom)
    out = tmp_path / "out"
    code = runner.run(data, cases=["re2ob_checkoutservice_cpu_1"], report_dir=out)

    assert code == 1
    summary = json.loads(next(out.glob("*/_summary.json")).read_text())
    assert summary["n_cases"] == 1
    assert summary["n_failed"] == 1
    case = summary["cases"][0]
    assert case["error"].startswith("RuntimeError")
    assert case["ranks"] == [] and case["gt_service_rank"] is None
    # the failed case counts as a miss, not as excluded from the average
    overall = summary["metrics"]["overall"]["service"]
    assert overall["ac@1"] == 0.0
    assert overall["retrieval@5"] == 0.0
