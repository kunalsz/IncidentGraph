import json
from types import SimpleNamespace

import pandas as pd
import pytest

from backend.app.eval import rcaeval_adapter as adapter
from backend.app.eval import rcaeval_runner as runner
from backend.app.eval.comparison import compare
from backend.app.eval.replay import freeze_record, mock_artifacts, rerank, is_eligible
from scripts.fetch_rcaeval import select_subset, download_file


def spans():
    rows = []
    for trace in range(800):
        for i in range(10):
            ts = (trace % 20)*60000+i
            rows.append(dict(traceID=str(trace), spanID=f"{trace}-{i}", parentSpanID=f"{trace}-{i-1}" if i else None,
                startTimeMillis=ts, serviceName=f"svc{trace%3}", duration=10, statusCode=2 if trace%4==0 else 0))
    rows.append(dict(traceID="boundary", spanID="outside", parentSpanID=None, startTimeMillis=-1,
                     serviceName="svc0", duration=10, statusCode=2))
    rows.append(dict(traceID="boundary", spanID="inside", parentSpanID="outside", startTimeMillis=1,
                     serviceName="svc0", duration=10, statusCode=2))
    return pd.DataFrame(rows)


def test_sampler_exact_cap_window_deterministic_periods_and_groups():
    frame = spans()
    selected, diag = adapter.select_trace_sample(frame, 0, 1199999, 600000)
    shuffled, _ = adapter.select_trace_sample(frame.sample(frac=1, random_state=42), 0, 1199999, 600000)
    pd.testing.assert_frame_equal(selected, shuffled)
    assert len(selected) <= 6000
    assert selected.startTimeMillis.between(0,1199999).all()
    assert diag["selected_period_counts"]["pre"] and diag["selected_period_counts"]["post"]
    assert len(diag["selected_services"]) == 3
    boundary, boundary_diag = adapter.select_trace_sample(frame[frame.traceID == "boundary"], 0, 1199999, 600000)
    assert boundary_diag["boundary_parent_loss"] == 1
    assert not boundary_diag["full_source_trace_complete"]
    window = frame[frame.startTimeMillis.between(0,1199999)]
    for tid, group in selected.groupby("traceID"):
        assert len(group) == len(window[window.traceID == tid])


def test_oversized_and_impossible_period_quality():
    frame = spans().head(10)
    selected, diag = adapter.select_trace_sample(frame,0,1199999,600000,budget=5)
    assert selected.empty and diag["oversized_groups"]
    assert diag["quality_status"] == "insufficient_period_coverage"
    with pytest.raises(ValueError):
        adapter.select_trace_sample(frame,0,1199999,600000,budget=6001)


def test_grid_diagnostics_and_missing_latency():
    frame = pd.DataFrame({"ts":[10,700],"service_id":["a","a"]})
    scores, diag = adapter.minute_z(frame,0,1199,600,return_diagnostics=True)
    assert diag["services"]["a"]["baseline_mean"] == .1
    assert diag["services"]["a"]["baseline_std"] == pytest.approx(.316227766)
    assert scores["a"] == pytest.approx(2.846049894)
    frame["duration_us"] = [10,20]
    scores, diag = adapter.latency_z(frame,0,1199,600,return_diagnostics=True)
    assert scores == {} and diag["services"]["a"]["status"] == "insufficient_statistics"
    assert adapter.minute_z(None,0,1199,600) == {}


@pytest.mark.parametrize("n",[1,20,30,31,60,90,100])
def test_nested_subsets_exact(n):
    cases = [f"re2ob_s{s}_{f}_{i}" for s in range(5) for f in ("cpu","mem","delay","disk","socket","loss") for i in (1,2,3)]
    order = select_subset(cases,100)
    assert select_subset(list(reversed(cases)),n) == order[:min(n,90)]
    assert len(set(order)) == 90
    with pytest.raises(ValueError):
        select_subset(cases,0)


def record():
    return {"case":"re2ob_a_cpu_1","gt_service":"a","gt_fault":"cpu","service_order":["a","b"],
            "baseline_order":["a","b"],"ranks":["a_cpu","b_cpu"]}


def test_historical_unreconstructible_and_mock_negative_control():
    frozen = freeze_record(record())
    assert not frozen.stage_artifacts_available
    with pytest.raises(ValueError, match="absent"):
        rerank(frozen,"corrected")
    mock = mock_artifacts(frozen)
    assert rerank(mock,"negative_control") == frozen.baseline_order
    assert rerank(mock,"corrected") == rerank(mock,"corrected")
    assert frozen.services[0].hypotheses == []


def test_paired_intent_missing_and_cluster_interval():
    left = {"one":record(), "two":{**record(),"case":"two"}}
    right = {"one":{**record(),"service_order":["b","a"]}}
    result = compare(left,right,draws=40)
    assert result["intent_to_run"]["n"] == 2
    assert result["intent_to_run"]["top1_lost"] == 2
    assert result["strata"]["failed_or_missing"]["n"] == 1


def test_runner_fresh_dirs_missing_duplicate_nonzero_and_atomic(tmp_path,monkeypatch):
    monkeypatch.setattr(runner,"client",lambda: (_ for _ in ()).throw(RuntimeError()))
    name = "re2ob_a_cpu_1"
    for _ in range(2):
        assert runner.run(tmp_path,cases=[name,name],report_dir=tmp_path/"out") == 1
    runs = list((tmp_path/"out").iterdir())
    assert len(runs) == 2
    for run in runs:
        summary = json.loads((run/"_summary.json").read_text())
        assert summary["n_cases"] == summary["n_failed"] == 2
        assert len(list(run.glob("*.record.json"))) == 2
        assert not list(run.glob("*.tmp"))
        assert runner.evaluate(summary["cases"]) == summary["metrics"]


def test_export_invalid_unknown_and_rejected():
    rank = SimpleNamespace(candidates=[SimpleNamespace(id="service:a"),SimpleNamespace(id="service:b")],
        indicators=[{"service_id":"service:a","indicator":"cpu","gamma":1},
                    {"service_id":"service:a","indicator":"cpu","gamma":2},
                    {"service_id":"service:evil","indicator":"cpu","gamma":9},
                    {"service_id":"service:a","indicator":"bad_label","gamma":9}])
    fused = SimpleNamespace(fused=[SimpleNamespace(candidate="service:b",status="disproved")])
    assert runner.export_ranks(rank,fused,"eligible_fusion") == ["a_cpu","b"]


def test_downloader_rejects_same_size_corruption(tmp_path):
    import hashlib
    import httpx
    content=b"correct"
    digest=hashlib.sha256(content).hexdigest()
    def handler(request):
        return httpx.Response(200,headers={"etag":digest,"content-length":str(len(content))},
                              content=b"" if request.method=="HEAD" else content)
    client=httpx.Client(transport=httpx.MockTransport(handler))
    target=tmp_path/"re2ob_a_cpu_1"/"metrics.parquet"
    target.parent.mkdir()
    target.write_bytes(b"corrupt")
    assert download_file(client,"re2ob_a_cpu_1","metrics.parquet",tmp_path,"fixed") == "downloaded"
    assert target.read_bytes()==content
    assert download_file(client,"re2ob_a_cpu_1","metrics.parquet",tmp_path,"fixed") == "skipped"


def test_replay_branch_identity_and_legacy_status_failclosed():
    source = record()
    source["artifacts"] = {"investigation": {"branches": [{"node_id": "service:a", "status": "closed"}]},
        "fusion": {"fused": [{"id": "h", "candidate": "service:a", "status": "validated"}]}}
    frozen = freeze_record(source)
    assert frozen.services[0].branch_status == "closed"
    assert not frozen.services[0].eligible
    assert not is_eligible({"status":"validated"})
    assert not is_eligible({"status":"confirmed","eligible":True})
    assert is_eligible({"status":"partial","eligible":True})


def test_comparison_failed_stale_rank_and_unknown_ledger():
    source = record()
    failed = {**source, "status":"failed", "resources":{"llm_calls":2,"tokens_total":10,
        "tokens_charged":100,"unknown_usage_attempts":1,"usage_complete":False}}
    result = compare({"a":source},{"a":failed},draws=20)
    assert result["intent_to_run"]["top1_lost"] == 1
    cost = result["costs"]["right"]
    assert cost["known_tokens"] == 10
    assert cost["charged_tokens_including_unknown_reservations"] == 100
    assert cost["unknown_usage_attempts"] == cost["unknown_expenditure_cases"] == 1


def test_runner_aggregate_known_charged_unknown():
    rows=[{"resources":{"llm_calls":2,"tokens_total":10,"tokens_charged":90,
           "unknown_usage_attempts":1,"usage_complete":False}}]
    cost=runner.aggregate_resources(rows)
    assert cost["known_tokens"]==10
    assert cost["charged_tokens_including_unknown_reservations"]==90
    assert cost["unknown_usage_attempts"]==cost["unknown_expenditure_cases"]==1


def test_replay_nonmatching_proposal_fingerprint_is_ineligible():
    source=record()
    ref={"incident_id":"i","revision":0,"fingerprint":"f"}
    source["artifacts"]={"ranking":{"scope_ref":ref},"scope":{"scope_ref":ref},
        "investigation":{"hypotheses":[{"schema_version":2,"id":"h","candidate":"service:a",
            "scope_ref":ref,"branch_id":"b","claim_kind":"service_abnormality","text":"x","evidence":[]}]},
        "fusion":{"fused":[{"id":"h","candidate":"service:a","status":"validated",
                              "confidence_derivation":{"eligible":True}}]},
        "validation":{"hypotheses":{"h":{"schema_version":2,"hypothesis_id":"h","candidate":"service:a",
            "scope_ref":ref,"proposal_fingerprint":"wrong","verdict":"pass","qualified_timing":True,"plausible_path":True}}}}
    assert not freeze_record(source).services[0].eligible
