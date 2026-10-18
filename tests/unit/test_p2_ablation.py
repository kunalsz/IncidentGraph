"""Offline P2 ablation: replay fidelity, tie bounds, statistics. No graph/LLM/data."""

import json

import pytest

from backend.app.eval import p2_ablation as ab
from backend.app.engine.p2_rank import rank_candidates
from backend.app.engine.contracts import ScopeResult, ScopeService

SERVICES = ["frontend", "checkoutservice", "currencyservice", "cartservice", "emailservice"]


def _components(severity, phi1=None, phi3=None, rwr=None):
    out = {}
    for i, name in enumerate(SERVICES):
        out["service:" + name] = {
            "phi1": (phi1 or {}).get(name, 0.0), "phi3": (phi3 or {}).get(name, 0.0),
            "severity": severity.get(name, 0.0), "rwr": (rwr or {}).get(name, 0.0),
            "deploy_proximity": 0.0}
    return out


def _record(case, gt, severity, **kw):
    return _record_from_components(case, gt, _components(severity, **kw))


def _record_from_components(case, gt, comps):
    scores = ab.fuse(comps, ab.ARMS["baseline_rrf"]["weights"])
    order = ab.deployed_order(scores)
    cands = [{"id": sid, "rank": i + 1, "score": scores[sid], "components": comps[sid],
              "component_ranks": {}, "evidence": [], "rule_flags": [], "selected": True}
             for i, sid in enumerate(order)]
    return {"case": case, "status": "ok", "gt_service": gt, "gt_fault": "cpu",
            "n_candidates_service": len(ab.service_names(order)), "answer_in_service_universe": True,
            "artifacts": {"ranking": {"candidates": cands}, "context": {"anchor_service": "service:frontend"}}}


# frontend wins on 3 anchor-correlated voters; the true service wins severity by far.
def symptom_case(case="re2ob_currencyservice_cpu_1"):
    return _record(case, "currencyservice",
                   severity={"currencyservice": 40.0, "frontend": 12.0, "checkoutservice": 9.0},
                   phi1={"frontend": 9.0, "checkoutservice": 5.0, "currencyservice": 2.0},
                   phi3={"frontend": 9e5, "checkoutservice": 5e5, "currencyservice": 1e5},
                   rwr={"frontend": .46, "currencyservice": .06, "checkoutservice": .05})


def test_baseline_replays_production_fusion_exactly():
    # Same scope fed to the real P2 and to the ablation fuse() must agree.
    svc = [ScopeService(id="service:" + n, sev_m=float(i + 1)) for i, n in enumerate(SERVICES)]
    scope = ScopeResult(incident_id="incident:x", anchor_service="service:frontend",
                        window={"t0": 0, "t1": 1}, services=svc, nodes=[
                            {"id": s.id, "type": "Service", "properties": {}} for s in svc],
                        edges=[{"source": "service:frontend", "target": "service:currencyservice",
                                "type": "DEPENDS_ON", "properties": {}}], evidence_gaps=[],
                        n_series_in_scope=0, truncated=False)
    real = rank_candidates(scope)
    comps = {c.id: c.components for c in real.candidates}
    scores = ab.fuse(comps, ab.ARMS["baseline_rrf"]["weights"])
    assert ab.deployed_order(scores) == [c.id for c in real.candidates]
    for c in real.candidates:
        assert scores[c.id] == pytest.approx(c.score)


def test_severity_only_fixes_the_symptom_case_and_baseline_does_not():
    rec = symptom_case()
    assert ab.replay_case(rec, ab.ARMS["baseline_rrf"])["names"][0] == "frontend"
    assert ab.replay_case(rec, ab.ARMS["severity_only"])["names"][0] == "currencyservice"


def test_rwr_excluding_anchor_removes_the_constant_prior():
    # Equal severity; the only difference is the anchor's random-walk mass.
    comps = _components({"currencyservice": 5.0, "frontend": 5.0},
                        rwr={"frontend": .46, "currencyservice": .06})
    with_anchor = ab.fuse(comps, ab.ARMS["baseline_rrf"]["weights"])
    without = ab.fuse(comps, ab.ARMS["rwr_excl_anchor"]["weights"], anchor="service:frontend",
                      rwr_excludes_anchor=True)
    assert with_anchor["service:frontend"] > with_anchor["service:currencyservice"]
    # With the anchor's restart mass removed, the prior no longer favours it: the
    # equal-severity services tie on severity and only currency keeps an rwr vote.
    assert without["service:currencyservice"] > without["service:frontend"]


def test_zero_weight_removes_component_from_ranking_entirely():
    comps = _components({"currencyservice": 5.0}, rwr={"frontend": .46})
    scores = ab.fuse(comps, {"severity": 1.0})
    assert scores["service:frontend"] == 0.0 and scores["service:currencyservice"] > 0


def test_tie_bounds_are_reported():
    # identical severity -> a pure tie; deployed order is alphabetical (cartservice first)
    rec = _record("re2ob_currencyservice_cpu_1", "currencyservice",
                  severity={n: 6.0 for n in SERVICES})
    r = ab.replay_case(rec, ab.ARMS["severity_only"])
    assert r["optimistic"] == 1 and r["pessimistic"] == len(SERVICES)
    assert r["names"][0] == "cartservice"  # deployed tie order, not the truth
    out = ab.evaluate_arm([rec], ab.ARMS["severity_only"])["overall"]
    assert out["ac@1"] == 0.0 and out["ac@1_optimistic"] == 1.0 and out["ac@1_pessimistic"] == 0.0
    assert out["cases_won_only_on_tie_order"] == 0


def test_deployed_top1_win_in_top_tie_is_counted():
    rec = _record("tie_win", "cartservice", severity={n: 6.0 for n in SERVICES})
    result = ab.evaluate_arm([rec], ab.ARMS["severity_only"])
    assert result["correct_at_1"] == [True]
    assert result["positions"] == [1]
    assert result["overall"]["ac@1"] == 1.0
    assert result["overall"]["ac@1_pessimistic"] == 0.0
    assert result["overall"]["cases_won_only_on_tie_order"] == 1


@pytest.mark.parametrize(
    "severity, truth, names, optimistic, pessimistic, tie_wins",
    [
        # The truth need not have an exact raw candidate ID.
        ({"service:carts-db": 9, "service:frontend": 1},
         "carts", ["carts", "frontend"], 1, 1, 0),
        # Best alias wins even if a lower-scoring alias was inserted first.
        ({"service:carts": 1, "service:carts-db_cpu": 9, "service:frontend": 5},
         "carts", ["carts", "frontend"], 1, 1, 0),
        # Multiple aliases of the truth do not make a nontrivial service tie.
        ({"service:carts": 9, "service:carts-db": 9, "service:frontend": 1},
         "carts", ["carts", "frontend"], 1, 1, 0),
        # Neither the truth nor a tied competitor counts twice in tie bounds.
        ({"service:carts": 9, "service:carts-db": 9,
          "service:frontend": 9, "service:frontend_cpu": 9},
         "carts", ["carts", "frontend"], 1, 2, 1),
        # A tied loss must not count as a win merely because an alias matches.
        ({"service:carts": 9, "service:carts-db": 9, "service:frontend-db": 9},
         "frontend", ["carts", "frontend"], 1, 2, 0),
        # Higher-ranked duplicate competitors consume only one service position.
        ({"service:carts": 9, "service:carts-db": 9, "service:frontend-db": 1},
         "frontend", ["carts", "frontend"], 2, 2, 0),
        ({"service:carts-db": 9}, "missing", ["carts"], None, None, 0),
    ],
)
def test_normalized_service_metrics_and_bounds(severity, truth, names,
                                               optimistic, pessimistic, tie_wins):
    comps = {sid: {"severity": value} for sid, value in severity.items()}
    rec = _record_from_components("aliases", truth, comps)
    arm = ab.ARMS["severity_only"]
    replay = ab.replay_case(rec, arm)
    assert replay["names"] == names
    assert replay["optimistic"] == optimistic
    assert replay["pessimistic"] == pessimistic
    # Normalization must never replace the raw ranking used for replay fidelity.
    assert replay["order"] == ab.recorded_order(rec)
    assert len(replay["order"]) == len(severity)
    assert ab.run_ablation([rec])["replay_fidelity"]["baseline_mismatches"] == 0

    result = ab.evaluate_arm([rec], arm)
    position = names.index(truth) + 1 if truth in names else None
    win = position == 1
    assert result["positions"] == [position]
    assert result["correct_at_1"] == [win]
    overall = result["overall"]
    assert overall["ac@1"] == float(win)
    assert overall["ac@3"] == float(position is not None)
    assert overall["mrr"] == (1 / position if position is not None else 0.0)
    assert overall["ac@1_ci95"] == ab.wilson(int(win), 1)
    assert overall["ac@1_optimistic"] == float(optimistic == 1)
    assert overall["ac@1_pessimistic"] == float(pessimistic == 1)
    assert overall["cases_won_only_on_tie_order"] == tie_wins
    assert result["per_class"]["cpu"]["ac@1"] == float(win)


def test_run_ablation_reports_fidelity_and_pairwise_test():
    records = [symptom_case(f"re2ob_currencyservice_cpu_{i}") for i in range(8)]
    summary = ab.run_ablation(records)
    assert summary["n_cases"] == 8
    assert summary["replay_fidelity"]["baseline_mismatches"] == 0
    arm = summary["arms"]["severity_only"]
    assert arm["ac@1"] == 1.0 and summary["arms"]["baseline_rrf"]["ac@1"] == 0.0
    cmp_ = arm["vs_baseline_ac@1"]
    assert (cmp_["arm_only_correct"], cmp_["baseline_only_correct"]) == (8, 0)
    assert cmp_["mcnemar_exact_p"] == pytest.approx(2 / 256, abs=1e-6)  # reported rounded to 6dp
    assert "baseline_rrf" in ab.render(summary)


def test_fidelity_mismatch_is_flagged():
    rec = symptom_case()
    rec["artifacts"]["ranking"]["candidates"].reverse()
    for i, c in enumerate(rec["artifacts"]["ranking"]["candidates"]):
        c["rank"] = i + 1  # recorded order now disagrees with the recomputed baseline
    summary = ab.run_ablation([rec])
    assert summary["replay_fidelity"]["baseline_mismatches"] == 1
    assert "DO NOT TRUST" in ab.render(summary)


def test_unusable_records_are_skipped_not_scored():
    bad = {"case": "x", "status": "failed", "gt_service": "a", "artifacts": {}}
    summary = ab.run_ablation([bad, symptom_case()])
    assert summary["n_cases"] == 1 and summary["skipped"] == ["x"]
    assert ab.run_ablation([bad])["n_cases"] == 0


def test_mcnemar_and_wilson():
    assert ab.mcnemar_exact(0, 0) == 1.0
    assert ab.mcnemar_exact(5, 5) == 1.0
    assert ab.mcnemar_exact(10, 0) == pytest.approx(2 / 1024)
    lo, hi = ab.wilson(16, 50)
    assert lo == pytest.approx(0.2077, abs=1e-3) and hi == pytest.approx(0.4580, abs=1e-3)


def test_severity_record_fidelity_and_explicit_policy_validation():
    rec = symptom_case()
    rec["rank_policy"] = "severity_only"
    ranking = rec["artifacts"]["ranking"]
    ranking["config"] = {"rank_policy": "severity_only", "rrf_epsilon": 7}
    order = ab.replay_case(rec, ab.ARMS["severity_only"])["order"]
    by_id = {c["id"]: c for c in ranking["candidates"]}
    ranking["candidates"] = [dict(by_id[sid], rank=i + 1) for i, sid in enumerate(order)]
    assert order != ab.replay_case(rec, ab.ARMS["baseline_rrf"])["order"]
    fidelity = ab.run_ablation([rec])["replay_fidelity"]
    assert fidelity["recorded_policy_mismatches"] == 0
    assert fidelity["baseline_mismatches"] == 0
    assert fidelity["recorded_policies"] == ["severity_only"]
    ranking["candidates"][0]["rank"] = 99
    report = ab.run_ablation([rec])
    assert report["replay_fidelity"]["recorded_policy_mismatches"] == 1
    assert "DO NOT TRUST" in ab.render(report)
    rec["rank_policy"] = "baseline_rrf"
    with pytest.raises(ValueError, match="disagree"):
        ab.run_ablation([rec])
    rec["rank_policy"] = ranking["config"]["rank_policy"] = "unknown"
    with pytest.raises(ValueError, match="unknown recorded"):
        ab.run_ablation([rec])


def test_replay_uses_recorded_epsilon_and_scoped_pagerank():
    rec = _record("ties", "frontend", severity={n: 6.0 for n in SERVICES})
    rec["artifacts"]["ranking"]["config"] = {"rrf_epsilon": 2}
    rec["artifacts"]["scope"] = {"nodes": [
        {"id": "service:frontend", "properties": {"pagerank": 10}}]}
    assert ab.replay_case(rec, ab.ARMS["severity_only"])["order"][0] == "service:frontend"
    # epsilon is passed through even for recorded policies, not the current default.
    rec["artifacts"]["ranking"]["config"]["rrf_epsilon"] = 0
    with pytest.raises(ValueError, match="epsilon"):
        ab.replay_case(rec, ab.ARMS["severity_only"])


def test_load_records_from_directory_and_filter(tmp_path):
    for i in range(3):
        rec = symptom_case(f"re2ob_currencyservice_cpu_{i}")
        (tmp_path / f"{rec['case']}.record.json").write_text(json.dumps(rec))
    assert len(ab.load_records(tmp_path)) == 3
    assert [r["case"] for r in ab.load_records(tmp_path, ["re2ob_currencyservice_cpu_1"])] == \
        ["re2ob_currencyservice_cpu_1"]
