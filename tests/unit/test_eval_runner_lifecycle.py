import json
from types import SimpleNamespace

import pytest

from backend.app.eval import rcaeval_runner as runner


def test_partial_load_cleanup_and_failed_compatibility_output(tmp_path,monkeypatch):
    case = tmp_path/"re2ob_a_cpu_1"
    case.mkdir()
    deleted=[]
    db=SimpleNamespace(execute_command=lambda *args:deleted.append(args))
    monkeypatch.setattr(runner,"_data_provenance",lambda p:{"sha256":{}})
    monkeypatch.setattr(runner.adapter,"load_case",lambda *args:(_ for _ in ()).throw(RuntimeError("secret=not_exported")))
    out=tmp_path/"out"
    out.mkdir()
    (out/f"{case.name}.json").write_text('{"0":["stale_cpu"]}')
    result=runner.run_case(case,out,graph_factory=lambda name:object(),db_factory=lambda:db)
    assert result["status"]=="failed"
    assert result["failed_stage"]=="load"
    assert result["cleanup"]=="deleted"
    assert len(deleted)==1 and deleted[0][0]=="GRAPH.DELETE"
    assert json.loads((out/f"{case.name}.json").read_text())=={"0":[]}
    assert "secret" not in json.dumps(result)
    assert result["resources"]["llm_calls"]==0


def test_sessions_are_unique_even_same_case(tmp_path,monkeypatch):
    case=tmp_path/"re2ob_a_cpu_1"
    case.mkdir()
    names=[]
    monkeypatch.setattr(runner,"_data_provenance",lambda p:{})
    monkeypatch.setattr(runner.adapter,"load_case",lambda *args:(_ for _ in ()).throw(RuntimeError()))
    def graph(name):
        names.append(name)
        return object()
    db=SimpleNamespace(execute_command=lambda *args:None)
    for _ in range(2):
        runner.run_case(case,tmp_path/"out",run_id="same",graph_factory=graph,db_factory=lambda:db)
    assert len(set(names))==2


@pytest.mark.parametrize("policy", ["baseline_rrf", "severity_only"])
def test_success_policy_forwarding_and_persisted_provenance(tmp_path, monkeypatch, policy):
    from backend.app.engine import runner as engine_runner
    from backend.app.engine.contracts import Investigation, ScopeResult, ScopeService
    from backend.app.engine.p2_rank import rank_candidates

    case = tmp_path / "re2ob_a_cpu_1"
    case.mkdir()
    monkeypatch.setattr(runner, "_data_provenance", lambda p: {})
    monkeypatch.setattr(runner.adapter, "load_case", lambda *a: {
        "incident_id": "policy", "t0_ms": 0, "t1_ms": 2, "counts": {},
        "sample_diagnostics": {}, "aggregate_diagnostics": {}, "window_frame": None})
    scope = ScopeResult(incident_id="policy", anchor_service="service:frontend",
        window={"t0": 0, "t1": 2}, services=[ScopeService(id="service:frontend")],
        nodes=[{"id": "service:frontend", "type": "Service", "properties": {}}],
        edges=[], evidence_gaps=[], n_series_in_scope=0, truncated=False)
    calls = []
    def rank(inv, scoped, *, rank_policy):
        calls.append(rank_policy)
        return rank_candidates(scoped, rank_policy=rank_policy)
    monkeypatch.setattr(runner, "rank_investigation", rank)
    def execute(*a, phase_functions, **kw):
        inv = phase_functions[0]("test")
        assert isinstance(inv, Investigation)
        ranked = phase_functions[2](inv, scope)
        return SimpleNamespace(status="completed", events=[], investigation=inv,
            scope=scope, ranking=ranked, invest=None, validated=None, fused=None)
    monkeypatch.setattr(engine_runner, "run_investigation", execute)
    monkeypatch.setattr(runner.ev, "candidate_universes", lambda *a: {
        "service": {"count": 1, "identities": ["frontend"]},
        "metric": {"count": 0, "identities": []}})
    db = SimpleNamespace(execute_command=lambda *a: None)
    result = runner.run_case(case, tmp_path / "out", rank_policy=policy,
                            graph_factory=lambda n: object(), db_factory=lambda: db)
    assert result["status"] == "ok"
    assert calls == [policy]
    stored = json.loads((tmp_path / "out" / f"{case.name}.record.json").read_text())
    assert stored["rank_policy"] == policy
    assert stored["artifacts"]["ranking"]["config"]["rank_policy"] == policy
    assert json.loads(stored["handoffs"]["rank"])["payload"]["config"]["rank_policy"] == policy


def test_failure_keeps_requested_policy(tmp_path, monkeypatch):
    case = tmp_path / "re2ob_a_cpu_1"
    case.mkdir()
    monkeypatch.setattr(runner.adapter, "load_case", lambda *a: (_ for _ in ()).throw(RuntimeError()))
    monkeypatch.setattr(runner, "_data_provenance", lambda p: {})
    result = runner.run_case(case, tmp_path / "out", rank_policy="severity_only",
        graph_factory=lambda n: object(), db_factory=lambda: SimpleNamespace(execute_command=lambda *a: None))
    assert result["status"] == "failed"
    assert result["rank_policy"] == "severity_only"
    assert result["rank_policy_version"]


@pytest.mark.parametrize("policy", ["baseline_rrf", "severity_only"])
def test_eval_cli_forwards_rank_policy_separately(tmp_path, monkeypatch, policy):
    import sys
    from scripts import eval as cli
    calls = []
    monkeypatch.setattr(runner, "run", lambda *a, **kw: calls.append(kw) or 0)
    monkeypatch.setattr(sys, "argv", ["eval", "--rcaeval", str(tmp_path), "--rank-policy", policy])
    assert cli.main() == 0
    assert calls[0]["rank_policy"] == policy
    monkeypatch.setattr(sys, "argv", ["eval", "--rcaeval", str(tmp_path)])
    assert cli.main() == 0
    assert calls[1]["rank_policy"] == "baseline_rrf"
    monkeypatch.setattr(sys, "argv", ["eval", "--replay", "records.json", "--rank-policy", "severity_only"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2


def test_eval_fixture_policy_and_replay_policy_are_independent(monkeypatch):
    import sys
    from scripts import eval as cli
    from backend.app.eval import replay
    seen = []
    def fixture(name, *, rank_policy):
        seen.append(rank_policy)
        return {"root_rank": 1, "top3_pass": True, "mrr": 1.0}
    monkeypatch.setattr(cli, "evaluate_fixture", fixture)
    monkeypatch.setattr(sys, "argv", ["eval", "--fixtures", "--rank-only",
                                     "--rank-policy", "severity_only"])
    assert cli.main() == 0
    assert seen == ["severity_only"] * len(cli.FIXTURES)
    forwarded = []
    monkeypatch.setattr(replay, "main", lambda argv: forwarded.append(argv) or 0)
    monkeypatch.setattr(sys, "argv", ["eval", "--replay", "frozen.json", "--policy", "shadow"])
    assert cli.main() == 0
    assert forwarded[0][-2:] == ["--policy", "shadow"]
