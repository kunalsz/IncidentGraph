import json
from types import SimpleNamespace

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
