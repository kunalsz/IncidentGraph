"""A second corpus must not require engine edits, and must not smuggle labels.

RE2-SS (SockShop) differs from RE2-OB in three ways that matter: a different case
prefix, a different entry service name, and NO traces at all. These tests pin the
generalisation that makes it runnable, and the invariants of the authored
SockShop topology (acyclic, monotone levels, no benchmark labels).
"""

import json
from pathlib import Path

import pytest

from backend.app.eval import rcaeval_adapter as adapter
from backend.app.eval import rcaeval_runner as runner

TOPOLOGY = json.loads(adapter.SUITE_TOPOLOGY["re2ss_"].read_text())


class TestCaseNamesAcrossSuites:
    @pytest.mark.parametrize("case,expected", [
        ("re2ob_checkoutservice_cpu_1", ("checkoutservice", "cpu", "1")),
        ("re2ss_carts_delay_2", ("carts", "delay", "2")),
        ("re2ss_front-end_loss_3", ("front-end", "loss", "3")),
        ("re2tt_ts-travel-service_mem_1", ("ts-travel-service", "mem", "1")),
    ])
    def test_parses_every_known_suite(self, case, expected):
        assert runner.parse_case_name(case) == expected
        assert runner.suite_of(case) is not None

    def test_rejects_unknown_or_malformed_names(self):
        for case in ("re9xx_a_cpu_1", "re2ob_a_cpu", "a_cpu_1", "re2ob_a_cpu_x"):
            with pytest.raises(ValueError):
                runner.parse_case_name(case)
        assert runner.suite_of("random_dir") is None

    def test_discovery_accepts_any_registered_suite(self, tmp_path):
        for name in ("re2ob_a_cpu_1", "re2ss_b_cpu_1", "notes", "re2ss_b_cpu_1.json"):
            (tmp_path / name).mkdir()
        (tmp_path / "re2ss_b_cpu_1.json").rmdir()
        (tmp_path / "re2ss_b_cpu_1.json").write_text("{}")
        found = [p.name for p in runner.discover_cases(tmp_path)]
        assert found == ["re2ob_a_cpu_1", "re2ss_b_cpu_1"]


class TestTopologyRegistry:
    def test_each_suite_resolves_to_its_own_topology(self):
        assert adapter.topology_for("re2ob_x_cpu_1").name == "rcaeval_ob_topology.json"
        assert adapter.topology_for("re2ss_carts_cpu_1").name == "rcaeval_ss_topology.json"

    def test_unknown_suite_raises_rather_than_inventing_edges(self):
        with pytest.raises(ValueError, match="no authored topology"):
            adapter.topology_for("re2zz_x_cpu_1")

    def test_missing_file_is_an_explicit_error(self, monkeypatch, tmp_path):
        monkeypatch.setitem(adapter.SUITE_TOPOLOGY, "re2ss_", tmp_path / "absent.json")
        with pytest.raises(FileNotFoundError, match="missing"):
            adapter.topology_for("re2ss_carts_cpu_1")


class TestDefaultAnchor:
    def test_prefers_the_corpus_entry_service(self):
        assert runner.default_anchor({"front-end", "carts", "orders"}) == "service:front-end"
        assert runner.default_anchor({"frontend", "cartservice"}) == "service:frontend"

    def test_falls_back_deterministically_without_an_entry_service(self):
        assert runner.default_anchor({"zeta", "alpha"}) == "service:alpha"


class TestSockShopTopology:
    def test_edges_reference_declared_services_and_are_acyclic(self):
        declared = {s["id"] for s in TOPOLOGY["services"]}
        edges = [(d["from"], d["to"]) for d in TOPOLOGY["depends_on"]]
        assert all(a in declared and b in declared for a, b in edges)
        assert len(set(edges)) == len(edges) and all(a != b for a, b in edges)
        order = {s["id"]: s["level"] for s in TOPOLOGY["services"]}
        # Kahn: no directed cycle, and every edge is non-decreasing in level
        adjacency = {}
        indegree = {sid: 0 for sid in declared}
        for a, b in edges:
            adjacency.setdefault(a, set()).add(b)
            indegree[b] += 1
        pending = [sid for sid, deg in indegree.items() if deg == 0]
        visited = 0
        while pending:
            node = pending.pop()
            visited += 1
            for neighbour in adjacency.get(node, ()):
                indegree[neighbour] -= 1
                if indegree[neighbour] == 0:
                    pending.append(neighbour)
        assert visited == len(declared), "cycle in the SockShop topology"
        assert all(order[b] >= order[a] for a, b in edges), "level decreases along an edge"

    def test_certification_marks_every_connected_service(self):
        certified = adapter.authoritative_levels(TOPOLOGY)
        names = {s["name"] for s in TOPOLOGY["services"] if s["id"] in certified}
        assert "front-end" in names and "rabbitmq" in names
        assert names == {s["name"] for s in TOPOLOGY["services"]} - {"rabbitmq-exporter"}

    def test_no_benchmark_labels_leak_into_the_graph_data(self):
        text = adapter.SUITE_TOPOLOGY["re2ss_"].read_text().lower()
        for label in ("cpu", "mem", "disk", "socket", "delay", "loss",
                      "cpu_1", "delay_2", "_1", "_2", "_3"):
            assert label not in text, f"fault/instance label {label!r} in topology"

    def test_provenance_states_the_basis_and_the_missing_modality(self):
        provenance = TOPOLOGY["_provenance"].lower()
        assert "no traces" in provenance
        assert "published_architecture" in provenance or "public sockshop" in provenance
        assert {d.get("basis") for d in TOPOLOGY["depends_on"]} <= {"published_architecture", "db_naming"}
