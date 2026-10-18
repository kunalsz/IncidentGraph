"""Conservative topology certification without changing shortest BFS depths."""

import json

import pytest

from backend.app.eval import rcaeval_adapter as adapter


def topology(levels, edges):
    return {
        "services": [
            {"id": f"service:{name}", "name": name, "level": level}
            for name, level in levels.items()
        ],
        "depends_on": [
            {"from": f"service:{a}", "to": f"service:{b}"} for a, b in edges
        ],
    }


def ids(*names):
    return {f"service:{name}" for name in names}


def test_dag_shortcut_decreasing_depth_rejects_whole_component():
    # The shortcut makes target shallower than deep despite deep -> target.
    graph = topology(
        {"root": 0, "middle": 1, "deep": 2, "target": 1, "leaf": 2},
        [("root", "middle"), ("middle", "deep"), ("deep", "target"),
         ("root", "target"), ("target", "leaf")],
    )
    assert adapter.dependency_depths(graph) == {
        "service:root": 0, "service:middle": 1, "service:deep": 2,
        "service:target": 1, "service:leaf": 2,
    }
    assert adapter.authoritative_levels(graph) == set()


@pytest.mark.parametrize("cycle_edges", [
    [("a", "b"), ("b", "a")],
    [("a", "a")],
])
def test_root_reachable_equal_depth_cycles_reject_whole_component(cycle_edges):
    # No edge decreases in BFS depth: explicit cycle detection is necessary.
    graph = topology(
        {"root": 0, "a": 1, "b": 1, "leaf": 2},
        [("root", "a"), ("root", "b"), ("b", "leaf")] + cycle_edges,
    )
    assert adapter.dependency_depths(graph) == {
        "service:root": 0, "service:a": 1, "service:b": 1, "service:leaf": 2,
    }
    assert adapter.authoritative_levels(graph) == set()


def test_rootless_cycle_and_its_reachable_tail_stay_unmarked():
    graph = topology(
        {"a": 0, "b": 1, "tail": 2},
        [("a", "b"), ("b", "a"), ("b", "tail")],
    )
    assert adapter.dependency_depths(graph) == {}
    assert adapter.authoritative_levels(graph) == set()


def test_unreachable_cycle_contaminates_root_reachable_part_of_component():
    graph = topology(
        {"root": 0, "leaf": 1, "a": 0, "b": 1},
        [("root", "leaf"), ("a", "b"), ("b", "a"), ("b", "leaf")],
    )
    assert adapter.dependency_depths(graph) == {"service:root": 0, "service:leaf": 1}
    assert adapter.authoritative_levels(graph) == set()


@pytest.mark.parametrize("unsafe_edges", [
    [("bad", "a"), ("a", "b"), ("b", "a")],
    [("bad", "a"), ("a", "b"), ("b", "target"), ("bad", "target")],
    [("bad", "missing")],
    [("missing", "bad")],
])
def test_separate_safe_component_is_retained(unsafe_edges):
    graph = topology(
        {"safe": 0, "leaf": 1, "bad": 0, "a": 1, "b": 2,
         "target": 1, "isolated": 0},
        [("safe", "leaf")] + unsafe_edges,
    )
    assert adapter.authoritative_levels(graph) == ids("safe", "leaf")


def test_equal_depth_dag_edges_are_safe_and_duplicate_edges_do_not_look_cyclic():
    graph = topology(
        {"root": 0, "a": 1, "b": 1},
        [("root", "a"), ("root", "b"), ("a", "b"), ("a", "b")],
    )
    assert adapter.authoritative_levels(graph) == ids("root", "a", "b")


def test_mismatched_level_stays_unmarked_without_rejecting_safe_neighbors():
    graph = topology(
        {"root": 0, "middle": 9, "leaf": 2, "isolated": 0},
        [("root", "middle"), ("middle", "leaf")],
    )
    assert adapter.authoritative_levels(graph) == ids("root", "leaf")


def test_entity_restriction_is_applied_to_depths_and_certification():
    graph = topology(
        {"root": 0, "leaf": 1},
        [("root", "leaf"), ("leaf", "missing")],
    )
    assert adapter.authoritative_levels(graph) == set()
    assert adapter.authoritative_levels(graph, {"root", "leaf"}) == ids("root", "leaf")
    assert adapter.dependency_depths(graph, {"leaf"}) == {}
    assert adapter.authoritative_levels(graph, {"leaf"}) == set()


def test_shipped_topology_keeps_all_previously_certified_services():
    graph = json.loads(adapter.OB_TOPOLOGY.read_text())
    depths = adapter.dependency_depths(graph)
    expected = {svc["id"] for svc in graph["services"] if svc["name"] != "istio-init"}
    assert expected == {
        svc["id"] for svc in graph["services"]
        if svc["id"] in depths and int(svc["level"]) == depths[svc["id"]]
    }
    assert depths["service:cartservice"] == 2  # BFS, not longest-path depth 3.
    assert depths["service:redis"] == 3
    assert "service:istio-init" not in depths
    assert adapter.authoritative_levels(graph) == expected
    assert adapter.authoritative_levels(graph, {svc["name"] for svc in graph["services"]}) == expected
