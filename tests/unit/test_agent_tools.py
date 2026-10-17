"""Unit tests for the tool registry: read-only boundary, surface budget,
search ranking, dispatch behavior (no graph needed)."""

import pytest

from backend.app.agent.tools import (TOOLS, WRITE_PATTERN, call_tool,
                                     search_tools, surface_tokens)


class TestRegistryInvariants:
    def test_no_write_pattern_names(self):
        for name in TOOLS:
            assert not WRITE_PATTERN.match(name), name

    def test_all_specs_complete(self):
        for spec in TOOLS.values():
            assert spec.signature and spec.description
            assert callable(spec.fn)
            assert spec.tier in (0, 1, 2)

    def test_meta_tools_not_dispatchable(self):
        assert "search_tools" not in TOOLS
        assert "call_tool" not in TOOLS

    def test_all_ten_tier1_tools_registered(self):
        # the §4.5 surface is complete: 8 graph tools + check_hypothesis +
        # get_evidence_path (registered at M3)
        assert "check_hypothesis" in TOOLS
        assert "get_evidence_path" in TOOLS

    def test_surface_budget(self):
        assert surface_tokens((1,)) < 2000
        assert surface_tokens((1,)) + surface_tokens((0,)) < 8000

    def test_tier1_count(self):
        assert len([s for s in TOOLS.values() if s.tier == 1]) == 10


class TestSearchTools:
    def test_runbook_query_top_hit(self):
        results = search_tools("runbook owner")
        assert results
        assert results[0]["name"] == "get_runbook_and_owner"

    def test_severity_query(self):
        assert search_tools("severity vector")[0]["name"] == "get_severity"

    def test_short_tokens_dropped(self):
        # "a", "of" < 3 chars -> no tokens -> full surface returned
        assert len(search_tools("a of")) == len(TOOLS)

    def test_tier_filter(self):
        results = search_tools("plan", filters={"tier": 0})
        assert [r["name"] for r in results] == ["draft_investigation_plan"]

    def test_every_result_carries_contract_fields(self):
        for row in search_tools("service"):
            assert set(row) >= {"name", "signature", "description", "tier",
                                "callable_now"}


class TestCallTool:
    def test_unknown_tool_returns_observation(self):
        result = call_tool("nope", graph=None)
        assert "error" in result and "known" in result

    def test_bad_arguments_return_observation(self):
        result = call_tool("get_severity", graph=object(), args={"bogus": 1})
        assert result["code"] == "context_required"

    def test_tool_failure_never_raises(self):
        result = call_tool("get_severity", graph=None, args={"node_id": "service:x"})
        assert "error" in result  # graph None -> internal failure reported

    def test_plan_tool_works_without_graph(self):
        result = call_tool("draft_investigation_plan", graph=None,
                           args={"symptom": "checkout latency up",
                                 "anchor": "service:checkout"})
        assert len(result["steps"]) >= 5
        assert result["anchor"] == "service:checkout"
