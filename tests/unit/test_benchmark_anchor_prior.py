"""Anchor provenance: the benchmark never runs an anchored prior on a cold scope.

The walk prior is only meaningful when the anchor is derived from the case
(an alert or a symptom). The benchmark harness fixes the anchor to an entry
service because RCAEval supplies no symptom, so it must not inherit the product
default. These tests pin that split, the audit labels, and the fact that the
policy is never switched silently.
"""

import inspect

import pytest

from backend.app.engine.contracts import ScopeResult, ScopeService
from backend.app.engine.p2_rank import rank_candidates
from backend.app.eval import rcaeval_runner
from config import tuning


def edge(source, target):
    return {"source": source, "target": target, "type": "DEPENDS_ON", "properties": {}}


def scope():
    return ScopeResult(
        incident_id="anchor-provenance", anchor_service="service:frontend", window={"t0": 0, "t1": 2},
        services=[ScopeService(id="service:frontend", sev_m=1.0),
                  ScopeService(id="service:checkout", sev_m=1.0),
                  ScopeService(id="service:payment", sev_m=1.0)],
        nodes=[{"id": f"service:{n}", "type": "Service", "properties": {}}
               for n in ("frontend", "checkout", "payment")],
        edges=[edge("service:frontend", "service:checkout"),
               edge("service:checkout", "service:payment")],
        evidence_gaps=[], n_series_in_scope=0, truncated=False)


class TestAnchorProvenance:
    def test_cold_scope_with_anchored_policy_is_labelled_and_warned(self):
        result = rank_candidates(scope(), rank_policy="baseline_rrf", case_anchor=False)
        assert result.config["anchor_source"] == "none"
        assert result.config["rwr_prior"] == "personalized"
        assert result.config["anchor_prior_warning"]
        # the warning is an audit entry, not a silent policy change
        assert result.config["rank_policy"] == "baseline_rrf"

    def test_case_anchor_is_labelled_without_warning(self):
        result = rank_candidates(scope(), rank_policy="baseline_rrf", case_anchor=True)
        assert result.config["anchor_source"] == "case"
        assert result.config["rwr_prior"] == "personalized"
        assert result.config["anchor_prior_warning"] is None

    def test_anchorless_policy_never_warns(self):
        for declared in (False, True, None):
            result = rank_candidates(scope(), rank_policy="rrf_anchor_neutral",
                                     case_anchor=declared)
            assert result.config["rwr_prior"] == "anchorless_uniform"
            assert result.config["anchor_prior_warning"] is None

    def test_non_voting_prior_is_not_warned(self):
        # severity_only gives rwr zero weight, so there is no anchored prior to
        # warn about even on a declared cold scope.
        result = rank_candidates(scope(), rank_policy="severity_only", case_anchor=False)
        assert result.config["component_weights"]["rwr"] == 0.0
        assert result.config["anchor_prior_warning"] is None

    def test_undeclared_keeps_historical_behaviour(self):
        # Existing callers that know nothing about anchors must be unaffected.
        plain = rank_candidates(scope())
        undeclared = rank_candidates(scope(), case_anchor=None)
        assert plain == undeclared
        assert plain.config["anchor_source"] == "undeclared"
        assert plain.config["anchor_prior_warning"] is None

    def test_declaration_does_not_move_scores(self):
        # Provenance only: the same policy must rank identically either way.
        cold = rank_candidates(scope(), rank_policy="baseline_rrf", case_anchor=False)
        described = rank_candidates(scope(), rank_policy="baseline_rrf", case_anchor=True)
        assert [(c.id, c.score) for c in cold.candidates] == \
               [(c.id, c.score) for c in described.candidates]


class TestRunnerDefaults:
    def test_benchmark_default_is_the_policy_that_transfers_across_corpora(self):
        # severity_only is the only policy that led on both SockShop (85.0%) and
        # Online Boutique (78.9%); either walk variant wins on one corpus and
        # loses on the other, so neither can be the default. The product default
        # is unchanged because real incidents have a case-derived anchor.
        assert rcaeval_runner.BENCHMARK_RANK_POLICY == "severity_only"
        assert tuning.P2_RANK_POLICY == "baseline_rrf"
        assert "rrf_anchor_neutral" in tuning.P2_RANK_POLICIES  # still selectable
        for function in (rcaeval_runner.run_case, rcaeval_runner.run):
            default = inspect.signature(function).parameters["rank_policy"].default
            assert default == rcaeval_runner.BENCHMARK_RANK_POLICY

    def test_benchmark_declares_the_cold_scope(self):
        source = inspect.getsource(rcaeval_runner.run_case)
        assert "case_anchor=False" in source
        assert "BENCHMARK_RANK_POLICY" in inspect.getsource(rcaeval_runner)

    def test_explicit_override_is_still_honoured(self):
        # Comparability with archived runs requires the override to work.
        result = rank_candidates(scope(), rank_policy="baseline_rrf", case_anchor=False)
        assert result.config["rank_policy"] == "baseline_rrf"
        assert rank_candidates(scope(), rank_policy="severity_only").config["rank_policy"] == "severity_only"
        with pytest.raises(ValueError):
            rank_candidates(scope(), rank_policy="baseline_rrf_v2")
