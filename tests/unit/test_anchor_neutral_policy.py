"""Anchor-neutral walk and its opt-in P2 policy.

Diagnosis behind it: `rcaeval_runner` hardcodes `anchor_service="service:frontend"`
because RCAEval supplies no symptom, so the anchored walk elects frontend in
90/90 archived cases and the true service is never its rank-1. Predeclared
candidate C3 (`output/anchor-prior-*/protocol.json`) replaces that prior with a
uniform-restart walk and scored 65.6% AC@1 versus baseline_rrf's 37.8% on the
same archived components. These tests pin the mechanism, not the number.
"""

import pytest

from backend.app.engine.contracts import ScopeResult, ScopeService
from backend.app.engine.p2_rank import rank_candidates
from backend.app.graph.scorers import neutral_walk, random_walk
from config import tuning


def edge(source, target):
    return {"source": source, "target": target, "type": "DEPENDS_ON", "properties": {}}


class TestNeutralWalk:
    def test_conserves_mass_and_ignores_anchor_preference(self):
        ids = ["service:frontend", "service:checkout", "service:payment"]
        edges = [edge("service:frontend", "service:checkout"),
                 edge("service:checkout", "service:payment")]
        mass = neutral_walk(ids, edges, restart=0.15, iterations=20)
        assert sum(mass.values()) == pytest.approx(1.0)
        # no service is privileged by identity: the deepest node carries more
        # mass than the entry node, the opposite of the anchored walk
        assert mass["service:payment"] > mass["service:frontend"]
        anchored = random_walk(ids, edges, "service:frontend", restart=0.15, iterations=20)
        assert anchored["service:frontend"] > anchored["service:payment"]

    def test_symmetric_topology_yields_equal_mass(self):
        mass = neutral_walk(["a", "b"], [edge("a", "b"), edge("b", "a")], iterations=5)
        assert mass["a"] == pytest.approx(mass["b"])

    def test_dead_end_keeps_its_own_mass(self):
        # The anchored walk leaks dead-end mass to the anchor; here it stays.
        mass = neutral_walk(["a", "b"], [], restart=0.15, iterations=3)
        assert mass["a"] == pytest.approx(mass["b"])

    def test_empty_scope_and_bad_parameters(self):
        assert neutral_walk([], []) == {}
        with pytest.raises(ValueError):
            neutral_walk(["a"], [], restart=0)
        with pytest.raises(ValueError):
            neutral_walk(["a"], [], iterations=0)


def _scope(anchor="service:frontend"):
    return ScopeResult(
        incident_id="anchor-neutral", anchor_service=anchor, window={"t0": 0, "t1": 2},
        services=[ScopeService(id="service:frontend", sev_m=1.0),
                  ScopeService(id="service:checkout", sev_m=1.0),
                  ScopeService(id="service:payment", sev_m=1.0)],
        nodes=[{"id": f"service:{n}", "type": "Service", "properties": {}}
               for n in ("frontend", "checkout", "payment")],
        edges=[edge("service:frontend", "service:checkout"),
               edge("service:checkout", "service:payment")],
        evidence_gaps=[], n_series_in_scope=0, truncated=False)


class TestAnchorNeutralPolicy:
    def test_policy_keeps_rwr_voting_and_only_changes_the_walk(self):
        baseline = rank_candidates(_scope())
        neutral = rank_candidates(_scope(), rank_policy="rrf_anchor_neutral")
        assert neutral.config["rank_policy"] == "rrf_anchor_neutral"
        # the component still votes; it is not dropped the way C1 drops it
        assert neutral.config["component_weights"]["rwr"] == 1.0
        assert neutral.config["component_weights"] == baseline.config["component_weights"]
        by_id = {c.id: c for c in baseline.candidates}
        for candidate in neutral.candidates:
            assert candidate.components["phi1"] == by_id[candidate.id].components["phi1"]
            assert candidate.components["severity"] == by_id[candidate.id].components["severity"]
            # same components, different walk mass
            assert candidate.components["rwr"] != by_id[candidate.id].components["rwr"]
            assert candidate.evidence == by_id[candidate.id].evidence

    def test_equal_severity_the_prior_decides_and_reverses(self):
        # Identical severities: under the anchored prior the anchor leads; under
        # the neutral walk the topology does. This is exactly the defect.
        baseline = rank_candidates(_scope())
        neutral = rank_candidates(_scope(), rank_policy="rrf_anchor_neutral")
        assert baseline.candidates[0].id == "service:frontend"
        assert neutral.candidates[0].id == "service:payment"
        assert {c.id for c in neutral.candidates} == {c.id for c in baseline.candidates}

    def test_default_is_unchanged_and_unknown_policies_still_reject(self):
        assert rank_candidates(_scope()) == rank_candidates(_scope(), rank_policy="baseline_rrf")
        assert tuning.P2_RANK_POLICY == "baseline_rrf"
        with pytest.raises(ValueError):
            rank_candidates(_scope(), rank_policy="not_a_policy")

    def test_walk_helper_matches_the_policy_components(self):
        # The policy must use the shared helper, not a private copy.
        mass = neutral_walk(["service:frontend", "service:checkout", "service:payment"],
                            _scope().edges)
        by_id = {c.id: c for c in rank_candidates(_scope(), rank_policy="rrf_anchor_neutral").candidates}
        for sid, value in mass.items():
            assert by_id[sid].components["rwr"] == pytest.approx(value)
