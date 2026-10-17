"""Unit tests for the P4 deterministic validators (pure functions)."""

import pytest

from backend.app.graph import validators as v


class TestLevelVeto:
    def test_deeper_cause_allowed(self):
        # payment (level 2) causing checkout (level 1): the physical direction
        assert v.level_veto(2, 1)["verdict"] == "pass"

    def test_same_level_allowed(self):
        assert v.level_veto(2, 2)["verdict"] == "pass"

    def test_shallower_cause_rejected(self):
        # frontend (level 0) claimed as cause of payment (level 2): veto
        check = v.level_veto(0, 2)
        assert check["verdict"] == "fail"
        assert "physically implausible" in check["detail"]


class TestCycleRejection:
    def test_direct_cycle_rejected(self):
        assert v.cycle_rejection([("a", "b")], ("b", "a"))["verdict"] == "fail"

    def test_transitive_cycle_rejected(self):
        edges = [("a", "b"), ("b", "c")]
        assert v.cycle_rejection(edges, ("c", "a"))["verdict"] == "fail"

    def test_acyclic_addition_allowed(self):
        edges = [("a", "b")]
        assert v.cycle_rejection(edges, ("c", "a"))["verdict"] == "pass"


class TestOnsetAlignment:
    def test_within_delta(self):
        onset = 1_000_000
        assert v.onset_alignment([onset - 10_000, onset + 60_000], onset)["verdict"] == "pass"

    def test_outside_delta_is_partial(self):
        onset = 1_000_000
        check = v.onset_alignment([onset + 10 * 60_000], onset)
        assert check["verdict"] == "partial"

    def test_no_evidence_fails(self):
        assert v.onset_alignment([], 0)["verdict"] == "fail"


class TestSvarLite:
    def test_present_excursion_passes(self):
        records = [{"z": 6.1, "gamma": 60.5, "direction": "up"},
                   {"z": 5.2, "gamma": 14.6, "direction": "up"}]
        assert v.svar_lite(records)["verdict"] == "pass"

    def test_no_excursion_fails(self):
        assert v.svar_lite([{"z": 1.2, "gamma": 2.0}])["verdict"] == "fail"

    def test_no_records_fails(self):
        assert v.svar_lite([])["verdict"] == "fail"

    def test_no_gamma_is_partial(self):
        assert v.svar_lite([{"z": 5.0, "gamma": None}])["verdict"] == "partial"

    def test_gamma_inside_baseline_spread_fails(self):
        # a 3-sigma z whose absolute excursion is inside the baseline IQR
        assert v.svar_lite([{"z": 4.0, "gamma": 1.0}])["verdict"] == "fail"

    def test_direction_flip_is_partial(self):
        records = [{"z": 5.0, "gamma": 9.0, "direction": "up"},
                   {"z": 4.0, "gamma": 9.0, "direction": "down"}]
        assert v.svar_lite(records)["verdict"] == "partial"

    def test_direction_check_is_per_series(self):
        # latency up on one series while an unrelated resource metric moves
        # down is legitimate — must not demote the candidate
        records = [
            {"z": 5.0, "gamma": 9.0, "direction": "up", "column": "svc_latency"},
            {"z": 4.0, "gamma": 9.0, "direction": "down", "column": "svc_cpu"},
        ]
        assert v.svar_lite(records)["verdict"] == "pass"

    def test_same_series_direction_flip_still_partial(self):
        records = [
            {"z": 5.0, "gamma": 9.0, "direction": "up", "column": "svc_latency"},
            {"z": 4.0, "gamma": 9.0, "direction": "down", "column": "svc_latency"},
        ]
        assert v.svar_lite(records)["verdict"] == "partial"


class TestSymptomExplanation:
    def test_own_anomaly_passes(self):
        assert v.symptom_explanation("service:payment", "service:checkout",
                                     True, False, False)["verdict"] == "pass"

    def test_forwarding_node_fails(self):
        assert v.symptom_explanation("service:frontend", "service:checkout",
                                     False, False, False)["verdict"] == "fail"

    def test_anchor_itself_is_partial(self):
        assert v.symptom_explanation("service:checkout", "service:checkout",
                                     True, False, False)["verdict"] == "partial"


class TestNoEvidenceCap:
    def test_cap_applied_without_decisive_modality(self):
        check = v.no_evidence_cap(False, False, False)
        assert check["verdict"] == "partial"
        assert check["confidence_cap"] == pytest.approx(0.4)

    def test_any_modality_passes(self):
        assert v.no_evidence_cap(True, False, False)["verdict"] == "pass"
        assert v.no_evidence_cap(False, True, False)["verdict"] == "pass"
        assert v.no_evidence_cap(False, False, True)["verdict"] == "pass"


class TestChainCoherence:
    def test_mapping(self):
        assert v.chain_coherence(2, 3) == 1.0
        assert v.chain_coherence(3, 3) == 1.0
        assert v.chain_coherence(1, 3) == 0.5
        assert v.chain_coherence(0, 3) == 0.0

    def test_unverifiable_is_neutral(self):
        assert v.chain_coherence(0, 0) == 0.5


class TestAggregation:
    def test_fail_dominates(self):
        checks = [{"verdict": "pass"}, {"verdict": "fail"}, {"verdict": "partial"}]
        assert v.aggregate_verdict(checks) == "fail"

    def test_partial_demotes(self):
        assert v.aggregate_verdict([{"verdict": "pass"}, {"verdict": "partial"}]) == "partial"

    def test_all_pass(self):
        assert v.aggregate_verdict([{"verdict": "pass"}]) == "pass"

    def test_lowest_cap_wins(self):
        checks = [{"confidence_cap": 0.7}, {"confidence_cap": 0.4}]
        assert v.confidence_cap_from_checks(checks) == pytest.approx(0.4)
        assert v.confidence_cap_from_checks([{"verdict": "pass"}]) is None
        assert v.apply_cap(0.9, 0.4) == pytest.approx(0.4)
        assert v.apply_cap(0.3, 0.4) == pytest.approx(0.3)
        assert v.apply_cap(0.3, None) == pytest.approx(0.3)


class TestFinite:
    def test_coercions(self):
        assert v.finite(None) == 0.0
        assert v.finite(float("nan")) == 0.0
        assert v.finite(float("inf")) == 0.0
        assert v.finite("3.2") == pytest.approx(3.2)
        assert v.finite(7) == 7.0
