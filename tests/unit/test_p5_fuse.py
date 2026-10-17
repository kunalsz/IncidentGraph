"""Unit tests for P5 fusion math and confidence typing (hand-computed)."""

import pytest

from backend.app.engine.contracts import (CheckResult, EvidencePath,
                                          HypothesisRecord, Investigation,
                                          InvestigationWindow, InvestResult,
                                          RankedCandidate, RankResult,
                                          ValidateResult, ValidatedCandidate, HypothesisValidation)
from backend.app.engine.p5_fuse import (fuse_investigation, fuse_score,
                                        rank_normalized_d, typed_confidence,
                                        rwr_top3_share)
from config import tuning
from backend.app.engine import policy
from backend.app.engine.contracts import ScopeRef
REF = ScopeRef(incident_id="i", fingerprint="a" * 64)

INV = Investigation(incident_id="i", session_graph="incidentgraph:session:t",
                    symptom="s", anchor_service="service:s0", budget={},
                    window=InvestigationWindow(t0=0, t1=1))


class TestFusionMath:
    def test_rank_normalized_d(self):
        assert rank_normalized_d(1, 7) == 1.0
        assert rank_normalized_d(7, 7) == 0.0
        assert rank_normalized_d(4, 7) == pytest.approx(0.5)
        assert rank_normalized_d(1, 1) == 1.0

    def test_fuse_score_hand_computed(self):
        # 0.35*1 + 0.25*0.8 + 0.20*0.5 + 0.20*1 = 0.85
        assert fuse_score(1.0, 0.8, 0.5, 1.0) == pytest.approx(0.85)

    def test_weights_match_published_config(self):
        assert tuning.FUSE_WEIGHTS == {"deterministic": 0.35, "reflection": 0.25,
                                       "chain_coherence": 0.20, "validator": 0.20}


class TestTypedConfidence:
    def test_rule_ceilings(self):
        assert typed_confidence("cross_modal_consensus") == pytest.approx(0.85)
        assert typed_confidence("single_modality") == pytest.approx(0.7)
        assert typed_confidence("inference_only") == pytest.approx(0.4)
        assert typed_confidence("unknown_rule") == pytest.approx(0.4)

    def test_extra_caps_only_lower(self):
        assert typed_confidence("cross_modal_consensus",
                                extra_cap=0.4) == pytest.approx(0.4)
        assert typed_confidence("inference_only",
                                extra_cap=0.9) == pytest.approx(0.4)


def _ranking(rwr_values):
    candidates = [RankedCandidate(id=f"service:s{i}", rank=i + 1, score=1.0 - i * 0.1,
                                  components={"rwr": value}, component_ranks={},
                                  evidence=[], rule_flags=[], selected=True)
                  for i, value in enumerate(rwr_values)]
    return RankResult(incident_id="i", scope_ref=REF, candidates=candidates, branch_ids=[c.id for c in candidates],
                      indicators=[], audit=[], n_candidates=len(candidates),
                      truncated=False, config={})


def _invest(*hypotheses):
    return InvestResult(incident_id="i", scope_ref=REF, branches=[], hypotheses=list(hypotheses),
                        n_branches=0, n_hypotheses=len(hypotheses))


def _hypothesis(hid, candidate, rule="cross_modal_consensus", confidence=0.85,
                reward=0.8):
    return HypothesisRecord(id=hid, scope_ref=REF, candidate=candidate, text="t",
                            confidence=confidence, confidence_rule=rule,
                            reward=reward, evidence=["m", "l"])


class TestRwrShare:
    def test_top3_share(self):
        ranking = _ranking([0.5, 0.25, 0.15, 0.05, 0.05])
        # (0.5 + 0.25 + 0.15) / 1.0
        assert rwr_top3_share(ranking) == pytest.approx(0.9)

    def test_zero_mass(self):
        assert rwr_top3_share(_ranking([0.0, 0.0])) == 0.0


class TestFuseInvestigation:
    def _validated(self, candidate, verdict="pass", coherence=0.5, cap=None):
        assessment = HypothesisValidation(
            hypothesis_id="h1", candidate=candidate, scope_ref=REF,
            proposal_fingerprint=policy.proposal_fingerprint(_hypothesis("h1", candidate)),
            checks=[CheckResult(name="level_veto", verdict="pass")],
            evidence_path=EvidencePath(from_id="a", to_id=candidate, found=True),
            chain_coherence=coherence, confidence_cap=cap, verdict=verdict,
            evidence_scope=INV.session_graph, qualified_timing=True, plausible_path=True,
            evidence_roles={"local_anomaly": ["m", "l"]}, modalities=["metrics", "logs"],
            source_fingerprints={"m": "test", "l": "test"})
        return ValidateResult(incident_id="i", scope_ref=REF, candidates=[], hypotheses={"h1": assessment})

    def test_confidence_caps_do_not_inflate(self):
        ranking = _ranking([0.9, 0.1])
        invest = _invest(_hypothesis("h1", "service:s0", confidence=0.85))
        result = fuse_investigation(INV, ranking, invest,
                                    self._validated("service:s0"))
        # Confidence comes from the verified assessment, not the branch value.
        assert result.fused[0].confidence == pytest.approx(0.85)

    def test_model_single_modality_label_cannot_lower_verified_class(self):
        ranking = _ranking([0.9])
        invest = _invest(_hypothesis("h1", "service:s0", rule="single_modality",
                                     confidence=0.85))
        result = fuse_investigation(INV, ranking, invest,
                                    self._validated("service:s0"))
        assert result.fused[0].confidence == pytest.approx(0.85)

    def test_ordering_and_winner_and_alternatives(self):
        ranking = _ranking([0.9, 0.1])
        invest = _invest(
            _hypothesis("h1", "service:s0", reward=0.9),
            _hypothesis("h2", "service:s1", rule="single_modality", reward=0.5))
        validated = self._validated("service:s0")
        result = fuse_investigation(INV, ranking, invest, validated)
        assert result.winner == "h1"
        assert result.winner_candidate == "service:s0"
        assert [item.rank for item in result.fused] == [1, 2]
        assert result.fused[0].h_score > result.fused[1].h_score
        # Confidence alone cannot make an unvalidated candidate viable.
        assert result.fused[1].status == "unvalidated"
        assert "h2" not in result.alternatives

    def test_validator_fail_scores_zero_and_disproves(self):
        ranking = _ranking([0.9])
        invest = _invest(_hypothesis("h1", "service:s0"))
        result = fuse_investigation(INV, ranking, invest,
                                    self._validated("service:s0", verdict="fail"))
        assert result.fused[0].components["V"] == 0.0
        assert result.fused[0].status == "disproved"

    def test_presentation_from_rwr_concentration(self):
        concentrated = fuse_investigation(
            INV, _ranking([0.8, 0.1, 0.05, 0.05]),
            _invest(_hypothesis("h1", "service:s0")),
            self._validated("service:s0"))
        assert concentrated.presentation == "single"  # top3 share = 0.95 > 0.7
        spread = fuse_investigation(
            INV, _ranking([0.2, 0.2, 0.2, 0.2, 0.2]),
            _invest(_hypothesis("h1", "service:s0")),
            self._validated("service:s0"))
        assert spread.presentation == "multi"  # top3 share = 0.6 <= 0.7
