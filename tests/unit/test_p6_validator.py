"""Unit tests for the P6 machine gate and renderer (hand-computed)."""

import pytest

from backend.app.engine.contracts import EvidencePath, ReportResult
from backend.app.report import render, validator


class TestGate:
    def test_no_winner_abstains(self):
        result = validator.gate(has_winner=False, confidence=0.0, coherence=0.0,
                                validator_all_pass=False, per_link_support=False)
        assert result["verdict"] == "insufficient_evidence"
        assert result["tier"] == "INCONCLUSIVE"

    def test_below_theta_abstains(self):
        result = validator.gate(has_winner=True, confidence=0.55, coherence=1.0,
                                validator_all_pass=True, per_link_support=True)
        assert result["verdict"] == "insufficient_evidence"
        assert result["tier"] == "INCONCLUSIVE"

    def test_independent_cause_not_erased_by_zero_cross_branch_coherence(self):
        result = validator.gate(has_winner=True, confidence=0.9, coherence=0.0,
                                validator_all_pass=True, per_link_support=True)
        assert result["tier"] == "CONFIRMED"
        assert result["verdict"] == "diagnosed"

    def test_all_conditions_confer_confirmed(self):
        result = validator.gate(has_winner=True, confidence=0.9, coherence=1.0,
                                validator_all_pass=True, per_link_support=True)
        assert result["tier"] == "CONFIRMED"

    def test_missing_per_link_support_downgrades_to_likely(self):
        result = validator.gate(has_winner=True, confidence=0.9, coherence=0.5,
                                validator_all_pass=True, per_link_support=False)
        assert result["tier"] == "LIKELY"
        assert any("supporting evidence" in reason for reason in result["reasons"])

    def test_ambiguous_failed_or_missing_validator_abstains(self):
        result = validator.gate(has_winner=True, confidence=0.9, coherence=0.5,
                                validator_all_pass=False, per_link_support=True)
        assert result["tier"] == "INCONCLUSIVE"
        assert result["verdict"] == "insufficient_evidence"
        assert any("validators" in reason for reason in result["reasons"])


class TestRender:
    def _report(self, **overrides):
        base = dict(incident_id="incident:T1", verdict="diagnosed",
                    tier="CONFIRMED", confidence=0.85,
                    confidence_rule="cross_modal_consensus",
                    winner_candidate="service:payment",
                    summary=["p1", "p2", "p3"], next_action="do the thing",
                    evidence_path=EvidencePath(
                        from_id="service:checkout", to_id="service:payment",
                        nodes=[{"id": "service:checkout", "role": "symptom"},
                               {"id": "service:payment", "role": "candidate_cause"}],
                        edges=[{"source": "service:checkout",
                                "target": "service:payment",
                                "type": "DEPENDS_ON", "support": ["metric:m1"]}],
                        found=True),
                    runbook={"runbooks": [{"id": "runbook:payment-failures",
                                           "title": "Payment failures", "approved": True,
                                           "steps": [{"step": 1, "action": "check",
                                                      "validate_cmd": "kubectl get pods",
                                                      "risk": "read-only",
                                                      "undo": "none"}],
                                           "params_resolved": []}]},
                    owner={"name": "Payments", "oncall_handle": "@oncall-payments"},
                    ruled_out=[], not_checked=[], unanswered_questions=[],
                    budget={"llm_calls": 40, "tokens_total": 12000, "wall_s": 3.2,
                            "hypotheses": 30, "compliant": True})
        base.update(overrides)
        return ReportResult(**base)

    def test_markdown_has_banner_and_sections(self):
        markdown = render.render_markdown(self._report())
        assert markdown.startswith("# Incident report — incident\\:T1")
        assert "## CONFIRMED" in markdown
        for section in ("## Summary", "## Evidence path", "## Alternatives",
                        "## Runbook", "## Owner", "## Next action",
                        "## Ruled out", "## Not checked", "## Cost"):
            assert section in markdown, section
        assert "Payment failures" in markdown
        assert "Payments" in markdown
        assert "1 verified support references" in markdown
        assert "LLM calls 40" in markdown

    @pytest.mark.parametrize("tier,banner", [
        ("CONFIRMED", "CONFIRMED"), ("LIKELY", "LIKELY"),
        ("INCONCLUSIVE", "INCONCLUSIVE")])
    def test_banners(self, tier, banner):
        assert banner in render.render_markdown(self._report(tier=tier))

    def test_missing_runbook_renders_honestly(self):
        markdown = render.render_markdown(self._report(runbook={}))
        assert "_no approved runbook retrieved for this service_" in markdown
