"""P4 deterministic validators (§4.3 P4, L8). Pure functions over typed dicts.

Knowledge asserts itself as a VETO here, never as a nudge (RADICE): a claim
that violates a check is rejected outright; nothing in P4 can talk a failed
check back into a pass — the LLM critique may only lower confidence.

Level convention (documented translation): our `level` property is the
topological depth over DEPENDS_ON in call direction (entry service = 0, its
callees deeper). Faults originate deeper and manifest upward, so the
physically possible CAUSES direction is same-or-deeper => shallower. RADICE's
veto (stated against their root-cause-lowest convention) therefore becomes:
**reject a causal claim whose cause sits strictly shallower than its effect.**
"""

from __future__ import annotations

import math

from config import tuning


def level_veto(cause_level: int, effect_level: int) -> dict:
    """Reject CAUSES(cause -> effect) when the cause is strictly shallower."""
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in (cause_level, effect_level)):
        return {"name": "level_veto", "verdict": "partial", "detail": "authoritative levels unavailable"}
    if cause_level < effect_level:
        return {"name": "level_veto", "verdict": "fail",
                "detail": f"cause level {cause_level} is shallower than effect "
                          f"level {effect_level}: physically implausible direction"}
    return {"name": "level_veto", "verdict": "pass",
            "detail": f"cause level {cause_level} >= effect level {effect_level}"}


def cycle_rejection(existing_edges: list[tuple[str, str]],
                    proposed: tuple[str, str]) -> dict:
    """Reject a proposed cause->effect edge that would close a cycle."""
    cause, effect = proposed
    adjacency: dict[str, set] = {}
    for source, target in existing_edges:
        adjacency.setdefault(source, set()).add(target)
    # would effect already reach cause? then cause->effect closes a loop
    frontier, seen = [effect], {effect}
    while frontier:
        node = frontier.pop()
        if node == cause:
            return {"name": "cycle_rejection", "verdict": "fail",
                    "detail": f"adding {cause} -> {effect} closes a causal cycle"}
        for neighbor in adjacency.get(node, ()):
            if neighbor not in seen:
                seen.add(neighbor)
                frontier.append(neighbor)
    return {"name": "cycle_rejection", "verdict": "pass",
            "detail": "no cycle introduced"}


def onset_alignment(evidence_ts: list[int], onset_ts: int,
                    delta_ms: int = tuning.ONSET_DELTA_MS) -> dict:
    """Evidence must land within +/-delta of the alert (onset) timestamp."""
    if not evidence_ts:
        return {"name": "onset_alignment", "verdict": "fail",
                "detail": "no timestamped evidence"}
    nearest = min(abs(ts - onset_ts) for ts in evidence_ts)
    if nearest <= delta_ms:
        return {"name": "onset_alignment", "verdict": "pass",
                "detail": f"nearest evidence {nearest} ms from onset "
                          f"(delta {delta_ms} ms)"}
    return {"name": "onset_alignment", "verdict": "partial",
            "detail": f"nearest evidence {nearest} ms from onset "
                      f"(outside delta {delta_ms} ms)"}


def svar_lite(records: list[dict]) -> dict:
    """Cheapened SVAR two-window falsifier (CORAL): the anomaly must be
    present in-window AND a real excursion against the baseline, not baseline
    noise. With summary-statistic evidence the baseline-side proxy is the
    IQR-normalized deviation gamma (>= n-sigma): a value inside the normal
    spread cannot be evidence of change."""
    if not records:
        return {"name": "svar_lite", "verdict": "fail",
                "detail": "no metric evidence for this candidate"}
    numeric = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
    if any(not numeric(r.get("z")) or (r.get("gamma") is not None and not numeric(r["gamma"])) for r in records):
        return {"name": "svar_lite", "verdict": "partial", "detail": "invalid finite series statistics"}
    present = [r for r in records if abs(r["z"]) >= tuning.NSIGMA_N]
    if not present:
        peak = max((abs(r.get("z") or 0.0) for r in records), default=0.0)
        return {"name": "svar_lite", "verdict": "fail",
                "detail": f"no in-window excursion (peak |z|={peak:.2f} "
                          f"< {tuning.NSIGMA_N})"}
    peak_record = max(present, key=lambda r: abs(r.get("z") or 0.0))
    gamma = peak_record.get("gamma")
    # direction consistency is a property of ONE series (peak vs onset), not of
    # the service's whole metric set: a service can legitimately have latency
    # up while an unrelated resource metric moves down
    series_key = peak_record.get("column") or peak_record.get("metric")
    series_records = [r for r in present
                      if (r.get("column") or r.get("metric")) == series_key]
    directions = {r.get("direction") for r in series_records if r.get("direction")}
    if gamma is None:
        return {"name": "svar_lite", "verdict": "partial",
                "detail": "in-window excursion present; gamma unavailable for "
                          "the baseline excursion check"}
    if gamma < tuning.NSIGMA_N:
        return {"name": "svar_lite", "verdict": "fail",
                "detail": f"excursion gamma={gamma:.2f} sits inside the "
                          "baseline spread"}
    if not directions or not directions <= {"up", "down"} or any(not r.get("direction") for r in series_records):
        return {"name": "svar_lite", "verdict": "partial",
                "detail": "in-window excursion present; direction unavailable"}
    if len(directions) > 1:
        return {"name": "svar_lite", "verdict": "partial",
                "detail": f"excursion flips direction within the series {directions}"}
    return {"name": "svar_lite", "verdict": "pass",
            "detail": f"excursion z={peak_record.get('z'):.2f}, gamma={gamma:.2f}, "
                      f"direction={next(iter(directions)) or 'unknown'} stable"}


def symptom_explanation(candidate: str, anchor: str, own_anomaly: bool,
                        self_time_flag: bool, error_span: bool) -> dict:
    """The candidate must explain the upstream symptom with its own signal —
    a node that merely forwards another's anomaly cannot close a branch."""
    if candidate == anchor:
        return {"name": "symptom_explanation", "verdict": "partial",
                "detail": "candidate is the symptom anchor itself"}
    if own_anomaly or self_time_flag or error_span:
        return {"name": "symptom_explanation", "verdict": "pass",
                "detail": "candidate carries its own anomaly "
                          f"(self_time={self_time_flag}, error_span={error_span})"}
    return {"name": "symptom_explanation", "verdict": "fail",
            "detail": "candidate shows no own anomaly to explain the symptom"}


def no_evidence_cap(has_error_logs: bool, has_metric_anomaly: bool,
                    has_error_span: bool) -> dict:
    """Candidates with no error logs and normal metrics get min(c, 0.4)."""
    if has_error_logs or has_metric_anomaly or has_error_span:
        return {"name": "no_evidence_cap", "verdict": "pass",
                "detail": "candidate carries at least one decisive modality"}
    return {"name": "no_evidence_cap", "verdict": "partial",
            "detail": "no error logs and no metric anomaly: confidence capped",
            "confidence_cap": tuning.NO_EVIDENCE_CONFIDENCE_CAP}


def chain_coherence(explained_other_branches: int, other_abnormal_branches: int) -> float:
    """CC in {0, 0.5, 1}: rank-1 must explain >=2 other branches' symptoms.
    With no other abnormal branches the check is unverifiable -> neutral 0.5."""
    if other_abnormal_branches == 0:
        return 0.5
    if explained_other_branches >= 2:
        return 1.0
    if explained_other_branches == 1:
        return 0.5
    return 0.0


def aggregate_verdict(checks: list[dict]) -> str:
    """Hard fails reject; partials demote; otherwise pass."""
    verdicts = {check["verdict"] for check in checks}
    if "fail" in verdicts:
        return "fail"
    if "partial" in verdicts:
        return "partial"
    return "pass"


def confidence_cap_from_checks(checks: list[dict]) -> float | None:
    """Lowest cap any check imposes, or None when none do."""
    caps = [check["confidence_cap"] for check in checks
            if isinstance(check.get("confidence_cap"), (int, float))]
    return min(caps) if caps else None


def apply_cap(confidence: float, cap: float | None) -> float:
    return confidence if cap is None else min(confidence, cap)


def finite(value) -> float:
    """Coerce to a finite float (graph nulls and NaN become 0.0)."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0
