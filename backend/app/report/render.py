"""Safe compatibility Markdown from authoritative deterministic report JSON."""
from __future__ import annotations

import html
import re

BANNER = {tier: tier for tier in ("CONFIRMED", "LIKELY", "INCONCLUSIVE")}


def safe(value):
    """Render every external value as text: no HTML, links, code or directives."""
    text = html.escape(str(value), quote=True)
    text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", text)
    text = text.replace("\n", " ").replace("\r", " ")
    # Escape all Markdown structure including citation-looking brackets.
    return re.sub(r"([\\`*_{}\[\]()#+.!|>~:-])", r"\\\1", text)


def _path_chain(path):
    if not path or not path.found:
        return ["_no verified evidence path was produced_"]
    nodes = [safe(n["id"]) for n in path.nodes if n.get("role") != "corroborating"]
    return [" → ".join(nodes)] + [
        f"- dependency {safe(e['source'])} → {safe(e['target'])}: {len(e.get('support') or [])} verified support references"
        for e in path.edges]


def _runbook_block(runbook):
    lines = []
    for entry in (runbook or {}).get("runbooks", []):
        if entry.get("approved") is not True and entry.get("approval_status") != "approved":
            continue
        lines.append(f"- {safe(entry.get('title') or entry.get('id'))} (human review required)")
        for step in entry.get("steps", []):
            lines.append(f"  - {safe(step.get('action', ''))}")
            if step.get("validate_cmd"):
                lines.append(f"  - Approved validation text (not executed): {safe(step['validate_cmd'])}")
    return lines or ["_no approved runbook retrieved for this service_"]


def _as_lines(items):
    return [safe((str(item.get("id") or item.get("node_id") or "") + ": " +
                  str(item.get("reason") or item.get("detail") or "")) if isinstance(item, dict) else item)
            for item in items]


def render_markdown(report):
    lines = [f"# Incident report — {safe(report.incident_id)}", "", f"## {BANNER[report.tier]}", "",
             f"- verdict: **{report.verdict}** · deterministic evidence strength **{report.confidence:.2f}** (not calibrated probability)",
             f"- leading candidate: {safe(report.winner_candidate or 'none')}",
             "- Tier describes verified abnormality/association only; causality and narrative mechanisms are not established.", "", "## Summary", ""]
    if report.facts:
        ids = {fact["id"] for fact in report.facts}
        if len(ids) != len(report.facts):
            raise ValueError("duplicate report fact identity")
        evidence = {sid for fact in report.facts for sid in fact.get("evidence_ids", [])}
        if set(report.citations) != evidence or sorted(report.citations.values()) != list(range(1, len(evidence) + 1)):
            raise ValueError("unknown or inconsistent report citations")
        for fact in report.facts:
            refs = " ".join(f"[{report.citations[sid]}]" for sid in sorted(fact.get("evidence_ids", [])))
            lines.append(f"- {safe(fact['text'])}" + (f" {refs}" if refs else ""))
    else:
        # Historical artifacts are viewable, explicitly nonauthoritative.
        lines.append("_Historical report text; not revalidated as trusted v2._")
        lines.extend(safe(s) for s in report.summary)
    lines += ["", "## Evidence path", ""] + _path_chain(report.evidence_path)
    lines += ["", "## Alternatives", ""]
    lines += [f"- {safe(a.get('candidate'))}: {safe(a.get('reason', ''))}" for a in report.alternatives] or ["_none live-revalidated_"]
    lines += ["", "## Runbook (approved, retrieved; never executed)", ""] + _runbook_block(report.runbook)
    lines += ["", "## Owner", "", safe(report.owner.get("name", "not retrieved")), "", "## Next action", "", safe(report.next_action or "none")]
    lines += ["", "## Ruled out", ""] + ([f"- {s}" for s in _as_lines(report.ruled_out)] or ["_none_"])
    lines += ["", "## Not checked", ""] + ([f"- {s}" for s in _as_lines(report.not_checked)] or ["_none_"])
    if report.citations:
        lines += ["", "## Evidence references", ""]
        lines += [f"[{number}] {safe(sid)}" for sid, number in sorted(report.citations.items(), key=lambda pair: pair[1])]
    budget = report.budget or {}
    lines += ["", "## Cost and finalization", "",
              f"- LLM calls {safe(budget.get('llm_calls', 0))} · tokens {safe(budget.get('tokens_total', 0))} · wall {safe(budget.get('wall_s', 0))} s",
              f"- compliant: {safe(budget.get('compliant'))} · persistence: {safe(report.config.get('persistence', 'unknown'))}",
              f"- Provider attempts: {safe(budget.get('provider_attempts', budget.get('llm_calls', 'unknown')))} · logical requests: {safe(budget.get('logical_requests', 'unknown'))} · cache hits: {safe(budget.get('cache_hits', 'unknown'))} · retries: {safe(budget.get('retries', 'unknown'))}",
              f"- Known tokens: {safe(budget.get('tokens_total', 'unknown'))} · charged tokens (includes conservative estimates for unknown usage): {safe(budget.get('tokens_charged', 'unknown'))} · committed tokens: {safe(budget.get('tokens_committed', 'unknown'))}",
              f"- Usage complete: {safe(budget.get('usage_complete', 'unknown'))} · unknown usage attempts: {safe(budget.get('unknown_usage_attempts', 'unknown'))} · in-flight attempts: {safe(budget.get('in_flight_attempts', 'unknown'))}",
              f"- Strict token enforcement: {safe(budget.get('strict_token_enforcement', 'unknown'))}",
              f"- Deadline enforcement: {safe(budget.get('deadline_enforcement', 'unknown; no hard deadline guarantee recorded'))}"]
    if report.unanswered_questions:
        lines += ["", "## Unanswered questions", ""] + [f"- {safe(q)}" for q in report.unanswered_questions]
    return "\n".join(lines) + "\n"
