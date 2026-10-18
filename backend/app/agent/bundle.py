"""Diagnostic bundle builder (§4.3 P3, L3): one bounded-context object per branch.

Bundle contents per the GALA lesson (native format, not charts): per-series
metric anomaly records as timestamped JSON, the 1-hop annotated adjacency,
distilled logs (error templates first, plus a few non-error rows to distinguish
silent failure from log absence), and recent changes.

The assembler is a pure function over fetched rows (unit-testable without a
graph); `fetch_branch_bundle` is the graph-facing wrapper.
"""

from __future__ import annotations

import hashlib
import json

from config import tuning

from backend.app.agent.llm import estimate_tokens

MAX_LOG_MESSAGE_CHARS = 200


def _trim_log_message(message: str) -> str:
    message = " ".join(str(message).split())
    return (message[:MAX_LOG_MESSAGE_CHARS] + "…"
            if len(message) > MAX_LOG_MESSAGE_CHARS else message)


def _representative_spans(spans, onset_ts=None, window=None, pairs=None, limit=15):
    """Onset-relative pre/post samples, with qualified pairs kept together.

    Prefer nearby error spans, but reserve representation of non-error spans on
    each side too. Unknown/UNSET status is context, never a decisive error.
    """
    from backend.app.engine import policy

    center = onset_ts if policy.finite_number(onset_ts) else (
        (window.t0 + window.t1) / 2 if window is not None else None)
    valid = {s['id']: s for s in spans if s.get('id') and policy.finite_number(s.get('ts'))
             and (window is None or window.t0 <= s['ts'] <= window.t1)}
    kept = []
    for pair in (pairs or [])[:3]:
        refs = pair['evidence_refs']
        additions = [valid[sid] for sid in refs if sid in valid and sid not in {s['id'] for s in kept}]
        if all(sid in valid for sid in refs) and len(kept) + len(additions) <= limit:
            kept.extend(additions)
    groups = [[], [], [], []]
    for span in valid.values():
        # Foreign context is never retained independently of its qualified pair.
        if span.get('evidence_role') == 'propagation_context':
            continue
        post = center is not None and span['ts'] >= center
        group = (0 if policy.error_status(span.get('status')) else 2) + (0 if post else 1)
        groups[group].append(span)
    for group in groups:
        group.sort(key=lambda s: (abs(s['ts'] - center) if center is not None else 0, s['id']))
    while len(kept) < limit and any(groups):
        for group in groups:
            while group and group[0]['id'] in {s['id'] for s in kept}:
                group.pop(0)
            if group and len(kept) < limit:
                kept.append(group.pop(0))
    return kept


def _representative_metrics(metrics, onset_ts, window, limit):
    """Reserve exact-series onset excursions before round-robin peak/context.

    Summary fields are descriptive only: qualification uses the actual sample
    and the existing modality/SVAR checks, never a summary's claimed timestamp,
    direction set or persistence count. No clock means legacy selection.
    """
    from backend.app.engine import policy
    from backend.app.graph.validators import svar_lite

    ranked = sorted(metrics, key=lambda m: (-abs(m.get("z") or 0.0),
                                            str(m.get("id", ""))))
    if not policy.finite_number(onset_ts):
        return ranked[:limit]
    groups = {}
    for index, sample in enumerate(ranked):
        # Never combine distinct raw columns just because their metric names match.
        key = sample.get("column") or sample.get("metric")
        key = ("series", key) if key else ("anonymous", index)
        groups.setdefault(key, []).append(sample)
    primaries, fallback, extras = [], [], []
    for samples in groups.values():
        qualified = [m for m in samples
                     if policy.finite_number(m.get("ts"))
                     and abs(m["ts"] - onset_ts) <= tuning.ONSET_DELTA_MS
                     and (window is None or window.t0 <= m["ts"] <= window.t1)
                     and policy.modality("MetricAnomaly", m)
                     and svar_lite([m])["verdict"] == "pass"
                     and (m.get("direction") == "up" and m["z"] > 0
                          or m.get("direction") == "down" and m["z"] < 0)]
        primary = min(qualified, key=lambda m: (m["ts"], str(m.get("id", "")))) if qualified else samples[0]
        (primaries if qualified else fallback).append(primary)
        extras.append([m for m in samples if m is not primary])
    # Tail trimming must discard duplicates before ANY series representative.
    ordered = primaries + fallback
    for round_index in range(max((len(rows) for rows in extras), default=0)):
        ordered.extend(rows[round_index] for rows in extras if round_index < len(rows))
    return ordered[:limit]


def assemble_bundle(service_id: str, metrics: list[dict], logs: list[dict],
                    adjacency: list[dict], changes: list[dict] | None = None,
                    token_cap: int | None = None, max_logs: int = 40,
                    max_metrics: int = 15, max_adjacency: int = 12,
                    spans: list[dict] | None = None, *, onset_ts=None,
                    window=None, span_pairs: list[dict] | None = None) -> tuple[dict, str]:
    """Assemble a bounded bundle; returns (bundle, fingerprint).

    Deterministic trimming under the token cap: metrics reserve onset-qualified
    series representatives when a clock exists (otherwise legacy |z| order),
    logs are template-deduplicated errors-first, adjacency by criticality.
    """
    cap = tuning.BUNDLE_TOKEN_CAP if token_cap is None else token_cap

    kept_metrics = _representative_metrics(metrics, onset_ts, window, max_metrics)

    errors, seen_templates, infos = [], set(), []
    for line in sorted(logs, key=lambda item: int(item.get("ts", 0))):
        entry = {
            "id": line.get("id"), "ts": line.get("ts"),
            "level": line.get("level", "info"),
            "template_id": line.get("template_id"),
            "message": _trim_log_message(line.get("message", "")),
            "is_error": bool(line.get("is_error")),
            "z": line.get("z", 0.0),
        }
        if entry["is_error"]:
            if entry["template_id"] not in seen_templates:
                seen_templates.add(entry["template_id"])
                errors.append(entry)
        elif len(infos) < 5:
            infos.append(entry)
    kept_logs = (errors + infos)[:max_logs]

    kept_adjacency = sorted(
        adjacency, key=lambda n: (-float(n.get("criticality", 0.0)),
                                  str(n.get("service", ""))))[:max_adjacency]
    def bounded(value):
        if isinstance(value, str):
            return value[:4096]
        if isinstance(value, dict):
            return {k: (v if k in ('id', 'service', 'service_id') else bounded(v)) for k, v in value.items()}
        if isinstance(value, list):
            return [bounded(v) for v in value[:100]]
        return value
    kept_metrics = bounded(kept_metrics)
    kept_adjacency = bounded(kept_adjacency)
    kept_changes = bounded(list(changes or [])[:10])
    kept_spans = bounded(_representative_spans(spans or [], onset_ts, window, span_pairs))

    bundle = {
        "service_id": service_id,
        "metrics": kept_metrics,
        "logs": kept_logs,
        "adjacency": kept_adjacency,
        "changes": kept_changes,
        "spans": kept_spans,
        "fingerprint_inputs": {
            "n_metrics_total": len(metrics), "n_logs_total": len(logs),
            "n_adjacency_total": len(adjacency),
        },
    }
    while True:
        if span_pairs:
            ids = {s['id'] for s in bundle['spans']}
            bundle['span_pairs'] = [dict(p) for p in span_pairs[:3]
                                    if set(p['evidence_refs']) <= ids]
            paired_ids = {sid for p in bundle['span_pairs'] for sid in p['evidence_refs']}
            bundle['spans'] = [s for s in bundle['spans']
                               if s.get('evidence_role') != 'propagation_context' or s['id'] in paired_ids]
        if estimate_tokens(json.dumps(bundle)) <= cap:
            break
        if bundle["spans"]:
            bundle["spans"].pop()
        elif len(bundle["metrics"]) > 1:
            bundle["metrics"].pop()
        elif bundle["logs"]:
            bundle["logs"].pop()
        elif bundle["adjacency"]:
            bundle["adjacency"].pop()
        elif bundle["changes"]:
            bundle["changes"].pop()
        elif bundle["metrics"]:
            bundle["metrics"].pop()
        else:
            raise ValueError("token cap is smaller than minimal bundle envelope")
    fingerprint = hashlib.sha1(
        json.dumps(bundle, sort_keys=True).encode()).hexdigest()
    return bundle, fingerprint


def bundle_evidence_ids(bundle: dict) -> list[str]:
    """Record ids carried by a bundle, for HAS_EVIDENCE linking."""
    ids = [m["id"] for m in bundle.get("metrics", []) if m.get("id")]
    ids += [line["id"] for line in bundle.get("logs", []) if line.get("id")]
    ids += [change["id"] for change in bundle.get("changes", []) if change.get("id")]
    ids += [s["id"] for s in bundle.get("spans", []) if s.get("id")]
    return sorted(set(ids))


def fetch_branch_bundle(graph, service_id: str, window, token_cap: int | None = None) -> dict:
    """Fetch and assemble the bundle for one service within the incident window.

    Returns {"bundle", "fingerprint", "evidence_ids"}.
    """
    from backend.app.engine.scope_policy import recalled_scope, admitted_records, admitted_span_pairs
    from backend.app.engine import policy
    snapshot = recalled_scope(graph, window)
    if snapshot is None:
        raise ValueError('P1 admitted scope required before building diagnostic bundles')
    scope, onset = snapshot
    if service_id not in {s.id for s in scope.services}:
        raise ValueError('scope expansion required')
    records = admitted_records(scope, service_id)
    metrics = [r['properties'] for r in records if r['type'] == 'MetricAnomaly']
    logs = [r['properties'] for r in records if r['type'] == 'LogEvent']
    spans = [dict(r['properties'], service_id=service_id,
                  evidence_role='local_anomaly' if policy.modality('Span', r['properties']) else 'context')
             for r in records if r['type'] == 'Span']
    pairs = admitted_span_pairs(scope, service_id, onset)
    foreign_ids = {sid for pair in pairs for sid in pair['evidence_refs']} - {s['id'] for s in spans}
    spans += [dict(r['properties'], service_id=r['service_id'], evidence_role='propagation_context')
              for r in admitted_records(scope) if r['id'] in foreign_ids and r['type'] == 'Span']
    by_span = {s['id']: s for s in spans}
    pairs = [dict(pair, association_only=True,
                  claim_eligible=policy.finite_number(onset) and all(
                      abs(by_span[sid]['ts'] - onset) <= tuning.ONSET_DELTA_MS
                      for sid in pair['evidence_refs'])) for pair in pairs]
    changes = [dict(r['properties'], kind=r['type']) for r in records
               if r['type'] in ('ConfigChange', 'Deployment') and policy.finite_number(onset)
               and r['properties']['ts'] <= onset]
    by_id = {n['id']: n['properties'] for n in scope.nodes if n['type'] == 'Service'}
    adjacency = []
    for edge in scope.edges:
        if edge['type'] != 'DEPENDS_ON' or service_id not in (edge['source'], edge['target']):
            continue
        other = edge['target'] if edge['source'] == service_id else edge['source']
        adjacency.append({'service': other, 'name': by_id[other].get('name'),
                          'criticality': by_id[other].get('criticality', 0),
                          'direction': 'calls' if edge['source'] == service_id else 'called_by'})
    bundle, fingerprint = assemble_bundle(service_id, metrics, logs, adjacency,
                                          changes, token_cap=token_cap, spans=spans,
                                          onset_ts=onset, window=window, span_pairs=pairs)
    return {'bundle': bundle, 'fingerprint': fingerprint,
            'evidence_ids': bundle_evidence_ids(bundle)}
