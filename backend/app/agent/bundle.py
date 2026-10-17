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


def assemble_bundle(service_id: str, metrics: list[dict], logs: list[dict],
                    adjacency: list[dict], changes: list[dict] | None = None,
                    token_cap: int | None = None, max_logs: int = 40,
                    max_metrics: int = 15, max_adjacency: int = 12,
                    spans: list[dict] | None = None) -> tuple[dict, str]:
    """Assemble a bounded bundle; returns (bundle, fingerprint).

    Deterministic trimming under the token cap: metrics are kept by |z| desc,
    logs are template-deduplicated errors-first, adjacency by criticality.
    """
    cap = tuning.BUNDLE_TOKEN_CAP if token_cap is None else token_cap

    kept_metrics = sorted(metrics, key=lambda m: (-abs(m.get("z") or 0.0),
                                                  str(m.get("id", ""))))[:max_metrics]

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

    bundle = {
        "service_id": service_id,
        "metrics": kept_metrics,
        "logs": kept_logs,
        "adjacency": kept_adjacency,
        "changes": kept_changes,
        "spans": sorted(bounded(spans or []), key=lambda s: (s.get("ts", 0), s.get("id", "")))[:15],
        "fingerprint_inputs": {
            "n_metrics_total": len(metrics), "n_logs_total": len(logs),
            "n_adjacency_total": len(adjacency),
        },
    }
    while estimate_tokens(json.dumps(bundle)) > cap:
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
    from backend.app.engine.scope_policy import recalled_scope, admitted_records
    snapshot = recalled_scope(graph, window)
    if snapshot is None:
        raise ValueError('P1 admitted scope required before building diagnostic bundles')
    scope, onset = snapshot
    if service_id not in {s.id for s in scope.services}:
        raise ValueError('scope expansion required')
    records = admitted_records(scope, service_id)
    metrics = [r['properties'] for r in records if r['type'] == 'MetricAnomaly']
    logs = [r['properties'] for r in records if r['type'] == 'LogEvent']
    spans = [r['properties'] for r in records if r['type'] == 'Span']
    changes = [dict(r['properties'], kind=r['type']) for r in records
               if r['type'] in ('ConfigChange', 'Deployment') and r['properties']['ts'] <= onset]
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
                                          changes, token_cap=token_cap, spans=spans)
    return {'bundle': bundle, 'fingerprint': fingerprint,
            'evidence_ids': bundle_evidence_ids(bundle)}
