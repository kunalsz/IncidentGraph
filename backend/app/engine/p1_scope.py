import math

from backend.app.engine.contracts import Investigation, ScopeResult, ScopeService
from backend.app.graph import queries
from backend.app.graph.client import graph_for
from config import tuning


def service_statistics(service_id, observations, changes, trigger_ts):
    records = [r for r in observations if r["service_id"] == service_id]
    vectors = {"MetricAnomaly": 0.0, "LogEvent": 0.0, "Span": 0.0}
    nsigma = []
    self_times = []
    spans = [r for r in observations if r["type"] == "Span"]
    for record in records:
        props = record["properties"]
        kind = record["type"]
        modality = "Span" if kind == "Trace" else kind
        z = props.get("z")
        if z is None and kind == "MetricAnomaly":
            value = props.get("value")
            mean = props.get("baseline_mean")
            std = props.get("baseline_std")
            if value is not None and mean is not None and std is not None:
                if std > 0:
                    z = (value - mean) / std
                elif std == 0:
                    z = 0.0 if value == mean else None
                    if value != mean and abs(props["ts"] - trigger_ts) <= tuning.NSIGMA_DELTA_MS:
                        nsigma.append(record["id"])
        if z is not None and modality in vectors:
            vectors[modality] = max(vectors[modality], abs(z))
            if (kind == "MetricAnomaly" and abs(z) > tuning.NSIGMA_N
                    and abs(props["ts"] - trigger_ts) <= tuning.NSIGMA_DELTA_MS):
                nsigma.append(record["id"])
        if kind == "Span" and props.get("duration_us") is not None:
            children = {
                child["id"]: child["properties"].get("duration_us", 0)
                for child in spans
                if props.get("trace_id") and props.get("span_id")
                and child["properties"].get("trace_id") == props["trace_id"]
                and child["properties"].get("parent_id") == props["span_id"]
                and child["id"] != record["id"]
            }
            self_times.append(max(0.0, props["duration_us"] - sum(children.values())) / 1000)
    latest = max((r["properties"]["ts"] for r in changes
                  if r["service_id"] == service_id
                  and r["properties"]["ts"] <= trigger_ts), default=None)
    hours = (trigger_ts - latest) / 3_600_000 if latest is not None else None
    self_time = max(self_times, default=0.0)
    return ScopeService(
        id=service_id, sev_m=vectors["MetricAnomaly"], sev_l=vectors["LogEvent"],
        sev_tc=vectors["Span"], self_time_ms=self_time,
        self_time_flag=self_time > tuning.SELF_TIME_THRESHOLD_MS,
        nsigma_evidence=sorted(set(nsigma)), hours_since_change=hours,
        deploy_proximity=math.exp(-hours / 24) if hours is not None else 0.0,
    )


def scope_investigation(investigation: Investigation) -> ScopeResult:
    if not investigation.session_graph.startswith("incidentgraph:session:"):
        raise ValueError("P1 requires an isolated session graph")
    if tuning.SCOPE_MAX_NODES < 1 or tuning.SCOPE_MAX_SERIES < 0:
        raise ValueError("scope limits must allow the anchor and nonnegative series")
    graph = graph_for(investigation.session_graph)
    service_rows = graph.query(
        queries.SCOPE_SERVICES, params={"anchor": investigation.anchor_service}
    ).result_set
    if not service_rows:
        raise ValueError("anchor service is absent from the session graph")
    all_services = [dict(row[0].properties) for row in service_rows]
    services = all_services[:tuning.SCOPE_MAX_NODES]
    ids = [s["id"] for s in services]
    priorities = {s["id"]: i for i, s in enumerate(services)}
    window = investigation.window
    trigger_row = graph.query(
        "MATCH (i:Incident {id: $id}) RETURN i.start_ts",
        params={"id": investigation.incident_id},
    ).result_set
    trigger_ts = (trigger_row[0][0] if trigger_row and trigger_row[0][0] is not None
                  else (window.t0 + window.t1) // 2)
    observations = []
    for node, sid, target in graph.query(
        queries.SCOPE_OBSERVATIONS,
        params={"ids": ids, "t0": window.t0, "t1": window.t1},
    ).result_set:
        observations.append({"id": node.properties["id"], "type": node.labels[0],
                             "properties": dict(node.properties),
                             "service_id": sid, "target_id": target})
    changes = []
    for node, sid, target in graph.query(
        queries.SCOPE_CHANGES, params={"ids": ids, "t1": trigger_ts}
    ).result_set:
        changes.append({"id": node.properties["id"], "type": node.labels[0],
                        "properties": dict(node.properties), "service_id": sid, "target_id": target})
    series = sorted({(r["service_id"], r["properties"].get("metric", r["id"]))
                     for r in observations if r["type"] == "MetricAnomaly"},
                    key=lambda item: (priorities[item[0]], item[1]))
    allowed_series = set(series[:tuning.SCOPE_MAX_SERIES])
    eligible = [r for r in observations if r["type"] != "MetricAnomaly"
                or (r["service_id"], r["properties"].get("metric", r["id"])) in allowed_series]
    eligible += [r for r in changes if window.t0 <= r["properties"]["ts"] <= window.t1]
    eligible.sort(key=lambda r: (r["id"] != investigation.anchor_alert,
                                 priorities[r["service_id"]],
                                 r["properties"]["ts"], r["id"]))
    nodes = [{"id": s["id"], "type": "Service", "properties": s} for s in services]
    selected = set(ids)
    edges = [{"source": a, "target": b, "type": "DEPENDS_ON", "properties": props}
             for a, b, props in graph.query(queries.SCOPE_EDGES, params={"ids": ids}).result_set]
    for record in eligible:
        if record["id"] not in selected:
            if len(selected) >= tuning.SCOPE_MAX_NODES:
                continue
            nodes.append({k: record[k] for k in ("id", "type", "properties")})
            selected.add(record["id"])
        edges.append({"source": record["id"], "target": record["service_id"],
                      "type": "CHANGED" if record["type"] == "ConfigChange" else
                              "DEPLOYED_AT" if record["type"] == "Deployment" else "OBSERVED_ON",
                      "properties": {"projected": True,
                                     "target_id": record.get("target_id", record["service_id"])}})
    truncated = (len(services) < len(all_services) or len(series) > len(allowed_series)
                 or any(r["id"] not in selected for r in eligible))
    gaps = []
    if truncated:
        gaps.append({"node_id": investigation.anchor_service, "kind": "scope_truncation",
                     "suggestion": "Expand the scope or query omitted telemetry in a bounded follow-up."})
    stats = []
    for service in services:
        sid = service["id"]
        coverage = {r["type"] for r in observations if r["service_id"] == sid}
        for kind, labels in (("no_traces", {"Span", "Trace"}), ("no_logs", {"LogEvent"})):
            if not coverage.intersection(labels):
                gaps.append({"node_id": sid, "kind": kind,
                             "suggestion": f"Query {kind[3:]} for {sid} within the incident window."})
        stat = service_statistics(sid, observations, changes, trigger_ts)
        stat.criticality = service.get("criticality", 0.0)
        stats.append(stat)
        next(n for n in nodes if n["id"] == sid)["properties"].update(
            stat.model_dump(exclude={"id", "nsigma_evidence"})
        )
        graph.query(queries.UPDATE_SCOPE_STATS, params={
            **stat.model_dump(exclude={"criticality", "nsigma_evidence"}),
            "nsigma_count": len(stat.nsigma_evidence), "t0": window.t0, "t1": window.t1,
        })
    for gap in gaps:
        gap["id"] = f"gap:{investigation.incident_id}:{gap['node_id']}:{gap['kind']}"
        graph.query(queries.UPSERT_SCOPE_GAP, params={
            **gap, "incident": investigation.incident_id, "ts": trigger_ts,
            "attempted": ["scope_observations"],
        })
    return ScopeResult(
        incident_id=investigation.incident_id, anchor_service=investigation.anchor_service,
        window=window, services=stats, nodes=nodes, edges=edges, evidence_gaps=gaps,
        n_series_in_scope=len({(r["service_id"], r["properties"].get("metric", r["id"]))
                               for r in observations
                               if r["type"] == "MetricAnomaly" and r["id"] in selected}),
        truncated=truncated,
    )
