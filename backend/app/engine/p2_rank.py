from backend.app.engine.contracts import Investigation, RankedCandidate, RankResult, ScopeResult, dump_handoff
from backend.app.graph.client import graph_for
from backend.app.graph.scorers import (indicator_scores, latency_scores, neutral_walk,
                                        random_walk, reciprocal_rank_fusion)
from backend.app.ingest.normalize import is_error_log
from config import tuning


def validate_rank_policy(rank_policy):
    if rank_policy not in tuning.P2_RANK_POLICIES:
        raise ValueError(f"unknown P2 rank policy: {rank_policy}")
    return rank_policy


def rank_candidates(scope: ScopeResult, *, rank_policy=tuning.P2_RANK_POLICY) -> RankResult:
    validate_rank_policy(rank_policy)
    ids = [service.id for service in scope.services]
    nodes = {node["id"]: node for node in scope.nodes}
    observations = []
    for edge in scope.edges:
        if (edge["type"] == "OBSERVED_ON" and edge["target"] in ids
                and edge["source"] in nodes):
            observations.append({**nodes[edge["source"]], "service_id": edge["target"]})
    latency = latency_scores(ids, observations, scope.edges)
    # Cold ranking (benchmark) has no case-derived anchor; the anchored walk
    # would then be a constant prior, so it is opt-in via the policy name.
    mass = (neutral_walk(ids, scope.edges) if rank_policy == "rrf_anchor_neutral"
            else random_walk(ids, scope.edges, scope.anchor_service))
    components = {}
    flags = {sid: [] for sid in ids}
    evidence = {sid: set() for sid in ids}
    for service in scope.services:
        sid = service.id
        components[sid] = {
            "phi1": latency[sid]["phi1"], "phi3": latency[sid]["phi3"],
            "severity": max(service.sev_m, service.sev_l, service.sev_tc),
            "rwr": mass[sid], "deploy_proximity": service.deploy_proximity,
        }
        if service.self_time_flag:
            flags[sid].append("self_time")
        if service.nsigma_evidence:
            flags[sid].append("nsigma")
            evidence[sid].update(service.nsigma_evidence)
    for edge in scope.edges:
        if edge["type"] in {"CHANGED", "DEPLOYED_AT"} and edge["target"] in evidence:
            evidence[edge["target"]].add(edge["source"])
    for record in observations:
        sid = record["service_id"]
        evidence[sid].add(record["id"])
        if record["type"] == "LogEvent" and (
            record["properties"].get("is_error") or is_error_log(record["properties"].get("level", "info"))
        ):
            if "error_logs" not in flags[sid]:
                flags[sid].append("error_logs")
    scores, ranks = reciprocal_rank_fusion(components)
    weights = {name: float(rank_policy in ("baseline_rrf", "rrf_anchor_neutral")
                          or name == "severity")
               for name in ("phi1", "phi3", "severity", "rwr", "deploy_proximity")}
    if rank_policy == "severity_only":
        # Keep all measured components/ranks as diagnostics; only severity votes.
        scores = {sid: 1 / (tuning.RRF_EPSILON + ranks[sid]["severity"])
                  if "severity" in ranks[sid] else 0.0 for sid in ids}
    ordered = sorted(ids, key=lambda sid: (-scores[sid], -nodes[sid]["properties"].get("pagerank", 0), sid))
    selected = set(ordered[:tuning.TOP_K_CANDIDATES]) | {sid for sid in ids if flags[sid]}
    candidates = [RankedCandidate(
        id=sid, rank=index + 1, score=scores[sid], components=components[sid],
        component_ranks=ranks[sid], evidence=sorted(evidence[sid]),
        rule_flags=sorted(flags[sid]), selected=sid in selected,
    ) for index, sid in enumerate(ordered)]
    audit = [{"id": sid, "reason": "outside_top_k_without_rule_flag"}
             for sid in ordered if sid not in selected]
    return RankResult(
        incident_id=scope.incident_id, scope_ref=scope.scope_ref, candidates=candidates,
        branch_ids=[sid for sid in ordered if sid in selected],
        indicators=indicator_scores(observations), audit=audit,
        n_candidates=len(ids), truncated=scope.truncated,
        config={"rank_policy": rank_policy, "rank_policy_version": tuning.P2_RANK_POLICY_VERSION,
                "component_weights": weights,
                "component_ranks": "all_measured_components",
                "candidate_tie_policy": "score_desc_pagerank_desc_id_asc",
                "rrf_epsilon": tuning.RRF_EPSILON, "rwr_restart": tuning.RWR_PHI,
                "rwr_iterations": tuning.RWR_ITERATIONS, "top_k": tuning.TOP_K_CANDIDATES,
                "walk_direction": "caller_to_dependency", "tie_policy": "average_rank",
                "missing_signal": "excluded", "latency_source": "scoped_service_spans",
                "correlation_pruning": "unavailable_without_aligned_raw_series"},
    )


def rank_investigation(investigation: Investigation, scope: ScopeResult, *,
                       rank_policy=tuning.P2_RANK_POLICY) -> RankResult:
    if not investigation.session_graph.startswith("incidentgraph:session:"):
        raise ValueError("P2 requires an isolated session graph")
    if (scope.incident_id != investigation.incident_id
            or scope.anchor_service != investigation.anchor_service
            or scope.window != investigation.window):
        raise ValueError("scope does not match investigation")
    result = rank_candidates(scope, rank_policy=rank_policy)
    graph_for(investigation.session_graph).query(
        "MERGE (a:Artifact {id: $id}) "
        "SET a.kind = 'stage-view', a.stage = 'strix', a.version = 2, "
        "a.incident_id = $incident, a.json = $payload",
        params={"id": f"artifact:{investigation.incident_id}:strix:0",
                "incident": investigation.incident_id, "payload": dump_handoff("rank", result)},
    )
    return result
