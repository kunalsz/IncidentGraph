import math

from config import tuning


def percentile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def latency_scores(service_ids, observations, edges):
    durations = {sid: {} for sid in service_ids}
    for record in observations:
        props = record["properties"]
        duration = props.get("duration_us")
        sid = record["service_id"]
        if record["type"] == "Span" and duration is not None and duration >= 0:
            durations[sid][record["id"]] = duration / 1000
    means = {sid: sum(items.values()) / len(items) if items else 0.0
             for sid, items in durations.items()}
    callees = {sid: set() for sid in service_ids}
    for edge in edges:
        if edge["type"] == "DEPENDS_ON":
            callees[edge["source"]].add(edge["target"])
    scores = {}
    for sid, items in durations.items():
        values = list(items.values())
        p50 = percentile(values, 0.5)
        p99 = percentile(values, 0.99)
        slowest = max((means[c] for c in callees[sid]), default=0.0)
        scores[sid] = {
            "phi1": p99 / p50 if p50 and p99 is not None else 0.0,
            "phi3": sum(values) * max(0.0, 1 - slowest / means[sid]) if means[sid] else 0.0,
            "latency_samples": len(values),
        }
    return scores


def random_walk(service_ids, edges, anchor, restart=None, iterations=None):
    restart = tuning.RWR_PHI if restart is None else restart
    iterations = tuning.RWR_ITERATIONS if iterations is None else iterations
    if anchor not in service_ids:
        raise ValueError("walk anchor must be in scope")
    if not 0 < restart <= 1 or iterations < 1:
        raise ValueError("invalid random-walk parameters")
    adjacency = {sid: set() for sid in service_ids}
    for edge in edges:
        if edge["type"] == "DEPENDS_ON":
            adjacency[edge["source"]].add(edge["target"])
    mass = {sid: float(sid == anchor) for sid in service_ids}
    for _ in range(iterations):
        updated = {sid: restart * float(sid == anchor) for sid in service_ids}
        for sid, value in mass.items():
            neighbors = adjacency[sid] or {anchor}
            for neighbor in neighbors:
                updated[neighbor] += (1 - restart) * value / len(neighbors)
        mass = updated
    return mass


def reciprocal_rank_fusion(components, epsilon=None):
    epsilon = tuning.RRF_EPSILON if epsilon is None else epsilon
    if epsilon <= 0:
        raise ValueError("RRF epsilon must be positive")
    ids = list(components)
    scores = {sid: 0.0 for sid in ids}
    ranks = {sid: {} for sid in ids}
    for component in ("phi1", "phi3", "severity", "rwr", "deploy_proximity"):
        values = sorted(((sid, components[sid][component]) for sid in ids
                         if components[sid][component] > 0), key=lambda item: (-item[1], item[0]))
        index = 0
        while index < len(values):
            end = index + 1
            while end < len(values) and values[end][1] == values[index][1]:
                end += 1
            rank = (index + 1 + end) / 2
            for sid, _ in values[index:end]:
                ranks[sid][component] = rank
                scores[sid] += 1 / (epsilon + rank)
            index = end
    return scores, ranks


def indicator_scores(observations):
    indicators = {}
    for record in observations:
        if record["type"] not in {"MetricAnomaly", "LogEvent", "Span", "Trace"}:
            continue
        props = record["properties"]
        indicator = props.get("metric") or props.get("template_id") or props.get("op")
        if not indicator:
            continue
        gamma = props.get("gamma")
        if gamma is None:
            value, median, iqr = (props.get(key) for key in ("value", "baseline_median", "baseline_iqr"))
            if value is not None and median is not None and iqr is not None and iqr > 0:
                gamma = abs(value - median) / iqr
        if gamma is None or gamma < 0 or not math.isfinite(gamma):
            continue
        key = (record["service_id"], indicator)
        entry = indicators.setdefault(key, {"service_id": key[0], "indicator": indicator,
                                            "gamma": 0.0, "evidence": []})
        entry["gamma"] = max(entry["gamma"], gamma)
        entry["evidence"].append(record["id"])
    return sorted(indicators.values(), key=lambda item: (-item["gamma"], item["service_id"], item["indicator"]))
