"""RCAEval RE2-OB adapter (§9.1 external benchmark, engine side).

Reads one case directory's telemetry — `metrics.parquet`, `logs.parquet`,
`traces.parquet`, `inject_time.txt` — and writes it into an isolated session
graph in the exact P1/P2 contract shape (Service / MetricAnomaly / LogEvent /
Span nodes with OBSERVED_ON edges; Incident for the window clock).

Ground-truth boundary: this module reads only telemetry files. It never reads
the case directory *name* (which encodes the injected service/fault) and never
receives labels. Run keys are sha1 digests so no readable label reaches the
graph. The runner derives ground truth separately, for the evaluator only.

Windowing mirrors RCAEval main.py: `--length 20` -> the last 600 metric rows
before the injection timestamp plus the first 600 after. Baseline statistics
(mean/std/median/IQR per series, per-minute log/trace aggregates) come from the
pre-injection side; anomaly records are emitted from the in-window side.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import pandas as pd

from backend.app.eval import baseline_scale as bscale
from backend.app.ingest import normalize
from config import tuning

OB_TOPOLOGY = Path(__file__).resolve().parent.parent / "seed" / "rcaeval_ob_topology.json"

WINDOW_HALF_S = 600          # RCAEval --length 20 -> 600 samples each side (1 Hz)
# Per-minute log/trace series with no observed spread get a scale floor of one
# count divided by the band width, so the unremarkable band is one count (an
# isolated extra log line is not an anomaly) while a persistent multi-count
# burst is. Latency series scale from 5% of their own level instead.
COUNT_FLOOR = 1.0 / 3.0
LEVEL_CONVENTION = "dependency_depth_v1"  # what policy._service_levels accepts as authoritative
SPAN_BUDGET = 6_000          # total spans written per case (deterministic selection)
ERROR_SPAN_BUDGET = 3_000    # of which reserved for traces carrying error spans
LOG_ERROR_CAP = 2_000        # error log records written per case
LOG_SAMPLE_PER_SERVICE = 20  # non-error rows per service (silent != absent)

TRACE_SERVICE_MAP = {"frontendservice": "frontend"}

_ERROR_RE = re.compile(
    r"error|fail|exception|timeout|refused|unavailable|deadline|panic|fatal|critical",
    re.IGNORECASE,
)


def run_key(case_dir: Path) -> str:
    """Deterministic, non-readable run key (no label leaks into graph ids)."""
    return hashlib.sha1(str(Path(case_dir).name).encode()).hexdigest()[:10]


# ---------------------------------------------------------------------------
# Reading + windowing
# ---------------------------------------------------------------------------

def read_case(case_dir: Path) -> dict:
    """Load a case's telemetry files (never the directory name)."""
    case_dir = Path(case_dir)
    inject_s = int((case_dir / "inject_time.txt").read_text().strip())
    metrics = pd.read_parquet(case_dir / "metrics.parquet")
    logs = traces = None
    if (case_dir / "logs.parquet").exists():
        logs = pd.read_parquet(case_dir / "logs.parquet")
    if (case_dir / "traces.parquet").exists():
        traces = pd.read_parquet(case_dir / "traces.parquet")
    return {"inject_s": inject_s, "metrics": metrics, "logs": logs, "traces": traces}


def window_frames(metrics: pd.DataFrame, inject_s: int, length_min: int = 20) -> dict:
    """RCAEval's normal/anomal slice (main.py, verbatim semantics).

    Indices are reset to positions: callers index `before`/`after` positionally
    (e.g. series_stats' peak sample), so index labels must not leak through.
    """
    rows = length_min * 60 // 2
    before = metrics[metrics["time"] < inject_s].tail(rows).reset_index(drop=True)
    after = metrics[metrics["time"] >= inject_s].head(rows).reset_index(drop=True)
    frame = pd.concat([before, after], ignore_index=True)
    return {
        "frame": frame,
        "before": before,
        "after": after,
        "t0_ms": int(frame["time"].min()) * 1000,
        "t1_ms": int(frame["time"].max()) * 1000,
    }


def metric_columns(frame: pd.DataFrame) -> list[tuple[str, str, str]]:
    """(column, entity, family) for every metric column we ingest.

    `_latency-50` columns are excluded (RCAEval drops them); `_latency-90`
    maps to the `latency` family (RCAEval renames it).
    """
    columns = []
    for column in frame.columns:
        if column in ("time", "Time", "time.1"):
            continue
        if column.endswith("_latency-50"):
            continue
        entity, family = column.split("_", 1)
        columns.append((column, entity, "latency" if family == "latency-90" else family))
    return columns


def series_stats(before: pd.Series, after: pd.Series, after_times: list[int],
                 onset_s: int | None = None) -> dict:
    """Baseline statistics + at most three distinct sample timestamps.

    Preserve the legacy peak and first finite post rows, then add the earliest
    qualifying deviation within ONSET_DELTA_MS (inclusive) of authoritative
    `onset_s` (seconds). Omission means no onset selection, not an inferred
    clock from the first observation. No service/fault labels enter selection.

    `summary` contains bounded scalar descriptive facts, not causal evidence or
    a persistence verdict. Onset counts cover finite observations in the onset
    window; runs count consecutive source rows qualifying in the same direction
    (missing/nonqualifying rows break runs), not elapsed seconds. Direction sets
    are sorted comma-separated strings (empty means none). Full-post deviation
    directions include subthreshold reversals relative to the baseline mean.
    The floor persistence guard remains global, not onset-local: a long global
    departure may score an isolated onset sample without proving persistence.

    Measured-spread baselines (std > 0 for z, IQR > 0 for gamma) are scored
    exactly as before. A baseline with no spread is scored by
    `baseline_scale` (IQR -> std -> relative floor), so `z` and `gamma`
    agree and an isolated blip against a flat baseline scores 0 (see that
    module). `z_capped` flags a floor-scaled (non-measured) deviation.
    Without finite baseline observations, samples retain their values but
    baseline statistics, z, and gamma remain unavailable (None).
    """
    base = before.astype(float).replace([math.inf, -math.inf], float("nan")).dropna()
    win = after.astype(float)
    valid = win.replace([math.inf, -math.inf], float("nan")).dropna()
    mean = float(base.mean()) if len(base) else 0.0
    std = float(base.std(ddof=1)) if len(base) > 1 else 0.0
    if not math.isfinite(std):
        std = 0.0
    median = float(base.median()) if len(base) else 0.0
    iqr = (float(base.quantile(0.75)) - float(base.quantile(0.25))) if len(base) else 0.0
    scale = bscale.baseline_scale(base.tolist())
    post = valid.tolist()
    if iqr > 0:
        gamma = float((valid - median).abs().max() / iqr) if len(valid) else 0.0
    else:
        gamma = bscale.gamma(post, scale)
    stats = {
        "baseline_mean": mean if len(base) else None,
        "baseline_std": std if len(base) else None,
        "baseline_median": median if len(base) else None,
        "baseline_iqr": (iqr if iqr > 0 else 0.0) if len(base) else None,
        "gamma": gamma, "scale_source": scale.source,
    }
    summary = {
        "onset_clock_available": onset_s is not None,
        "onset_valid_count": 0,
        "onset_qualified_count": 0,
        "onset_longest_same_direction_run": 0,
        "onset_qualified_directions": "",
        "onset_representative_ts": None,
        "onset_representative_direction": "",
        "post_deviation_directions": "",
    }
    if valid.empty:
        return {"stats": stats, "samples": [], "summary": summary}
    floor_path = std <= 0
    floor_persistent = floor_path and bscale.persistent(post, scale)

    def z_of(value: float) -> tuple[float | None, bool]:
        if scale.source == "unavailable":
            return None, False
        if not floor_path:
            return (value - mean) / std, False
        if value == mean or not floor_persistent:
            return 0.0, False
        return bscale.deviation(value, scale), True

    onset_directions, post_directions = set(), set()
    representative = None
    run_direction, run_length = None, 0
    gamma_qualified = gamma is not None and math.isfinite(gamma) and gamma >= tuning.NSIGMA_N
    for position, value in win.items():
        # Iterate source rows, not dropna(), so missing observations break runs.
        if not math.isfinite(value):
            run_direction, run_length = None, 0
            continue
        direction = "up" if value >= mean else "down"
        if len(base) and value != mean:
            post_directions.add(direction)
        ts = int(after_times[position]) * 1000
        if onset_s is None or abs(ts - onset_s * 1000) > tuning.ONSET_DELTA_MS:
            run_direction, run_length = None, 0
            continue
        summary["onset_valid_count"] += 1
        z_value, _ = z_of(float(value))
        qualified = (gamma_qualified and z_value is not None and math.isfinite(z_value)
                     and abs(z_value) >= tuning.NSIGMA_N
                     and (z_value > 0 if direction == "up" else z_value < 0))
        if not qualified:
            run_direction, run_length = None, 0
            continue
        summary["onset_qualified_count"] += 1
        onset_directions.add(direction)
        run_length = run_length + 1 if run_direction == direction else 1
        run_direction = direction
        summary["onset_longest_same_direction_run"] = max(
            summary["onset_longest_same_direction_run"], run_length)
        if representative is None or ts < summary["onset_representative_ts"]:
            representative = int(position)
            summary["onset_representative_ts"] = ts
            summary["onset_representative_direction"] = direction
    summary["onset_qualified_directions"] = ",".join(sorted(onset_directions))
    summary["post_deviation_directions"] = ",".join(sorted(post_directions))

    # Keep legacy peak/first priority even when timestamps are duplicated.
    positions = [int(valid.sub(mean).abs().idxmax()), int(valid.index[0])]
    if representative is not None:
        positions.append(representative)
    samples, seen_times = [], set()
    for position in positions:
        ts = int(after_times[position]) * 1000
        if ts in seen_times:
            continue
        seen_times.add(ts)
        value = float(win.loc[position])
        z_value, capped = z_of(value)
        samples.append({
            "ts": ts,
            "value": value, "z": z_value, "z_capped": capped,
            "direction": "up" if value >= mean else "down",
        })
    return {"stats": stats, "samples": samples, "summary": summary}


SAMPLER_VERSION = "stratified-window-traces-v2"


def _frame_hash(frame: pd.DataFrame) -> str:
    rows = sorted(frame.astype(str).to_json(orient="records").splitlines())
    return hashlib.sha256("\n".join(rows).encode()).hexdigest()


def select_trace_sample(traces: pd.DataFrame, t0_ms: int, t1_ms: int,
                        inject_ms: int, budget: int = SPAN_BUDGET,
                        seed: int = 0) -> tuple[pd.DataFrame, dict]:
    """Pure bounded selection of complete *in-window* groups, never source expansion.

    Round robin over injection-aligned minute/period/service strata; errors
    lead within each stratum and seeded hashes break ties. The two period
    coverage reservations precede the general round robin. No group is sliced.
    """
    if not 0 < budget <= SPAN_BUDGET:
        raise ValueError("trace budget must be in 1..6000")
    source = traces[traces["startTimeMillis"].notna()].copy()
    window = source[source["startTimeMillis"].between(t0_ms, t1_ms)].copy()
    window["service_id"] = window["serviceName"].astype(str).map(lambda x: TRACE_SERVICE_MAP.get(x, x))
    window["ts"] = (window["startTimeMillis"] // 1000).astype("int64")
    window["duration_us"] = window["duration"]
    # OpenTelemetry UNSET=0, OK=1, ERROR=2; OK is not an error.
    window["error"] = window["statusCode"].map(lambda value: str(value).lower() in ("2", "2.0", "error", "failed", "failure"))
    groups = {str(tid): group for tid, group in window.groupby("traceID", sort=True)}
    def priority(tid):
        group = groups[tid]
        return (not bool(group["error"].any()),
                hashlib.sha256(f"{seed}:{tid}".encode()).hexdigest(), tid)
    strata = {}
    for tid, group in groups.items():
        if len(group) > budget:
            continue
        # Membership in all covered strata makes service coverage feasible.
        for row in group.itertuples():
            period = "pre" if row.startTimeMillis < inject_ms else "post"
            key = (period, int((row.startTimeMillis - inject_ms) // 60000), row.service_id)
            strata.setdefault(key, set()).add(tid)
    queues = {key: sorted(ids, key=priority) for key, ids in sorted(strata.items())}
    selected, consumed = set(), 0
    def take(tid):
        nonlocal consumed
        if tid not in selected and consumed + len(groups[tid]) <= budget:
            selected.add(tid)
            consumed += len(groups[tid])
            return True
        return False
    # Reserve one feasible group from each period before filling time bins.
    for period in ("pre", "post"):
        ids = {tid for key, ids in strata.items() if key[0] == period for tid in ids}
        # Smallest complete groups reserve feasible period coverage before
        # error-first stratified filling; a large baseline group must not
        # consume the incident quota when a feasible pair exists.
        for tid in sorted(ids, key=lambda tid: (len(groups[tid]), priority(tid))):
            if take(tid):
                break
    while any(queues.values()):
        for key in queues:
            queue = queues[key]
            while queue:
                tid = queue.pop(0)
                if take(tid):
                    break
    included = window[window["traceID"].astype(str).isin(selected)].copy()
    included = included.sort_values(["startTimeMillis", "traceID", "spanID"]).reset_index(drop=True)
    def counts(frame):
        return {"pre": int((frame["startTimeMillis"] < inject_ms).sum()),
                "post": int((frame["startTimeMillis"] >= inject_ms).sum()),
                "errors": int(frame["error"].sum())}
    def parent_losses(frame, reference):
        ids = set(zip(reference["traceID"].astype(str), reference["spanID"].astype(str)))
        return sum(1 for row in frame.itertuples() if pd.notna(row.parentSpanID)
                   and str(row.parentSpanID) not in ("", "0", "nan")
                   and (str(row.traceID), str(row.parentSpanID)) not in ids)
    selected_bins = sorted({int((t-inject_ms)//60000) for t in included["startTimeMillis"]})
    unmet = [period for period in ("pre", "post") if counts(included)[period] == 0]
    source_sizes = traces.groupby(traces["traceID"].astype(str)).size().to_dict()
    # Closure is relative to this supplied source, not a claim about upstream
    # collection. A boundary-sliced, orphaned or conflicting group proves nothing.
    complete_groups = set()
    for tid in selected:
        group = groups[tid]
        explicit = group.get("trace_complete")
        if (tid not in ('', 'nan', 'None')
                and len(group) == source_sizes.get(tid, 0)
                and not group["spanID"].isna().any()
                and not group["spanID"].astype(str).duplicated().any()
                and not parent_losses(group, group)
                and (explicit is None or explicit.map(lambda v: v is True).all())):
            complete_groups.add(tid)
    included["trace_complete"] = included["traceID"].astype(str).isin(complete_groups)
    diagnostics = {
        "sampler_version": SAMPLER_VERSION, "seed": seed, "hard_cap": budget,
        "source_count": len(source), "window_count": len(window), "selected_count": len(included),
        "window_sha256": _frame_hash(window.sort_values(["traceID", "spanID"])),
        "selected_sha256": _frame_hash(included), "window_period_counts": counts(window),
        "selected_period_counts": counts(included), "selected_trace_ids": sorted(selected),
        "window_services": sorted(window["service_id"].unique()),
        "selected_services": sorted(included["service_id"].unique()),
        "selected_time_bins": selected_bins,
        "oversized_groups": sorted(tid for tid,g in groups.items() if len(g) > budget),
        "skipped_groups": len(groups) - len(selected),
        "orphan_parent_count": parent_losses(included, included),
        "boundary_parent_loss": parent_losses(included, window) - parent_losses(included, source),
        "in_window_group_complete": True,
        "source_sampled": len(included) < len(window),
        "window_records_omitted": len(window) - len(included),
        "source_coverage": "supplied-file-only-upstream-unknown",
        "selected_source_closed_groups": len(complete_groups),
        "selected_source_unproven_groups": len(selected) - len(complete_groups),
        "trace_complete_basis": "whole-supplied-source-group-no-orphans-or-duplicate-spans",
        "full_source_trace_complete": all(len(groups[tid]) == source_sizes.get(tid, 0) for tid in selected),
        "unmet_quotas": unmet, "quality_status": "insufficient_period_coverage" if unmet else "ok",
    }
    return included, diagnostics


def _normal_log_sample(plain: pd.DataFrame, inject_s: int) -> pd.DataFrame:
    """Same per-service volume, round-robin periods then minute/template strata."""
    queues = {False: {}, True: {}}
    for index, row in plain.sort_values(["ts", "message"], kind="stable").iterrows():
        post = row["ts"] >= inject_s
        key = (int((row["ts"] - inject_s) // 60), normalize.template_for(str(row["message"])))
        queues[post].setdefault(key, []).append(index)
    # Rotate temporal/template strata instead of retaining a baseline prefix.
    strata = {period: list(sorted(groups)) for period, groups in queues.items()}
    selected = []
    while len(selected) < min(LOG_SAMPLE_PER_SERVICE, len(plain)):
        for period in (False, True):
            if not strata[period] or len(selected) >= LOG_SAMPLE_PER_SERVICE:
                continue
            key = strata[period].pop(0)
            queue = queues[period][key]
            selected.append(queue.pop(len(queue) // 2))
            if queue:
                strata[period].append(key)
    return plain.loc[selected]


def _minute_score(pre: list[float], post: list[float], *,
                  absolute_floor: float) -> dict:
    """z for one per-minute series under the shared baseline-scale policy.

    Measured spread scores a plain z (no sentinel cap: log/trace severity must
    be comparable with metric severity, which is unbounded). A baseline with no
    observed spread scores from the floor scale and only when the departure
    persists across minutes, so one quiet extra observation is not an anomaly.
    """
    mean = sum(pre) / len(pre)
    std = math.sqrt(sum((v - mean) ** 2 for v in pre) / (len(pre) - 1))
    peak = max(post)
    if std > 0:
        raw = (peak - mean) / std
        return {"z": max(-bscale.SCALE_Z_CEILING, min(bscale.SCALE_Z_CEILING, raw)),
                "source": "measured", "persistent": True, "floor_scale": None}
    scale = bscale.baseline_scale(pre, abs_floor=absolute_floor)
    persistent = bscale.persistent(post, scale)
    z = bscale.deviation(peak, scale) if persistent else 0.0
    return {"z": z, "source": scale.source, "persistent": persistent,
            "floor_scale": scale.scale}


def _minute_stats(events, t0_s, t1_s, inject_s, latency=False, services=None):
    # Observed interval is inclusive at second resolution. Exclude partial
    # bins; grids are aligned to injection (never to the first event).
    if events is None:
        return {}, {"status": "missing_modality"}
    first = math.ceil((t0_s - inject_s) / 60)
    last = math.floor((t1_s + 1 - inject_s) / 60) - 1
    bins = list(range(first, last + 1))
    pre_bins, post_bins = [b for b in bins if b < 0], [b for b in bins if b >= 0]
    frame = events[events["ts"].between(t0_s, t1_s)].copy()
    frame["_bin"] = ((frame["ts"] - inject_s) // 60).astype(int)
    scores, diagnostics = {}, {"partial_bin_policy": "exclude", "aligned_to": "injection", "services": {}}
    names = sorted(set(services or []) | set(frame["service_id"].astype(str)))
    for service in names:
        group = frame[frame["service_id"] == service]
        observed = (group.groupby("_bin")["duration_us"].quantile(.95) if latency
                    else group.groupby("_bin").size())
        values = observed.to_dict()
        pre = [float(values[b]) for b in pre_bins if b in values] if latency else [float(values.get(b, 0)) for b in pre_bins]
        post = [float(values[b]) for b in post_bins if b in values] if latency else [float(values.get(b, 0)) for b in post_bins]
        info = {"baseline_count": len(pre), "post_count": len(post), "status": "ok"}
        if len(pre) < 2 or not post:
            info["status"] = "insufficient_statistics"
        else:
            mean = sum(pre) / len(pre)
            std = math.sqrt(sum((v-mean)**2 for v in pre)/(len(pre)-1))
            scored = _minute_score(pre, post, absolute_floor=COUNT_FLOOR if not latency
                                   else bscale.ABS_FLOOR)
            scores[service] = scored["z"]
            info.update(baseline_mean=mean, baseline_std=std, peak_post=max(post),
                        z=scores[service], score_source=scored["source"],
                        score_persistent=scored["persistent"],
                        floor_scale=scored["floor_scale"])
        diagnostics["services"][service] = info
    return scores, diagnostics


def minute_z(events: pd.DataFrame, t0_s: int, t1_s: int, inject_s: int,
             *, services=None, return_diagnostics=False):
    result = _minute_stats(events, t0_s, t1_s, inject_s, services=services)
    return result if return_diagnostics else result[0]


def latency_z(spans: pd.DataFrame, t0_s: int, t1_s: int, inject_s: int,
              *, return_diagnostics=False):
    result = _minute_stats(spans, t0_s, t1_s, inject_s, latency=True)
    return result if return_diagnostics else result[0]


# ---------------------------------------------------------------------------
# Topology levels
# ---------------------------------------------------------------------------

def dependency_depths(topology: dict, entities: set[str] | None = None) -> dict[str, int]:
    """Shortest dependency depth from the DEPENDS_ON roots (BFS), by service id.

    Roots are services that call others but are never called. Isolated
    services (no edges) get no depth: nothing in the topology justifies one.
    Edges are restricted to `entities` when given, matching what `load_case`
    actually writes.
    """
    def name(value: str) -> str:
        return value.replace("service:", "")
    edges = [(dep["from"], dep["to"]) for dep in topology["depends_on"]
             if entities is None or (name(dep["from"]) in entities and name(dep["to"]) in entities)]
    callers = {a for a, _ in edges}
    callees = {b for _, b in edges}
    depth = {root: 0 for root in sorted(callers - callees)}
    frontier = list(depth)
    while frontier:
        nxt = []
        for node in frontier:
            for a, b in edges:
                if a == node and b not in depth:
                    depth[b] = depth[node] + 1
                    nxt.append(b)
        frontier = nxt
    return depth


def authoritative_levels(topology: dict, entities: set[str] | None = None) -> set[str]:
    """Matching BFS levels in dependency components safe for the P4 level veto.

    Reject a whole weakly connected component if it has a directed cycle,
    an unknown/unreachable endpoint, or an edge decreasing in BFS depth.
    Nondecreasing edges then guarantee that no dependency path contradicts
    the levels of a marked candidate/anchor pair. Equal-depth DAG edges are
    safe (and occur in the shipped topology); isolated services stay unmarked.
    """
    depths = dependency_depths(topology, entities)
    known = {svc["id"] for svc in topology["services"]}
    outgoing, neighbors = {}, {}
    for dep in topology["depends_on"]:
        a, b = dep["from"], dep["to"]
        if entities is not None and (a.replace("service:", "") not in entities
                                     or b.replace("service:", "") not in entities):
            continue
        outgoing.setdefault(a, set()).add(b)
        outgoing.setdefault(b, set())
        neighbors.setdefault(a, set()).add(b)
        neighbors.setdefault(b, set()).add(a)

    safe = set()
    unseen = set(neighbors)
    while unseen:
        start = unseen.pop()
        component, pending = {start}, [start]
        while pending:
            node = pending.pop()
            for neighbor in neighbors[node]:
                if neighbor not in component:
                    component.add(neighbor)
                    unseen.discard(neighbor)
                    pending.append(neighbor)

        if any(node not in known or node not in depths for node in component):
            continue
        if any(depths[b] < depths[a] for a in component for b in outgoing[a]):
            continue

        # Kahn's algorithm also rejects equal-depth cycles, which a depth
        # monotonicity check alone cannot detect. Use sets to deduplicate edges.
        indegree = dict.fromkeys(component, 0)
        for a in component:
            for b in outgoing[a]:
                indegree[b] += 1
        pending = [node for node in component if indegree[node] == 0]
        visited = 0
        while pending:
            node = pending.pop()
            visited += 1
            for neighbor in outgoing[node]:
                indegree[neighbor] -= 1
                if indegree[neighbor] == 0:
                    pending.append(neighbor)
        if visited == len(component):
            safe.update(component)

    return {svc["id"] for svc in topology["services"]
            if svc["id"] in safe and int(svc.get("level", -1)) == depths[svc["id"]]}


# ---------------------------------------------------------------------------
# Graph writes (batched UNWIND; MERGE-on-id)
# ---------------------------------------------------------------------------

def _write_nodes(graph, label: str, rows: list[dict], batch: int = 500) -> None:
    for start in range(0, len(rows), batch):
        graph.query(
            f"UNWIND $rows AS row MERGE (n:{label} {{id: row.id}}) SET n += row",
            params={"rows": rows[start:start + batch]},
        )


def _write_edges(graph, rel: str, pairs: list[dict], src_label: str, dst_label: str,
                 batch: int = 1000) -> None:
    for start in range(0, len(pairs), batch):
        graph.query(
            f"UNWIND $rows AS row "
            f"MATCH (a:{src_label} {{id: row.src}}), (b:{dst_label} {{id: row.dst}}) "
            f"MERGE (a)-[:{rel}]->(b)",
            params={"rows": pairs[start:start + batch]},
        )


# ---------------------------------------------------------------------------
# Case loading
# ---------------------------------------------------------------------------

def load_case(graph, case_dir: Path, topology_path: Path = OB_TOPOLOGY,
              length_min: int = 20) -> dict:
    """Write one RE2-OB case into `graph`; return window + telemetry facts.

    Returns {"incident_id", "t0_ms", "t1_ms", "entities", "window_frame",
    "counts"} — the runner builds the Investigation from this and uses the
    frame only for RCAEval candidate counting.
    """
    case_dir = Path(case_dir)
    key = run_key(case_dir)
    case = read_case(case_dir)
    inject_s = case["inject_s"]
    inject_ms = inject_s * 1000
    slices = window_frames(case["metrics"], inject_s, length_min)
    frame, before, after = slices["frame"], slices["before"], slices["after"]
    t0_s, t1_s = slices["t0_ms"] // 1000, slices["t1_ms"] // 1000
    after_times = [int(t) for t in after["time"].tolist()]

    # --- services: authored topology + every entity the telemetry names ---
    topology = json.loads(Path(topology_path).read_text())
    known = {svc["name"]: svc for svc in topology["services"]}
    entities = {entity for _, entity, _ in metric_columns(frame)}
    if case["logs"] is not None:
        entities |= set(case["logs"]["container_name"].astype(str).unique())
    if case["traces"] is not None:
        entities |= {TRACE_SERVICE_MAP.get(name, name) for name in
                     case["traces"]["serviceName"].astype(str).unique()}
    # Depth is an architectural fact: derive it from the full authored edge set,
    # never from whichever services happened to emit telemetry in this case.
    authoritative = authoritative_levels(topology)
    service_rows = []
    for entity in sorted(entities):
        meta = known.get(entity, {})
        row = {
            "id": f"service:{entity}", "name": entity,
            "aliases": json.dumps(meta.get("aliases", [])),
            "criticality": float(meta.get("criticality", 0.0)),
            "level": int(meta.get("level", 0)),
        }
        if f"service:{entity}" in authoritative:
            row["level_convention"] = LEVEL_CONVENTION
        service_rows.append(row)
    _write_nodes(graph, "Service", service_rows)
    dep_pairs = [{"src": dep["from"], "dst": dep["to"]}
                 for dep in topology["depends_on"]
                 if dep["from"].replace("service:", "") in entities
                 and dep["to"].replace("service:", "") in entities]
    _write_edges(graph, "DEPENDS_ON", dep_pairs, "Service", "Service")

    counts = {"services": len(service_rows), "metrics": 0, "logs": 0, "spans": 0}

    # --- Incident: the window clock P1 reads ---
    incident_id = f"incident:RCABENCH:{key}"
    graph.query(
        "MERGE (i:Incident {id: $id}) SET i.symptom = 'benchmark case', "
        "i.start_ts = $ts, i.status = 'investigating'",
        params={"id": incident_id, "ts": inject_ms},
    )

    # --- metrics -> MetricAnomaly (per series: peak + first + qualifying onset) ---
    metric_rows, metric_edges = [], []
    for column, entity, family in metric_columns(frame):
        stats = series_stats(before[column], after[column], after_times, onset_s=inject_s)
        for sample in stats["samples"]:
            record_id = f"metric:{key}/{column}/{sample['ts']}"
            metric_rows.append({
                "id": record_id, "ts": sample["ts"], "metric": family, "column": column,
                "value": sample["value"], "z": sample["z"],
                "direction": sample["direction"],
                "baseline_mean": stats["stats"]["baseline_mean"],
                "baseline_std": stats["stats"]["baseline_std"],
                "baseline_median": stats["stats"]["baseline_median"],
                "baseline_iqr": stats["stats"]["baseline_iqr"],
                "gamma": stats["stats"]["gamma"],
                "z_capped": bool(sample["z_capped"]),
                **stats["summary"],
            })
            metric_edges.append({"src": record_id, "dst": f"service:{entity}"})
        counts["metrics"] += len(stats["samples"])
    _write_nodes(graph, "MetricAnomaly", metric_rows)
    _write_edges(graph, "OBSERVED_ON", metric_edges, "MetricAnomaly", "Service")

    # --- logs -> LogEvent (errors carry the per-service log z) ---
    log_rows, log_edges = [], []
    log_diagnostics = {}
    if case["logs"] is not None and not case["logs"].empty:
        logs = case["logs"].rename(columns={"timestamp": "ts", "container_name": "service_id"}).reset_index(drop=True)
        logs["service_id"] = logs["service_id"].astype(str)
        logs["is_error"] = logs["message"].astype(str).str.contains(_ERROR_RE)
        logs = logs[(logs["ts"] >= t0_s) & (logs["ts"] <= t1_s)]
        errors = logs[logs["is_error"]]
        z_by_service, log_diagnostics = minute_z(errors, t0_s, t1_s, inject_s,
            services=logs["service_id"].unique().tolist(), return_diagnostics=True)
        for service, group in logs.groupby("service_id"):
            service_errors = group[group["is_error"]]
            if len(service_errors) > LOG_ERROR_CAP:
                stride = max(1, len(service_errors) // LOG_ERROR_CAP)
                service_errors = service_errors.iloc[::stride].head(LOG_ERROR_CAP)
            plain = _normal_log_sample(group[~group["is_error"]], inject_s)
            for index, row in service_errors.iterrows():
                record_id = f"log:{key}/{service}/{int(row['ts'])}/{index}"
                log_rows.append({
                    "id": record_id, "ts": int(row["ts"]) * 1000, "level": "major",
                    "template_id": normalize.template_for(str(row["message"])),
                    "message": str(row["message"])[:300], "service_id": f"service:{service}",
                    "is_error": True, "z": float(z_by_service.get(service, 0.0)),
                })
                log_edges.append({"src": record_id, "dst": f"service:{service}"})
            for index, row in plain.iterrows():
                record_id = f"log:{key}/{service}/{int(row['ts'])}/{index}"
                log_rows.append({
                    "id": record_id, "ts": int(row["ts"]) * 1000, "level": "info",
                    "template_id": normalize.template_for(str(row["message"])),
                    "message": str(row["message"])[:300], "service_id": f"service:{service}",
                    "is_error": False, "z": 0.0,
                })
                log_edges.append({"src": record_id, "dst": f"service:{service}"})
        counts["logs"] = len(log_rows)
        log_diagnostics.update(
            sampling="prepost-minute-template-round-robin-v1",
            window_count=len(logs), selected_count=len(log_rows),
            normal_records_omitted=int((~logs["is_error"]).sum()) - sum(not r["is_error"] for r in log_rows),
            error_records_omitted=len(errors) - sum(r["is_error"] for r in log_rows),
            source_sampled=len(log_rows) < len(logs))
    _write_nodes(graph, "LogEvent", log_rows)
    _write_edges(graph, "OBSERVED_ON", log_edges, "LogEvent", "Service")

    # --- traces -> Span (deterministic budget; error traces first) ---
    span_rows, span_edges = [], []
    sample_diagnostics = {"quality_status": "missing_modality"}
    latency_diagnostics = error_diagnostics = {}
    if case["traces"] is not None and not case["traces"].empty:
        included, sample_diagnostics = select_trace_sample(
            case["traces"], slices["t0_ms"], slices["t1_ms"], inject_ms)
        lat, latency_diagnostics = latency_z(included, t0_s, t1_s, inject_s, return_diagnostics=True)
        err, error_diagnostics = minute_z(included[included["error"]][["ts", "service_id"]],
            t0_s, t1_s, inject_s, services=included["service_id"].unique().tolist(), return_diagnostics=True)
        per_service_z = {service: max(lat.get(service, 0.0), err.get(service, 0.0))
                         for service in set(lat) | set(err)}
        for _, row in included.iterrows():
            service = str(row["service_id"])
            parent = row["parentSpanID"]
            record_id = f"span:{key}/{row['spanID']}"
            span_rows.append({
                "id": record_id, "ts": int(row["startTimeMillis"]),
                # RCAEval startTime is epoch microseconds; never reconstruct it
                # from rounded milliseconds when source precision is available.
                "start_us": int(row["startTime"]) if pd.notna(row.get("startTime")) else int(row["startTimeMillis"] * 1000),
                "trace_complete": bool(row["trace_complete"]),
                "duration_us": int(row["duration"]) if pd.notna(row["duration"]) else 0,
                "trace_id": str(row["traceID"]), "span_id": str(row["spanID"]),
                "parent_id": str(parent) if pd.notna(parent) else "",
                "service_id": f"service:{service}",
                "op": str(row["operationName"]),
                "status": int(row["statusCode"]) if pd.notna(row["statusCode"]) else 0,
                "error": bool(row["error"]),
                "z": float(per_service_z.get(service, 0.0)),
            })
            span_edges.append({"src": record_id, "dst": f"service:{service}"})
        counts["spans"] = len(span_rows)
    _write_nodes(graph, "Span", span_rows)
    _write_edges(graph, "OBSERVED_ON", span_edges, "Span", "Service")

    return {
        "incident_id": incident_id,
        "onset_ts": inject_ms,
        "t0_ms": slices["t0_ms"],
        "t1_ms": slices["t1_ms"],
        "entities": sorted(entities),
        "window_frame": frame,
        "counts": counts,
        "sample_diagnostics": sample_diagnostics,
        "aggregate_diagnostics": {"logs": log_diagnostics, "latency": latency_diagnostics, "trace_errors": error_diagnostics},
    }
