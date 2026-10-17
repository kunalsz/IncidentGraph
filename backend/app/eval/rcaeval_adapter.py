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

from backend.app.ingest import normalize

OB_TOPOLOGY = Path(__file__).resolve().parent.parent / "seed" / "rcaeval_ob_topology.json"

WINDOW_HALF_S = 600          # RCAEval --length 20 -> 600 samples each side (1 Hz)
Z_CAP = 6.0                  # sentinel for a zero-variance baseline with deviation
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


def series_stats(before: pd.Series, after: pd.Series, after_times: list[int]) -> dict:
    """Baseline statistics + peak/onset anomaly samples for one metric series.

    Emits up to two samples: the peak |z| post-injection row and the first
    post-injection row (so P1's ±60 s n-sigma rule sees the onset). A
    zero-variance baseline with a deviating window admits no z; we cap at
    ±Z_CAP rather than divide by zero (`z_capped` flags the sentinel).
    """
    base = before.astype(float).dropna()
    win = after.astype(float)
    valid = win.dropna()
    mean = float(base.mean()) if len(base) else 0.0
    std = float(base.std(ddof=1)) if len(base) > 1 else 0.0
    if not math.isfinite(std):
        std = 0.0
    median = float(base.median()) if len(base) else 0.0
    iqr = (float(base.quantile(0.75)) - float(base.quantile(0.25))) if len(base) else 0.0
    stats = {
        "baseline_mean": mean, "baseline_std": std,
        "baseline_median": median, "baseline_iqr": iqr if iqr > 0 else 0.0,
        "gamma": float((valid - median).abs().max() / iqr) if iqr > 0 and len(valid) else 0.0,
    }
    if valid.empty:
        return {"stats": stats, "samples": []}

    def z_of(value: float) -> tuple[float, bool]:
        if std > 0:
            return (value - mean) / std, False
        return (Z_CAP if value != mean else 0.0), value != mean

    positions = []
    peak_position = int(valid.sub(mean).abs().idxmax())
    positions.append(peak_position)
    onset = valid.index[0]
    if onset != peak_position:
        positions.append(int(onset))
    samples = []
    for position in positions:
        value = float(win.loc[position])
        z_value, capped = z_of(value)
        samples.append({
            "ts": int(after_times[position]) * 1000,
            "value": value, "z": z_value, "z_capped": capped,
            "direction": "up" if value >= mean else "down",
        })
    return {"stats": stats, "samples": samples}


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
    source_sizes = source.groupby(source["traceID"].astype(str)).size().to_dict()
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
        "full_source_trace_complete": all(len(groups[tid]) == source_sizes.get(tid, 0) for tid in selected),
        "unmet_quotas": unmet, "quality_status": "insufficient_period_coverage" if unmet else "ok",
    }
    return included, diagnostics


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
            peak = max(post)
            scores[service] = min((peak-mean)/std, Z_CAP) if std else (Z_CAP if peak > mean else 0.0)
            info.update(baseline_mean=mean, baseline_std=std, peak_post=peak, z=scores[service])
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
    service_rows = []
    for entity in sorted(entities):
        meta = known.get(entity, {})
        service_rows.append({
            "id": f"service:{entity}", "name": entity,
            "aliases": json.dumps(meta.get("aliases", [])),
            "criticality": float(meta.get("criticality", 0.0)),
            "level": int(meta.get("level", 0)),
        })
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

    # --- metrics -> MetricAnomaly (per series: peak + onset samples) ---
    metric_rows, metric_edges = [], []
    for column, entity, family in metric_columns(frame):
        stats = series_stats(before[column], after[column], after_times)
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
            })
            metric_edges.append({"src": record_id, "dst": f"service:{entity}"})
        counts["metrics"] += len(stats["samples"])
    _write_nodes(graph, "MetricAnomaly", metric_rows)
    _write_edges(graph, "OBSERVED_ON", metric_edges, "MetricAnomaly", "Service")

    # --- logs -> LogEvent (errors carry the per-service log z) ---
    log_rows, log_edges = [], []
    log_diagnostics = {}
    if case["logs"] is not None and not case["logs"].empty:
        logs = case["logs"].rename(columns={"timestamp": "ts", "container_name": "service_id"})
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
            plain = group[~group["is_error"]].head(LOG_SAMPLE_PER_SERVICE)
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
        "t0_ms": slices["t0_ms"],
        "t1_ms": slices["t1_ms"],
        "entities": sorted(entities),
        "window_frame": frame,
        "counts": counts,
        "sample_diagnostics": sample_diagnostics,
        "aggregate_diagnostics": {"logs": log_diagnostics, "latency": latency_diagnostics, "trace_errors": error_diagnostics},
    }
