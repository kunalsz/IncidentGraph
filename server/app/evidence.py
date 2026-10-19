"""What a case actually looks like in its raw telemetry.

Read-only and independent of any run: this parses the same parquet files with
the same helpers the RCAEval adapter uses, so the numbers on screen are the
numbers the pipeline sees — not a second interpretation of the data.
"""
from __future__ import annotations

import threading
from typing import Any

from . import config

METRIC_TOP = 12
LOG_TOP = 12
SERVICE_TOP = 8
ONSET_BAND_S = 300  # the same ±5 minutes the engine's timing check uses
MESSAGE_CHARS = 200
ERROR_STATUSES = ("2", "2.0", "error", "failed", "failure")  # OpenTelemetry ERROR

_CACHE: dict[str, dict[str, Any]] = {}
_LOCK = threading.Lock()
_CACHE_LIMIT = 12


def known_case(case_id: str) -> bool:
    return any(item["case"] == case_id for item in config.available_cases())


def case_evidence(case_id: str) -> dict[str, Any] | None:
    """Bounded summary, cached per case (parsing parquet is the expensive part)."""
    if not known_case(case_id):
        return None
    with _LOCK:
        cached = _CACHE.get(case_id)
    if cached is not None:
        return cached
    built = _build(case_id)
    with _LOCK:
        _CACHE[case_id] = built
        while len(_CACHE) > _CACHE_LIMIT:
            _CACHE.pop(next(iter(_CACHE)))
    return built


def _metric_rows(adapter: Any, frame: Any, before: Any, after: Any,
                 inject_s: int) -> list[dict[str, Any]]:
    after_times = [int(value) for value in after["time"].tolist()]
    rows: list[dict[str, Any]] = []
    for column, entity, family in adapter.metric_columns(frame):
        stats = adapter.series_stats(before[column], after[column], after_times, onset_s=inject_s)
        samples = stats.get("samples") or []
        if not samples:
            continue
        peak = max(samples, key=lambda item: abs(float(item.get("z") or 0.0)))
        rows.append({
            "service": entity,
            "column": column,
            "metric": family,
            "z": round(float(peak.get("z") or 0.0), 2),
            "direction": peak.get("direction"),
            "value": float(peak.get("value") or 0.0),
            "ts": int(peak.get("ts") or 0),
            "baseline_mean": stats["stats"].get("baseline_mean"),
            "baseline_std": stats["stats"].get("baseline_std"),
        })
    rows.sort(key=lambda item: -abs(item["z"]))
    return rows[:METRIC_TOP]


def _log_summary(adapter: Any, logs: Any, t0_s: int, t1_s: int,
                 inject_s: int) -> dict[str, Any]:
    if logs is None or logs.empty:
        return {"total_errors": 0, "by_service": [], "samples": [], "band_s": ONSET_BAND_S}
    frame = logs.rename(columns={"timestamp": "ts", "container_name": "service_id"}).copy()
    frame["message"] = frame["message"].astype(str)
    frame["is_error"] = frame["message"].str.contains(adapter._ERROR_RE)
    window = frame[(frame["ts"] >= t0_s) & (frame["ts"] <= t1_s)]
    errors = window[window["is_error"]]
    near = errors[(errors["ts"] >= inject_s - ONSET_BAND_S)
                  & (errors["ts"] <= inject_s + ONSET_BAND_S)]
    chosen = near if len(near) else errors
    by_service = (
        errors.groupby("service_id").size().sort_values(ascending=False)
        .head(SERVICE_TOP).reset_index(name="count").to_dict("records")
    )
    return {
        "total_errors": int(len(errors)),
        "by_service": [{"service": str(row["service_id"]), "count": int(row["count"])}
                       for row in by_service],
        "samples": [
            {"ts": int(row.ts) * 1000, "service": str(row.service_id),
             "message": row.message[:MESSAGE_CHARS]}
            for row in chosen.head(LOG_TOP).itertuples()
        ],
        "band_s": ONSET_BAND_S,
    }


def _span_summary(traces: Any, t0_ms: int, t1_ms: int) -> dict[str, Any]:
    from backend.app.eval.rcaeval_adapter import SPAN_BUDGET, TRACE_SERVICE_MAP

    empty = {"total": 0, "errors": 0, "services": [], "slowest": [], "admitted": SPAN_BUDGET}
    if traces is None or traces.empty:
        return empty
    source = traces[traces["startTimeMillis"].notna()].copy()
    window = source[source["startTimeMillis"].between(t0_ms, t1_ms)]
    if window.empty:
        return empty
    window = window.assign(
        service_id=window["serviceName"].astype(str).map(
            lambda name: TRACE_SERVICE_MAP.get(name, name)),
        error=window["statusCode"].map(
            lambda value: str(value).lower() in ERROR_STATUSES),
        duration_us=window["duration"],
    )
    grouped = (
        window.groupby("service_id")
        .agg(spans=("duration_us", "size"), errors=("error", "sum"),
             p95_us=("duration_us", lambda values: float(values.quantile(0.95))))
        .sort_values(["errors", "p95_us"], ascending=False)
        .head(SERVICE_TOP)
    )
    slowest = window.nlargest(6, "duration_us")
    return {
        "total": int(len(window)),
        "errors": int(window["error"].sum()),
        # What the pipeline actually admits, versus what sits in the file.
        "admitted": SPAN_BUDGET,
        "services": [
            {"service": str(index), "spans": int(row.spans), "errors": int(row.errors),
             "p95_ms": round(float(row.p95_us) / 1000.0, 1)}
            for index, row in grouped.iterrows()
        ],
        "slowest": [
            {"service": str(row.service_id), "op": str(row.operationName)[:80],
             "ms": round(float(row.duration_us) / 1000.0, 1), "error": bool(row.error),
             "ts": int(row.startTimeMillis)}
            for row in slowest.itertuples()
        ],
    }


def _build(case_id: str) -> dict[str, Any]:
    from backend.app.eval import rcaeval_adapter as adapter

    case = adapter.read_case(config.DATA_DIR / case_id)
    inject_s = case["inject_s"]
    slices = adapter.window_frames(case["metrics"], inject_s)
    frame, before, after = slices["frame"], slices["before"], slices["after"]
    t0_ms, t1_ms = slices["t0_ms"], slices["t1_ms"]

    metrics = _metric_rows(adapter, frame, before, after, inject_s)
    logs = _log_summary(adapter, case["logs"], t0_ms // 1000, t1_ms // 1000, inject_s)
    spans = _span_summary(case["traces"], t0_ms, t1_ms)
    return {
        "case": case_id,
        "onset_ms": inject_s * 1000,
        "window": {"t0": t0_ms, "t1": t1_ms},
        "metrics": metrics,
        "logs": logs,
        "spans": spans,
        "counts": {"metric_series": len(metrics), "error_logs": logs["total_errors"],
                   "spans": spans["total"], "span_errors": spans["errors"]},
    }
