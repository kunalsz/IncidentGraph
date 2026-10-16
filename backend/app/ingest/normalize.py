"""Input normalization pipeline (§4.7).

Every source passes through this before touching the graph:
  1. Timestamps -> UTC epoch-ms integers (timezone ambiguity resolved at
     ingest, never at query time - LATS-RCA production normalization).
  2. Severity -> canonical hierarchy; unknown levels mapped and logged.
  3. Service/instance identity -> canonical ids via the single matcher
     (RCLAgent-V2 shipped three divergent matchers; we ship one).
  4. Logs: stack traces aggregated, message-dedup with template_id; error
     entries flagged; non-error rows retained (silent failure != absence).
  5. Metrics: unavailable metrics marked, never synthesized.
"""

import logging
import re
from datetime import datetime, timezone

from config import tuning

logger = logging.getLogger("incidentgraph.ingest")

# Canonical severity hierarchy, most severe first.
SEVERITY_ORDER = ["critical", "major", "minor", "warning", "info"]

# Raw level -> canonical. Unknown levels map to "info" and are logged,
# never silently dropped or guessed.
_SEVERITY_MAP = {
    "critical": "critical", "crit": "critical", "sev1": "critical", "p0": "critical",
    "high": "major", "error": "major", "sev2": "major", "p1": "major",
    "medium": "minor", "warn": "warning", "warning": "warning",
    "low": "warning", "sev3": "warning", "p2": "warning",
    "info": "info", "informational": "info", "debug": "info", "trace": "info",
    "notice": "info",
}

_POD_RE = re.compile(tuning.POD_PATTERN)

# Multiline stack-trace aggregation: a line starting with whitespace + "at"
# continues the previous event.
_STACKTRACE_CONT_RE = re.compile(r"^\s+(at |\.\.\. \d+ more|Caused by:)")


def to_epoch_ms(ts) -> int:
    """Normalize any timestamp representation to epoch-milliseconds UTC.

    Accepts: int/float epoch-ms, ISO-8601 strings (naive treated as UTC),
    datetime objects (naive treated as UTC).
    """
    if isinstance(ts, bool):
        raise ValueError(f"invalid timestamp: {ts!r}")
    if isinstance(ts, (int, float)):
        if ts < 0:
            raise ValueError(f"invalid timestamp: {ts!r}")
        return int(ts)
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return int(ts.timestamp() * 1000)
    if isinstance(ts, str):
        s = ts.strip()
        if s.isdigit():
            return int(s)
        iso = s.replace("Z", "+00:00")
        try:
            return to_epoch_ms(datetime.fromisoformat(iso))
        except ValueError:
            raise ValueError(f"unparseable timestamp: {ts!r}")
    raise ValueError(f"invalid timestamp type: {type(ts).__name__}")


def canonical_severity(level: str | None) -> str:
    """Map a raw severity level to the canonical hierarchy.

    Unknown levels map to "info" and are logged (mapped, never guessed).
    """
    if not level:
        return "info"
    mapped = _SEVERITY_MAP.get(str(level).strip().lower())
    if mapped is None:
        logger.warning("unknown severity level %r mapped to 'info'", level)
        return "info"
    return mapped


def service_from_pod(pod_id: str) -> str | None:
    """The single canonical pod->service matcher.

    Pattern: `^(.+?)-[a-z0-9]{5,}-[a-z0-9]{4,}$` (RCLAgent-V2's regex,
    adopted repo-wide by ingest, eval and UI).
    """
    m = _POD_RE.match(pod_id or "")
    if not m:
        return None
    name = m.group(1)
    return f"service:{name}" if name else None


def aggregate_stack_traces(lines: list[str]) -> list[str]:
    """Join multiline stack traces into single events (§4.7 item 4)."""
    events: list[str] = []
    for line in lines:
        if events and _STACKTRACE_CONT_RE.match(line):
            events[-1] = f"{events[-1]}\n{line}"
        else:
            events.append(line)
    return events


def is_error_log(level: str) -> bool:
    """Error/exception entries flag used for polarity pre-filtering."""
    return canonical_severity(level) in ("critical", "major")


def normalize_log(raw: dict) -> dict:
    """Normalize one raw log record into a LogEvent property dict."""
    message = raw.get("message", "")
    lines = aggregate_stack_traces(str(message).splitlines())
    return {
        "id": raw["id"],
        "ts": to_epoch_ms(raw["ts"]),
        "level": canonical_severity(raw.get("level", "info")),
        "template_id": raw.get("template_id") or template_for(message),
        "message": "\n".join(lines),
        "service_id": raw.get("service_id"),
        "is_error": is_error_log(raw.get("level", "info")),
    }


def template_for(message: str) -> str:
    """Derive a log template id by masking variable tokens (dedup key).

    Order matters: IPs and hex hashes first (they contain digits), then
    bare numbers, so compound tokens collapse cleanly.
    """
    masked = re.sub(r"\b\d{1,3}(\.\d{1,3}){3}(:\d+)?\b", "<ip>", message or "")
    masked = re.sub(r"\b[0-9a-f]{8,}\b", "<hex>", masked)
    masked = re.sub(r"\d+(\.\d+)?", "<num>", masked)
    return re.sub(r"\s+", " ", masked.strip())[:120] or "empty"


def normalize_alert(raw: dict) -> dict:
    return {
        "id": raw["id"],
        "ts": to_epoch_ms(raw["ts"]),
        "kind": raw.get("kind", "unknown"),
        "severity": canonical_severity(raw.get("severity")),
        "value": raw.get("value"),
        "threshold": raw.get("threshold"),
        "unit": raw.get("unit"),
    }


def normalize_metric_anomaly(raw: dict) -> dict:
    """Metrics: per-record normalization; unavailable series are marked by
    the caller (missing_metric list), never synthesized."""
    return {
        "id": raw["id"],
        "ts": to_epoch_ms(raw["ts"]),
        "metric": raw["metric"],
        "z": raw.get("z"),
        "direction": raw.get("direction"),
        "baseline_median": raw.get("baseline_median"),
        "baseline_iqr": raw.get("baseline_iqr"),
        "gamma": raw.get("gamma"),
        "delta": raw.get("delta"),
    }
