"""Response models for the live pipeline API."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class _Model(BaseModel):
    model_config = ConfigDict(extra="allow")


class DatabaseStatus(_Model):
    state: Literal["ok", "offline"]
    detail: str | None = None
    graphs: int | None = None


class ProviderStatus(_Model):
    name: str = "unavailable"
    model: str | None = None
    available: bool = False
    detail: str | None = None


class Health(_Model):
    database: DatabaseStatus
    provider: ProviderStatus
    active_run: str | None = None


class CaseInfo(_Model):
    case: str
    service: str
    fault: str
    instance: str
    source_url: str | None = None


class ActivityEntry(_Model):
    """One model call, as it happened."""

    seq: int
    stage: str | None = None
    task: str | None = None
    ms: int | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    model: str | None = None
    usage: str | None = None
    finish: str | None = None
    status: str = "ok"


class StageState(_Model):
    id: str
    name: str
    status: Literal["idle", "running", "done", "failed"] = "idle"
    line: str | None = None
    detail: list[dict[str, Any]] = Field(default_factory=list)


class CostState(_Model):
    calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    tokens_total: int = 0
    tokens_charged: int = 0
    wall_s: float = 0.0
    unknown_usage_attempts: int = 0
    compliant: bool = False
    budget: dict[str, Any] = Field(default_factory=dict)


class RunResult(_Model):
    tier: str | None = None
    verdict: str | None = None
    confidence: float | None = None
    confidence_rule: str | None = None
    winner: str | None = None
    reasons: list[str] = Field(default_factory=list)
    next_action: str | None = None
    summary: list[str] = Field(default_factory=list)
    abstained: bool = False
    markdown: str | None = None
    session_graph: str | None = None
    mode: Literal["full", "preview"] = "full"
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    truncated: bool = False


class RunSnapshot(_Model):
    run_id: str
    case_id: str
    kind: Literal["benchmark", "custom"] = "benchmark"
    mode: Literal["full", "preview"] = "full"
    status: Literal["running", "completed", "failed"]
    started_at: float
    finished_at: float | None = None
    server_time: float | None = None
    duration_s: float
    stages: list[StageState] = Field(default_factory=list)
    activity: list[ActivityEntry] = Field(default_factory=list)
    cost: CostState | None = None
    result: RunResult | None = None
    error: str | None = None
    cleanup: str | None = None


class StartRunRequest(BaseModel):
    case_id: str


class TopologyNode(_Model):
    id: str
    name: str
    level: int = 0
    criticality: float = 0.0


class TopologyEdge(_Model):
    source: str
    target: str


class Topology(_Model):
    """The dependency map every case graph is built from."""

    nodes: list[TopologyNode] = Field(default_factory=list)
    edges: list[TopologyEdge] = Field(default_factory=list)
    excluded: list[str] = Field(default_factory=list)
    provenance: str | None = None
    note: str | None = None
    caveat: str | None = None
    derived_levels: bool = False


class MetricExcursion(_Model):
    service: str
    column: str
    metric: str | None = None
    z: float = 0.0
    direction: str | None = None
    value: float | None = None
    ts: int | None = None
    baseline_mean: float | None = None
    baseline_std: float | None = None


class LogSample(_Model):
    ts: int
    service: str
    message: str


class ServiceCount(_Model):
    service: str
    count: int


class LogSummary(_Model):
    total_errors: int = 0
    by_service: list[ServiceCount] = Field(default_factory=list)
    samples: list[LogSample] = Field(default_factory=list)
    band_s: int | None = None


class SpanService(_Model):
    service: str
    spans: int
    errors: int
    p95_ms: float


class SlowSpan(_Model):
    service: str
    op: str
    ms: float
    error: bool
    ts: int


class SpanSummary(_Model):
    total: int = 0
    errors: int = 0
    admitted: int | None = None
    services: list[SpanService] = Field(default_factory=list)
    slowest: list[SlowSpan] = Field(default_factory=list)


class CaseEvidence(_Model):
    """What the incident looks like in the case's own telemetry files."""

    case: str
    onset_ms: int | None = None
    window: dict[str, int] = Field(default_factory=dict)
    metrics: list[MetricExcursion] = Field(default_factory=list)
    logs: LogSummary = Field(default_factory=LogSummary)
    spans: SpanSummary = Field(default_factory=SpanSummary)
    counts: dict[str, int] = Field(default_factory=dict)


class ProbeRequest(BaseModel):
    """Everything needed to reach and verify a caller's graph."""

    host: str
    port: int = 6379
    graph: str
    anchor_service: str | None = None
    username: str | None = None
    password: str | None = None


class SchemaCheck(_Model):
    name: str
    status: Literal["pass", "warn", "fail"]
    detail: str


class ProbeResult(_Model):
    state: Literal["ok", "error"]
    verdict: Literal["ready", "incomplete", "unreachable", "blocked"]
    checks: list[SchemaCheck] = Field(default_factory=list)
    graphs: list[str] = Field(default_factory=list)
    counts: dict[str, int] | None = None
    anchor: str | None = None
    host: str | None = None
    port: int | None = None
    graph: str | None = None
    time_range: dict[str, int] | None = None
    topology: Topology | None = None
    detail: str | None = None


class CustomRunRequest(ProbeRequest):
    """A probe request plus the incident scope and run mode."""

    anchor_service: str
    window_t0: int
    window_t1: int
    onset_ts: int | None = None
    symptom: str | None = None
    incident_id: str | None = None
    mode: Literal["preview", "full"] = "full"
    label: str | None = None


class StoredRun(_Model):
    run_id: str
    case_id: str
    status: str
    started_at: float
    duration_s: float | None = None
    tier: str | None = None
    winner: str | None = None
    calls: int | None = None
    tokens_total: int | None = None


class ResultsPage(_Model):
    runs: list[StoredRun] = Field(default_factory=list)
