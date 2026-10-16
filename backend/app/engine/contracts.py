"""Frozen handoff contracts (§4.6) — typed, small, no transcripts move.

These Pydantic models are the P0 output and the API/E2E contract surface.
"""

from typing import Literal

from pydantic import BaseModel, Field


class ResolvedAnchor(BaseModel):
    query: str
    matched: bool
    anchor_service: str | None = None
    via_instance: str | None = None
    candidates: list[dict] = Field(default_factory=list)


class InvestigationWindow(BaseModel):
    t0: int  # epoch-ms, inclusive
    t1: int  # epoch-ms, inclusive


class Investigation(BaseModel):
    """P0 output (§4.3): the object every later phase consumes."""
    incident_id: str
    session_graph: str
    symptom: str
    anchor_service: str
    window: InvestigationWindow
    budget: dict
    anchor_alert: str | None = None


class ScopeService(BaseModel):
    id: str
    criticality: float = 0.0
    sev_m: float = 0.0
    sev_l: float = 0.0
    sev_tc: float = 0.0
    self_time_ms: float = 0.0
    self_time_flag: bool = False
    nsigma_evidence: list[str] = Field(default_factory=list)
    hours_since_change: float | None = None
    deploy_proximity: float = 0.0


class ScopeResult(BaseModel):
    incident_id: str
    anchor_service: str
    window: InvestigationWindow
    services: list[ScopeService]
    nodes: list[dict]
    edges: list[dict]
    evidence_gaps: list[dict]
    n_series_in_scope: int
    truncated: bool


class RankedCandidate(BaseModel):
    id: str
    rank: int
    score: float
    components: dict[str, float]
    component_ranks: dict[str, float]
    evidence: list[str]
    rule_flags: list[str]
    selected: bool


class RankResult(BaseModel):
    incident_id: str
    stage: Literal["strix"] = "strix"
    iteration: int = 0
    candidates: list[RankedCandidate]
    branch_ids: list[str]
    indicators: list[dict]
    audit: list[dict]
    n_candidates: int
    truncated: bool
    config: dict


class AgentRunResult(BaseModel):
    """LATS-style frozen handoff (summary/confidence/completeness/escalate)."""
    summary: str
    confidence: float
    completeness: float
    evidence_count: int
    escalate: bool
