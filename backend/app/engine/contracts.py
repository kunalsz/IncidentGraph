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


class AgentRunResult(BaseModel):
    """LATS-style frozen handoff (summary/confidence/completeness/escalate)."""
    summary: str
    confidence: float
    completeness: float
    evidence_count: int
    escalate: bool
