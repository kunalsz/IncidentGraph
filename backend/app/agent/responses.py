"""Versioned strict provider boundary, independent of persisted contracts.
Batch envelopes are strict; malformed proposals are rejected individually by P3.
"""
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field
from backend.app.engine.contracts import NodeVerdict, BranchVerdict, AgentRunResult, ProposalClaim

SCHEMA_VERSION = "task-responses-v2"
Score = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
Text = Annotated[str, Field(max_length=16000)]

class StrictResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

class SelfVerify(NodeVerdict):
    model_config = ConfigDict(extra="forbid", strict=True)
    confidence_rule: Literal["cross_modal_consensus", "single_modality", "inference_only"] = "inference_only"

class Consolidate(BranchVerdict):
    model_config = ConfigDict(extra="forbid", strict=True)
    confidence_rule: Literal["cross_modal_consensus", "single_modality", "inference_only"] = "inference_only"

class Escalate(AgentRunResult):
    model_config = ConfigDict(extra="forbid", strict=True)

class Proposal(ProposalClaim, StrictResponse):
    text: Text
    candidate: str
    evidence_refs: list[str] = Field(max_length=100)
    initial_score: Score = 0

class Batch(StrictResponse):
    hypotheses: list[dict] = Field(max_length=100)

class Reflection(StrictResponse):
    e: Annotated[float, Field(ge=0, le=10, allow_inf_nan=False)]
    c_comp: Annotated[float, Field(ge=0, le=10, allow_inf_nan=False)]
    k: Annotated[float, Field(ge=0, le=10, allow_inf_nan=False)]
    reason: Text

class Critique(StrictResponse):
    verdict: Literal["pass", "partial", "fail"]
    reason: Text
    confidence_cap: Score | None = None

class Report(StrictResponse):
    summary: list[Text] = Field(max_length=10)
    next_action: Text | None = None
    unanswered_questions: list[Text] = Field(default_factory=list, max_length=100)

TASK_SCHEMAS = {"self_verify": SelfVerify, "hypothesis_batch": Batch,
                "consolidate": Consolidate, "escalate": Escalate,
                "reflection": Reflection, "critique": Critique, "report": Report}

def validate_task(task, payload):
    schema = TASK_SCHEMAS.get(task)
    return schema.model_validate(payload).model_dump() if schema else payload
