"""
These Pydantic models are the P0 output and the API/E2E contract surface.
"""

from typing import Annotated, Literal

from pydantic import BaseModel as PydanticBaseModel, Field, ConfigDict, model_validator
import hashlib
import json

UnitScore = Annotated[float, Field(strict=True, ge=0.0, le=1.0, allow_inf_nan=False)]
NonnegativeInt = Annotated[int, Field(strict=True, ge=0)]


class BaseModel(PydanticBaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", strict=True)

    @model_validator(mode="before")
    @classmethod
    def bounded_json(cls, values):
        # Compatibility metadata is JSON, not an unbounded object/transcript tunnel.
        def visit(value, depth=0):
            if depth > 16:
                raise ValueError("handoff nesting limit exceeded")
            if isinstance(value, str) and len(value) > 131072:
                raise ValueError("handoff string limit exceeded")
            if isinstance(value, (list, dict)):
                if len(value) > 16384:
                    raise ValueError("handoff collection limit exceeded")
                for item in (value.values() if isinstance(value, dict) else value):
                    visit(item, depth + 1)
            elif isinstance(value, float) and not __import__("math").isfinite(value):
                raise ValueError("handoff requires finite JSON numbers")
        visit(values)
        return values

    @model_validator(mode="before")
    @classmethod
    def numeric_not_boolean(cls, values):
        if isinstance(values, dict):
            for key, value in values.items():
                if isinstance(value, bool) and key in {"confidence", "completeness", "reward", "sc", "score", "h_score", "confidence_cap", "chain_coherence", "initial_score", "value", "criticality", "rwr_top3_share"}:
                    raise ValueError(f"{key} must not be boolean")
        return values


class ScopeRef(BaseModel):
    incident_id: str
    revision: NonnegativeInt = 0
    fingerprint: str = Field(min_length=64, max_length=64, pattern=r"^[a-f0-9]+$")


class Versioned(BaseModel):
    schema_version: Literal[1, 2] = 2
    scope_ref: ScopeRef | None = None



class ResolvedAnchor(BaseModel):
    query: str
    matched: bool
    anchor_service: str | None = None
    via_instance: str | None = None
    candidates: list[dict] = Field(default_factory=list)


class InvestigationWindow(BaseModel):
    t0: NonnegativeInt  # epoch-ms, inclusive
    t1: NonnegativeInt  # epoch-ms, inclusive

    @model_validator(mode="after")
    def ordered(self):
        if self.t0 > self.t1:
            raise ValueError("investigation window is reversed")
        return self


class Investigation(Versioned):
    """P0 output : the object every later phase consumes."""
    incident_id: str
    session_graph: str
    symptom: str
    anchor_service: str
    window: InvestigationWindow
    budget: dict
    anchor_alert: str | None = None
    onset_ts: NonnegativeInt | None = None
    time_source: str = "unknown"
    clock_provenance: dict = Field(default_factory=dict)


class ScopeService(BaseModel):
    id: str
    criticality: float = 0.0
    sev_m: float = 0.0
    sev_l: float = 0.0
    sev_tc: float = 0.0
    self_time_ms: Annotated[float, Field(ge=0)] = 0.0
    self_time_upper_bound_ms: Annotated[float, Field(ge=0)] | None = None
    self_time_quality: Literal["complete", "partial", "unknown"] = "unknown"
    self_time_flag: bool = False
    nsigma_evidence: list[str] = Field(default_factory=list)
    hours_since_change: float | None = None
    deploy_proximity: float = 0.0


class ScopeResult(Versioned):
    incident_id: str
    anchor_service: str
    window: InvestigationWindow
    services: list[ScopeService]
    nodes: list[dict]
    edges: list[dict]
    evidence_gaps: list[dict]
    n_series_in_scope: NonnegativeInt
    truncated: bool
    config: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def bind_scope(self):
        ids = [s.id for s in self.services]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate scoped service")
        payload = [self.incident_id, self.anchor_service, self.window.model_dump(),
                   [service.model_dump() for service in sorted(self.services, key=lambda s: s.id)],
                   sorted(self.nodes, key=lambda n: str(n.get("id"))),
                   sorted(self.edges, key=lambda e: json.dumps(e, sort_keys=True)),
                   {k: v for k, v in self.config.items() if k not in {"fingerprint", "scope_ref", "scope_fingerprint"}}]
        fp = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
        if self.scope_ref is None:
            self.scope_ref = ScopeRef(incident_id=self.incident_id, fingerprint=fp)
        elif self.scope_ref.incident_id != self.incident_id or self.scope_ref.fingerprint != fp:
            raise ValueError("scope fingerprint or incident mismatch")
        return self


class RankedCandidate(BaseModel):
    id: str
    rank: Annotated[int, Field(strict=True, ge=0)]
    score: float
    components: dict[str, float]
    component_ranks: dict[str, float]
    evidence: list[str]
    rule_flags: list[str]
    selected: bool


class RankResult(Versioned):
    incident_id: str
    stage: Literal["strix"] = "strix"
    iteration: NonnegativeInt = 0
    candidates: list[RankedCandidate]
    branch_ids: list[str]
    indicators: list[dict]
    audit: list[dict]
    n_candidates: NonnegativeInt
    truncated: bool
    config: dict


class AgentRunResult(BaseModel):
    """LATS-style frozen handoff (summary/confidence/completeness/escalate)."""
    summary: str
    confidence: UnitScore
    completeness: UnitScore
    evidence_count: NonnegativeInt
    escalate: bool


# ---------------------------------------------------------------------------
# P3-P6 contracts (§4.6): typed, small; no transcripts move between phases.
# ---------------------------------------------------------------------------

class NodeVerdict(BaseModel):
    """P3 phase-1 (self-verify): is this node abnormal, and why."""
    node_id: str
    is_abnormal: bool
    has_error_logs: bool
    has_metric_anomaly: bool
    key_symptoms: list[str] = Field(default_factory=list)
    hypothesis: str = ""
    confidence: UnitScore = 0.0
    confidence_rule: str = "inference_only"
    completeness: UnitScore = 0.0


class StructuredClaim(BaseModel):
    """Explicit targets only; prose never supplies missing claim parameters."""
    metric: str | None = Field(default=None, max_length=512)
    direction: Literal["up", "down"] | None = None
    dependency_source: str | None = Field(default=None, max_length=512)
    dependency_target: str | None = Field(default=None, max_length=512)
    change_id: str | None = Field(default=None, max_length=512)


class HypothesisRecord(Versioned):
    """Persisted hypothesis , also the Track-02 memory surface."""
    id: str
    candidate: str
    branch_id: str | None = None
    claim_kind: Literal["service_abnormality", "metric_excursion", "dependency_propagation", "change_association"] = "service_abnormality"
    claim: StructuredClaim | None = None
    text: str = Field(max_length=16000)
    status: Literal["unexplored", "partial", "closed"] = "partial"
    initial_score: float = 0.0
    confidence: UnitScore = 0.0
    confidence_rule: str = "inference_only"
    depth: NonnegativeInt = 0
    visits: NonnegativeInt = 0
    value: float = 0.0
    reward: UnitScore = 0.0
    reflection: dict = Field(default_factory=dict)
    sc: UnitScore = 0.0
    evidence: list[str] = Field(default_factory=list)
    contradicting: list[str] = Field(default_factory=list)


class BranchVerdict(BaseModel):
    """P3 phase-2 (consolidate): branch-level result entering P4/P5."""
    node_id: str
    local_root_cause: str | None = None
    reason: str = ""
    confidence: UnitScore = 0.0
    confidence_rule: str = "inference_only"
    evidence_ids: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    status: Literal["unexplored", "partial", "closed"] = "closed"


class InvestResult(Versioned):
    """P3 output: all branches, persisted hypotheses, escalations, budget."""
    incident_id: str
    branches: list[BranchVerdict]
    hypotheses: list[HypothesisRecord]
    escalations: list[dict] = Field(default_factory=list)
    expansions: list[dict] = Field(default_factory=list)
    n_branches: NonnegativeInt = 0
    n_hypotheses: NonnegativeInt = 0
    budget: dict = Field(default_factory=dict)
    truncated: bool = False


class CheckResult(BaseModel):
    """One deterministic validator verdict (P4)."""
    name: str
    verdict: Literal["pass", "partial", "fail", "not_applicable"]
    detail: str = ""


class EvidencePath(BaseModel):
    """The report artifact as a query product (L10)."""
    from_id: str
    to_id: str
    nodes: list[dict] = Field(default_factory=list)   # [{id, role}]
    edges: list[dict] = Field(default_factory=list)   # [{source, target, type, support[]}]
    found: bool = False
    status: Literal["found", "not_found", "truncated"] = "not_found"
    diagnostics: dict = Field(default_factory=dict)


class ValidatedCandidate(BaseModel):
    """P4 output per candidate: checks, path, coherence, cap, verdict."""
    candidate: str
    checks: list[CheckResult] = Field(default_factory=list)
    evidence_path: EvidencePath | None = None
    chain_coherence: UnitScore = 0.0
    confidence_cap: UnitScore | None = None
    contradictions: list[str] = Field(default_factory=list)
    verdict: Literal["pass", "partial", "fail"] = "partial"
    critique: dict = Field(default_factory=dict)


class HypothesisValidation(ValidatedCandidate):
    """Exact hypothesis assessment; candidate summaries are not eligibility proof."""
    schema_version: Literal[1, 2] = 2
    scope_ref: ScopeRef | None = None
    hypothesis_id: str
    claim_type: Literal["service_abnormality", "metric_excursion", "dependency_propagation", "change_association"] = "service_abnormality"
    evidence_roles: dict[str, list[str]] = Field(default_factory=dict)
    modalities: list[Literal["metrics", "logs", "traces"]] = Field(default_factory=list)
    qualified_timing: bool = False
    plausible_path: bool = False
    source_fingerprints: dict[str, str] = Field(default_factory=dict)
    evidence_scope: str = ""
    proposal_fingerprint: str = ""


class ValidateResult(Versioned):
    """P4 output: validated candidates entering P5 fusion."""
    incident_id: str
    candidates: list[ValidatedCandidate]
    n_candidates: NonnegativeInt = 0
    hypotheses: dict[str, HypothesisValidation] = Field(default_factory=dict)
    config: dict = Field(default_factory=dict)


class FusedHypothesis(BaseModel):
    """One hypothesis after H(h) fusion with typed confidence."""
    id: str
    candidate: str
    rank: Annotated[int, Field(strict=True, ge=0)]
    h_score: UnitScore
    components: dict[str, float] = Field(default_factory=dict)  # D, R, CC, V
    confidence: UnitScore = 0.0
    confidence_rule: str = "inference_only"
    confidence_cap: UnitScore | None = None
    status: Literal["validated", "confirmed", "disproved", "partial", "unvalidated", "invalid", "unsupported"] = "partial"
    evidence: list[str] = Field(default_factory=list)
    confidence_derivation: dict = Field(default_factory=dict)


class FuseResult(Versioned):
    """P5 output: the fused ranking and its presentation contract."""
    incident_id: str
    presentation: Literal["single", "multi"]
    fused: list[FusedHypothesis]
    winner: str | None = None
    winner_candidate: str | None = None
    alternatives: list[str] = Field(default_factory=list)  # hypothesis ids, conf >= theta
    rwr_top3_share: UnitScore = 0.0
    service_ranking: list[dict] = Field(default_factory=list)
    ranking_policy: Literal["deterministic_first_shadow", "corrected_h_eligible_first", "bounded_p2_update_v1"] = "deterministic_first_shadow"
    config: dict = Field(default_factory=dict)


class ReportResult(Versioned):
    """P6 output: the rendered, machine-gated incident report (API/E2E surface)."""
    incident_id: str
    verdict: Literal["diagnosed", "insufficient_evidence"]
    tier: Literal["CONFIRMED", "LIKELY", "INCONCLUSIVE"]
    confidence: UnitScore = 0.0
    confidence_rule: str = "inference_only"
    winner_candidate: str | None = None
    summary: list[str] = Field(default_factory=list)
    next_action: str | None = None
    evidence_path: EvidencePath | None = None
    alternatives: list[dict] = Field(default_factory=list)
    runbook: dict = Field(default_factory=dict)
    owner: dict = Field(default_factory=dict)
    ruled_out: list[dict] = Field(default_factory=list)
    not_checked: list[dict] = Field(default_factory=list)
    unanswered_questions: list[str] = Field(default_factory=list)
    budget: dict = Field(default_factory=dict)
    markdown: str = ""
    report_version: str = ""
    facts: list[dict] = Field(default_factory=list)
    action_ref: dict = Field(default_factory=dict)
    citations: dict[str, int] = Field(default_factory=dict)
    config: dict = Field(default_factory=dict)


MAX_HANDOFF_BYTES = 8 * 1024 * 1024
STAGE_MODELS = {"resolve": Investigation, "scope": ScopeResult, "rank": RankResult,
                "investigate": InvestResult, "validate": ValidateResult,
                "fuse": FuseResult, "report": ReportResult}


class PersistedHandoff(BaseModel):
    schema_version: Literal[2] = 2
    stage: Literal["resolve", "scope", "rank", "investigate", "validate", "fuse", "report"]
    incident_id: str = Field(min_length=1, max_length=512)
    payload_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    payload: dict


def _bounded_load(raw):
    if not isinstance(raw, (str, bytes)) or len(raw.encode() if isinstance(raw, str) else raw) > MAX_HANDOFF_BYTES:
        raise ValueError("persisted handoff byte limit exceeded")
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    try:
        return json.loads(raw, object_pairs_hook=pairs,
                          parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
    except RecursionError as exc:
        raise ValueError("persisted handoff nesting limit exceeded") from exc


def dump_handoff(stage, result):
    """Integrity envelope, not authentication. Live evidence is still revalidated."""
    model = STAGE_MODELS[stage].model_validate(result.model_dump())
    if model.schema_version != 2:
        raise ValueError("legacy objects cannot be persisted as authoritative v2")
    payload = model.model_dump()
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    raw = PersistedHandoff(stage=stage, incident_id=model.incident_id,
                           payload_sha256=digest, payload=payload).model_dump_json()
    if len(raw.encode()) > MAX_HANDOFF_BYTES:
        raise ValueError("persisted handoff byte limit exceeded")
    return raw


def load_handoff(raw, *, stage, incident_id, scope_ref=None):
    envelope = PersistedHandoff.model_validate(_bounded_load(raw))
    if envelope.stage != stage or envelope.incident_id != incident_id:
        raise ValueError("persisted handoff stage/incident mismatch")
    digest = hashlib.sha256(json.dumps(envelope.payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if digest != envelope.payload_sha256:
        raise ValueError("persisted handoff digest mismatch")
    if envelope.payload.get("schema_version") != 2:
        raise ValueError("persisted handoff requires explicit v2 payload")
    model = STAGE_MODELS[stage].model_validate(envelope.payload)
    if model.schema_version != 2 or model.incident_id != incident_id:
        raise ValueError("persisted handoff requires matching explicit v2 payload")
    if scope_ref is not None and model.scope_ref != scope_ref:
        raise ValueError("persisted handoff scope mismatch")
    return model


# Immutable historical input allowlist, checked against actual archived bytes.
# These are evaluation outputs, NOT P3/P4 evidence or a recoverable diagnosis.
HISTORICAL_V1_SHA256 = {
    "041d2d409956d4001931c9c7937b2542c0b478765a69fae7a34012cd3be174fd",
    "c4094e37c4051e38098e5cfd47747bb459bbc3b14cc123b319cd9a105a3ea3e0",
}


def historical_v1_gate(raw, *, mode="VIEW", investigation=None, scope=None, ranking=None, invest=None, graph=None):
    """VIEW preserves real legacy results. MIGRATION recomputes P4 from live v2 inputs.

    Never relabel a v1 assessment as v2; missing historic P3/P4 is unrecoverable.
    Fresh caller-supplied proposals are explicitly a new revalidation run.
    """
    if mode not in {"VIEW", "MIGRATION"}:
        raise ValueError("unknown historical mode")
    encoded = raw.encode() if isinstance(raw, str) else raw
    if hashlib.sha256(encoded).hexdigest() not in HISTORICAL_V1_SHA256:
        raise ValueError("not a pinned historical v1 artifact")
    data = _bounded_load(raw)
    if mode == "VIEW":
        return {"schema_version": 1, "mode": "VIEW", "authoritative": False,
                "diagnosis_eligible": False, "historical": data,
                "missing": ["historical P3 proposals", "historical P4 evidence assessments"]}
    if any(v is None for v in (investigation, scope, ranking, invest, graph)):
        raise ValueError("MIGRATION requires fresh v2 context and live evidence; historical P3/P4 missing")
    from backend.app.engine import policy
    from backend.app.engine.p4_validate import assess_hypothesis
    policy.verify_scope(scope, investigation)
    if (any(v.schema_version != 2 or v.incident_id != investigation.incident_id for v in (investigation, ranking, invest))
            or ranking.scope_ref != scope.scope_ref or invest.scope_ref != scope.scope_ref
            or investigation.onset_ts is None):
        raise ValueError("migration requires scope-bound v2 inputs and authoritative onset")
    allowed = {c.id for c in ranking.candidates} & {s.id for s in scope.services}
    assessments = {h.id: assess_hypothesis(graph, investigation, scope, h, investigation.onset_ts)
                   for h in invest.hypotheses if h.candidate in allowed}
    return ValidateResult(incident_id=investigation.incident_id, scope_ref=scope.scope_ref,
        candidates=[], hypotheses=assessments,
        config={"migration": "fresh_live_revalidation_not_historical_recovery",
                "onset_ts": investigation.onset_ts, "historical_sha256": hashlib.sha256(encoded).hexdigest()})
