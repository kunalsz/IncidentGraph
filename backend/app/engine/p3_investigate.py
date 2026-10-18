"""P3 — INVESTIGATE (§4.3): bounded LLM branches over the P2 candidate set.

One BranchInvestigator per candidate, each with a fixed-size bundle (L3) —
never a telemetry dump. Every LLM call goes through the metered, cached
`LLMClient`; hypotheses are persisted as `Hypothesis` nodes with
`HAS_EVIDENCE` edges to `Evidence` nodes immediately (the tree is the
artifact, not a throwaway trace — LATS/GALA lesson).

Execution shape (two passes instead of a depth-ordered blocking pool, so a
slow branch can never deadlock its parents):
  pass A (ranking order): bundle -> self-verify -> escalation hop (if weak)
          -> hypothesis batch (fair preallocated slots, zero allowed)
          reflection/escalation disabled in v2 repair baseline
  pass B: consolidation over the verdicts pass A produced (children = callees
          inside the branch set, already verified in pass A)

Scoring (L4):  r = (e + c_comp + k)/30  ;  sc = signature self-consistency over
each hypothesis's own cited modalities, with a >=1-graph-read floor (sentinel
loophole closed)  ;  R = a*r + (1-a)*sc.

Budget (L11): every visit/hypothesis/expansion/LLM call is metered; a trip
marks the run truncated and stops cleanly — never a silent stall.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

from config import tuning

from backend.app.agent import bundle as bundle_mod
from backend.app.agent.llm import LLMClient, LLMError, load_prompt
from backend.app.engine.budget import BudgetExceeded, BudgetMeter
from backend.app.engine.contracts import (AgentRunResult, BranchVerdict,
                                          HypothesisRecord, InvestResult,
                                          Investigation, NodeVerdict,
                                          RankResult, ScopeResult)

REAL_PROVIDERS = ("openai", "anthropic")


def _max_parallel(llm: LLMClient) -> int:
    """Branch-pool size: the frozen 8 for mock/local; a gentle 3 for live
    free-tier APIs (their limits are account-wide, so per-key rotation cannot
    outrun parallel bursts); P3_MAX_PARALLEL overrides either way."""
    from backend.app.settings import RuntimeSettings
    if isinstance(getattr(llm, "runtime_settings", None), RuntimeSettings):
        return llm.runtime_settings.parallelism
    override = os.environ.get("P3_MAX_PARALLEL")
    if override:
        value = int(override)
        if value < 1:
            raise ValueError("P3_MAX_PARALLEL must be positive")
        return min(tuning.MAX_AGENT_PARALLEL, value)
    if getattr(llm.provider, "name", "") in REAL_PROVIDERS:
        return 3
    return tuning.MAX_AGENT_PARALLEL


# ---------------------------------------------------------------------------
# Pure policy helpers (unit-tested without a graph or LLM)
# ---------------------------------------------------------------------------

def reflection_reward(reflection: dict) -> float:
    """r = (e + c_comp + k)/30, each judged 0-10 (L4)."""
    e = min(10.0, max(0.0, float(reflection.get("e", 0))))
    c_comp = min(10.0, max(0.0, float(reflection.get("c_comp", 0))))
    k = min(10.0, max(0.0, float(reflection.get("k", 0))))
    return (e + c_comp + k) / 30.0


def combined_reward(reflection: dict, sc: float,
                    alpha: float = tuning.ALPHA_REFLECTION) -> float:
    """R = a*r + (1-a)*sc."""
    return alpha * reflection_reward(reflection) + (1 - alpha) * sc


def hypothesis_signature(kinds: list[str]) -> tuple:
    """Cypher-native signature analogue: (query template id, sorted labels)."""
    return ("bundle:fetch", tuple(sorted(set(kinds))))


def signature_self_consistency(signatures: list[tuple],
                               has_graph_read: list[bool]) -> list[float]:
    """Zero-token stability score; floor: no graph read => sc = 0."""
    if not signatures:
        return []
    scores = []
    for index, signature in enumerate(signatures):
        if not has_graph_read[index]:
            scores.append(0.0)
            continue
        same = sum(1 for other in signatures if other == signature)
        scores.append(same / len(signatures))
    return scores


def expansion_batches(neighbors: list[str],
                      batch_size: int = tuning.EXPANSION_BATCH_SIZE,
                      max_batches: int = tuning.MAX_EXPANSIONS_PER_BRANCH,
                      seed: int = 0) -> list[list[str]]:
    """Random batches of b neighbors, <= max_batches, seeded — random beats
    score-ordered (GALA) and the seed keeps runs reproducible."""
    rng = random.Random(seed)
    shuffled = sorted(neighbors)
    rng.shuffle(shuffled)
    batches = [shuffled[i:i + batch_size]
               for i in range(0, len(shuffled), batch_size)]
    return batches[:max_batches]


def _stable_seed(text: str) -> int:
    """Process-stable seed (Python's hash() is salted per interpreter)."""
    return int(hashlib.sha1(text.encode()).hexdigest()[:8], 16)


# ---------------------------------------------------------------------------
# Graph persistence (main thread only)
# ---------------------------------------------------------------------------

def _persist_evidence(graph, evidence_rows: list[dict]) -> None:
    for start in range(0, len(evidence_rows), 500):
        graph.query(
            "UNWIND $rows AS row MERGE (e:Evidence {id: row.id}) SET e += row",
            params={"rows": evidence_rows[start:start + 500]})


def _persist_hypotheses(graph, hypotheses: list[HypothesisRecord],
                        evidence_rows: list[dict]) -> None:
    rows = []
    for hypothesis in hypotheses:
        row = hypothesis.model_dump()
        for field in ("scope_ref", "claim"):
            if isinstance(row.get(field), dict):
                row[field] = json.dumps(row[field], sort_keys=True)
        row["reflection"] = json.dumps(row["reflection"])
        row["evidence"] = json.dumps(row["evidence"])
        row["contradicting"] = json.dumps(row["contradicting"])
        rows.append(row)
    _persist_evidence(graph, evidence_rows)
    for start in range(0, len(rows), 300):
        graph.query(
            "UNWIND $rows AS row MERGE (h:Hypothesis {id: row.id}) SET h += row",
            params={"rows": rows[start:start + 300]})
    weights = {row["source_node"]: row["weight"] for row in evidence_rows}
    wrappers = {row["source_node"]: row["id"] for row in evidence_rows}
    edges = [{"h": row["id"], "ev": wrappers.get(source, ""),
              "polarity": "support", "weight": weights.get(source, 1.0)}
             for row in rows for source in json.loads(row["evidence"])]
    for start in range(0, len(edges), 500):
        graph.query(
            "UNWIND $rows AS row "
            "MATCH (h:Hypothesis {id: row.h}), (e:Evidence {id: row.ev}) "
            "MERGE (h)-[r:HAS_EVIDENCE]->(e) "
            "SET r.polarity = row.polarity, r.weight = row.weight",
            params={"rows": edges[start:start + 500]})


def _persist_expansions(graph, parent_hypothesis: str, children: list[str]) -> None:
    graph.query(
        "UNWIND $children AS child "
        "MATCH (p:Hypothesis {id: $parent}), (h:Hypothesis {id: child}) "
        "MERGE (p)-[:EXPANDED_TO {depth: 1}]->(h)",
        params={"parent": parent_hypothesis, "children": children})


# ---------------------------------------------------------------------------
# Branch work (graph reads + LLM calls only; no writes — thread-safe)
# ---------------------------------------------------------------------------

def _available_kinds(graph, service_id: str, window) -> list[str]:
    found = []
    for label, kind in (("LogEvent", "logs"), ("Span", "traces"),
                        ("MetricAnomaly", "metrics")):
        rows = graph.query(
            f"MATCH (e:{label})-[:OBSERVED_ON]->(s:Service {{id: $sid}}) "
            f"WHERE e.ts >= $t0 AND e.ts <= $t1 RETURN count(e)",
            params={"sid": service_id, "t0": window.t0, "t1": window.t1},
        ).result_set
        if rows and rows[0][0]:
            found.append(kind)
    return found


def _source_kinds(bundle: dict) -> dict[str, str]:
    """id -> evidence kind for every citable record in a bundle."""
    kinds = {}
    for metric in bundle.get("metrics", []):
        if metric.get("id"):
            kinds[metric["id"]] = metric.get("modality", "metric")
    for span in bundle.get("spans", []):
        if span.get("id"):
            kinds[span["id"]] = "traces"
    for line in bundle.get("logs", []):
        if line.get("id"):
            kinds[line["id"]] = "logs"
    for change in bundle.get("changes", []):
        if change.get("id"):
            kinds[change["id"]] = "changes"
    return kinds


def _evidence_rows(bundle: dict, evidence_ids: list[str],
                   incident_id: str) -> list[dict]:
    """Wrap cited observation nodes as first-class Evidence nodes (L6)."""
    allowed = set(evidence_ids)
    sources: dict[str, dict] = {}
    for metric in bundle.get("metrics", []):
        if metric.get("id") in allowed:
            sources[metric["id"]] = {
                "kind": "MetricAnomaly", "ts": metric.get("ts"),
                "weight": abs(metric.get("z", 0.0)),
                "stat": json.dumps({"z": metric.get("z"),
                                    "gamma": metric.get("gamma")})}
    for span in bundle.get("spans", []):
        if span.get("id") in allowed:
            sources[span["id"]] = {"kind": "Span", "ts": span.get("ts"),
                                   "weight": 1.0, "stat": "{}"}
    for line in bundle.get("logs", []):
        if line.get("id") in allowed:
            sources[line["id"]] = {
                "kind": "LogEvent", "ts": line.get("ts"),
                "weight": 1.0,
                "stat": json.dumps({"template": line.get("template_id")})}
    for change in bundle.get("changes", []):
        if change.get("id") in allowed:
            sources[change["id"]] = {
                "kind": change.get("kind", "ConfigChange"), "ts": change.get("ts"),
                "weight": 1.0, "stat": json.dumps({"flag": change.get("flag")})}
    return [{"id": f"ev:{incident_id}:{source_id}", "incident_id": incident_id,
             "kind": meta["kind"], "source_node": source_id, "ts": meta["ts"],
             "polarity": "support", "weight": meta["weight"], "stat": meta["stat"],
             "raw": source_id}
            for source_id, meta in sorted(sources.items())]


def _metric_claim_options(bundle: dict, evidence_ids: set[str],
                          investigation: Investigation, branch_id: str) -> list[dict]:
    """Bounded exact-column facts from the admitted, displayed branch evidence.

    Match P4's qualification/onset rules, not prose or generic metric aliases.
    These are selectable associations, never inferred causal mechanisms.
    """
    from backend.app.engine import policy
    from backend.app.graph.validators import svar_lite

    onset = investigation.onset_ts
    if not policy.finite_number(onset) or bundle.get("service_id", branch_id) != branch_id:
        return []
    groups = {}
    for metric in bundle.get("metrics", []):
        column, ts = metric.get("column"), metric.get("ts")
        if (metric.get("id") not in evidence_ids or not isinstance(column, str)
                or not column.strip() or len(column) > 512
                or metric.get("service_id", branch_id) != branch_id
                or not policy.finite_number(ts)
                or not investigation.window.t0 <= ts <= investigation.window.t1
                or abs(ts - onset) > tuning.ONSET_DELTA_MS
                or not policy.modality("MetricAnomaly", metric)
                or svar_lite([metric])["verdict"] != "pass"):
            continue
        groups.setdefault(column, []).append(metric)
    options = []
    for column, records in sorted(groups.items()):
        # Pin P4's peak-selector order independent of source row order.
        records = sorted(records, key=lambda r: (-abs(r["z"]), r["id"]))
        direction = records[0].get("direction")
        if (direction not in ("up", "down") or svar_lite(records)["verdict"] != "pass"
                or not all(r.get("direction") == direction and
                           (r["z"] > 0 if direction == "up" else r["z"] < 0)
                           for r in records)):
            continue
        options.append({"claim_kind": "metric_excursion",
                        "claim": {"metric": column, "direction": direction},
                        "evidence_refs": sorted({r["id"] for r in records})[:100]})
    return options[:15]


def _dependency_claim_options(bundle: dict, evidence_ids: set[str], branch_id: str) -> list[dict]:
    """Offer only retained admitted same-trace pairs; never invent pairings."""
    retained = {s.get("id") for s in bundle.get("spans", [])} & evidence_ids
    options = {}
    for pair in bundle.get("span_pairs", []):
        source, target = pair.get("dependency_source"), pair.get("dependency_target")
        refs = pair.get("evidence_refs")
        if (pair.get("claim_eligible") is not True or target != branch_id
                or not isinstance(source, str) or not source.strip() or len(source) > 512
                or source == target or not isinstance(refs, list) or len(refs) != 2
                or not all(isinstance(ref, str) for ref in refs) or len(set(refs)) != 2
                or not set(refs) <= retained):
            continue
        key = (source, target, tuple(sorted(refs)))
        options[key] = {"claim_kind": "dependency_propagation",
                        "claim": {"dependency_source": source, "dependency_target": target},
                        "evidence_refs": sorted(refs)}
    return [options[key] for key in sorted(options)[:3]]


def _admit_supported_claim(candidate: dict, options: list[dict]) -> None:
    kind = candidate["claim_kind"]
    if kind not in ("metric_excursion", "dependency_propagation"):
        return
    claim = {k: v for k, v in candidate["claim"].items() if v is not None}
    refs = set(candidate["evidence_refs"])
    matching = [o for o in options if o["claim_kind"] == kind and o["claim"] == claim]
    if kind == "metric_excursion":
        if not any(refs & set(o["evidence_refs"]) for o in matching):
            raise ValueError("metric claim does not match a supported exact-column option and citation")
    elif not any(set(o["evidence_refs"]) <= refs for o in matching):
        raise ValueError("dependency claim requires an offered exact pair and both supporting citations")


def _run_branch_calls(graph, investigation: Investigation, branch_id: str,
                      llm: LLMClient, meter: BudgetMeter, slots: int = 8) -> dict:
    """Pass A for one branch: bundle -> verify -> escalate -> hypotheses -> reflect."""
    meter.check()
    meter.visit(branch_id)
    fetched = bundle_mod.fetch_branch_bundle(graph, branch_id, investigation.window)
    bundle = fetched["bundle"]
    evidence_ids = set(fetched["evidence_ids"])
    meter.visit(branch_id)

    verify_response = llm.complete(
        "self_verify", load_prompt("self_verify"),
        json.dumps({"service_id": branch_id,
                    "window": investigation.window.model_dump(),
                    "bundle": bundle}),
        tier="cheap", max_tokens=900)
    verdict = NodeVerdict(**verify_response.json())
    if verdict.node_id != branch_id:
        raise ValueError("self-verify returned a different branch identity")

    # Repair baseline disables optional escalation and per-proposal reflection.
    # They cannot promote model confidence or consume slots before admission.
    escalation = None
    claim_options = (_metric_claim_options(bundle, evidence_ids, investigation, branch_id)
                     + _dependency_claim_options(bundle, evidence_ids, branch_id))
    try:
        batch_response = llm.complete(
        "hypothesis_batch", load_prompt("hypothesis_batch"),
        json.dumps({"service_id": branch_id,
                    "self_verify": verdict.model_dump(), "bundle": bundle,
                    "allowed_candidate_ids": [branch_id], "max_hypotheses": slots,
                    "supported_claim_options": claim_options}),
        tier="default", max_tokens=1600)
    except (BudgetExceeded, LLMError, ValueError) as exc:
        return {"branch_id": branch_id, "bundle": bundle,
                "evidence_ids": fetched["evidence_ids"], "verdict": verdict,
                "escalation": None, "hypotheses": [], "generated": 0,
                "rejected": [str(exc)], "slots": slots, "partial_error": str(exc)}
    from backend.app.agent.responses import Proposal
    raw_candidates = batch_response.json()["hypotheses"]
    candidates, rejected, seen = [], [], set()
    for raw in raw_candidates:
        try:
            candidate = Proposal.model_validate(raw).model_dump()
            if candidate["candidate"] != branch_id:
                raise ValueError("undisclosed candidate")
            if not set(candidate["evidence_refs"]) <= evidence_ids:
                raise ValueError("undisclosed evidence reference")
            _admit_supported_claim(candidate, claim_options)
            key = (candidate["candidate"], candidate["claim_kind"],
                   json.dumps(candidate["claim"], sort_keys=True), candidate["text"],
                   tuple(sorted(candidate["evidence_refs"])))
            if key in seen:
                raise ValueError("duplicate proposal")
            seen.add(key)
            if len(candidates) >= slots:
                raise ValueError("branch slot allocation exhausted")
            candidates.append(candidate)
        except (ValueError, TypeError) as exc:
            rejected.append(str(exc))

    source_kinds = _source_kinds(bundle)
    signatures, has_read = [], []
    for candidate in candidates:
        refs = [ref for ref in (candidate.get("evidence_refs") or [])
                if ref in evidence_ids]
        kinds = [source_kinds[ref] for ref in refs if ref in source_kinds]
        signatures.append(hypothesis_signature(kinds))
        has_read.append(bool(refs))
    consistency = signature_self_consistency(signatures, has_read)

    scored = []
    for index, candidate in enumerate(candidates):
        reflection = {"mode": "disabled-v2"}
        scored.append({"candidate": candidate, "reflection": reflection,
                       "sc": round(consistency[index], 4),
                       "reward": round((1 - tuning.ALPHA_REFLECTION) * consistency[index], 4)})
    return {"branch_id": branch_id, "bundle": bundle,
            "evidence_ids": fetched["evidence_ids"], "verdict": verdict,
            "escalation": escalation, "hypotheses": scored,
            "generated": len(raw_candidates), "rejected": rejected, "slots": slots,
            **({"partial_error": "all proposals rejected: " + "; ".join(rejected)}
               if rejected and not candidates else {})}


def _consolidate(branch_id: str, verdict: NodeVerdict, evidence_ids: list[str],
                 children: list[dict], llm: LLMClient) -> BranchVerdict:
    """Pass B: frozen payloads only — never a transcript."""
    payload = {
        "node_id": branch_id,
        "verdict": {**verdict.model_dump(), "evidence_ids": evidence_ids},
        "children": children,
        "allowed_candidate_ids": [branch_id] + [child["node_id"] for child in children],
    }
    system, user = load_prompt("consolidate"), json.dumps(payload)
    # IDs can dominate the structured output. Allow room for both evidence
    # lists plus prose, with a per-call ceiling and the existing ledger as the
    # authoritative (atomic) admission check. This is planning, not tokenization.
    refs = sorted(set(evidence_ids).union(
        *(child.get("evidence_ids", []) for child in children)))
    output_limit = min(4096, max(1600, 1024 + 2 * len(json.dumps(refs).encode())))
    if isinstance(llm, LLMClient):
        ledger = llm.meter.as_dict()
        available = (llm.meter.budget.max_tokens - ledger["tokens_charged"]
                     - ledger["tokens_committed"]
                     - len(system.encode()) - len(user.encode()) - 256)
        if available <= 0:
            raise BudgetExceeded("tokens", llm.meter.budget.max_tokens + 1,
                                 llm.meter.budget.max_tokens)
        output_limit = min(output_limit, available)
    response = llm.complete("consolidate", system, user,
                            tier="default", max_tokens=output_limit)
    result = BranchVerdict(**response.json())
    allowed = {branch_id} | {child["node_id"] for child in children}
    if result.node_id != branch_id or (result.local_root_cause is not None
                                      and result.local_root_cause not in allowed):
        raise ValueError("consolidation returned an undisclosed candidate/branch")
    return result


def _child_payload(child_id: str, verdict: NodeVerdict,
                   evidence_ids: list[str]) -> dict:
    return {"node_id": child_id,
            "is_abnormal": bool(verdict.is_abnormal),
            "confidence": verdict.confidence,
            "confidence_rule": verdict.confidence_rule,
            "hypothesis": verdict.hypothesis,
            "evidence_ids": evidence_ids}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def investigate_investigation(investigation: Investigation, scope: ScopeResult,
                              ranking: RankResult,
                              meter: BudgetMeter | None = None,
                              llm: LLMClient | None = None, *,
                              minimum_primary_coverage: int | None = None) -> InvestResult:
    """P3 entry point; optional coverage floor borrows from expansion, not budget."""
    if minimum_primary_coverage is not None and (
            type(minimum_primary_coverage) is not int or minimum_primary_coverage < 0):
        raise ValueError("minimum_primary_coverage must be a nonnegative integer or None")
    if not investigation.session_graph.startswith("incidentgraph:session:"):
        raise ValueError("P3 requires an isolated session graph")
    if ranking.incident_id != investigation.incident_id:
        raise ValueError("ranking does not match investigation")
    from backend.app.graph.client import graph_for
    graph = graph_for(investigation.session_graph)
    meter = meter or BudgetMeter()
    llm = llm or LLMClient(meter)

    if scope.incident_id != investigation.incident_id:
        raise ValueError("scope does not match investigation")
    if ranking.scope_ref != scope.scope_ref:
        raise ValueError("ranking does not match admitted scope")
    allowed = {service.id for service in scope.services} & {
        candidate.id for candidate in ranking.candidates}
    branch_ids = list(dict.fromkeys(branch for branch in ranking.branch_ids
                                    if branch in allowed))
    callees: dict[str, list[str]] = {service.id: [] for service in scope.services}
    for edge in scope.edges:
        if edge["type"] == "DEPENDS_ON" and edge["source"] in callees:
            if edge["target"] in allowed:
                callees[edge["source"]].append(edge["target"])

    # Reserve fair branch capacity BEFORE bundles/provider work. Expansion has
    # a separate initial share; unused primary slots are released for expansion.
    expansion_share = int(meter.budget.max_hypotheses * meter.budget.expansion_slot_fraction)
    primary_limit = meter.budget.max_hypotheses - expansion_share
    coverage = min(minimum_primary_coverage or 0, len(branch_ids))
    if minimum_primary_coverage is not None:
        ledger = meter.as_dict()
        available = (meter.budget.max_hypotheses - ledger["hypotheses"]
                     - ledger["hypothesis_slots_reserved"])
        if available < coverage:
            raise ValueError("requested minimum primary coverage exceeds available hypothesis slots")
        primary_limit = max(primary_limit, coverage)
    plan = meter.allocate_hypothesis_slots(branch_ids, primary_limit)
    if any(plan[node] < 1 for node in branch_ids[:coverage]):
        for node in branch_ids:
            meter.release_hypothesis_slots(node)
        raise ValueError("requested minimum primary coverage could not be allocated")
    truncated = False
    results = []
    expansions = []
    hypotheses = []
    evidence_rows = []
    node_top_hypothesis = {}
    node_verdicts = {}
    branch_evidence = {}
    branches = []
    execution_errors = {}

    def run_nodes(ids, allocation):
        # At most workers futures exist; do not submit an unbounded queue and
        # never stop collecting successful siblings after another future fails.
        active_ids = [node for node in ids if allocation.get(node, 0)]
        workers = min(_max_parallel(llm), max(1, len(active_ids)))
        ordered = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending = {}
            iterator = iter(active_ids)
            def submit():
                node = next(iterator, None)
                if node is not None:
                    future = pool.submit(_run_branch_calls, graph, investigation,
                                         node, llm, meter, allocation[node])
                    pending[future] = node
            for _ in range(workers):
                submit()
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    node = pending.pop(future)
                    try:
                        ordered[node] = future.result()
                    except (BudgetExceeded, LLMError, KeyError, ValueError) as exc:
                        ordered[node] = {"branch_id": node, "error": str(exc),
                                         "exhausted": isinstance(exc, BudgetExceeded)}
                    submit()
        for node in ids:
            if node not in ordered:
                ordered[node] = {"branch_id": node, "error": "no allocated hypothesis slots", "exhausted": True}
        return [ordered[node] for node in ids]

    def retain(result, depth):
        nonlocal truncated
        node = result["branch_id"]
        if "error" in result:
            execution_errors[node] = result["error"]
            truncated = True
            meter.release_hypothesis_slots(node)
            branches.append(BranchVerdict(node_id=node, reason=result["error"], status="unexplored"))
            return
        truncated = truncated or bool(result.get("partial_error"))
        if result.get("partial_error"):
            execution_errors[node] = result["partial_error"]
        node_verdicts[node] = result["verdict"]
        branch_evidence[node] = result["evidence_ids"]
        evidence_rows.extend(_evidence_rows(result["bundle"], result["evidence_ids"], investigation.incident_id))
        for index, scored in enumerate(result["hypotheses"]):
            meter.new_hypothesis(node)
            candidate = scored["candidate"]
            record = HypothesisRecord(
                id=f"h:{investigation.incident_id}:{node}:{index}",
                candidate=node, branch_id=node, scope_ref=scope.scope_ref,
                claim_kind=candidate.get("claim_kind", "service_abnormality"),
                claim=candidate.get("claim"),
                text=candidate["text"], initial_score=candidate.get("initial_score", 0.0),
                confidence=result["verdict"].confidence,
                confidence_rule=result["verdict"].confidence_rule,
                depth=depth, visits=1, value=scored["reward"], reward=scored["reward"],
                reflection=scored["reflection"], sc=scored["sc"], evidence=candidate.get("evidence_refs", []))
            hypotheses.append(record)
            node_top_hypothesis.setdefault(node, record.id)
        meter.release_hypothesis_slots(node)

    results = run_nodes(branch_ids, plan)
    for result in results:
        retain(result, 0)

    def consolidate(node):
        nonlocal truncated
        if node in execution_errors:
            return BranchVerdict(node_id=node, reason=f"branch execution failed: {execution_errors[node]}",
                                 status="partial")
        children = [_child_payload(child, node_verdicts[child], branch_evidence[child])
                    for child in sorted(set(callees.get(node, [])))
                    if child in node_verdicts and child not in execution_errors]
        try:
            return _consolidate(node, node_verdicts[node], branch_evidence[node], children, llm)
        except (BudgetExceeded, LLMError, ValueError) as exc:
            truncated = True
            execution_errors[node] = f"consolidation skipped: {exc}"
            return BranchVerdict(node_id=node, reason=execution_errors[node], status="partial")

    branches.extend(consolidate(node) for node in branch_ids if node in node_verdicts)
    # One outgoing round only. Shared neighbors execute once, but every parent
    # retains its relationship. No out-of-scope evidence/candidates are admitted.
    parents = {}
    for branch in list(branches):
        if (branch.node_id in execution_errors or branch.status == "unexplored"
                or branch.confidence >= tuning.CONFIDENCE_GATE_THETA):
            continue
        neighbors = sorted(set(callees.get(branch.node_id, [])) - set(branch_ids))
        if neighbors:
            parents[branch.node_id] = neighbors
    expansion_ids = sorted({node for nodes in parents.values() for node in nodes})
    xplan = meter.allocate_hypothesis_slots(expansion_ids)
    admitted = set()
    for parent, nodes in parents.items():
        used = []
        for batch in expansion_batches(nodes, seed=_stable_seed(parent)):
            batch = [node for node in batch if xplan.get(node)]
            if not batch:
                continue
            try:
                meter.expansion(parent)
            except BudgetExceeded:
                truncated = True
                break
            used.append(batch)
            admitted.update(batch)
        expansions.append({"branch": parent, "batches": used,
                           "status": "scheduled", "new_nodes": len({n for b in used for n in b})})
    xresults = run_nodes(sorted(admitted), xplan) if admitted else []
    for result in xresults:
        retain(result, 1)
        if "error" not in result:
            branches.append(consolidate(result["branch_id"]))
    for node in expansion_ids:
        meter.release_hypothesis_slots(node)
    for event in expansions:
        scheduled = {node for batch in event["batches"] for node in batch}
        event["completed_ids"] = sorted((scheduled & set(node_verdicts)) - set(execution_errors))
        event["failed_ids"] = sorted(scheduled & set(execution_errors))
        event["status"] = "partial" if event["failed_ids"] else "success"
        event["errors"] = [{"branch_id": node, "error": execution_errors[node]}
                           for node in event["failed_ids"]]
    # Reconsolidate affected parents against all retained expanded siblings.
    affected = set(parents) & set(node_verdicts)
    if xresults:
        branches = [branch for branch in branches if branch.node_id not in affected]
        branches.extend(consolidate(node) for node in sorted(affected))

    persistence_errors = []
    if meter.remaining_seconds() > 0:
        try:
            _persist_hypotheses(graph, hypotheses, evidence_rows)
            for event in expansions:
                parent = node_top_hypothesis.get(event["branch"])
                children = sorted({node_top_hypothesis[node] for batch in event["batches"]
                                   for node in batch if node in node_top_hypothesis})
                if parent and children:
                    _persist_expansions(graph, parent, children)
        except Exception as exc:
            persistence_errors.append(type(exc).__name__)
            truncated = True
    else:
        persistence_errors.append("work_deadline_exhausted")
        truncated = True
    budget = meter.as_dict()
    budget["persistence_errors"] = persistence_errors
    budget["p3_execution_errors"] = dict(execution_errors)
    budget["p3_scheduling"] = {"policy": "fair-slots-v2-reflection-off", "primary_slots": plan,
        "expansion_slots": xplan,
        "branches": [{"branch_id": r["branch_id"], "generated": r.get("generated", 0),
                      "rejected": r.get("rejected", []), "admitted": len(r.get("hypotheses", []))}
                     for r in results + xresults]}
    invest = InvestResult(incident_id=investigation.incident_id, scope_ref=scope.scope_ref, branches=branches,
        hypotheses=hypotheses, escalations=[], expansions=expansions,
        n_branches=len(branches), n_hypotheses=len(hypotheses), budget=budget, truncated=truncated)
    if meter.remaining_seconds() > 0:
        try:
            from backend.app.engine.contracts import dump_handoff
            graph.query("MERGE (a:Artifact {id: $id}) SET a.kind = 'stage-view', "
                        "a.stage = 'investigate', a.version = 2, a.incident_id = $incident, a.json = $payload",
                        params={"id": f"artifact:{investigation.incident_id}:investigate:0",
                                "incident": investigation.incident_id, "payload": dump_handoff("investigate", invest)})
        except Exception as exc:
            invest.truncated = True
            invest.budget["persistence_errors"].append(type(exc).__name__)
    return invest
