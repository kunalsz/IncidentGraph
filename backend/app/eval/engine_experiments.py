"""Actual local P3/P4/P5 experiments; never configures a live provider."""
from __future__ import annotations
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path

from backend.app.agent.llm import LLMClient, MockProvider, LLMResponse
from backend.app.engine import contracts as c
from backend.app.engine.p5_fuse import fuse_investigation
from backend.app.eval.comparison import compare, load_results
from backend.app.eval.protocol import freeze_protocol
from backend.app.eval.rcaeval_runner import (BENCH_BUDGET, atomic_json, provenance,
    run_case, export_ranks)
from backend.app.eval import evaluator as ev


class EvidenceRuleProvider(MockProvider):
    """Deterministic mock with low scheduling confidence to exercise expansion.

    Confidence is NOT evidence strength and cannot pass the P4 veto. Lowering it
    here is a frozen mechanism-test rule, not a label-dependent intervention.
    """
    def complete(self, **kwargs):
        response = super().complete(**kwargs)
        if kwargs["task"] == "consolidate":
            payload = response.json()
            payload["confidence"] = .1
            return LLMResponse(json.dumps(payload), response.tokens_in, response.tokens_out,
                               response.model, usage_status="estimated")
        return response


def deterministic_client(meter):
    return LLMClient(meter, provider=EvidenceRuleProvider())


def replay_fusion(record, **options):
    """Re-execute real P5 on exact typed recorded inputs; no synthetic eligibility."""
    a = record.get("artifacts", {})
    required = ("context", "ranking", "investigation", "validation", "scope")
    if any(key not in a for key in required):
        raise ValueError("complete typed context/P2/P3/P4/scope artifacts required")
    handoffs = record.get("handoffs", {})
    if any(stage not in handoffs for stage in ("resolve", "scope", "rank", "investigate", "validate")):
        raise ValueError("complete typed v2 integrity handoffs required; historical raw stages are nonauthoritative")
    incident_id = a["context"]["incident_id"]
    context = c.load_handoff(handoffs["resolve"], stage="resolve", incident_id=incident_id)
    scope = c.load_handoff(handoffs["scope"], stage="scope", incident_id=incident_id)
    ranking = c.load_handoff(handoffs["rank"], stage="rank", incident_id=incident_id, scope_ref=scope.scope_ref)
    invest = c.load_handoff(handoffs["investigate"], stage="investigate", incident_id=incident_id, scope_ref=scope.scope_ref)
    validated = c.load_handoff(handoffs["validate"], stage="validate", incident_id=incident_id, scope_ref=scope.scope_ref)
    from backend.app.engine.policy import verify_scope
    verify_scope(scope, context)
    if not (scope.scope_ref == ranking.scope_ref == invest.scope_ref == validated.scope_ref):
        raise ValueError("stage scope mismatch")
    fused = fuse_investigation(context, ranking, invest, validated, **options)
    return fused


def arm_record(record, arm, fused=None, order=None, *, reused=False):
    if order is None:
        order = [row["candidate"].removeprefix("service:") for row in fused.service_ranking]
    ranking = c.RankResult.model_validate(record["artifacts"]["ranking"])
    ranking.candidates.sort(key=lambda x: order.index(x.id.removeprefix("service:")))
    ranks = export_ranks(ranking)
    result = {**record, "experiment_arm": arm, "service_order": order,
        "final_order": order, "ranks": ranks, "mock": record.get("provider_kind") == "mock",
        "ranking_source": arm, "stage_reuse": reused}
    result["artifacts"] = {**record["artifacts"]}
    result["top5"] = ranks[:5]
    if fused is not None:
        result["artifacts"]["fusion"] = fused.model_dump(mode="json")
        result["handoffs"] = {**record.get("handoffs", {}), "fuse":c.dump_handoff("fuse", fused)}
    result["gt_service_rank"] = order.index(record["gt_service"])+1 if record["gt_service"] in order else None
    return result


def matched_recorded_comparison(mock_records, recorded_source):
    if recorded_source is None:
        return {"status": "unavailable", "reason": "no explicitly supplied real-provider records"}
    real = load_results(recorded_source)
    pairs, excluded = {}, {}
    for name, mock in mock_records.items():
        row = real.get(name)
        if row is None:
            excluded[name] = "missing real record"
            continue
        if row.get("provider_kind") not in ("openai", "anthropic", "authorized_live"):
            excluded[name] = "real provider provenance not explicit"
            continue
        if row.get("protocol") != mock.get("protocol") or row.get("data") != mock.get("data"):
            excluded[name] = "input protocol/hash mismatch"
            continue
        if row.get("experiment_protocol_sha256") != mock.get("experiment_protocol_sha256"):
            excluded[name] = "frozen experiment protocol mismatch"
            continue
        try:
            fused = replay_fusion(row, ranking_policy="corrected_h_eligible_first")
            pairs[name] = arm_record(row, "E_recorded", fused)
        except (ValueError, KeyError):
            excluded[name] = "incomplete or invalid recorded stages"
    return {"status": "available" if pairs else "unavailable", "matched": len(pairs),
        "excluded": excluded, "comparison": compare({k:mock_records[k] for k in pairs}, pairs) if pairs else None,
        "claim": "paired recorded-provider diagnostic only; promotion gate remains independent"}


def run_engine_experiments(data_dir, output, cases, *, recorded_source=None):
    if os.environ.get("FALKORDB_HOST") not in ("localhost", "127.0.0.1") or not os.environ.get("FALKORDB_PORT") or os.environ["FALKORDB_PORT"] == "6380":
        raise ValueError("engine experiments require explicit isolated localhost DB port, never 6380")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if not cases or len(cases) != len(set(cases)):
        raise ValueError("provide nonempty unique case list")
    protocol = freeze_protocol(cases)
    atomic_json(output/"_protocol.json", protocol)  # Before ANY graph/provider work.
    source_provenance = provenance()
    atomic_json(output/"_manifest.json", {"provenance": source_provenance, "protocol_sha256": protocol["sha256"]})
    arms = {name:{} for name in ("A_p2", "B_deterministic_eligibility", "C_engine_shadow",
        "D_engine_corrected", "D_recorded_legacy_membership", "F_reflection_off", "F_coherence_off",
        "F_validator_score_off", "G_expansion_off", "H_two_fair_slots")}
    failures = []
    for name in cases:
        common = dict(primary_branches=protocol["primary_branches"])
        # At two initial branches, primary share=2 fixes ONE slot each; extra
        # capacity is reserved for expansion. H grants exactly TWO primary slots.
        base_budget = replace(BENCH_BUDGET, max_hypotheses=10, expansion_slot_fraction=.8)
        for label, agent, budget in (("A_p2", False, base_budget),
                ("B_deterministic_eligibility", True, base_budget),
                ("G_expansion_off", True, replace(base_budget, max_expansions_per_branch=0)),
                ("H_two_fair_slots", True, replace(base_budget, expansion_slot_fraction=.6))):
            row = run_case(Path(data_dir)/name, output/label, with_agent=agent,
                llm_factory=deterministic_client if agent else None, budget=budget, **common)
            row.update(provider_kind="mock", experiment_protocol_sha256=protocol["sha256"])
            atomic_json(output/label/f"{name}.record.json", row)
            if row["status"] != "ok":
                failures.append({"case":name,"arm":label,"stage":row.get("failed_stage")})
                arms[label][name] = row
                continue
            if not agent:
                arms[label][name] = row
                continue
            fused = replay_fusion(row, ranking_policy="corrected_h_eligible_first", per_service_slots=2 if label.startswith("H") else 1)
            arms[label][name] = arm_record(row, label, fused)
        row = arms["B_deterministic_eligibility"][name]
        if row["status"] != "ok":
            continue
        # B uses eligibility only, with D weight 1; no proposal-membership reward.
        arms["B_deterministic_eligibility"][name] = arm_record(row, "B_deterministic_eligibility",
            replay_fusion(row, ranking_policy="corrected_h_eligible_first",
                weights={"deterministic":1., "reflection":0., "chain_coherence":0., "validator":0.}))
        shadow = replay_fusion(row)
        arms["C_engine_shadow"][name] = arm_record(row, "C_engine_shadow", shadow, reused=True)
        arms["D_engine_corrected"][name] = arm_record(row, "D_engine_corrected",
            replay_fusion(row, ranking_policy="corrected_h_eligible_first"), reused=True)
        # The historical policy is explicitly a membership diagnostic; it is
        # intentionally NOT an eligible/winner export and cannot be promoted.
        legacy = [h.candidate.removeprefix("service:") for h in shadow.fused]
        legacy = list(dict.fromkeys(legacy + row["baseline_order"]))
        arms["D_recorded_legacy_membership"][name] = arm_record(row, "D_recorded_legacy_membership", order=legacy, reused=True)
        for key, component in (("reflection", "reflection"), ("coherence", "chain_coherence"), ("validator_score", "validator")):
            weights = dict(protocol["weights"])
            weights[component] = 0.
            total = sum(weights.values())
            weights = {k:v/total for k,v in weights.items()}
            label = f"F_{key}_off"
            arms[label][name] = arm_record(row, label, replay_fusion(row,
                ranking_policy="corrected_h_eligible_first", weights=weights), reused=True)
    equality = len(arms["C_engine_shadow"]) == len(cases) and all(
        arms["C_engine_shadow"][n]["service_order"] == arms["A_p2"][n]["service_order"] for n in arms["C_engine_shadow"])
    for label, records in arms.items():
        atomic_json(output/f"{label}.json", {"cases":list(records.values())})
    result = {"schema_version":3, "n_cases":len(cases), "engine_executed":True,
        "negative_control_full_order_equal":equality, "failures":failures,
        "protocol_sha256":protocol["sha256"],
        "source_stable_at_completion":provenance()["source_sha256"] == source_provenance["source_sha256"],
        "comparisons":{label:compare(arms["A_p2"],records) for label,records in arms.items()},
        "E_recorded_vs_deterministic":matched_recorded_comparison(arms["D_engine_corrected"],recorded_source),
        "mechanism_diagnostics": {label:{name:{
            "n_hypotheses":row.get("artifacts",{}).get("investigation",{}).get("n_hypotheses"),
            "scheduling":row.get("artifacts",{}).get("investigation",{}).get("budget",{}).get("p3_scheduling"),
            "expansions":row.get("artifacts",{}).get("investigation",{}).get("expansions",[]),
            "fusion_config":row.get("artifacts",{}).get("fusion",{}).get("config"),
            "cleanup":row.get("cleanup"), "phase_events":row.get("phase_events",[])}
            for name,row in records.items()} for label,records in arms.items()},
        "promotion_gate":protocol["promotion_gate"],
        "limitations":["All new proposals are deterministic local mocks; no model gain claim.",
            "F reflection-off may be an exact null: repaired P3 reflection already disabled.",
            "Reused C/D/F stages share B acquisition cost; do not sum ledgers as new provider spend.",
            "G/H execute independently; differences include declared capacity/expansion policy.",
            "Legacy D membership output deliberately preserves historical failure mode; never a safe production policy."]}
    atomic_json(output/"_experiments.json",result)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--cases",required=True)
    parser.add_argument("--recorded",type=Path)
    args=parser.parse_args(argv)
    result=run_engine_experiments(args.data,args.output,args.cases.split(","),recorded_source=args.recorded)
    print(json.dumps({k:v for k,v in result.items() if k != "comparisons"},indent=2))
    return int(bool(result["failures"]) or not result["negative_control_full_order_equal"])


if __name__ == "__main__":
    raise SystemExit(main())
