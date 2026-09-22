"""Prospective promotion evidence; physical work counts corroborate latency.

This is a conservative engineering gate, not a statistical noninferiority test.
Missing or unsuccessful transport evidence cannot establish an efficiency gain.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import math

PROMOTION_CONTRACT = "quality_gain_or_measured_efficiency_v2"
EFFICIENCY_CONTRACT = "paired_work_pareto_8cases_75pctquality_20pctgain_v1"
WORK_METRICS = ("model_calls", "input_tokens", "output_tokens")


def _integer(value, *, positive=False):
    return type(value) is int and (value > 0 if positive else value >= 0)


def _number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def transport_evidence(details, observation, epoch):
    """Recompute work from hash-verified successful broker records, never totals alone."""
    reasons = []
    unavailable = lambda reason: {"verified": False, "reasons": [reason]}
    if not isinstance(details, dict) or not isinstance(epoch, dict):
        return unavailable("missing_transport_details_or_epoch")
    if details.get("error") or observation.get("output", {}).get("execution_error"):
        return unavailable("execution_failure")
    if details.get("runtime_epoch_end_sha256") != epoch["epoch_sha256"]:
        return unavailable("runtime_identity_not_verified_after_answer")
    calls = details.get("calls")
    if not isinstance(calls, list) or not calls:
        return unavailable("missing_transport_records")
    descriptor = epoch["descriptor"]
    ids = set()
    measured = {name: 0 for name in WORK_METRICS}
    for call in calls:
        if not isinstance(call, dict) or not isinstance(call.get("call_id"), str) or not call["call_id"] or call["call_id"] in ids:
            return unavailable("missing_or_duplicate_transport_call_id")
        ids.add(call["call_id"])
        metrics = call.get("metrics")
        if call.get("status") != "ok" or call.get("error") or not isinstance(metrics, dict):
            return unavailable("unsuccessful_denied_or_partial_transport")
        if metrics.get("budget_admitted") is not True:
            return unavailable("transport_not_bound_to_shared_budget")
        role = call.get("role")
        expected = descriptor["model_roles"].get(role, {}).get("name")
        if not expected or str(call.get("model", "")).removesuffix(":latest") != expected:
            return unavailable("transport_role_model_mismatch")
        if metrics.get("context_tokens") != descriptor["runtime_config"]["context_tokens"]:
            return unavailable("transport_context_mismatch")
        if metrics.get("stream_incomplete") != 0 or metrics.get("done_reason") != "stop":
            return unavailable("transport_not_verified_complete")
        if not _integer(metrics.get("eval_count"), positive=True) or not _integer(metrics.get("prompt_eval_count"), positive=True):
            return unavailable("missing_or_invalid_measured_tokens")
        if (metrics.get("budget_charged_output_tokens") != metrics["eval_count"] or
                not _integer(metrics.get("budget_output_ceiling"), positive=True) or
                metrics["eval_count"] > metrics["budget_output_ceiling"]):
            return unavailable("measured_tokens_disagree_with_budget_charge")
        measured["model_calls"] += 1
        measured["input_tokens"] += metrics["prompt_eval_count"]
        measured["output_tokens"] += metrics["eval_count"]
    used = observation["budget_used"]
    account = details.get("budget_reservation")
    if not isinstance(account, dict) or account.get("descriptor") != descriptor["budget_regime"]:
        reasons.append("missing_or_mismatched_shared_budget_snapshot")
    elif (any(not _integer(account.get(key)) for key in ("denied_requests", "model_calls", "charged_output_tokens")) or
          account.get("denied_requests") != 0 or account.get("model_calls") != measured["model_calls"] or
          account.get("charged_output_tokens") != measured["output_tokens"]):
        reasons.append("budget_snapshot_disagrees_with_successful_transport")
    if (any(not _integer(details.get(key)) for key in ("model_call_attempts", "charged_output_tokens", "measured_input_tokens", "measured_output_tokens")) or
            used["model_calls"] != measured["model_calls"] or used["output_tokens"] != measured["output_tokens"] or
            details.get("model_call_attempts") != measured["model_calls"] or
            details.get("charged_output_tokens") != measured["output_tokens"] or
            details.get("measured_input_tokens") != measured["input_tokens"] or
            details.get("measured_output_tokens") != measured["output_tokens"]):
        reasons.append("reported_totals_disagree_with_transport_records")
    question = details.get("question")
    if not isinstance(question, str) or not question.strip():
        reasons.append("missing_literal_question")
    if details.get("elapsed_s") != observation["elapsed_s"]:
        reasons.append("answer_timing_disagrees_with_details")
    return {"verified": not reasons, "reasons": reasons, **measured, "call_ids": sorted(ids),
            "question_sha256": hashlib.sha256(question.encode("utf-8")).hexdigest() if isinstance(question, str) else None}


def efficiency_gate(pairs, collections):
    """Evaluate this fixed contract on unique matched development case pairs."""
    reasons = []
    groups = defaultdict(list)
    for pair in pairs:
        groups[pair["collection_id"]].append(pair)
    if len(pairs) < 8 or len(groups) < 2:
        reasons.append("needs_eight_pairs_and_two_collections")
    for cid, rows in groups.items():
        for arm in ("baseline", "candidate"):
            if sum(bool(p[arm]["complete"]) for p in rows) / len(rows) < .75:
                reasons.append(f"quality_floor_below_75pct:{cid}/{arm}")
    for pair in pairs:
        before, after = pair["baseline"], pair["candidate"]
        left, right = before.get("check_results"), after.get("check_results")
        if (not isinstance(left, list) or not isinstance(right, list) or len(left) != len(right) or
                any(type(value) is not bool for value in (*left, *right))):
            reasons.append("missing_individual_oracle_checks")
        elif any(a and not b for a, b in zip(left, right)):
            reasons.append(f"per_check_quality_regression:{pair['collection_id']}/{pair['case_id']}")
        if after["critical_failures"] > before["critical_failures"]:
            reasons.append(f"material_error_regression:{pair['collection_id']}/{pair['case_id']}")
    missing = [(p["collection_id"], p["case_id"], arm)
               for p in pairs for arm in ("baseline", "candidate")
               if not p[arm].get("transport_evidence", {}).get("verified")]
    if missing:
        reasons.append("missing_successful_hash_bound_transport_evidence")
        return {"contract": EFFICIENCY_CONTRACT, "eligible": False, "reasons": reasons,
                "paired_cases": len(pairs), "missing_transport_cases": missing}
    sources = {c["collection_id"]: tuple(c["document_sha256"]) for c in collections}
    identities = set()
    call_ids = set()
    for pair in pairs:
        before, after = (pair[arm]["transport_evidence"] for arm in ("baseline", "candidate"))
        if before["question_sha256"] != after["question_sha256"]:
            reasons.append("paired_literal_question_mismatch")
        identities.add((sources[pair["collection_id"]], before["question_sha256"]))
        for row in (before, after):
            if call_ids.intersection(row["call_ids"]):
                reasons.append("transport_record_reused_between_answers")
            call_ids.update(row["call_ids"])
    if len(identities) != len(pairs):
        reasons.append("repeated_question_and_sources_do_not_add_independent_case_support")
    totals = {arm: {name: sum(p[arm]["transport_evidence"][name] for p in pairs)
                    for name in WORK_METRICS} for arm in ("baseline", "candidate")}
    ratios = {name: totals["candidate"][name] / totals["baseline"][name] for name in WORK_METRICS}
    if any(value > 1 + 1e-12 for value in ratios.values()):
        reasons.append("aggregate_work_metric_regression")
    if sum(value <= .8 + 1e-12 for value in ratios.values()) < 2:
        reasons.append("needs_20pct_reduction_in_two_work_metrics")
    work_wins = 0
    for pair in pairs:
        before, after = (pair[arm]["transport_evidence"] for arm in ("baseline", "candidate"))
        if after["model_calls"] > before["model_calls"]:
            reasons.append(f"paired_model_call_increase:{pair['collection_id']}/{pair['case_id']}")
        if any(after[k] > before[k] * 1.10 + 1e-12 for k in ("input_tokens", "output_tokens")):
            reasons.append(f"paired_token_increase_above_10pct:{pair['collection_id']}/{pair['case_id']}")
        work_wins += any(after[k] <= .9 * before[k] + 1e-12 for k in WORK_METRICS)
        if pair["candidate"]["elapsed_s"] > pair["baseline"]["elapsed_s"] * 1.10 + 1e-12:
            reasons.append(f"paired_latency_increase_above_10pct:{pair['collection_id']}/{pair['case_id']}")
    if work_wins < math.ceil(.75 * len(pairs)):
        reasons.append("needs_work_reduction_in_75pct_of_pairs")
    elapsed = {arm: sum(p[arm]["elapsed_s"] for p in pairs) for arm in ("baseline", "candidate")}
    elapsed_ratio = elapsed["candidate"] / elapsed["baseline"] if elapsed["baseline"] else None
    if elapsed_ratio is None or elapsed_ratio > .8 + 1e-12:
        reasons.append("needs_20pct_total_answer_latency_reduction")
    return {"contract": EFFICIENCY_CONTRACT, "eligible": not reasons, "reasons": reasons,
            "paired_cases": len(pairs), "unique_question_source_pairs": len(identities),
            "work_reduction_pairs": work_wins, "work_totals": totals, "work_ratios": ratios,
            "elapsed_ratio": elapsed_ratio, "statistical_noninferiority_claim": False}
