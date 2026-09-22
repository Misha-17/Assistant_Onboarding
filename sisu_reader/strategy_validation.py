"""Hash-bound development evidence validation with executable finite oracles."""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from .runtime_compatibility import compatible, validate_runtime_compatibility
from .learning_policy import validate_trajectory
from .efficiency_gate import PROMOTION_CONTRACT, efficiency_gate, transport_evidence


class StrategyValidationError(ValueError):
    pass


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise StrategyValidationError(message)


def _keys(value: Any, required: set[str], optional: set[str] | None = None) -> None:
    _need(isinstance(value, dict), "expected a JSON object")
    _need(required <= set(value), "required evidence fields are missing")
    _need(set(value) <= required | (optional or set()), "unexpected evidence fields")


def _hash(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _identifier(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.:@/+\-]{1,256}", value) is not None


def _json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        _need(key not in result, "duplicate JSON object key")
        result[key] = value
    return result


def read_artifact(root: Path, reference: Any) -> tuple[Any, str]:
    _keys(reference, {"path", "sha256"})
    _need(_hash(reference["sha256"]), "invalid artifact SHA-256")
    _need(isinstance(reference["path"], str) and len(reference["path"]) <= 2048, "invalid artifact path")
    base = root.resolve(strict=True)
    path = (base / reference["path"]).resolve(strict=True)
    _need(path.is_relative_to(base) and path.is_file(), "artifact escapes evidence root")
    _need(path.stat().st_size <= 8 * 1024 * 1024, "evidence artifact is too large")
    data = path.read_bytes()
    _need(hashlib.sha256(data).hexdigest() == reference["sha256"], "artifact hash mismatch")
    try:
        value = json.loads(data.decode("utf-8"), object_pairs_hook=_json_pairs,
                           parse_constant=lambda _: (_ for _ in ()).throw(StrategyValidationError("non-finite JSON number")))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise StrategyValidationError("invalid evidence JSON") from exc
    return value, str(path)


def _number(value: Any, *, positive: bool = False) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and (value > 0 if positive else value >= 0)


def _check(output: dict[str, Any], rule: dict[str, Any]) -> bool:
    _keys(rule, {"field", "op", "value", "critical"})
    _need(type(rule["critical"]) is bool, "oracle critical flag must be boolean")
    _need(isinstance(rule["field"], str) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*){0,7}", rule["field"]) is not None,
          "invalid oracle field selector")
    op = rule["op"]
    _need(op in {"equals", "contains_all", "contains_none", "set_equals", "lte", "gte"}, "oracle operator is not allowed")
    expected = rule["value"]
    if op in {"contains_all", "contains_none", "set_equals"}:
        _need(isinstance(expected, list) and bool(expected) and all(isinstance(v, str) for v in expected), "oracle needs nonempty string list")
    if op in {"lte", "gte"}:
        _need(type(expected) in (int, float) and math.isfinite(expected), "numeric oracle threshold required")
    actual: Any = output
    for part in rule["field"].split("."):
        if not isinstance(actual, dict) or part not in actual:
            return False
        actual = actual[part]
    if op == "equals":
        if isinstance(actual, bool) != isinstance(expected, bool):
            return False
        return actual == expected
    if op in {"contains_all", "contains_none"}:
        if not isinstance(actual, str):
            return False
        return all(v in actual for v in expected) if op == "contains_all" else all(v not in actual for v in expected)
    if op == "set_equals":
        return isinstance(actual, list) and all(isinstance(v, str) for v in actual) and set(actual) == set(expected)
    if type(actual) not in (int, float) or not math.isfinite(actual):
        return False
    return actual <= expected if op == "lte" else actual >= expected


def _observation(value: Any, *, collection: str, case: str, arm: str, spec_hash: str, checks: list[dict[str, Any]]) -> dict[str, Any]:
    _keys(value, {"schema_version", "kind", "collection_id", "case_id", "split", "arm", "candidate_spec_sha256", "budget", "budget_used", "elapsed_s", "output"},
          {"trajectory", "episode_id", "source_code_sha256", "model_digest", "run_id", "principal", "authorization_scope", "snapshot_id", "question_sha256", "policy_checkpoint_id", "action", "details", "runtime_epoch_sha256"})
    _need(type(value["schema_version"]) is int and value["schema_version"] == 1 and value["kind"] == "strategy_observation", "unsupported observation schema")
    _need(value["split"] == "development", "held-out/final/test evidence cannot enter learning")
    _need(value["collection_id"] == collection and value["case_id"] == case and value["arm"] == arm,
          "paired observation provenance mismatch")
    _need(value["candidate_spec_sha256"] == spec_hash, "observation strategy specification mismatch")
    budget, used = value["budget"], value["budget_used"]
    _keys(budget, {"deadline_s", "max_model_calls", "max_output_tokens", "max_documents"})
    _keys(used, {"model_calls", "output_tokens", "documents"})
    _need(_number(budget["deadline_s"], positive=True), "invalid deadline budget")
    for key in ("max_model_calls", "max_output_tokens", "max_documents"):
        _need(type(budget[key]) is int and 0 <= budget[key] <= 1_000_000_000, "invalid operation budget")
    for key in used:
        _need(type(used[key]) is int and 0 <= used[key] <= budget["max_" + key], "observed operation exceeds budget")
    _need(_number(value["elapsed_s"]) and value["elapsed_s"] <= budget["deadline_s"], "observed elapsed time exceeds budget")
    _need(isinstance(value["output"], dict), "observation output must be a JSON object")
    if "trajectory" in value:
        try:
            trajectory = validate_trajectory(value["trajectory"])
        except ValueError as exc:
            raise StrategyValidationError(str(exc)) from exc
        limits = {"elapsed_fraction": value["elapsed_s"] / budget["deadline_s"],
                  "token_fraction": used["output_tokens"] / budget["max_output_tokens"] if budget["max_output_tokens"] else 0.0,
                  "call_fraction": used["model_calls"] / budget["max_model_calls"] if budget["max_model_calls"] else 0.0}
        for name, maximum in limits.items():
            _need(sum(step["cost"][name] for step in trajectory) <= maximum + 1e-6, "trajectory costs exceed recorded usage")
    passed = [_check(value["output"], c) for c in checks]
    critical = [i for i, c in enumerate(checks) if c["critical"]]
    _need(bool(critical), "oracle requires at least one critical check")
    return {"complete": all(passed), "checks_passed": sum(passed), "checks_total": len(checks), "check_results": passed,
            "critical_failures": sum(not passed[i] for i in critical), "critical_checks": len(critical),
            "elapsed_s": float(value["elapsed_s"]), "budget": budget, "budget_used": used,
            "has_trajectory": "trajectory" in value}


def _comparison_baseline(protocol: dict[str, Any], evidence_root: Path, *, candidate_id: str,
                         principal: str | None, authorization_scope: str | None) -> dict[str, Any]:
    reference = protocol.get("baseline_checkpoint")
    if reference is None:
        return {"baseline_checkpoint_id": None, "baseline_spec_sha256": None, "baseline_artifact": None}
    checkpoint, path = read_artifact(evidence_root, reference)
    spec_fields = {"policy_contract_version", "principal", "authorization_scope", "snapshot_id", "parent_checkpoint_id", "model", "training"}
    _keys(checkpoint, spec_fields | {"id", "checkpoint_id", "spec_sha256"},
          {"status", "created_at", "promoted_validation_id", "action_support", "training_validation_ids", "training_collection_ids"})
    _need(type(checkpoint["policy_contract_version"]) is int and checkpoint["policy_contract_version"] == 1, "unsupported baseline policy contract")
    _need(_identifier(checkpoint["id"]) and checkpoint["id"].startswith("policy_") and checkpoint["checkpoint_id"] == checkpoint["id"], "invalid baseline policy identity")
    _need(checkpoint["id"] != candidate_id, "candidate cannot be its own comparison baseline")
    _need(_hash(checkpoint["spec_sha256"]) and digest({key: checkpoint[key] for key in spec_fields}) == checkpoint["spec_sha256"], "baseline policy specification hash mismatch")
    for key, expected in (("principal", principal), ("authorization_scope", authorization_scope)):
        _need(isinstance(checkpoint[key], str) and bool(checkpoint[key]), "invalid baseline owner")
        if expected is not None:
            _need(checkpoint[key] == expected, "baseline belongs to another authorization scope")
        if key in protocol:
            _need(checkpoint[key] == protocol[key], "baseline owner differs from protocol")
    if "status" in checkpoint:
        _need(checkpoint["status"] == "active", "comparison baseline was not an active policy")
    return {"baseline_checkpoint_id": checkpoint["id"], "baseline_spec_sha256": checkpoint["spec_sha256"],
            "baseline_snapshot_id": checkpoint["snapshot_id"],
            "baseline_artifact": {"path": path, "sha256": reference["sha256"]}}


def validate_receipt(receipt: Any, *, candidate_id: str, spec_hash: str, evidence_root: Path,
                     excluded_collection_ids: set[str] | None = None,
                     excluded_document_hashes: set[str] | None = None,
                     principal: str | None = None, authorization_scope: str | None = None, runtime_compatibility=None) -> dict[str, Any]:
    try:
        return _validate_receipt(receipt, candidate_id=candidate_id, spec_hash=spec_hash,
                                 evidence_root=Path(evidence_root), excluded_collection_ids=excluded_collection_ids or set(),
                                 excluded_document_hashes=excluded_document_hashes or set(), principal=principal, authorization_scope=authorization_scope, runtime_compatibility=runtime_compatibility)
    except (OSError, TypeError, KeyError, OverflowError) as exc:
        raise StrategyValidationError(f"invalid evaluation evidence: {type(exc).__name__}") from exc


def _validate_receipt(receipt: Any, *, candidate_id: str, spec_hash: str, evidence_root: Path,
                      excluded_collection_ids: set[str], excluded_document_hashes: set[str],
                      principal: str | None, authorization_scope: str | None, runtime_compatibility=None) -> dict[str, Any]:
    _keys(receipt, {"schema_version", "kind", "candidate_id", "candidate_spec_sha256", "split", "evaluator", "protocol", "collections"})
    _need(type(receipt["schema_version"]) is int and receipt["schema_version"] == 1 and receipt["kind"] == "controlled_strategy_validation", "unsupported receipt schema")
    _need(receipt["candidate_id"] == candidate_id and receipt["candidate_spec_sha256"] == spec_hash, "receipt does not bind this candidate")
    _need(receipt["split"] == "development", "held-out/final/test receipts cannot enter learning")
    evaluator = receipt["evaluator"]
    _keys(evaluator, {"id", "method", "origin"}, {"code"})
    _need(_identifier(evaluator["id"]) and evaluator["method"] == "executable_oracle" and evaluator["origin"] == "external_evaluation", "unverified or self-rated evaluator")
    if "code" in evaluator:
        # Optional JSON producer manifest binds a script hash/source revision.
        read_artifact(evidence_root, evaluator["code"])
    protocol, _ = read_artifact(evidence_root, receipt["protocol"])
    _keys(protocol, {"schema_version", "kind", "split", "candidate_spec_sha256", "collections"},
          {"declared_at", "description", "source_code_sha256", "candidate", "tasks", "config", "budget",
           "authorization_scope", "principal", "source_sha256", "materials", "strategy_database", "runtime_actions",
           "ordering", "oracle_limits", "cost_accounting", "checkpoint", "selected_oracles", "baseline_checkpoint", "runtime_compatibility", "promotion_contract", "oracle_contract"})
    _need(type(protocol["schema_version"]) is int and protocol["schema_version"] == 1 and protocol["kind"] == "controlled_strategy_validation" and protocol["split"] == "development", "invalid development protocol")
    _need(protocol["candidate_spec_sha256"] == spec_hash, "protocol candidate mismatch")
    if runtime_compatibility is not None:
        current = validate_runtime_compatibility(runtime_compatibility)
        _need(compatible(protocol, current), "protocol runtime compatibility differs from current epoch")
        _need(protocol.get("promotion_contract") == current["descriptor"]["promotion_contract"], "protocol promotion contract differs from current epoch")
        if current["descriptor"]["oracle_contract"] == "source_join_decisions_v4_grounded_complete_material_error":
            _need(protocol.get("oracle_contract") == current["descriptor"]["oracle_contract"],
                  "protocol oracle contract differs from current epoch; legacy receipts cannot be relabeled")
        regime = current["descriptor"]["budget_regime"]
        execution_budget = protocol.get("budget", {})
        _need(execution_budget.get("deadline_s") == current["descriptor"]["runtime_config"]["total_deadline_s"], "protocol deadline differs from compatibility epoch")
        _need(regime["kind"] == "bounded_calls_tokens" and all(
            execution_budget.get(key) == regime[key] for key in ("max_model_calls", "max_output_tokens")),
            "protocol enforced budgets differ from compatibility epoch")
    else:
        current = None
    if protocol.get("runtime_compatibility") is not None:
        declared = validate_runtime_compatibility(protocol["runtime_compatibility"])
        _need(current is not None and current == declared, "explicit current epoch required to verify compatible learning")
    for key, expected in (("principal", principal), ("authorization_scope", authorization_scope)):
        if expected is not None and key in protocol:
            _need(protocol[key] == expected, "protocol belongs to another authorization scope")
    baseline = _comparison_baseline(protocol, evidence_root, candidate_id=candidate_id, principal=principal, authorization_scope=authorization_scope)
    if current is not None and protocol.get("baseline_checkpoint") is not None:
        baseline_checkpoint, _ = read_artifact(evidence_root, protocol["baseline_checkpoint"])
        _need(compatible(baseline_checkpoint, current), "baseline checkpoint belongs to another runtime epoch")
    _need(isinstance(protocol["collections"], dict) and 2 <= len(protocol["collections"]) <= 100, "protocol needs separate development collections")
    collections = receipt["collections"]
    _need(isinstance(collections, list) and 2 <= len(collections) <= 100, "need at least two development collections")
    seen_collections: set[str] = set()
    seen_hashes: set[str] = set()
    seen_artifacts: set[str] = set()
    pairs_out = []
    collection_out = []
    failures = []
    for collection in collections:
        _keys(collection, {"collection_id", "split", "manifest", "oracle", "pairs"})
        cid = collection["collection_id"]
        _need(_identifier(cid) and cid not in seen_collections, "duplicate or invalid development collection")
        _need(cid not in excluded_collection_ids, "promotion validation reuses replay collection")
        seen_collections.add(cid)
        _need(collection["split"] == "development", "held-out collection cannot enter learning")
        manifest, manifest_path = read_artifact(evidence_root, collection["manifest"])
        _keys(manifest, {"schema_version", "collection_id", "split", "document_sha256"}, {"sources", "declared_at", "name", "purpose", "synthetic", "seed", "documents"})
        _need(type(manifest["schema_version"]) is int and manifest["schema_version"] == 1 and manifest["collection_id"] == cid and manifest["split"] == "development", "manifest identity/split mismatch")
        hashes = manifest["document_sha256"]
        _need(isinstance(hashes, list) and 1 <= len(hashes) <= 100000 and all(_hash(h) for h in hashes), "invalid source document hashes")
        if "documents" in manifest:
            documents = manifest["documents"]
            _need(isinstance(documents, list) and len(documents) == len(hashes), "document manifest count mismatch")
            measured_hashes = []
            for document in documents:
                _keys(document, {"path", "sha256", "kind", "bytes"})
                _need(isinstance(document["path"], str) and _hash(document["sha256"]), "invalid source document identity")
                path = (Path(manifest_path).parent / "documents" / document["path"]).resolve(strict=True)
                _need(path.is_relative_to(evidence_root.resolve()) and path.is_file(), "source document escapes evidence root")
                _need(type(document["bytes"]) is int and 0 <= document["bytes"] == path.stat().st_size <= 64 * 1024 * 1024, "source document byte count mismatch")
                measured = hashlib.sha256(path.read_bytes()).hexdigest()
                _need(measured == document["sha256"], "source document hash mismatch")
                measured_hashes.append(measured)
            _need(sorted(measured_hashes) == sorted(hashes), "manifest content hashes disagree with documents")
        _need(not (set(hashes) & seen_hashes), "development collections share source document content")
        _need(not (set(hashes) & excluded_document_hashes), "promotion documents overlap replay sources")
        seen_hashes.update(hashes)
        oracle, _ = read_artifact(evidence_root, collection["oracle"])
        if "selected_oracles" in protocol:
            _need(isinstance(protocol["selected_oracles"], dict) and protocol["selected_oracles"].get(cid) == collection["oracle"], "receipt oracle differs from frozen selected oracle")
        _keys(oracle, {"schema_version", "collection_id", "split", "cases"})
        _need(type(oracle["schema_version"]) is int and oracle["schema_version"] == 1 and oracle["collection_id"] == cid and oracle["split"] == "development", "oracle identity/split mismatch")
        _need(isinstance(oracle["cases"], list) and 1 <= len(oracle["cases"]) <= 10000, "invalid oracle cases")
        rules = {}
        for item in oracle["cases"]:
            _keys(item, {"case_id", "checks"})
            _need(_identifier(item["case_id"]) and item["case_id"] not in rules, "duplicate oracle case")
            _need(isinstance(item["checks"], list) and 1 <= len(item["checks"]) <= 100, "invalid oracle check list")
            rules[item["case_id"]] = item["checks"]
        planned = protocol["collections"].get(cid)
        _need(isinstance(planned, list) and len(planned) == len(set(planned)) and set(planned) == set(rules), "oracle differs from frozen protocol cases")
        pairs = collection["pairs"]
        _need(isinstance(pairs, list) and len(pairs) == len(planned), "incomplete paired development evaluation")
        case_ids = set()
        base_complete = candidate_complete = 0
        for pair in pairs:
            _keys(pair, {"case_id", "baseline", "candidate"})
            case_id = pair["case_id"]
            _need(case_id in rules and case_id not in case_ids, "duplicate/unplanned paired case")
            case_ids.add(case_id)
            observations = {}
            for arm in ("baseline", "candidate"):
                obs, absolute_path = read_artifact(evidence_root, pair[arm])
                if arm == "baseline":
                    _need(obs.get("policy_checkpoint_id") == baseline["baseline_checkpoint_id"], "observation differs from frozen baseline policy")
                    if baseline.get("baseline_snapshot_id") is not None:
                        _need(obs.get("snapshot_id") == baseline["baseline_snapshot_id"], "baseline policy is inapplicable to observation snapshot")
                for key, expected in (("principal", principal), ("authorization_scope", authorization_scope)):
                    if expected is not None and key in obs:
                        _need(obs[key] == expected, "observation belongs to another authorization scope")
                _need(pair[arm]["sha256"] not in seen_artifacts, "observation artifact reused across arms/cases")
                seen_artifacts.add(pair[arm]["sha256"])
                if current is not None:
                    _need(obs.get("runtime_epoch_sha256") == current["epoch_sha256"], "observation runtime differs from protocol")
                observations[arm] = _observation(obs, collection=cid, case=case_id, arm=arm, spec_hash=spec_hash, checks=rules[case_id])
                observations[arm]["artifact"] = {"path": absolute_path, "sha256": pair[arm]["sha256"]}
                if "budget" in protocol:
                    _need(obs["budget"] == protocol["budget"], "observation budget differs from frozen protocol")
                if "details" in obs:
                    details, detail_path = read_artifact(evidence_root, obs["details"])
                    _need(isinstance(details, dict), "invalid detailed execution evidence")
                    if current is not None:
                        _need(details.get("runtime_epoch_sha256") == current["epoch_sha256"], "raw execution runtime differs from protocol")
                    if "policy_checkpoint_id" in details:
                        _need(details["policy_checkpoint_id"] == obs.get("policy_checkpoint_id"), "detailed policy provenance mismatch")
                    for key, expected in (("case_id", case_id), ("arm", arm)):
                        _need(details.get(key) == expected, "detailed execution provenance mismatch")
                    if "authorization_scope" in protocol:
                        _need(details.get("authorization_scope") == protocol["authorization_scope"], "execution authorization differs from protocol")
                    if "trajectory" in obs:
                        raw_steps = details.get("raw_trajectory")
                        _need(isinstance(raw_steps, list) and len(raw_steps) == len(obs["trajectory"]), "missing linked raw trajectory")
                        for raw_step, normalized in zip(raw_steps, obs["trajectory"]):
                            _need(isinstance(raw_step, dict), "invalid raw trajectory step")
                            for key in ("step", "action", "context", "next_context", "evidence_gain"):
                                _need(raw_step.get(key) == normalized.get(key), "trajectory differs from raw controller decisions")
                            usage = raw_step.get("usage")
                            _need(isinstance(usage, dict) and set(usage) == {"elapsed_s", "output_tokens", "model_calls"}, "raw trajectory lacks measured usage")
                            _need(all(_number(v) for v in usage.values()), "invalid raw trajectory usage")
                            if "usage" in normalized:
                                _need(normalized["usage"] == usage, "normalized trajectory changes measured usage")
                            expected_cost = {
                                "elapsed_fraction": usage["elapsed_s"] / obs["budget"]["deadline_s"],
                                "token_fraction": usage["output_tokens"] / obs["budget"]["max_output_tokens"] if obs["budget"]["max_output_tokens"] else 0,
                                "call_fraction": usage["model_calls"] / obs["budget"]["max_model_calls"] if obs["budget"]["max_model_calls"] else 0,
                            }
                            _need(all(abs(normalized["cost"][key] - number) <= 1e-8 for key, number in expected_cost.items()), "trajectory cost is not measured usage divided by frozen budget")
                    if protocol.get("promotion_contract") == PROMOTION_CONTRACT:
                        tasks = protocol.get("tasks", [])
                        declared_task = next((task for task in tasks if task.get("case_id") == case_id and task.get("collection_id") == cid), None)
                        measured = transport_evidence(details, obs, current)
                        if declared_task is None or details.get("question") != declared_task.get("question"):
                            measured["verified"] = False
                            measured["reasons"].append("literal_question_not_bound_to_frozen_protocol")
                        observations[arm]["transport_evidence"] = measured
                    observations[arm]["details_artifact"] = {"path": detail_path, "sha256": obs["details"]["sha256"]}
            before, after = observations["baseline"], observations["candidate"]
            _need(before["budget"] == after["budget"], "comparison budgets differ")
            if after["critical_failures"] > before["critical_failures"]:
                failures.append(f"material_oracle_regression:{cid}/{case_id}")
            base_complete += int(before["complete"])
            candidate_complete += int(after["complete"])
            pairs_out.append({"collection_id": cid, "case_id": case_id, **observations})
        if candidate_complete < base_complete:
            failures.append(f"collection_complete_regression:{cid}")
        collection_out.append({"collection_id": cid, "document_sha256": sorted(set(hashes)),
                               "manifest_sha256": collection["manifest"]["sha256"], "oracle_sha256": collection["oracle"]["sha256"],
                               "baseline_complete": base_complete, "candidate_complete": candidate_complete})
    _need(seen_collections == set(protocol["collections"]), "receipt omits protocol collections")
    _need(len(pairs_out) >= 4, "at least four paired development cases required")
    base_count = sum(p["baseline"]["complete"] for p in pairs_out)
    candidate_count = sum(p["candidate"]["complete"] for p in pairs_out)
    if candidate_count <= base_count:
        failures.append("no_complete_case_gain")
    before_elapsed = sum(p["baseline"]["elapsed_s"] for p in pairs_out)
    after_elapsed = sum(p["candidate"]["elapsed_s"] for p in pairs_out)
    ratio = after_elapsed / before_elapsed if before_elapsed else (1.0 if not after_elapsed else None)
    if ratio is None or ratio > 1.25:
        failures.append("latency_regression_above_1.25")
    quality_failures = list(failures)
    contract = protocol.get("promotion_contract", "quality_gain_v1")
    efficiency = efficiency_gate(pairs_out, collection_out) if contract == PROMOTION_CONTRACT else None
    lane = "quality_improvement" if not quality_failures else "measured_efficiency" if efficiency and efficiency["eligible"] else None
    if lane == "measured_efficiency":
        failures = []
    elif lane is None and efficiency is not None:
        failures += ["efficiency:" + reason for reason in efficiency["reasons"]]
    return {"schema_version": 1, "runtime_compatibility": current, "receipt_sha256": digest(receipt),
            "promotion_contract": contract, "promotion_lane": lane, "efficiency": efficiency, "quality_eligibility_reasons": quality_failures, "evidence_root": str(evidence_root.resolve()),
            "protocol_sha256": receipt["protocol"]["sha256"], "evaluator": evaluator, **baseline,
            "promotion_eligible": not failures, "eligibility_reasons": failures,
            "pairs_count": len(pairs_out), "collections_count": len(collection_out),
            "baseline_complete": base_count, "candidate_complete": candidate_count,
            "complete_case_gain": candidate_count - base_count, "mean_elapsed_ratio": ratio,
            "collections": collection_out, "pairs": pairs_out}
