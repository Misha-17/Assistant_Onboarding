"""Durable controller learning with externally verified replay and gated policies.

Runtime signals are observations, never correctness labels. Only hash-verified
development artifacts with finite executable oracles may supply training returns.
"""
from __future__ import annotations

import json
import math
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .runtime_compatibility import compatible, epoch_of, validate_runtime_compatibility
from .learning_policy import (ACTIONS, FEATURES, CONTEXT_NAMES, encode_context, fit_policy,
                              rank_policy, replay_returns, validate_context, validate_trajectory)
from .strategy_validation import StrategyValidationError, canonical, digest, read_artifact, validate_receipt

GAP_KINDS = ("missing_evidence", "incomplete_coverage", "scope_ambiguity", "temporal_ambiguity",
             "reference_unresolved", "failed_read", "duplicate_evidence", "search_miss")
OUTCOME_CODES = ("new_evidence", "no_progress", "deadline", "error", "answered", "abstained", "reading_complete")
REVOCATION_REASONS = ("manual", "regression", "stale", "scope_changed", "evaluation_invalid")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS strategy_candidates (
 id TEXT PRIMARY KEY, spec_json TEXT NOT NULL, status TEXT NOT NULL,
 created_at TEXT NOT NULL, updated_at TEXT NOT NULL, promoted_validation_id TEXT);
CREATE TABLE IF NOT EXISTS strategy_validations (
 id TEXT PRIMARY KEY, subject_id TEXT NOT NULL, subject_kind TEXT NOT NULL,
 principal TEXT NOT NULL, authorization_scope TEXT NOT NULL,
 summary_json TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS strategy_observations (
 id TEXT PRIMARY KEY, candidate_id TEXT NOT NULL, observation_json TEXT NOT NULL,
 created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS learning_episodes (
 id TEXT PRIMARY KEY, principal TEXT NOT NULL, authorization_scope TEXT NOT NULL,
 snapshot_id TEXT, episode_json TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS learning_replay (
 id TEXT PRIMARY KEY, validation_id TEXT NOT NULL, collection_id TEXT NOT NULL,
 case_id TEXT NOT NULL, arm TEXT NOT NULL, principal TEXT NOT NULL,
 authorization_scope TEXT NOT NULL, experience_json TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS learning_policies (
 id TEXT PRIMARY KEY, principal TEXT NOT NULL, authorization_scope TEXT NOT NULL,
 snapshot_id TEXT, parent_id TEXT, spec_sha256 TEXT NOT NULL, model_json TEXT NOT NULL,
 training_json TEXT NOT NULL, status TEXT NOT NULL, promoted_validation_id TEXT,
 created_at TEXT NOT NULL);
CREATE UNIQUE INDEX IF NOT EXISTS learning_one_active_policy
 ON learning_policies(principal, authorization_scope) WHERE status = 'active';
CREATE INDEX IF NOT EXISTS learning_replay_owner ON learning_replay(principal, authorization_scope);
CREATE TABLE IF NOT EXISTS learning_audit (
 sequence INTEGER PRIMARY KEY AUTOINCREMENT, event TEXT NOT NULL, subject_id TEXT,
 principal TEXT NOT NULL, authorization_scope TEXT NOT NULL, details_json TEXT NOT NULL,
 created_at TEXT NOT NULL);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _text(value: Any, name: str, *, maximum: int = 256) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum or any(ord(c) < 32 for c in value):
        raise ValueError(f"invalid {name}")
    return value


def _labels(values: Iterable[str], allowed: Iterable[str], name: str) -> list[str]:
    if isinstance(values, str):
        raise ValueError(f"{name} must be a sequence")
    result = sorted(set(values))
    if any(v not in allowed for v in result):
        raise ValueError(f"unknown {name}")
    return result


def _owner(principal: str, authorization_scope: str) -> tuple[str, str]:
    return _text(principal, "principal"), _text(authorization_scope, "authorization scope", maximum=1024)


def _question_hash(value: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("question_sha256 must be a SHA-256, not question text")
    return value


class StrategyStore:
    def __init__(self, path: str | Path, *, runtime_compatibility=None):
        self.runtime_compatibility = validate_runtime_compatibility(runtime_compatibility) if runtime_compatibility is not None else None
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path, timeout=15)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(_SCHEMA)
        self._db.commit()

    def compatible(self, record):
        return compatible(record, self.runtime_compatibility)

    def _require_epoch(self):
        if self.runtime_compatibility is None:
            raise StrategyValidationError("A verified current runtime compatibility epoch is required; legacy state is archive-only")
        return self.runtime_compatibility

    def _require_compatible(self, record):
        self._require_epoch()
        if not self.compatible(record):
            raise StrategyValidationError("Learning record is incompatible with the current runtime epoch")

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> "StrategyStore":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def _audit(self, event: str, subject_id: str, principal: str, authorization_scope: str, details: dict[str, Any]) -> None:
        self._db.execute("INSERT INTO learning_audit(event,subject_id,principal,authorization_scope,details_json,created_at) VALUES(?,?,?,?,?,?)",
                         (event, subject_id, principal, authorization_scope, canonical(details), _now()))

    @staticmethod
    def _filter(principal: str | None, authorization_scope: str | None, *, prefix: str = "") -> tuple[str, list[str]]:
        parts, args = [], []
        for key, value in (("principal", principal), ("authorization_scope", authorization_scope)):
            if value is not None:
                _text(value, key, maximum=1024)
                parts.append(prefix + key + " = ?")
                args.append(value)
        return (" WHERE " + " AND ".join(parts) if parts else ""), args

    def get(self, candidate_id: str) -> dict[str, Any]:
        row = self._db.execute("SELECT * FROM strategy_candidates WHERE id=?", (candidate_id,)).fetchone()
        if row is None:
            raise KeyError(candidate_id)
        spec = json.loads(row["spec_json"])
        return {**spec, "id": row["id"], "candidate_id": row["id"], "spec_sha256": digest(spec),
                "status": row["status"], "created_at": row["created_at"], "updated_at": row["updated_at"],
                "promoted_validation_id": row["promoted_validation_id"]}

    def propose(self, action: str, *, principal: str, authorization_scope: str,
                features: Iterable[str] = (), gap_kinds: Iterable[str] = (), snapshot_id: str | None = None,
                source_document_ids: Iterable[str] = ()) -> dict[str, Any]:
        principal, authorization_scope = _owner(principal, authorization_scope)
        if action not in ACTIONS:
            raise ValueError("action is not in the bounded tool vocabulary")
        if isinstance(source_document_ids, str):
            raise ValueError("source_document_ids must be a sequence")
        documents = sorted(set(source_document_ids))
        if len(documents) > 32 or any(not isinstance(v, str) or re.fullmatch(r"[A-Za-z0-9_.:\-]{1,128}", v) is None for v in documents):
            raise ValueError("source hints must contain bounded opaque document IDs")
        if bool(documents) != bool(snapshot_id):
            raise ValueError("source-specific hints require both snapshot and document IDs")
        if snapshot_id is not None:
            _text(snapshot_id, "snapshot ID")
        self._require_epoch()
        spec = {"runtime_compatibility": self.runtime_compatibility, "policy_contract_version": 1, "action": action, "principal": principal,
                "authorization_scope": authorization_scope, "features": _labels(features, FEATURES, "feature"),
                "gap_kinds": _labels(gap_kinds, GAP_KINDS, "gap kind"), "snapshot_id": snapshot_id,
                "source_document_ids": documents}
        candidate_id = "strategy_" + digest(spec)[:32]
        with self._db:
            inserted = self._db.execute("INSERT OR IGNORE INTO strategy_candidates VALUES(?,?,?,?,?,NULL)",
                                       (candidate_id, canonical(spec), "candidate", _now(), _now())).rowcount
            if inserted:
                self._audit("candidate_proposed", candidate_id, principal, authorization_scope, {"action": action})
        return self.get(candidate_id)

    def list(self, *, principal: str | None = None, authorization_scope: str | None = None,
             status: str | None = None) -> list[dict[str, Any]]:
        if status is not None and status not in {"candidate", "promoted", "revoked"}:
            raise ValueError("unknown candidate status")
        result = [self.get(row[0]) for row in self._db.execute("SELECT id FROM strategy_candidates ORDER BY created_at,id")]
        return [item for item in result if (principal is None or item["principal"] == principal)
                and (authorization_scope is None or item["authorization_scope"] == authorization_scope)
                and (status is None or item["status"] == status)]

    def observe(self, candidate_id: str, *, trace_id: str, question_sha256: str,
                outcome_code: str, gaps: Iterable[str] = (), metrics: Mapping[str, int | float] | None = None) -> dict[str, Any]:
        candidate = self.get(candidate_id)
        self._require_compatible(candidate)
        _text(trace_id, "trace ID")
        _question_hash(question_sha256)
        if outcome_code not in OUTCOME_CODES:
            raise ValueError("unknown process outcome code")
        values = dict(metrics or {})
        if set(values) - {"documents_read", "new_cards", "waves", "elapsed_ms"} or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1_000_000_000 for v in values.values()):
            raise ValueError("invalid process metrics")
        observation = {"trace_id": trace_id, "question_sha256": question_sha256, "outcome_code": outcome_code,
                       "gaps": _labels(gaps, GAP_KINDS, "gap kind"), "metrics": values, "verified": False}
        oid = "observation_" + digest({"candidate": candidate_id, **observation})[:32]
        with self._db:
            self._db.execute("INSERT OR IGNORE INTO strategy_observations VALUES(?,?,?,?)", (oid, candidate_id, canonical(observation), _now()))
        return {"id": oid, "candidate_id": candidate_id, **observation}

    def _save_validation(self, subject: dict[str, Any], kind: str, receipt: dict[str, Any], evidence_root: Path,
                         excluded_ids: set[str] | None = None, excluded_hashes: set[str] | None = None) -> dict[str, Any]:
        self._require_compatible(subject)
        summary = validate_receipt(receipt, runtime_compatibility=self.runtime_compatibility, candidate_id=subject["id"], spec_hash=subject["spec_sha256"],
                                   evidence_root=evidence_root, excluded_collection_ids=excluded_ids,
                                   excluded_document_hashes=excluded_hashes, principal=subject["principal"], authorization_scope=subject["authorization_scope"])
        vid = "validation_" + digest({"subject": subject["id"], "receipt": summary["receipt_sha256"]})[:32]
        with self._db:
            self._db.execute("INSERT OR IGNORE INTO strategy_validations VALUES(?,?,?,?,?,?,?)",
                             (vid, subject["id"], kind, subject["principal"], subject["authorization_scope"], canonical(summary), _now()))
            self._audit("validation_recorded", subject["id"], subject["principal"], subject["authorization_scope"],
                        {"validation_id": vid, "promotion_eligible": summary["promotion_eligible"]})
        return {"id": vid, "validation_id": vid, "subject_id": subject["id"], "subject_kind": kind, **summary}

    def record_validation(self, candidate_id: str, receipt: dict[str, Any], *, evidence_root: str | Path) -> dict[str, Any]:
        return self._save_validation(self.get(candidate_id), "candidate", receipt, Path(evidence_root))

    def get_validation(self, validation_id: str) -> dict[str, Any]:
        row = self._db.execute("SELECT * FROM strategy_validations WHERE id=?", (validation_id,)).fetchone()
        if row is None:
            raise KeyError(validation_id)
        return {"id": row["id"], "validation_id": row["id"], "subject_id": row["subject_id"],
                "subject_kind": row["subject_kind"], "principal": row["principal"],
                "authorization_scope": row["authorization_scope"], "created_at": row["created_at"],
                **json.loads(row["summary_json"])}

    def promote(self, candidate_id: str, validation_id: str) -> dict[str, Any]:
        candidate, validation = self.get(candidate_id), self.get_validation(validation_id)
        self._require_compatible(candidate)
        self._require_compatible(validation)
        if candidate["status"] == "revoked" or validation["subject_id"] != candidate_id or validation["subject_kind"] != "candidate" or not validation["promotion_eligible"]:
            raise StrategyValidationError("candidate promotion requires its own eligible validation and an unrevoked candidate")
        with self._db:
            self._db.execute("UPDATE strategy_candidates SET status='promoted',updated_at=?,promoted_validation_id=? WHERE id=?", (_now(), validation_id, candidate_id))
            self._audit("candidate_promoted", candidate_id, candidate["principal"], candidate["authorization_scope"], {"validation_id": validation_id})
        return self.get(candidate_id)

    def revoke(self, candidate_id: str, *, reason: str = "manual") -> dict[str, Any]:
        if reason not in REVOCATION_REASONS:
            raise ValueError("unknown typed revocation reason")
        candidate = self.get(candidate_id)
        with self._db:
            self._db.execute("UPDATE strategy_candidates SET status='revoked',updated_at=? WHERE id=?", (_now(), candidate_id))
            self._audit("candidate_revoked", candidate_id, candidate["principal"], candidate["authorization_scope"], {"reason": reason})
        return self.get(candidate_id)

    def recommend(self, *, principal: str, authorization_scope: str, snapshot_id: str | None,
                  features: Iterable[str] = (), gap_kinds: Iterable[str] = (), limit: int = 3) -> list[dict[str, Any]]:
        _owner(principal, authorization_scope)
        current_features, gaps = set(_labels(features, FEATURES, "feature")), set(_labels(gap_kinds, GAP_KINDS, "gap kind"))
        if type(limit) is not int or not 1 <= limit <= len(ACTIONS):
            raise ValueError("invalid recommendation limit")
        result = []
        for item in self.list(principal=principal, authorization_scope=authorization_scope, status="promoted"):
            if not self.compatible(item):
                continue
            if item["snapshot_id"] is not None and item["snapshot_id"] != snapshot_id:
                continue
            if not set(item["features"]) <= current_features or (item["gap_kinds"] and not gaps.intersection(item["gap_kinds"])):
                continue
            result.append(item)
        return sorted(result, key=lambda item: item["id"])[:limit]

    def record_episode(self, *, principal: str, authorization_scope: str, snapshot_id: str | None,
                       trace_id: str, question_sha256: str, steps: list[dict[str, Any]], outcome_code: str) -> dict[str, Any]:
        _owner(principal, authorization_scope)
        _text(trace_id, "trace ID")
        _question_hash(question_sha256)
        if outcome_code not in OUTCOME_CODES:
            raise ValueError("unknown process outcome")
        trajectory = validate_trajectory(steps)
        episode = {"runtime_compatibility": self.runtime_compatibility, "principal": principal, "authorization_scope": authorization_scope, "snapshot_id": snapshot_id,
                   "trace_id": trace_id, "question_sha256": question_sha256, "steps": trajectory,
                   "outcome_code": outcome_code, "verified": False}
        eid = "episode_" + digest({"principal": principal, "authorization_scope": authorization_scope, "trace_id": trace_id, "runtime_compatibility": self.runtime_compatibility})[:32]
        previous = self._db.execute("SELECT episode_json FROM learning_episodes WHERE id=?", (eid,)).fetchone()
        if previous is not None and previous[0] != canonical(episode):
            raise ValueError("immutable episode ID already has different telemetry")
        with self._db:
            self._db.execute("INSERT OR IGNORE INTO learning_episodes VALUES(?,?,?,?,?,?)", (eid, principal, authorization_scope, snapshot_id, canonical(episode), _now()))
        return {"id": eid, "episode_id": eid, **episode}

    def record_trajectory(self, validation_id: str, *, collection_id: str, case_id: str, arm: str,
                          principal: str, authorization_scope: str) -> dict[str, Any]:
        _owner(principal, authorization_scope)
        validation = self.get_validation(validation_id)
        self._require_compatible(validation)
        if validation["principal"] != principal or validation["authorization_scope"] != authorization_scope:
            raise StrategyValidationError("validation is outside the authorized learning scope")
        if arm not in {"baseline", "candidate"}:
            raise ValueError("unknown trajectory arm")
        pair = next((p for p in validation["pairs"] if p["collection_id"] == collection_id and p["case_id"] == case_id), None)
        if pair is None:
            raise StrategyValidationError("case is not in verified development evidence")
        record = pair[arm]
        observation, _ = read_artifact(Path(validation["evidence_root"]), record["artifact"])
        if "details_artifact" in record:
            read_artifact(Path(validation["evidence_root"]), record["details_artifact"])
        if "trajectory" not in observation:
            raise StrategyValidationError("verified observation has no controller trajectory")
        if observation.get("runtime_epoch_sha256") != self._require_epoch()["epoch_sha256"]:
            raise StrategyValidationError("Verified trajectory runtime differs from the current epoch")
        steps = validate_trajectory(observation["trajectory"])
        if "episode_id" in observation:
            saved = self._db.execute("SELECT episode_json FROM learning_episodes WHERE id=?", (observation["episode_id"],)).fetchone()
            if saved is None:
                raise StrategyValidationError("receipt refers to an unknown runtime episode")
            episode = json.loads(saved[0])
            self._require_compatible(episode)
            matching_steps = len(episode["steps"]) == len(steps) and all(
                all(before.get(key) == after.get(key) for key in ("step", "action", "context", "next_context", "evidence_gain"))
                and ("usage" not in after or before.get("usage") == after["usage"])
                for before, after in zip(episode["steps"], steps))
            for before, after in zip(episode["steps"], steps):
                if "usage" in before:
                    usage, budget = before["usage"], observation["budget"]
                    expected_cost = {"elapsed_fraction": usage["elapsed_s"] / budget["deadline_s"],
                                     "token_fraction": usage["output_tokens"] / budget["max_output_tokens"] if budget["max_output_tokens"] else 0,
                                     "call_fraction": usage["model_calls"] / budget["max_model_calls"] if budget["max_model_calls"] else 0}
                    matching_steps = matching_steps and all(abs(after["cost"][k] - v) <= 1e-8 for k, v in expected_cost.items())
                else:
                    matching_steps = matching_steps and before["cost"] == after["cost"]
            if episode["principal"] != principal or episode["authorization_scope"] != authorization_scope or not matching_steps:
                raise StrategyValidationError("receipt trajectory differs from authorized runtime episode")
        returns = replay_returns(steps, complete=record["complete"], critical_failures=record["critical_failures"], critical_checks=record["critical_checks"])
        source = next(c for c in validation["collections"] if c["collection_id"] == collection_id)
        ids = []
        with self._db:
            for item in returns:
                identity = {"artifact_sha256": record["artifact"]["sha256"], "step": item["step"], "principal": principal, "authorization_scope": authorization_scope}
                rid = "replay_" + digest(identity)[:32]
                data = {**item, "runtime_compatibility": self.runtime_compatibility, "artifact_sha256": record["artifact"]["sha256"], "document_sha256": source["document_sha256"],
                        "complete": record["complete"], "critical_failures": record["critical_failures"], "critical_checks": record["critical_checks"],
                        "verification": "external_executable_oracle", "split": "development"}
                self._db.execute("INSERT OR IGNORE INTO learning_replay VALUES(?,?,?,?,?,?,?,?,?)",
                                 (rid, validation_id, collection_id, case_id, arm, principal, authorization_scope, canonical(data), _now()))
                ids.append(rid)
            self._audit("verified_trajectory_imported", validation_id, principal, authorization_scope,
                        {"collection_id": collection_id, "case_id": case_id, "arm": arm, "replay_ids": ids})
        return {"validation_id": validation_id, "replay_ids": ids, "steps": len(ids),
                "complete": record["complete"], "critical_failures": record["critical_failures"], "rewards": [r["reward"] for r in returns]}

    def replay(self, *, principal: str, authorization_scope: str) -> list[dict[str, Any]]:
        _owner(principal, authorization_scope)
        return [{"id": row["id"], "validation_id": row["validation_id"], "collection_id": row["collection_id"],
                 "case_id": row["case_id"], "arm": row["arm"], **json.loads(row["experience_json"])}
                for row in self._db.execute("SELECT * FROM learning_replay WHERE principal=? AND authorization_scope=? ORDER BY created_at,id", (principal, authorization_scope))
                if self.compatible(json.loads(row["experience_json"]))]

    def train_policy(self, *, principal: str, authorization_scope: str, parent_checkpoint_id: str | None = None,
                     ridge: float = 1.0, snapshot_id: str | None = None) -> dict[str, Any]:
        _owner(principal, authorization_scope)
        self._require_epoch()
        if parent_checkpoint_id is not None:
            parent = self.get_policy(parent_checkpoint_id)
            self._require_compatible(parent)
            if parent["principal"] != principal or parent["authorization_scope"] != authorization_scope:
                raise StrategyValidationError("parent checkpoint belongs to a different authorization scope")
        rows = self.replay(principal=principal, authorization_scope=authorization_scope)
        if len({(r["collection_id"], r["case_id"]) for r in rows}) < 4 or len({r["collection_id"] for r in rows}) < 2 or len({r["action"] for r in rows}) < 2:
            raise StrategyValidationError("policy fitting requires verified replay across two actions and two development collections")
        model = fit_policy(rows, ridge=ridge)
        training = {"runtime_compatibility": self.runtime_compatibility, "replay_ids": sorted(r["id"] for r in rows), "replay_sha256": digest(rows),
                    "validation_ids": sorted({r["validation_id"] for r in rows}),
                    "collection_ids": sorted({r["collection_id"] for r in rows}),
                    "document_sha256": sorted({h for r in rows for h in r["document_sha256"]}),
                    "reward_definition": "discounted_complete_minus_critical_error_fraction_minus_bounded_step_cost_v1",
                    "discount": 0.95, "samples": len(rows)}
        spec = {"policy_contract_version": 1, "principal": principal, "authorization_scope": authorization_scope,
                "snapshot_id": snapshot_id, "parent_checkpoint_id": parent_checkpoint_id, "model": model, "training": training}
        spec_hash = digest(spec)
        pid = "policy_" + spec_hash[:32]
        with self._db:
            self._db.execute("INSERT OR IGNORE INTO learning_policies VALUES(?,?,?,?,?,?,?,?,?,NULL,?)",
                             (pid, principal, authorization_scope, snapshot_id, parent_checkpoint_id, spec_hash,
                              canonical(model), canonical(training), "staged", _now()))
            self._audit("policy_trained", pid, principal, authorization_scope, {"samples": len(rows), "spec_sha256": spec_hash})
        return self.get_policy(pid)

    def get_policy(self, checkpoint_id: str) -> dict[str, Any]:
        row = self._db.execute("SELECT * FROM learning_policies WHERE id=?", (checkpoint_id,)).fetchone()
        if row is None:
            raise KeyError(checkpoint_id)
        model, training = json.loads(row["model_json"]), json.loads(row["training_json"])
        spec = {"policy_contract_version": 1, "principal": row["principal"], "authorization_scope": row["authorization_scope"],
                "snapshot_id": row["snapshot_id"], "parent_checkpoint_id": row["parent_id"], "model": model, "training": training}
        if digest(spec) != row["spec_sha256"]:
            raise StrategyValidationError("checkpoint content does not match its immutable specification hash")
        return {"id": row["id"], "checkpoint_id": row["id"], "spec_sha256": row["spec_sha256"], **spec,
                "status": row["status"], "created_at": row["created_at"], "promoted_validation_id": row["promoted_validation_id"],
                "action_support": {k: v["support"] for k, v in model["actions"].items()},
                "training_validation_ids": training["validation_ids"], "training_collection_ids": training["collection_ids"]}

    def policies(self, *, principal: str | None = None, authorization_scope: str | None = None) -> list[dict[str, Any]]:
        where, args = self._filter(principal, authorization_scope)
        return [self.get_policy(row[0]) for row in self._db.execute("SELECT id FROM learning_policies" + where + " ORDER BY created_at,id", args)]

    list_policies = policies

    def record_policy_validation(self, checkpoint_id: str, receipt: dict[str, Any], *, evidence_root: str | Path) -> dict[str, Any]:
        policy = self.get_policy(checkpoint_id)
        training = policy["training"]
        return self._save_validation(policy, "policy", receipt, Path(evidence_root),
                                     set(training["collection_ids"]), set(training["document_sha256"]))

    def _activate_policy(self, policy: dict[str, Any], validation_id: str, event: str) -> None:
        # Caller holds BEGIN IMMEDIATE across checking and activation.
        self._db.execute("UPDATE learning_policies SET status='retired' WHERE principal=? AND authorization_scope=? AND status='active'", (policy["principal"], policy["authorization_scope"]))
        self._db.execute("UPDATE learning_policies SET status='active',promoted_validation_id=? WHERE id=?", (validation_id, policy["id"]))
        self._audit(event, policy["id"], policy["principal"], policy["authorization_scope"], {"validation_id": validation_id})

    def promote_policy(self, checkpoint_id: str, validation_id: str) -> dict[str, Any]:
        with self._db:
            # Acquire the writer lock BEFORE reading active policy state. Two
            # valid challengers of A cannot both replace it concurrently.
            self._db.execute("BEGIN IMMEDIATE")
            policy, validation = self.get_policy(checkpoint_id), self.get_validation(validation_id)
            self._require_compatible(policy)
            self._require_compatible(validation)
            if policy["status"] == "revoked" or validation["subject_id"] != checkpoint_id or validation["subject_kind"] != "policy" or not validation["promotion_eligible"]:
                raise StrategyValidationError("policy promotion requires its own independent eligible validation")
            current = self._db.execute("SELECT id FROM learning_policies WHERE principal=? AND authorization_scope=? AND status='active'",
                                       (policy["principal"], policy["authorization_scope"])).fetchone()
            if current is not None and not self.compatible(self.get_policy(current["id"])):
                current = None  # An old epoch is inactive for this runtime.
            if current is not None and current["id"] == checkpoint_id and policy["promoted_validation_id"] == validation_id:
                return policy  # Idempotent resume of this committed activation.
            expected_id = validation.get("baseline_checkpoint_id")
            if (current["id"] if current is not None else None) != expected_id:
                raise StrategyValidationError("active policy differs from the validated comparison baseline")
            if current is not None:
                active = self.get_policy(current["id"])
                if active["spec_sha256"] != validation.get("baseline_spec_sha256"):
                    raise StrategyValidationError("active policy specification differs from the validated baseline")
            self._activate_policy(policy, validation_id, "policy_promoted")
        return self.get_policy(checkpoint_id)

    def rollback_policy(self, checkpoint_id: str, *, principal: str | None = None, authorization_scope: str | None = None) -> dict[str, Any]:
        # Restoration is an explicit operation, distinct from claiming a new
        # candidate improved on the currently deployed policy.
        with self._db:
            self._db.execute("BEGIN IMMEDIATE")
            policy = self.get_policy(checkpoint_id)
            self._require_compatible(policy)
            if (principal is not None and principal != policy["principal"]) or (authorization_scope is not None and authorization_scope != policy["authorization_scope"]):
                raise StrategyValidationError("rollback target is outside the authorized scope")
            if policy["status"] not in {"retired", "active"} or not policy["promoted_validation_id"]:
                raise StrategyValidationError("rollback requires a previously validated, unrevoked checkpoint")
            validation = self.get_validation(policy["promoted_validation_id"])
            self._require_compatible(validation)
            if validation["subject_id"] != checkpoint_id or validation["subject_kind"] != "policy" or not validation["promotion_eligible"]:
                raise StrategyValidationError("rollback checkpoint lacks eligible validation")
            self._activate_policy(policy, policy["promoted_validation_id"], "policy_rollback")
        return self.get_policy(checkpoint_id)

    def revoke_policy(self, checkpoint_id: str, *, reason: str = "manual") -> dict[str, Any]:
        if reason not in REVOCATION_REASONS:
            raise ValueError("unknown typed revocation reason")
        policy = self.get_policy(checkpoint_id)
        with self._db:
            self._db.execute("UPDATE learning_policies SET status='revoked' WHERE id=?", (checkpoint_id,))
            self._audit("policy_revoked", checkpoint_id, policy["principal"], policy["authorization_scope"], {"reason": reason})
        return self.get_policy(checkpoint_id)

    def rank_actions(self, *, principal: str, authorization_scope: str, context: list[float],
                     snapshot_id: str | None = None, allowed_actions: Iterable[str] = ACTIONS,
                     exploration: float = 0.0) -> list[dict[str, Any]]:
        _owner(principal, authorization_scope)
        validate_context(context)
        rows = self._db.execute("SELECT id FROM learning_policies WHERE principal=? AND authorization_scope=? AND status='active'", (principal, authorization_scope)).fetchall()
        if not rows:
            return []
        policy = self.get_policy(rows[0][0])
        if not self.compatible(policy):
            return []
        if policy["snapshot_id"] is not None and policy["snapshot_id"] != snapshot_id:
            return []
        return [{**ranked, "checkpoint_id": policy["id"], "spec_sha256": policy["spec_sha256"]}
                for ranked in rank_policy(policy["model"], context, allowed_actions=allowed_actions, exploration=exploration)]

    def status(self, *, principal: str | None = None, authorization_scope: str | None = None) -> dict[str, Any]:
        candidates = self.list(principal=principal, authorization_scope=authorization_scope)
        where, args = self._filter(principal, authorization_scope)
        counts = {}
        for table, label in (("strategy_validations", "verified_comparisons"), ("learning_episodes", "unverified_episodes"), ("learning_replay", "verified_replay_steps")):
            counts[label] = self._db.execute("SELECT COUNT(*) FROM " + table + where, args).fetchone()[0]
        policies = self.policies(principal=principal, authorization_scope=authorization_scope)
        candidate_ids = {c["id"] for c in candidates}
        observations = sum(row[0] in candidate_ids for row in self._db.execute("SELECT candidate_id FROM strategy_observations"))
        return {"path": str(self.path), "candidates": len(candidates),
                "candidate_statuses": {s: sum(c["status"] == s for c in candidates) for s in ("candidate", "promoted", "revoked")},
                "unverified_observations": observations, **counts, "policy_checkpoints": len(policies),
                "active_policy_ids": [p["id"] for p in policies if p["status"] == "active" and self.compatible(p)],
                "archived_or_incompatible_policy_ids": [p["id"] for p in policies if not self.compatible(p)],
                "runtime_compatibility": self.runtime_compatibility,
                "compatible_replay_steps": len(self.replay(principal=principal, authorization_scope=authorization_scope)) if principal is not None and authorization_scope is not None else None,
                "learned_parameters": sum(len(a["weights"]) for p in policies for a in p["model"]["actions"].values()),
                "context_dimension": len(CONTEXT_NAMES)}

    def export(self, path: str | Path | None = None, *, principal: str | None = None,
               authorization_scope: str | None = None) -> dict[str, Any]:
        where, args = self._filter(principal, authorization_scope)
        candidates = self.list(principal=principal, authorization_scope=authorization_scope)
        ids = {c["id"] for c in candidates}
        result = {"schema_version": 1, "exported_at": _now(), "runtime_compatibility": self.runtime_compatibility, "candidates": candidates,
                  "policies": self.policies(principal=principal, authorization_scope=authorization_scope),
                  "validations": [self.get_validation(r[0]) for r in self._db.execute("SELECT id FROM strategy_validations" + where, args)],
                  "observations": [json.loads(r["observation_json"]) | {"id": r["id"], "candidate_id": r["candidate_id"]}
                                   for r in self._db.execute("SELECT * FROM strategy_observations") if r["candidate_id"] in ids],
                  "episodes": [json.loads(r["episode_json"]) | {"id": r["id"]} for r in self._db.execute("SELECT * FROM learning_episodes" + where, args)],
                  "replay": [json.loads(r["experience_json"]) | {key: r[key] for key in ("id", "validation_id", "collection_id", "case_id", "arm", "principal", "authorization_scope", "created_at")}
                             for r in self._db.execute("SELECT * FROM learning_replay" + where + " ORDER BY created_at,id", args)],
                  "audit": [dict(r) for r in self._db.execute("SELECT * FROM learning_audit" + where + " ORDER BY sequence", args)]}
        if path is not None:
            Path(path).write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return result

    def reset(self, *, principal: str | None = None, authorization_scope: str | None = None) -> dict[str, Any]:
        # Explicit administrative operation. Files and evaluation artifacts remain.
        before = self.status(principal=principal, authorization_scope=authorization_scope)
        ids = [c["id"] for c in self.list(principal=principal, authorization_scope=authorization_scope)]
        where, args = self._filter(principal, authorization_scope)
        with self._db:
            for candidate_id in ids:
                self._db.execute("DELETE FROM strategy_observations WHERE candidate_id=?", (candidate_id,))
                self._db.execute("DELETE FROM strategy_candidates WHERE id=?", (candidate_id,))
            for table in ("strategy_validations", "learning_episodes", "learning_replay", "learning_policies", "learning_audit"):
                self._db.execute("DELETE FROM " + table + where, args)
        return {"reset": True, "previous": before, "current": self.status(principal=principal, authorization_scope=authorization_scope)}
