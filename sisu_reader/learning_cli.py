"""Trusted local operator interface for scoped, auditable policy learning."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


def add_learning_parser(subparsers):
    parser = subparsers.add_parser("learning", help="Inspect, train and validate the persistent research controller")
    actions = parser.add_subparsers(dest="learning_action", required=True)
    for name, help_text in (
        ("status", "Show verified replay and active checkpoints"),
        ("list", "List this authorization scope's strategies and checkpoints"),
        ("reset", "Explicitly erase this authorization scope's learning state"),
    ):
        actions.add_parser(name, help=help_text)
    propose = actions.add_parser("propose", help="Register a typed action; this does not activate it")
    propose.add_argument("action", choices=("balance_documents", "broaden_search", "focused_gap_search"))
    propose.add_argument("--feature", action="append", default=[])
    propose.add_argument("--gap", action="append", default=[])
    export = actions.add_parser("export", help="Export scoped checkpoints and provenance")
    export.add_argument("path", type=Path)
    for name in ("validate", "policy-validate"):
        validate = actions.add_parser(name, help="Recompute a frozen comparison from its evidence artifacts")
        validate.add_argument("id")
        validate.add_argument("receipt", type=Path)
        validate.add_argument("--evidence-root", type=Path, required=True)
    for name in ("promote", "policy-promote"):
        promote = actions.add_parser(name, help="Activate only after an eligible validation")
        promote.add_argument("id")
        promote.add_argument("validation_id")
    for name in ("revoke", "policy-revoke", "rollback"):
        action = actions.add_parser(name)
        action.add_argument("id")
    replay = actions.add_parser("import-replay", help="Import verified state/action trajectories from a comparison")
    replay.add_argument("validation_id")
    replay.add_argument("--collection", required=True)
    replay.add_argument("--case", required=True)
    replay.add_argument("--arm", choices=("baseline", "candidate"), required=True)
    train = actions.add_parser("train", help="Fit numerical policy parameters from accumulated verified replay")
    train.add_argument("--parent", help="Previous checkpoint in this learning lineage")
    train.add_argument("--ridge", type=float, default=1.0)


def _owned(item, principal, scope):
    if item.get("principal") != principal or item.get("authorization_scope") != scope:
        raise PermissionError("The learning record is unavailable in the current authorization scope")
    return item


def _summary(item):
    if isinstance(item, dict) and "model" in item and "checkpoint_id" in item:
        return {k: item[k] for k in ("checkpoint_id", "spec_sha256", "status", "parent_checkpoint_id",
                                    "action_support", "training_collection_ids", "training_validation_ids")}
    return item


def command_learning(config, args):
    from .access import AccessManager
    from .strategy_learning import StrategyStore
    from .runtime_compatibility import build_runtime_compatibility
    try:
        access = AccessManager(config)
        principal = config.principal_id
        auth = access.principal_snapshot(principal)
        if not auth.known or not auth.active or not access.has_permission(auth, "question.ask"):
            raise PermissionError("Learning is unavailable for this local user")
        scope = access.authorization_scope_hash(auth)
        owner = dict(principal=principal, authorization_scope=scope)
        epoch = None
        if args.learning_action not in {"reset", "revoke", "policy-revoke", "export", "list"}:
            try:
                epoch = build_runtime_compatibility(config, refresh_models=True)
            except Exception:
                if args.learning_action != "status":
                    raise ValueError("Current model/runtime identity is unavailable; learning mutations require verified compatibility")
        with StrategyStore(config.workspace_dir / "strategy_learning.sqlite3", runtime_compatibility=epoch) as memory:
            action = args.learning_action
            if action == "status":
                result = memory.status(**owner)
            elif action == "list":
                result = {"strategies": memory.list(**owner), "policies": [_summary(p) for p in memory.policies(**owner)]}
            elif action == "propose":
                result = memory.propose(args.action, **owner, features=args.feature, gap_kinds=args.gap)
            elif action == "export":
                # A scoped export cannot reveal another local user's experience.
                memory.export(args.path, **owner)
                result = {"exported": str(args.path.resolve())}
            elif action == "reset":
                result = memory.reset(**owner)
            elif action == "train":
                result = memory.train_policy(**owner, parent_checkpoint_id=args.parent, ridge=args.ridge)
            elif action == "import-replay":
                _owned(memory.get_validation(args.validation_id), principal, scope)
                result = memory.record_trajectory(args.validation_id, collection_id=args.collection,
                                                  case_id=args.case, arm=args.arm, **owner)
            else:
                is_policy = action.startswith("policy-") or action == "rollback"
                item = memory.get_policy(args.id) if is_policy else memory.get(args.id)
                _owned(item, principal, scope)
                if action in ("validate", "policy-validate"):
                    # The store's artifact parser rejects duplicate keys, escaping paths,
                    # tampered hashes, non-finite values and executable oracle content.
                    from .strategy_validation import read_artifact
                    receipt_path = args.receipt.resolve(strict=True)
                    receipt, _ = read_artifact(receipt_path.parent, {
                        "path": receipt_path.name,
                        "sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
                    })
                    method = memory.record_policy_validation if is_policy else memory.record_validation
                    result = method(args.id, receipt, evidence_root=args.evidence_root)
                elif action in ("promote", "policy-promote"):
                    _owned(memory.get_validation(args.validation_id), principal, scope)
                    method = memory.promote_policy if is_policy else memory.promote
                    result = method(args.id, args.validation_id)
                elif action == "rollback":
                    result = memory.rollback_policy(args.id, **owner)
                elif action in ("revoke", "policy-revoke"):
                    result = (memory.revoke_policy if is_policy else memory.revoke)(args.id)
                else:
                    raise ValueError("Unknown learning operation")
            print(json.dumps(_summary(result), ensure_ascii=False, indent=2, default=str, allow_nan=False))
            return 0
    except Exception as exc:
        print(f"error: {type(exc).__name__}: {' '.join(str(exc).split())}", file=sys.stderr)
        return 1
