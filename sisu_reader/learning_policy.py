"""Small auditable contextual ridge policy trained from verified replay.

This learns controller parameters, never language-model weights or document facts.
No optional numerical package or remotely supplied executable is required.
"""
from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

ACTIONS = (
    "broaden_search", "balance_documents", "focused_gap_search",
    "verify_scope", "check_recency", "resolve_references",
)
FEATURES = (
    "mode:targeted", "mode:exhaustive", "corpus:small", "corpus:large",
    "question:comparison", "question:temporal", "question:procedure",
    "question:exact", "question:general",
)
CONTEXT_NAMES = (
    "bias", "remaining_time_fraction", "read_document_fraction",
    "remaining_catalogue_fraction", "unique_block_fraction", "unique_card_fraction",
    "missing_obligation_fraction", "scope_issue_fraction", "pending_packet_fraction",
    "wave_fraction", "question_comparison", "question_temporal",
    "question_procedure", "question_exact", "corpus_large", "mode_exhaustive",
)
CONTEXT_DIMENSION = len(CONTEXT_NAMES)


def validate_context(context: Sequence[float]) -> list[float]:
    if not isinstance(context, (list, tuple)) or len(context) != CONTEXT_DIMENSION:
        raise ValueError("context must contain exactly 16 numeric features")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
           or not 0 <= v <= 1 for v in context):
        raise ValueError("context features must be finite numbers in [0, 1]")
    if context[0] != 1:
        raise ValueError("context bias must equal 1")
    return [float(v) for v in context]


def encode_context(*, features: Iterable[str] = (), stats: Mapping[str, float] | None = None) -> list[float]:
    labels = set(features)
    if labels - set(FEATURES):
        raise ValueError("unknown typed question features")
    values = dict(stats or {})
    if set(values) - set(CONTEXT_NAMES[1:10]):
        raise ValueError("unknown controller state features")
    context = [1.0] + [values.get(k, 0.0) for k in CONTEXT_NAMES[1:10]]
    context += [float(k in labels) for k in (
        "question:comparison", "question:temporal", "question:procedure",
        "question:exact", "corpus:large", "mode:exhaustive")]
    return validate_context(context)


def _inverse(matrix: list[list[float]]) -> list[list[float]]:
    """Gauss-Jordan with partial pivoting, on a ridge-positive matrix."""
    n = len(matrix)
    work = [row[:] + [float(i == j) for j in range(n)] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda i: abs(work[i][col]))
        if abs(work[pivot][col]) < 1e-12:
            raise ValueError("singular policy precision matrix")
        work[col], work[pivot] = work[pivot], work[col]
        divisor = work[col][col]
        work[col] = [v / divisor for v in work[col]]
        for row in range(n):
            if row == col:
                continue
            multiplier = work[row][col]
            work[row] = [a - multiplier * b for a, b in zip(work[row], work[col])]
    return [row[n:] for row in work]


def fit_policy(experiences: Iterable[Mapping[str, Any]], *, ridge: float = 1.0) -> dict[str, Any]:
    if isinstance(ridge, bool) or not isinstance(ridge, (int, float)) or not math.isfinite(ridge) or not 0.001 <= ridge <= 100:
        raise ValueError("ridge must be finite and between 0.001 and 100")
    samples: dict[str, list[tuple[list[float], float]]] = {}
    for item in experiences:
        action = item.get("action")
        reward = item.get("reward")
        if action not in ACTIONS:
            raise ValueError("unknown replay action")
        if isinstance(reward, bool) or not isinstance(reward, (float, int)) or not math.isfinite(reward) or not -2 <= reward <= 1:
            raise ValueError("invalid verified replay reward")
        samples.setdefault(action, []).append((validate_context(item["context"]), float(reward)))
    if not samples:
        raise ValueError("no verified replay available")
    models = {}
    for action, rows in sorted(samples.items()):
        n = CONTEXT_DIMENSION
        precision = [[float(ridge) if i == j else 0.0 for j in range(n)] for i in range(n)]
        target = [0.0] * n
        for x, y in rows:
            for i in range(n):
                target[i] += x[i] * y
                for j in range(n):
                    precision[i][j] += x[i] * x[j]
        covariance = _inverse(precision)
        weights = [sum(covariance[i][j] * target[j] for j in range(n)) for i in range(n)]
        residual = sum((y - sum(a * b for a, b in zip(weights, x))) ** 2 for x, y in rows)
        models[action] = {"weights": weights, "inverse_precision": covariance,
                          "support": len(rows), "residual_mean_square": residual / len(rows)}
    return {"schema_version": 1, "model_kind": "contextual_ridge_v1", "context_names": list(CONTEXT_NAMES),
            "ridge": float(ridge), "samples": sum(len(x) for x in samples.values()), "actions": models}


def rank_policy(model: Mapping[str, Any], context: Sequence[float], *,
                allowed_actions: Iterable[str] = ACTIONS, exploration: float = 0.0) -> list[dict[str, Any]]:
    x = validate_context(context)
    if model.get("model_kind") != "contextual_ridge_v1" or model.get("context_names") != list(CONTEXT_NAMES):
        raise ValueError("incompatible learned policy checkpoint")
    if isinstance(exploration, bool) or not isinstance(exploration, (int, float)) or not math.isfinite(exploration) or not 0 <= exploration <= 0.2:
        raise ValueError("exploration coefficient must be between 0 and 0.2")
    allowed = set(allowed_actions)
    if allowed - set(ACTIONS):
        raise ValueError("unknown allowed action")
    result = []
    for action, params in model["actions"].items():
        if action not in allowed or params["support"] < 1:
            continue
        prediction = sum(a * b for a, b in zip(params["weights"], x))
        covariance = params["inverse_precision"]
        variance = sum(x[i] * covariance[i][j] * x[j] for i in range(len(x)) for j in range(len(x)))
        uncertainty = math.sqrt(max(0.0, variance))
        result.append({"action": action, "score": prediction + exploration * uncertainty,
                       "predicted_reward": prediction, "uncertainty": uncertainty,
                       "support": params["support"], "exploration": float(exploration)})
    return sorted(result, key=lambda row: (-row["score"], row["action"]))


def validate_trajectory(steps: Any) -> list[dict[str, Any]]:
    if not isinstance(steps, list) or not 1 <= len(steps) <= 32:
        raise ValueError("trajectory must have 1 to 32 bounded steps")
    result = []
    for ordinal, step in enumerate(steps):
        required = {"step", "action", "context", "next_context", "cost"}
        if not isinstance(step, dict) or not required <= set(step) or set(step) - required - {"evidence_gain", "usage"}:
            raise ValueError("invalid trajectory step fields")
        if type(step["step"]) is not int or step["step"] != ordinal or step["action"] not in ACTIONS:
            raise ValueError("invalid trajectory action or order")
        cost = step["cost"]
        if not isinstance(cost, dict) or set(cost) != {"elapsed_fraction", "token_fraction", "call_fraction"}:
            raise ValueError("invalid trajectory cost fields")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1 for v in cost.values()):
            raise ValueError("trajectory costs must be finite fractions")
        result.append({"step": ordinal, "action": step["action"],
                       "context": validate_context(step["context"]),
                       "next_context": validate_context(step["next_context"]),
                       "cost": {k: float(v) for k, v in cost.items()}})
        if "evidence_gain" in step:
            gain = step["evidence_gain"]
            if not isinstance(gain, dict) or set(gain) != {"new_blocks", "new_source_bound_cards", "new_documents"} or any(type(v) is not int or not 0 <= v <= 1_000_000 for v in gain.values()):
                raise ValueError("invalid process evidence-gain counters")
            result[-1]["evidence_gain"] = dict(gain)
        if "usage" in step:
            usage = step["usage"]
            if not isinstance(usage, dict) or set(usage) != {"elapsed_s", "output_tokens", "model_calls"}:
                raise ValueError("invalid step usage fields")
            if type(usage["elapsed_s"]) not in (int, float) or not math.isfinite(usage["elapsed_s"]) or usage["elapsed_s"] < 0:
                raise ValueError("invalid step elapsed time")
            if any(type(usage[k]) is not int or not 0 <= usage[k] <= 1_000_000_000 for k in ("output_tokens", "model_calls")):
                raise ValueError("invalid step operation usage")
            result[-1]["usage"] = dict(usage)
    for key in ("elapsed_fraction", "token_fraction", "call_fraction"):
        if sum(row["cost"][key] for row in result) > 1.000001:
            raise ValueError("trajectory total cost exceeds episode budget")
    for left, right in zip(result, result[1:]):
        if left["next_context"] != right["context"]:
            raise ValueError("trajectory state transitions do not join")
    return result


def replay_returns(steps: Any, *, complete: bool, critical_failures: int, critical_checks: int,
                   discount: float = 0.95) -> list[dict[str, Any]]:
    trajectory = validate_trajectory(steps)
    if type(complete) is not bool or type(critical_failures) is not int or type(critical_checks) is not int or not 0 <= critical_failures <= critical_checks or critical_checks < 1:
        raise ValueError("invalid externally verified terminal outcome")
    if not 0 < discount <= 1:
        raise ValueError("invalid discount")
    terminal = float(complete) - critical_failures / critical_checks
    future = terminal
    result = []
    for index, step in reversed(list(enumerate(trajectory))):
        cost = step["cost"]
        penalty = 0.05 * cost["elapsed_fraction"] + 0.03 * cost["token_fraction"] + 0.02 * cost["call_fraction"]
        future = (future if index == len(trajectory) - 1 else discount * future) - penalty
        result.append({**step, "reward": future, "terminal_grounded_reward": terminal})
    return list(reversed(result))
