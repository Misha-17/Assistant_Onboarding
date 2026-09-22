"""Stable, explicit compatibility epochs for learned research controllers.

No document text, corpus identity, authorization identity, path or timestamp
enters the descriptor. Model lookup reads local Ollama metadata only.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import json
import math
from pathlib import Path
import re
import threading
import time

from .learning_policy import ACTIONS, CONTEXT_NAMES
from .efficiency_gate import PROMOTION_CONTRACT

ROLES = ("screen", "reader", "synthesis", "review")
REWARD_CONTRACT = "discounted_complete_minus_critical_error_fraction_minus_bounded_step_cost_v1"
ORACLE_CONTRACT = "source_join_decisions_v4_grounded_complete_material_error"
CONFIG_FIELDS = (
    "context_tokens", "screen_output_tokens", "reader_output_tokens", "synthesis_output_tokens",
    "review_output_tokens", "reasoning", "request_timeout_s", "total_deadline_s",
    "synthesis_reserve_s", "finalize_reserve_s", "minimum_screen_window_s",
    "minimum_reader_window_s", "screen_batch_token_budget", "reader_context_reserve_tokens",
    "source_block_hard_tokens", "read_wave_size", "manifest_full_scan_threshold",
    "discovery_page_size", "session_turns", "contradiction_probe_documents", "reference_closure",
    "adaptive_research", "claim_review", "adaptive_max_additional_waves", "adaptive_packets_per_wave",
    "evidence_review_timeout_s", "claim_review_reserve_s",
)
_CACHE = {}
_CACHE_LOCK = threading.Lock()
_HEX = re.compile(r"[0-9a-f]{64}")
_ANSWER_EPOCH = ContextVar("sisu_answer_runtime_epoch", default=None)
_MODULE_DIRECTORY = Path(__file__).resolve().parent
def _runtime_sources(base):
    """Bind executable helpers as well as Python modules; exclude UI assets."""
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(base).iterdir())
            if path.is_file() and path.suffix in {".py", ".ps1"}}


_IMPORTED_RUNTIME_SOURCES = _runtime_sources(_MODULE_DIRECTORY)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _hash(value):
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _sha(value):
    return isinstance(value, str) and _HEX.fullmatch(value) is not None


def validate_runtime_compatibility(value):
    if not isinstance(value, dict) or set(value) != {"schema_version", "epoch_sha256", "descriptor"}:
        raise ValueError("Missing or malformed runtime compatibility epoch")
    descriptor = value["descriptor"]
    required = {"runtime_source_sha256", "action_vocabulary", "context_schema", "prompt_versions",
                "model_roles", "runtime_config", "budget_regime", "reward_contract", "oracle_contract", "promotion_contract"}
    if type(value["schema_version"]) is not int or value["schema_version"] != 1 or not isinstance(descriptor, dict) or set(descriptor) != required:
        raise ValueError("Unsupported runtime compatibility descriptor")
    if not _sha(value["epoch_sha256"]) or _hash(descriptor) != value["epoch_sha256"]:
        raise ValueError("Runtime compatibility descriptor hash mismatch")
    files = descriptor["runtime_source_sha256"]
    if not isinstance(files, dict) or not files or any(not re.fullmatch(r"[A-Za-z0-9_]+\.(?:py|ps1)", name) or not _sha(sha) for name, sha in files.items()):
        raise ValueError("Runtime sources require stable module names and SHA-256 digests")
    if descriptor["action_vocabulary"] != list(ACTIONS) or descriptor["context_schema"] != list(CONTEXT_NAMES):
        raise ValueError("Runtime action or context contract differs")
    if not isinstance(descriptor["prompt_versions"], dict) or set(descriptor["prompt_versions"]) != {"screen", "reader", "synthesis", "review", "repair"}:
        raise ValueError("Incomplete prompt-version contract")
    if any(not isinstance(value, str) or not value or len(value) > 128 for value in descriptor["prompt_versions"].values()):
        raise ValueError("Invalid prompt version")
    roles = descriptor["model_roles"]
    if not isinstance(roles, dict) or set(roles) != set(ROLES):
        raise ValueError("Incomplete model-role contract")
    for row in roles.values():
        if not isinstance(row, dict) or set(row) != {"name", "sha256"} or not isinstance(row["name"], str) or not row["name"] or len(row["name"]) > 256 or not _sha(row["sha256"]):
            raise ValueError("Each model role requires an exact local model digest")
    config = descriptor["runtime_config"]
    if not isinstance(config, dict) or set(config) != set(CONFIG_FIELDS):
        raise ValueError("Incomplete runtime configuration contract")
    from .config import Config
    for name, scalar in config.items():
        expected_type = type(Config.__dataclass_fields__[name].default)
        good_type = type(scalar) in (int, float) if expected_type is float else type(scalar) is expected_type
        if not good_type or (type(scalar) in (int, float) and (not math.isfinite(scalar) or scalar < 0)):
            raise ValueError("Invalid runtime configuration scalar")
        if name == "reasoning" and scalar not in {"off", "low", "medium", "high"}:
            raise ValueError("Invalid reasoning contract")
    budget = descriptor["budget_regime"]
    if not isinstance(budget, dict) or set(budget) != {"kind", "max_model_calls", "max_output_tokens", "reservation_contract"}:
        raise ValueError("Incomplete execution budget regime")
    if budget["kind"] not in {"deadline_only", "bounded_calls_tokens"}:
        raise ValueError("Unknown execution budget regime")
    for key in ("max_model_calls", "max_output_tokens"):
        if budget["kind"] == "deadline_only":
            if budget[key] is not None:
                raise ValueError("Deadline-only runtime cannot claim enforced total budgets")
        elif type(budget[key]) is not int or budget[key] <= 0:
            raise ValueError("Bounded runtime requires positive enforced budgets")
    if budget["reservation_contract"] not in {"none", "synthesis_and_final_repair_v1", "synthesis_and_source_audit_v10"}:
        raise ValueError("Unknown finalization reservation contract")
    if not isinstance(descriptor["reward_contract"], str) or not descriptor["reward_contract"] or not isinstance(descriptor["oracle_contract"], str) or not descriptor["oracle_contract"]:
        raise ValueError("Missing reward or oracle contract")
    if descriptor["promotion_contract"] != PROMOTION_CONTRACT:
        raise ValueError("Unsupported promotion contract; legacy epochs are archive-only")
    return json.loads(_canonical(value))


def resolve_role_digests(config, *, refresh=False, cache_seconds=30.0, fetch=None):
    """Read /api/tags once per cache interval; never generate or download.

    Explicit refresh is used at evaluation boundaries. Failed refreshes never
    return expired data. Cache contents do not enter the compatibility hash.
    """
    from .ollama import OllamaClient, _validated_loopback_url
    endpoint = _validated_loopback_url(config.ollama_url)
    names = {role: getattr(config, "effective_" + role + "_model") for role in ROLES}
    key = (endpoint, tuple(sorted(names.items())))
    with _CACHE_LOCK:
        saved = _CACHE.get(key)
        if not refresh and saved and time.monotonic() - saved[0] < cache_seconds:
            return json.loads(_canonical(saved[1]))
        _CACHE.pop(key, None)  # A failed refresh must not resurrect an old mapping.
        if fetch is None:
            fetch = lambda: OllamaClient(config)._request_json("GET", "/api/tags", timeout_s=3.0)
        response = fetch()
        available = {}
        if not isinstance(response, dict) or not isinstance(response.get("models"), list):
            raise ValueError("Model metadata omitted installed-model digests")
        for item in response["models"]:
            name = item.get("name") or item.get("model")
            sha = str(item.get("digest", "")).removeprefix("sha256:")
            if isinstance(name, str) and _sha(sha):
                canonical = name[:-7] if name.endswith(":latest") else name
                if canonical in available and available[canonical] != sha:
                    raise ValueError("Conflicting local model digests")
                available[canonical] = sha
        result = {}
        for role, name in names.items():
            canonical = name[:-7] if name.endswith(":latest") else name
            if canonical not in available:
                raise ValueError("Exact digest unavailable for configured role model")
            result[role] = {"name": canonical, "sha256": available[canonical]}
        _CACHE[key] = (time.monotonic(), result)
        return json.loads(_canonical(result))


def build_runtime_compatibility(config, *, model_roles=None, budget_regime=None,
                                source_root=None, refresh_models=False,
                                oracle_contract=ORACLE_CONTRACT, reward_contract=REWARD_CONTRACT):
    from .grounded_answer import PLAN_PROMPT_VERSION, DRAFT_PROMPT_VERSION, AUDIT_PROMPT_VERSION
    from .answer_budget import budget_descriptor
    base = Path(source_root) if source_root is not None else Path(__file__).resolve().parent
    files = _runtime_sources(base)
    if source_root is None and files != _IMPORTED_RUNTIME_SOURCES:
        raise ValueError("Runtime files changed after import; restart before using learned policies")
    resolved_models = model_roles if model_roles is not None else resolve_role_digests(config, refresh=refresh_models)
    for role in ROLES:
        expected = getattr(config, "effective_" + role + "_model")
        expected = expected[:-7] if expected.endswith(":latest") else expected
        if resolved_models[role]["name"] != expected:
            raise ValueError("Supplied model digest belongs to another configured role model")
    descriptor = {"runtime_source_sha256": files, "action_vocabulary": list(ACTIONS),
        "context_schema": list(CONTEXT_NAMES),
        "prompt_versions": {"screen": PLAN_PROMPT_VERSION, "reader": "unused-direct-original-passages-v10",
                            "synthesis": DRAFT_PROMPT_VERSION, "review": AUDIT_PROMPT_VERSION,
                            "repair": AUDIT_PROMPT_VERSION},
        "model_roles": resolved_models,
        "runtime_config": {name: getattr(config, name) for name in CONFIG_FIELDS},
        "budget_regime": budget_regime or budget_descriptor(config),
        "reward_contract": reward_contract, "oracle_contract": oracle_contract, "promotion_contract": PROMOTION_CONTRACT}
    return validate_runtime_compatibility({"schema_version": 1, "epoch_sha256": _hash(descriptor), "descriptor": descriptor})


def epoch_of(record):
    value = record.get("runtime_compatibility")
    if value is None and isinstance(record.get("training"), dict):
        value = record["training"].get("runtime_compatibility")
    return value


def compatible(record, current):
    try:
        return validate_runtime_compatibility(epoch_of(record)) == validate_runtime_compatibility(current)
    except (TypeError, ValueError):
        return False


@contextmanager
def runtime_epoch_scope(config):
    """Pin a fresh metadata snapshot per answer, with safe default fallback.

    Answer-scoped caching avoids repeated metadata calls inside controller waves.
    Thread/task-local storage prevents overlapping answers sharing epochs.
    """
    try:
        epoch = build_runtime_compatibility(config, refresh_models=True) if config.strategy_learning else None
    except Exception:
        epoch = None  # No validated model identity: disable learned decisions only.
    token = _ANSWER_EPOCH.set(epoch)
    try:
        yield epoch
    finally:
        _ANSWER_EPOCH.reset(token)


def current_runtime_compatibility():
    value = _ANSWER_EPOCH.get()
    return json.loads(_canonical(value)) if value is not None else None
