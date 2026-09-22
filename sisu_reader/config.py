from __future__ import annotations

import os
import re
from dataclasses import dataclass, replace
from pathlib import Path


def _bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    return default if raw is None else raw.strip().casefold() not in {"0", "false", "no", "off"}


def _int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.environ.get(name)
    try:
        value = default if raw is None else int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _float(name: str, default: float, minimum: float, maximum: float) -> float:
    raw = os.environ.get(name)
    try:
        value = default if raw is None else float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


@dataclass(frozen=True, slots=True)
class Config:
    """Runtime limits for the local document-reading pipeline."""

    project_dir: Path
    workspace_dir: Path
    ollama_url: str = "http://127.0.0.1:11434"
    model: str = "gemma4:12b"
    screen_model: str = ""
    reader_model: str = ""
    synthesis_model: str = ""
    review_model: str = ""
    context_tokens: int = 32_768
    max_model_calls: int = 30
    max_output_tokens: int = 32_768
    screen_output_tokens: int = 1_280
    reader_output_tokens: int = 1_024
    synthesis_output_tokens: int = 3_000
    review_output_tokens: int = 2_000
    reasoning: str = "off"
    request_timeout_s: float = 88.0
    total_deadline_s: float = 95.0
    synthesis_reserve_s: float = 18.0
    finalize_reserve_s: float = 3.0
    minimum_screen_window_s: float = 10.0
    minimum_reader_window_s: float = 15.0
    inference_lock_timeout_s: float = 120.0
    keep_alive: str = "30m"
    screen_batch_token_budget: int = 8_000
    reader_context_reserve_tokens: int = 2_500
    source_block_hard_tokens: int = 4_096
    read_wave_size: int = 2
    manifest_full_scan_threshold: int = 200
    discovery_page_size: int = 48
    session_turns: int = 6
    trace_mode: str = "metrics"
    trace_reasoning: bool = False
    trace_retention_days: int = 30
    trace_max_total_mb: int = 512
    principal_id: str = "local-admin"
    contradiction_probe_documents: int = 1
    reference_closure: bool = True
    adaptive_research: bool = True
    claim_review: bool = True
    strategy_learning: bool = False
    adaptive_max_additional_waves: int = 2
    adaptive_packets_per_wave: int = 8
    evidence_review_timeout_s: float = 10.0
    claim_review_reserve_s: float = 6.0
    web_host: str = "127.0.0.1"
    web_port: int = 8780

    @property
    def db_path(self) -> Path:
        return self.workspace_dir / "reader.sqlite3"

    @property
    def trace_dir(self) -> Path:
        return self.workspace_dir / "traces"

    @property
    def profile_path(self) -> Path:
        return self.workspace_dir / "profile.json"

    @property
    def entity_registry_path(self) -> Path:
        return self.workspace_dir / "entities.json"

    @property
    def inference_lock_path(self) -> Path:
        return self.workspace_dir / "inference.lock"

    @property
    def effective_screen_model(self) -> str:
        return self.screen_model or self.model

    @property
    def effective_reader_model(self) -> str:
        return self.reader_model or self.model

    @property
    def effective_synthesis_model(self) -> str:
        return self.synthesis_model or self.model

    @property
    def effective_review_model(self) -> str:
        return self.review_model or self.effective_synthesis_model

    @property
    def reader_input_tokens(self) -> int:
        return max(2_048, self.context_tokens - self.reader_context_reserve_tokens)

    @property
    def ollama_timeout_s(self) -> float:
        return self.request_timeout_s

    @property
    def ollama_keep_alive(self) -> str:
        return self.keep_alive

    def with_overrides(self, **values: object) -> "Config":
        return replace(self, **values)

    @classmethod
    def load(cls, project_dir: str | Path | None = None) -> "Config":
        if project_dir is not None:
            base = Path(project_dir).resolve()
        else:
            source_root = Path(__file__).resolve().parent.parent
            base = source_root if (source_root / "sisu-reader.cmd").is_file() else Path.cwd().resolve()
        workspace = Path(os.environ.get("SISU_READER_WORKSPACE", base / "workspace"))
        if not workspace.is_absolute():
            workspace = (base / workspace).resolve()

        trace_mode = os.environ.get("SISU_READER_TRACE_MODE", "metrics").strip().casefold()
        if trace_mode not in {"off", "metrics", "answer", "diagnostic"}:
            raise ValueError("SISU_READER_TRACE_MODE must be off, metrics, answer, or diagnostic")
        reasoning = os.environ.get("SISU_READER_REASONING", "off").strip().casefold()
        if reasoning not in {"off", "low", "medium", "high"}:
            raise ValueError("SISU_READER_REASONING must be off, low, medium, or high")
        context_tokens = _int("SISU_READER_CONTEXT_TOKENS", 32_768, 4_096, 131_072)
        context_reserve = _int("SISU_READER_CONTEXT_RESERVE", 2_500, 1_000, 16_000)
        if context_reserve >= context_tokens - 2_048:
            raise ValueError("SISU_READER_CONTEXT_RESERVE leaves too little room for source text")
        safe_block_default = min(
            4_096,
            max(512, int((context_tokens - context_reserve) * 0.78) - 1_300),
        )
        request_timeout_s = _float("SISU_READER_REQUEST_TIMEOUT_S", 88.0, 10.0, 300.0)
        total_deadline_s = _float("SISU_READER_DEADLINE_S", 95.0, 20.0, 300.0)
        synthesis_reserve_s = _float("SISU_READER_SYNTHESIS_RESERVE_S", 18.0, 5.0, 90.0)
        finalize_reserve_s = _float("SISU_READER_FINALIZE_RESERVE_S", 3.0, 1.0, 30.0)
        minimum_reader_window_s = _float(
            "SISU_READER_MINIMUM_READER_WINDOW_S", 15.0, 2.0, 90.0
        )
        minimum_screen_window_s = _float(
            "SISU_READER_MINIMUM_SCREEN_WINDOW_S", 10.0, 2.0, 90.0
        )
        if synthesis_reserve_s + finalize_reserve_s + 5.0 >= total_deadline_s:
            raise ValueError(
                "Synthesis and finalization reserves must leave at least 5 seconds for research"
            )
        research_window_s = total_deadline_s - synthesis_reserve_s - finalize_reserve_s
        if minimum_reader_window_s >= research_window_s:
            raise ValueError(
                "SISU_READER_MINIMUM_READER_WINDOW_S must be shorter than the research window"
            )
        if minimum_screen_window_s + minimum_reader_window_s >= research_window_s:
            raise ValueError(
                "Minimum screen and reader windows must fit inside the research window"
            )

        principal_id = os.environ.get("SISU_READER_USER_ID", "local-admin").strip()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}", principal_id):
            raise ValueError("SISU_READER_USER_ID is not a valid local user identifier")

        return cls(
            project_dir=base,
            workspace_dir=workspace,
            ollama_url=os.environ.get("SISU_READER_OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/"),
            model=os.environ.get("SISU_READER_MODEL", "gemma4:12b").strip(),
            screen_model=os.environ.get("SISU_READER_SCREEN_MODEL", "").strip(),
            reader_model=os.environ.get("SISU_READER_DOCUMENT_MODEL", "").strip(),
            synthesis_model=os.environ.get("SISU_READER_SYNTHESIS_MODEL", "").strip(),
            review_model=os.environ.get("SISU_READER_REVIEW_MODEL", "").strip(),
            context_tokens=context_tokens,
            max_model_calls=_int("SISU_READER_MAX_MODEL_CALLS", 30, 1, 1_000),
            max_output_tokens=_int("SISU_READER_MAX_OUTPUT_TOKENS", 32_768, 256, 10_000_000),
            screen_output_tokens=_int("SISU_READER_SCREEN_OUTPUT", 1_280, 128, 2_048),
            reader_output_tokens=_int("SISU_READER_DOCUMENT_OUTPUT", 1_024, 256, 4_096),
            synthesis_output_tokens=_int("SISU_READER_SYNTHESIS_OUTPUT", 3_000, 256, 4_096),
            review_output_tokens=_int("SISU_READER_REVIEW_OUTPUT", 2_000, 256, 4_096),
            reasoning=reasoning,
            request_timeout_s=request_timeout_s,
            total_deadline_s=total_deadline_s,
            synthesis_reserve_s=synthesis_reserve_s,
            finalize_reserve_s=finalize_reserve_s,
            minimum_screen_window_s=minimum_screen_window_s,
            minimum_reader_window_s=minimum_reader_window_s,
            inference_lock_timeout_s=_float("SISU_READER_LOCK_TIMEOUT_S", 120.0, 5.0, 600.0),
            keep_alive=os.environ.get("SISU_READER_KEEP_ALIVE", "30m"),
            screen_batch_token_budget=_int("SISU_READER_SCREEN_BUDGET", 8_000, 2_000, 32_000),
            reader_context_reserve_tokens=context_reserve,
            source_block_hard_tokens=_int("SISU_READER_BLOCK_HARD_TOKENS", safe_block_default, 512, 16_384),
            read_wave_size=_int("SISU_READER_READ_WAVE", 2, 1, 64),
            manifest_full_scan_threshold=_int(
                "SISU_READER_FULL_SCAN_DOCUMENTS", 200, 1, 100_000
            ),
            discovery_page_size=_int(
                "SISU_READER_DISCOVERY_PAGE", 48, 4, 1_000
            ),
            session_turns=_int("SISU_READER_HISTORY_TURNS", 6, 0, 20),
            trace_mode=trace_mode,
            trace_reasoning=_bool("SISU_READER_TRACE_REASONING", False),
            trace_retention_days=_int("SISU_READER_TRACE_DAYS", 30, 1, 3650),
            trace_max_total_mb=_int("SISU_READER_TRACE_MAX_MB", 512, 16, 16_384),
            principal_id=principal_id,
            contradiction_probe_documents=_int(
                "SISU_READER_CONTRADICTION_PROBES", 1, 0, 8
            ),
            reference_closure=_bool("SISU_READER_REFERENCE_CLOSURE", True),
            adaptive_research=_bool("SISU_READER_ADAPTIVE_RESEARCH", True),
            claim_review=_bool("SISU_READER_CLAIM_REVIEW", True),
            strategy_learning=_bool("SISU_READER_STRATEGY_LEARNING", False),
            adaptive_max_additional_waves=_int(
                "SISU_READER_ADAPTIVE_ADDITIONAL_WAVES", 2, 0, 2
            ),
            adaptive_packets_per_wave=_int("SISU_READER_ADAPTIVE_PACKETS", 8, 1, 64),
            evidence_review_timeout_s=_float("SISU_READER_EVIDENCE_REVIEW_TIMEOUT_S", 10.0, 2.0, 30.0),
            claim_review_reserve_s=_float("SISU_READER_CLAIM_REVIEW_RESERVE_S", 6.0, 2.0, 30.0),
            web_host=os.environ.get("SISU_READER_WEB_HOST", "127.0.0.1"),
            web_port=_int("SISU_READER_WEB_PORT", 8780, 1_024, 65_535),
        )
