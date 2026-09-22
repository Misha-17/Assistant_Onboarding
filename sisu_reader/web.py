"""Loopback-only asynchronous web transport for SISU Reader."""

from __future__ import annotations

import ipaddress
import json
import math
import queue
import re
import secrets
import threading
import time
import webbrowser
from dataclasses import asdict, dataclass, field, is_dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit

from .access import AccessManager
from .config import Config
from .session import Session


_MAX_BODY_BYTES = 64 * 1024
_MAX_QUESTION_CHARS = 4_000
_MAX_TRACE_BYTES = 8 * 1024 * 1024
_TOKEN = re.compile(r"^[A-Za-z0-9_-]{24,160}$")
_REQUEST_ID = re.compile(r"^[A-Za-z0-9_.:-]{1,160}$")
_STAGES = (
    "authorizing",
    "exact_search",
    "screening",
    "adaptive_wave",
    "reading",
    "evidence_review",
    "synthesizing",
    "claim_review",
    "binding_citations",
)
_STAGE_SET = frozenset(_STAGES)
_PUBLIC_STAGE_MESSAGES = {
    "authorizing": "Checking access",
    "exact_search": "Searching the authorized catalogue",
    "screening": "Checking likely sources",
    "reading": "Reading authorized sources",
    "adaptive_wave": "Searching for more evidence",
    "evidence_review": "Checking evidence gaps",
    "claim_review": "Checking claims against sources",
    "synthesizing": "Writing the answer",
    "binding_citations": "Checking citations",
}
_STAGE_ALIASES = {
    "authorization": "authorizing",
    "authorize": "authorizing",
    "exact": "exact_search",
    "searching": "exact_search",
    "search": "exact_search",
    "screen": "screening",
    "read": "reading",
    "writing": "synthesizing",
    "synthesis": "synthesizing",
    "citations": "binding_citations",
    "citation_binding": "binding_citations",
}
_ASSETS = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/assets/styles.css": ("styles.css", "text/css; charset=utf-8"),
    "/assets/app.js": ("app.js", "text/javascript; charset=utf-8"),
}


def _value(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(name, default)
    return getattr(item, name, default)


def _text(value: Any, maximum: int = 20_000) -> str:
    return str(value if value is not None else "")[:maximum]


def _json_safe(value: Any, *, depth: int = 0) -> Any:
    if depth > 10:
        return "<truncated>"
    if is_dataclass(value):
        return _json_safe(asdict(value), depth=depth + 1)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in list(value.items())[:500]:
            result[_text(key, 160)] = _json_safe(item, depth=depth + 1)
        return result
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item, depth=depth + 1) for item in list(value)[:500]]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, str):
        return value[:100_000]
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return _text(value, 20_000)


def _warnings(value: Any) -> list[str]:
    if isinstance(value, (str, bytes, Mapping)):
        return []
    try:
        rows = tuple(value or ())[:64]
    except TypeError:
        return []
    return [" ".join(_text(row, 1_000).split()) for row in rows if _text(row).strip()]


def _timings(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, float] = {}
    for key, raw in list(value.items())[:64]:
        if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", key):
            continue
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            number = float(raw)
            if math.isfinite(number):
                result[key] = number
    return result


def _sources(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, (str, bytes, Mapping)):
        return []
    try:
        rows = tuple(value or ())[:96]
    except TypeError:
        return []
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for index, source in enumerate(rows, 1):
        row = {
            "id": _text(_value(source, "source_id", "") or f"S{index}", 100),
            "document_id": _text(
                _value(source, "document_revision_id", "")
                or _value(source, "document_id", ""),
                240,
            ),
            "span_id": _text(
                _value(source, "span_id", "")
                or _value(source, "block_id", "")
                or _value(source, "window_id", "")
                or _value(source, "passage_id", "")
                or _value(source, "chunk_id", ""),
                240,
            ),
            "title": _text(_value(source, "title", "Untitled source"), 1_000),
            "locator": _text(_value(source, "locator", ""), 1_000),
            "quote": _text(_value(source, "quote", ""), 24_000),
            "resource_type": _text(_value(source, "resource_type", "document"), 80),
            "resource_id": _text(_value(source, "resource_id", ""), 240),
            "uri": _text(_value(source, "uri", ""), 4_000),
            "timestamp_seconds": _value(source, "timestamp_seconds", None),
            "provenance": _text(_value(source, "provenance", ""), 500),
        }
        if not row["quote"]:
            continue
        timestamp = row["timestamp_seconds"]
        if not isinstance(timestamp, int) or isinstance(timestamp, bool) or timestamp < 0:
            row["timestamp_seconds"] = None
        key = tuple(str(row[item]) for item in sorted(row))
        if key not in seen:
            result.append(row)
            seen.add(key)
    return result


def _answer_payload(answer: Any) -> dict[str, Any]:
    debug = _value(answer, "debug", {})
    coverage = _value(answer, "coverage", None)
    if coverage is None and isinstance(debug, Mapping):
        coverage = debug.get("coverage", {})
    trace_path = _text(_value(answer, "trace_path", ""), 4_000)
    return {
        "status": _text(_value(answer, "status", "error"), 80),
        "text": _text(_value(answer, "text", ""), 100_000),
        "sources": _sources(_value(answer, "sources", ())),
        "warnings": _warnings(_value(answer, "warnings", ())),
        "timings": _timings(_value(answer, "timings", {})),
        "coverage": _json_safe(coverage or {}),
        "trace_available": bool(trace_path),
        "debug": _json_safe(debug if isinstance(debug, Mapping) else {}),
    }


def _friendly_error(exc: BaseException, config: Config) -> dict[str, str]:
    raw = " ".join(str(exc).split())
    folded = raw.casefold()
    if isinstance(exc, FileNotFoundError) or "index" in folded and "not found" in folded:
        return {
            "code": "missing_index",
            "message": "The local document index has not been built yet.",
            "action": "Index a document folder, then retry the engine.",
            "command": 'sisu-reader index "C:\\path\\to\\documents"',
        }
    if "model" in folded and any(word in folded for word in ("missing", "not found", "unknown", "unavailable", "pull")):
        return {
            "code": "missing_model",
            "message": f"The local model {config.model!r} is not installed.",
            "action": "Install the configured Ollama model, then retry the engine.",
            "command": f"ollama pull {config.model}",
        }
    if any(word in folded for word in ("connection refused", "failed to establish", "ollama")):
        return {
            "code": "ollama_unavailable",
            "message": "Ollama is not reachable on this computer.",
            "action": "Start Ollama, then retry the engine.",
            "command": "ollama serve",
        }
    return {
        "code": "engine_error",
        "message": (raw or type(exc).__name__)[:700],
        "action": "Run the local doctor command for details, then retry.",
        "command": "sisu-reader doctor",
    }


def _normal_stage(value: Any) -> str:
    stage = _text(value, 100).strip().casefold().replace("-", "_").replace(" ", "_")
    stage = _STAGE_ALIASES.get(stage, stage)
    return stage if stage in _STAGE_SET else ""


@dataclass
class _BrowserSession:
    token: str
    conversation: Session
    principal_id: str
    authorization_revision: int
    created: float = field(default_factory=time.monotonic)
    touched: float = field(default_factory=time.monotonic)


class _SessionStore:
    def __init__(
        self,
        *,
        max_turns: int,
        access: AccessManager,
        principal_id: str,
        ttl_s: float = 4 * 60 * 60,
        maximum: int = 128,
    ) -> None:
        self.max_turns = max_turns
        self.access = access
        self.principal_id = principal_id
        self.ttl_s = ttl_s
        self.maximum = maximum
        self._items: dict[str, _BrowserSession] = {}
        self._lock = threading.RLock()

    def _purge(self) -> None:
        cutoff = time.monotonic() - self.ttl_s
        for token in [key for key, item in self._items.items() if item.touched < cutoff]:
            self._items.pop(token, None)
        while len(self._items) >= self.maximum and self._items:
            oldest = min(self._items.values(), key=lambda item: item.touched)
            self._items.pop(oldest.token, None)

    def create(self) -> _BrowserSession:
        with self._lock:
            self._purge()
            principal = self.access.principal_snapshot(self.principal_id)
            if not principal.known or not principal.active or not self.access.has_permission(
                principal, "question.ask"
            ):
                raise PermissionError("The configured local user is unavailable")
            token = secrets.token_urlsafe(32)
            conversation = Session(max_turns=self.max_turns)
            conversation.bind_authorization(principal.user_id, principal.revision)
            item = _BrowserSession(
                token,
                conversation,
                principal.user_id,
                principal.revision,
            )
            self._items[token] = item
            return item

    def get(self, token: str) -> _BrowserSession | None:
        if not _TOKEN.fullmatch(token):
            return None
        with self._lock:
            self._purge()
            item = self._items.get(token)
            if item is not None:
                item.touched = time.monotonic()
            return item


@dataclass
class _Job:
    job_id: str
    session_token: str
    question: str
    request_id: str
    principal_id: str = ""
    authorization_revision: int = 0
    status: str = "queued"
    stage: str = ""
    progress: dict[str, Any] = field(default_factory=dict)
    answer: dict[str, Any] | None = None
    trace_path: str = ""
    error: dict[str, str] | None = None
    created: float = field(default_factory=time.monotonic)
    updated: float = field(default_factory=time.monotonic)
    started: float | None = None
    finished: float | None = None

    def payload(self, *, queue_position: int | None = None) -> dict[str, Any]:
        elapsed_end = self.finished or time.monotonic()
        elapsed = elapsed_end - (self.started or self.created)
        result: dict[str, Any] = {
            "id": self.job_id,
            "status": self.status,
            "stage": self.stage,
            "progress": _json_safe(self.progress),
            "elapsed_s": max(0.0, elapsed),
        }
        if queue_position is not None and self.status == "queued":
            result["queue_position"] = queue_position
        if self.answer is not None:
            result["answer"] = self.answer
        if self.error is not None:
            result["error"] = self.error
        return result


class _JobStore:
    def __init__(
        self,
        *,
        maximum: int = 16,
        per_session_active: int = 2,
        ttl_s: float = 60 * 60,
    ) -> None:
        self.maximum = maximum
        self.per_session_active = per_session_active
        self.ttl_s = ttl_s
        self._jobs: dict[str, _Job] = {}
        self._requests: dict[tuple[str, str], str] = {}
        self._order: list[str] = []
        self._lock = threading.RLock()

    def _purge(self) -> None:
        cutoff = time.monotonic() - self.ttl_s
        removable = [
            job_id
            for job_id in self._order
            if (job := self._jobs.get(job_id)) is not None
            and job.status in {"complete", "failed"}
            and job.updated < cutoff
        ]
        for job_id in removable:
            job = self._jobs.pop(job_id, None)
            if job is not None:
                self._requests.pop((job.session_token, job.request_id), None)
            if job_id in self._order:
                self._order.remove(job_id)
        while len(self._jobs) >= self.maximum:
            candidate = next((
                job_id for job_id in self._order
                if self._jobs[job_id].status in {"complete", "failed"}
            ), None)
            if candidate is None:
                raise RuntimeError("The local job queue is full")
            job = self._jobs.pop(candidate)
            self._requests.pop((job.session_token, job.request_id), None)
            self._order.remove(candidate)

    def create(
        self,
        session_token: str,
        question: str,
        request_id: str,
        *,
        principal_id: str = "",
        authorization_revision: int = 0,
    ) -> tuple[_Job, bool]:
        with self._lock:
            self._purge()
            prior_id = self._requests.get((session_token, request_id))
            if prior_id and prior_id in self._jobs:
                return self._jobs[prior_id], False
            active_for_session = sum(
                1
                for item in self._jobs.values()
                if item.session_token == session_token
                and item.status in {"queued", "running"}
            )
            if active_for_session >= self.per_session_active:
                raise RuntimeError(
                    "This conversation already has the maximum number of active questions"
                )
            job = _Job(
                job_id="job_" + secrets.token_urlsafe(18),
                session_token=session_token,
                question=question,
                request_id=request_id,
                principal_id=principal_id,
                authorization_revision=int(authorization_revision),
            )
            self._jobs[job.job_id] = job
            self._requests[(session_token, request_id)] = job.job_id
            self._order.append(job.job_id)
            return job, True

    def get(self, job_id: str, session_token: str) -> _Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or not secrets.compare_digest(job.session_token, session_token):
                return None
            return job

    def mutate(self, job: _Job, **values: Any) -> None:
        with self._lock:
            if job.job_id not in self._jobs:
                return
            for key, value in values.items():
                setattr(job, key, value)
            job.updated = time.monotonic()

    def queue_position(self, job: _Job) -> int | None:
        with self._lock:
            queued = [
                self._jobs[job_id]
                for job_id in self._order
                if job_id in self._jobs and self._jobs[job_id].status == "queued"
            ]
            try:
                return queued.index(job) + 1
            except ValueError:
                return None


class _Application:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.access = AccessManager(config)
        self.assets_dir = Path(__file__).resolve().parent / "web_assets"
        self.sessions = _SessionStore(
            max_turns=config.session_turns,
            access=self.access,
            principal_id=config.principal_id,
        )
        self.jobs = _JobStore()
        self._queue: queue.Queue[_Job | None] = queue.Queue(maxsize=16)
        self._engine: Any = None
        self._engine_state = "idle"
        self._engine_error: dict[str, str] | None = None
        self._engine_lock = threading.RLock()
        self._closed = threading.Event()
        self._worker = threading.Thread(target=self._work, name="sisu-reader-jobs", daemon=True)
        self._worker.start()
        self.start_engine()

    def _new_engine(self) -> Any:
        from .engine import SisuReader

        return SisuReader(self.config)

    def start_engine(self) -> None:
        with self._engine_lock:
            if self._engine_state == "loading":
                return
            prior = self._engine
            self._engine = None
            self._engine_state = "loading"
            self._engine_error = None
        if prior is not None:
            try:
                prior.close()
            except Exception:
                pass

        def load() -> None:
            engine = None
            try:
                engine = self._new_engine()
                # Publish the object while it loads so a second retry cannot
                # create another engine concurrently.
                with self._engine_lock:
                    self._engine = engine
                engine.load()
                status = engine.status()
                if _value(status, "ready", True) is False:
                    raise RuntimeError(_value(status, "error", "engine reported not ready"))
                with self._engine_lock:
                    self._engine_state = "ready"
                    self._engine_error = None
            except Exception as exc:
                if engine is not None:
                    try:
                        engine.close()
                    except Exception:
                        pass
                with self._engine_lock:
                    self._engine = None
                    self._engine_state = "failed"
                    self._engine_error = _friendly_error(exc, self.config)

        threading.Thread(target=load, name="sisu-reader-startup", daemon=True).start()

    def engine_status(self) -> dict[str, Any]:
        with self._engine_lock:
            state = self._engine_state
            error = dict(self._engine_error) if self._engine_error else None
            engine = self._engine
        detail: Any = {}
        if state == "ready" and engine is not None:
            try:
                raw = engine.status()
                # This endpoint is intentionally available before a browser
                # session exists. Never expose corpus or resource counts here.
                detail = {
                    "ready": bool(_value(raw, "ready", True)),
                    "local_only": bool(_value(raw, "local_only", True)),
                    "pipeline": _text(_value(raw, "pipeline", ""), 160),
                }
            except Exception as exc:
                detail = {"status_error": " ".join(str(exc).split())[:500]}
        return {
            "state": state,
            "ready": state == "ready",
            "error": error,
            "model": self.config.model,
            "detail": detail,
            "progress_stages": list(_STAGES),
        }

    def _ready_engine(self) -> Any:
        with self._engine_lock:
            if self._engine_state != "ready" or self._engine is None:
                message = (self._engine_error or {}).get("message", "The local engine is not ready.")
                raise RuntimeError(message)
            return self._engine

    def job_authorized(self, job: _Job, browser: _BrowserSession) -> bool:
        """Reauthorize cached answer dependencies at delivery time.

        Revision checks catch policy mutations. Concrete checks also catch
        grants or memberships that expire merely because time passed.
        """

        current = self.access.principal_snapshot(browser.principal_id)
        if (
            current.user_id != job.principal_id
            or not current.known
            or not current.active
            or not self.access.has_permission(current, "question.ask")
            or current.revision != job.authorization_revision
        ):
            return False
        # Queued and running jobs do not have final answer dependencies yet.
        # Their principal, question permission, and authorization revision are
        # still checked above. The engine performs a fresh dependency check
        # before publishing a completed answer.
        if job.status != "complete":
            return True
        answer = job.answer or {}
        debug = answer.get("debug", {}) if isinstance(answer, Mapping) else {}
        if isinstance(debug, Mapping):
            dependencies = {
                str(item)
                for key in ("read_document_ids", "screened_document_ids")
                for item in (
                    debug.get(key, ())
                    if isinstance(debug.get(key, ()), (list, tuple))
                    else ()
                )
                if item
            }
            if dependencies:
                allowed = set(self.access.allowed_document_revision_ids(current, "document.read"))
                allowed.intersection_update(
                    self.access.allowed_document_revision_ids(current, "document.search")
                )
                allowed.intersection_update(
                    self.access.allowed_document_revision_ids(current, "document.cite")
                )
                if not dependencies.issubset(allowed):
                    return False
        sources = answer.get("sources", ()) if isinstance(answer, Mapping) else ()
        if not isinstance(sources, list):
            return False
        for source in sources:
            if not isinstance(source, Mapping):
                return False
            kind = str(source.get("resource_type") or "document")
            resource_id = str(source.get("resource_id") or "")
            if kind == "document":
                if resource_id and not all(
                    self.access.can(current, "document", resource_id, action)
                    for action in ("document.search", "document.read", "document.cite")
                ):
                    return False
            elif kind == "org_role":
                if (
                    not self.access.has_permission(current, "org_role.search")
                    or not resource_id
                    or not self.access.can(
                        current, "org_role", resource_id, "org_role.read"
                    )
                ):
                    return False
            elif kind == "video":
                if (
                    not self.access.has_permission(current, "video.search")
                    or not resource_id
                    or not self.access.can(
                        current, "video", resource_id, "video.metadata.read"
                    )
                ):
                    return False
                if source.get("uri") and not self.access.can(
                    current, "video", resource_id, "video.open"
                ):
                    return False
            else:
                return False
        return True

    def create_job(self, browser: _BrowserSession, question: str, request_id: str) -> _Job:
        principal = self.access.principal_snapshot(browser.principal_id)
        if not principal.known or not principal.active or not self.access.has_permission(
            principal, "question.ask"
        ):
            raise PermissionError("The configured local user is unavailable")
        if principal.revision != browser.authorization_revision:
            browser.conversation.bind_authorization(principal.user_id, principal.revision)
            browser.authorization_revision = principal.revision
        job, created = self.jobs.create(
            browser.token,
            question,
            request_id,
            principal_id=principal.user_id,
            authorization_revision=principal.revision,
        )
        if created:
            try:
                self._queue.put_nowait(job)
            except queue.Full:
                self.jobs.mutate(
                    job,
                    status="failed",
                    error={
                        "code": "queue_full",
                        "message": "The local answer queue is full.",
                        "action": "Wait for the current answer to finish and retry.",
                        "command": "",
                    },
                    finished=time.monotonic(),
                )
        return job

    def _progress_callback(self, job: _Job):
        def callback(event: Any = None, *args: Any, **kwargs: Any) -> None:
            del args
            payload: dict[str, Any] = {}
            stage = ""
            if isinstance(event, str):
                stage = _normal_stage(event)
            elif isinstance(event, Mapping):
                stage = _normal_stage(
                    event.get("stage") or event.get("status") or event.get("name")
                )
                payload.update(event)
            elif event is not None:
                stage = _normal_stage(
                    getattr(event, "stage", "")
                    or getattr(event, "status", "")
                    or getattr(event, "name", "")
                )
                for key in ("label", "coverage", "current", "elapsed_s"):
                    if hasattr(event, key):
                        payload[key] = getattr(event, key)
            if not stage:
                stage = _normal_stage(
                    kwargs.get("stage") or kwargs.get("status") or kwargs.get("name")
                )
            payload.update(kwargs)
            if not stage:
                return
            # Progress is deliberately generic. Document titles, labels, and
            # details can otherwise remain pollable during a revocation race.
            safe_progress: dict[str, Any] = {
                "message": _PUBLIC_STAGE_MESSAGES[stage],
            }
            if "elapsed_s" in payload:
                safe_progress["elapsed_s"] = _json_safe(payload["elapsed_s"])
            current = payload.get("current")
            if isinstance(current, Mapping):
                numeric_current = {
                    key: value
                    for key, value in current.items()
                    if key in {"batch", "batches", "manifests", "index", "total"}
                    and isinstance(value, int)
                    and not isinstance(value, bool)
                }
                if numeric_current:
                    safe_progress["current"] = _json_safe(numeric_current)
            coverage = payload.get("coverage")
            if isinstance(coverage, Mapping):
                numeric_coverage = {
                    key: value
                    for key, value in coverage.items()
                    if isinstance(value, (int, float, bool))
                }
                if numeric_coverage:
                    safe_progress["coverage"] = _json_safe(numeric_coverage)
            self.jobs.mutate(job, stage=stage, progress=safe_progress)

        return callback

    def _work(self) -> None:
        while not self._closed.is_set():
            job = self._queue.get()
            if job is None:
                self._queue.task_done()
                return
            self.jobs.mutate(
                job,
                status="running",
                stage="authorizing",
                started=time.monotonic(),
            )
            try:
                browser = self.sessions.get(job.session_token)
                if browser is None:
                    raise RuntimeError("The browser conversation expired. Start a new conversation.")
                engine = self._ready_engine()
                answer = engine.ask(
                    job.question,
                    session=browser.conversation,
                    mode="auto",
                    progress=self._progress_callback(job),
                    principal=job.principal_id,
                )
                browser.conversation.record(job.question, answer)
                payload = _answer_payload(answer)
                final_revision = int(
                    _value(_value(answer, "debug", {}), "authorization_revision", 0)
                    or self.access.authorization_revision()
                )
                self.jobs.mutate(
                    job,
                    status="complete",
                    stage="binding_citations",
                    progress={},
                    answer=payload,
                    trace_path=_text(_value(answer, "trace_path", ""), 4_000),
                    authorization_revision=final_revision,
                    finished=time.monotonic(),
                )
            except Exception as exc:
                self.jobs.mutate(
                    job,
                    status="failed",
                    error=_friendly_error(exc, self.config),
                    finished=time.monotonic(),
                )
            finally:
                self._queue.task_done()

    def read_trace(
        self,
        job: _Job,
        browser: _BrowserSession,
    ) -> tuple[dict[str, Any] | None, str | None]:
        if not job.trace_path:
            return None, "This answer did not save a trace."
        if not self.job_authorized(job, browser):
            return None, "This trace is unavailable under the current access policy."
        run_id = _text((job.answer or {}).get("debug", {}).get("run_id", ""), 160)
        current = self.access.principal_snapshot(browser.principal_id)
        if (
            not run_id
            or current.revision != job.authorization_revision
            or not (
                self.access.can(current, "trace", run_id, "trace.read_own")
                or self.access.can(current, "trace", run_id, "trace.read_any")
            )
        ):
            return None, "This trace is unavailable under the current access policy."
        try:
            root = self.config.trace_dir.resolve()
            path = Path(job.trace_path).resolve(strict=True)
            path.relative_to(root)
            if path.suffix.casefold() != ".json":
                raise ValueError("trace is not JSON")
            if path.stat().st_size > _MAX_TRACE_BYTES:
                return None, "The saved trace is too large to display in the browser."
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            return {"trace": _json_safe(payload)}, None
        except (OSError, ValueError, json.JSONDecodeError):
            return None, "The saved trace is unavailable or outside the private trace directory."

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        with self._engine_lock:
            engine = self._engine
            self._engine = None
            self._engine_state = "closed"
        if engine is not None:
            try:
                engine.close()
            except Exception:
                pass


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class _Handler(BaseHTTPRequestHandler):
    application: _Application
    server_version = "SisuReader/0.1"
    sys_version = ""

    def log_message(self, format: str, *args: Any) -> None:
        # Keep the local console useful without logging request bodies or
        # session tokens.
        print(f"[{self.log_date_time_string()}] {format % args}")

    def _security_headers(self, *, api: bool = False) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "connect-src 'self'; img-src 'self' data:; font-src 'self'; "
            "object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'",
        )
        self.send_header("Cache-Control", "no-store" if api else "no-cache")

    def _send_json(self, status: int, payload: Mapping[str, Any]) -> None:
        body = json.dumps(_json_safe(payload), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._security_headers(api=True)
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: int, code: str, message: str) -> None:
        self._send_json(status, {"error": {"code": code, "message": message}})

    def _guard(self) -> bool:
        try:
            if not ipaddress.ip_address(self.client_address[0]).is_loopback:
                self._error(HTTPStatus.FORBIDDEN, "loopback_only", "This service accepts loopback requests only.")
                return False
        except ValueError:
            self._error(HTTPStatus.FORBIDDEN, "loopback_only", "This service accepts loopback requests only.")
            return False

        host = self.headers.get("Host", "")
        try:
            hostname = urlsplit("//" + host).hostname or ""
            allowed_host = hostname.casefold() == "localhost" or ipaddress.ip_address(hostname).is_loopback
        except ValueError:
            allowed_host = False
        if not allowed_host:
            self._error(HTTPStatus.FORBIDDEN, "invalid_host", "Invalid local Host header.")
            return False

        if self.headers.get("Sec-Fetch-Site", "").casefold() == "cross-site":
            self._error(HTTPStatus.FORBIDDEN, "cross_site", "Cross-site requests are not accepted.")
            return False
        origin = self.headers.get("Origin")
        if origin:
            parsed = urlsplit(origin)
            origin_host = parsed.hostname or ""
            try:
                loopback = origin_host.casefold() == "localhost" or ipaddress.ip_address(origin_host).is_loopback
            except ValueError:
                loopback = False
            if parsed.scheme != "http" or not loopback or parsed.netloc.casefold() != host.casefold():
                self._error(HTTPStatus.FORBIDDEN, "invalid_origin", "Cross-origin requests are not accepted.")
                return False
        return True

    def _read_json(self) -> dict[str, Any] | None:
        content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip().casefold()
        if content_type != "application/json":
            self._error(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "content_type", "Expected application/json.")
            return None
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._error(HTTPStatus.BAD_REQUEST, "content_length", "Invalid Content-Length.")
            return None
        if length < 0 or length > _MAX_BODY_BYTES:
            self._error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "body_too_large", "Request body is too large.")
            return None
        try:
            value = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._error(HTTPStatus.BAD_REQUEST, "invalid_json", "Request body is not valid JSON.")
            return None
        if not isinstance(value, dict):
            self._error(HTTPStatus.BAD_REQUEST, "invalid_json", "Expected a JSON object.")
            return None
        return value

    def _browser(self) -> _BrowserSession | None:
        token = self.headers.get("X-SISU-Session", "")
        browser = self.application.sessions.get(token)
        if browser is None:
            self._error(HTTPStatus.UNAUTHORIZED, "session_expired", "Start a new browser conversation.")
        return browser

    def do_OPTIONS(self) -> None:
        if self._guard():
            self._error(HTTPStatus.METHOD_NOT_ALLOWED, "method_not_allowed", "CORS is not enabled.")

    def do_GET(self) -> None:
        if not self._guard():
            return
        path = urlsplit(self.path).path
        if path in _ASSETS:
            filename, content_type = _ASSETS[path]
            asset = self.application.assets_dir / filename
            try:
                body = asset.read_bytes()
            except OSError:
                self._error(HTTPStatus.NOT_FOUND, "asset_missing", "Web asset is missing.")
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self._security_headers()
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/api/status":
            self._send_json(HTTPStatus.OK, {"engine": self.application.engine_status()})
            return

        match = re.fullmatch(r"/api/jobs/(?P<job>job_[A-Za-z0-9_-]{20,160})", path)
        trace_match = re.fullmatch(r"/api/jobs/(?P<job>job_[A-Za-z0-9_-]{20,160})/trace", path)
        if match or trace_match:
            browser = self._browser()
            if browser is None:
                return
            job_id = (trace_match or match).group("job")
            job = self.application.jobs.get(job_id, browser.token)
            if job is None:
                self._error(HTTPStatus.NOT_FOUND, "job_not_found", "That local job was not found.")
                return
            if not self.application.job_authorized(job, browser):
                self._error(
                    HTTPStatus.CONFLICT,
                    "access_changed",
                    "Access changed after this answer was created. Ask the question again.",
                )
                return
            if trace_match:
                payload, error = self.application.read_trace(job, browser)
                if payload is None:
                    self._error(HTTPStatus.NOT_FOUND, "trace_unavailable", error or "Trace unavailable.")
                else:
                    self._send_json(HTTPStatus.OK, payload)
                return
            position = self.application.jobs.queue_position(job)
            self._send_json(HTTPStatus.OK, {"job": job.payload(queue_position=position)})
            return
        self._error(HTTPStatus.NOT_FOUND, "not_found", "Route not found.")

    def do_POST(self) -> None:
        if not self._guard():
            return
        path = urlsplit(self.path).path
        payload = self._read_json()
        if payload is None:
            return
        if path == "/api/session":
            try:
                browser = self.application.sessions.create()
            except PermissionError:
                self._error(
                    HTTPStatus.FORBIDDEN,
                    "user_unavailable",
                    "The configured local user is unavailable.",
                )
                return
            self._send_json(HTTPStatus.CREATED, {"session_id": browser.token})
            return
        if path == "/api/retry":
            self.application.start_engine()
            self._send_json(HTTPStatus.ACCEPTED, {"engine": self.application.engine_status()})
            return
        if path == "/api/session/reset":
            browser = self._browser()
            if browser is None:
                return
            browser.conversation.reset()
            self._send_json(HTTPStatus.OK, {"ok": True})
            return
        if path == "/api/ask":
            browser = self._browser()
            if browser is None:
                return
            state = self.application.engine_status()
            if not state["ready"]:
                self._send_json(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    {"error": state.get("error") or {
                        "code": "engine_loading",
                        "message": "The private local engine is still loading.",
                        "action": "Wait a moment and retry.",
                        "command": "",
                    }},
                )
                return
            question = str(payload.get("message") or "").strip()
            if not question:
                self._error(HTTPStatus.BAD_REQUEST, "empty_question", "Enter a question first.")
                return
            if len(question) > _MAX_QUESTION_CHARS:
                self._error(HTTPStatus.BAD_REQUEST, "question_too_long", "Question is too long.")
                return
            request_id = str(payload.get("request_id") or "")
            if not _REQUEST_ID.fullmatch(request_id):
                self._error(HTTPStatus.BAD_REQUEST, "invalid_request_id", "Invalid request identifier.")
                return
            try:
                job = self.application.create_job(browser, question, request_id)
            except (RuntimeError, PermissionError) as exc:
                self._error(HTTPStatus.SERVICE_UNAVAILABLE, "queue_full", str(exc))
                return
            position = self.application.jobs.queue_position(job)
            self._send_json(HTTPStatus.ACCEPTED, {"job": job.payload(queue_position=position)})
            return
        self._error(HTTPStatus.NOT_FOUND, "not_found", "Route not found.")


def _loopback_host(host: str) -> bool:
    if host.casefold() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def run_web(config: Config, *, open_browser: bool = True) -> None:
    """Serve the UI on a loopback address until interrupted."""

    if not _loopback_host(config.web_host):
        raise ValueError("SISU Reader UI must bind to localhost or a loopback IP")
    application = _Application(config)
    handler = type("SisuReaderHandler", (_Handler,), {"application": application})
    try:
        server = _Server((config.web_host, config.web_port), handler)
    except Exception:
        application.close()
        raise
    display_host = f"[{config.web_host}]" if ":" in config.web_host else config.web_host
    url = f"http://{display_host}:{config.web_port}/"
    print(f"SISU Reader UI is available at {url}")
    print("The page stays usable while the local engine starts. Press Ctrl+C to stop.")
    if open_browser:
        threading.Timer(0.25, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        server.server_close()
        application.close()


serve = run_web


__all__ = ["run_web", "serve"]
