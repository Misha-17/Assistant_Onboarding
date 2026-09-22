from __future__ import annotations

import errno
import math
import os
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeVar

from .config import Config
from .answer_budget import AnswerBudget, AnswerBudgetExceeded
from .models import GenerationResult, ModelCallRecord
from .ollama import OllamaClient


T = TypeVar("T")
RoleName = Literal["screen", "reader", "synthesis", "review"]


class InferenceLockTimeout(TimeoutError):
    """The shared local inference device did not become available in time."""


class BrokerCallError(RuntimeError):
    """A role call failed; ``record`` is safe to retain in the run trace."""

    def __init__(self, message: str, *, record: ModelCallRecord) -> None:
        self.record = record
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class BrokerResult:
    generation: GenerationResult
    call: ModelCallRecord


@dataclass
class _PathState:
    lock: threading.RLock = field(default_factory=threading.RLock)
    local: threading.local = field(default_factory=threading.local)


_STATES: dict[str, _PathState] = {}
_STATES_LOCK = threading.Lock()


def _state_for(path: Path) -> _PathState:
    key = os.path.normcase(str(path.resolve()))
    with _STATES_LOCK:
        state = _STATES.get(key)
        if state is None:
            state = _PathState()
            _STATES[key] = state
        return state


def _prepare_lock_file(handle: Any) -> None:
    handle.seek(0, os.SEEK_END)
    if handle.tell() == 0:
        handle.write(b"\0")
        handle.flush()
        try:
            os.fsync(handle.fileno())
        except OSError:
            pass
    handle.seek(0)


if os.name == "nt":
    import msvcrt

    def _try_lock(handle: Any) -> bool:
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            return True
        except OSError as exc:
            if exc.errno in {errno.EACCES, errno.EAGAIN, errno.EDEADLK, 13, 36}:
                return False
            raise

    def _unlock(handle: Any) -> None:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    def _try_lock(handle: Any) -> bool:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError as exc:
            if exc.errno in {errno.EACCES, errno.EAGAIN}:
                return False
            raise

    def _unlock(handle: Any) -> None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class InferenceBroker:
    """Serialize and route local inference across threads, processes, and roles.

    ``chat`` is the normal public entry point. It selects the configured model
    and output ceiling for ``screen``, ``reader``, or ``synthesis``, holds one
    shared advisory lease, and returns both the generation and a trace-ready
    :class:`~sisu_reader.models.ModelCallRecord`.
    """

    def __init__(
        self,
        config: Config | None = None,
        *,
        lock_path: str | Path | None = None,
        timeout_s: float | None = None,
        poll_interval_s: float = 0.05,
        client_factory: Callable[..., OllamaClient] | None = None,
    ) -> None:
        self.config = config or Config.load()
        self.lock_path = Path(lock_path or self.config.inference_lock_path).resolve()
        self.timeout_s = (
            self.config.inference_lock_timeout_s
            if timeout_s is None
            else float(timeout_s)
        )
        self.poll_interval_s = float(poll_interval_s)
        if not math.isfinite(self.timeout_s) or self.timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if (
            not math.isfinite(self.poll_interval_s)
            or not 0.01 <= self.poll_interval_s <= 1.0
        ):
            raise ValueError("poll_interval_s must be between 0.01 and 1.0")
        self._state = _state_for(self.lock_path)
        self._client_factory = client_factory or OllamaClient
        self._clients: dict[str, OllamaClient] = {}
        self._clients_lock = threading.Lock()
        self._answer_budget: ContextVar[AnswerBudget | None] = ContextVar("answer_budget", default=None)

    @property
    def current_answer_budget(self) -> AnswerBudget | None:
        return self._answer_budget.get()

    @contextmanager
    def answer_budget(self, config: Config | None = None) -> Iterator[AnswerBudget]:
        from .grounded_answer import AUDIT_PROMPT_VERSION as REPAIR_PROMPT_VERSION
        active = self.current_answer_budget
        if active is not None:
            # Nested orchestration shares the same request account, never a reset.
            if not active.matches(config or self.config, repair_prompt_version=REPAIR_PROMPT_VERSION):
                raise ValueError("Nested answer budgets must have the same contract")
            yield active
            return
        account = AnswerBudget(config or self.config, repair_prompt_version=REPAIR_PROMPT_VERSION)
        token = self._answer_budget.set(account)
        try:
            yield account
        finally:
            account.close()
            self._answer_budget.reset(token)

    @staticmethod
    def _role(role: str) -> RoleName:
        normalized = role.strip().casefold()
        aliases = {
            "screener": "screen",
            "document": "reader",
            "synthesizer": "synthesis",
        }
        normalized = aliases.get(normalized, normalized)
        if normalized not in {"screen", "reader", "synthesis", "review"}:
            raise ValueError("role must be screen, reader, synthesis, or review")
        return normalized  # type: ignore[return-value]

    def model_for(self, role: str) -> str:
        normalized = self._role(role)
        attribute = {
            "screen": "effective_screen_model",
            "reader": "effective_reader_model",
            "synthesis": "effective_synthesis_model",
            "review": "effective_review_model",
        }[normalized]
        value = getattr(self.config, attribute, None)
        if normalized == "review" and not value:
            value = getattr(self.config, "effective_synthesis_model", None)
        value = value or self.config.model
        clean = str(value).strip()
        if not clean:
            raise ValueError(f"No model is configured for role {normalized!r}")
        return clean

    def output_tokens_for(self, role: str) -> int:
        normalized = self._role(role)
        attribute = {
            "screen": "screen_output_tokens",
            "reader": "reader_output_tokens",
            "synthesis": "synthesis_output_tokens",
            "review": "review_output_tokens",
        }[normalized]
        value = int(getattr(self.config, attribute, 1400) if normalized == "review" else getattr(self.config, attribute))
        if value <= 0:
            raise ValueError(f"Configured {attribute} must be positive")
        return value

    def _client(self, model: str) -> OllamaClient:
        with self._clients_lock:
            client = self._clients.get(model)
            if client is None:
                client = self._client_factory(self.config, model=model)
                self._clients[model] = client
            return client

    @contextmanager
    def lease(
        self,
        *,
        timeout_s: float | None = None,
        cancel_event: threading.Event | None = None,
    ) -> Iterator[None]:
        """Hold the re-entrant local and cross-process inference lease."""

        timeout = self.timeout_s if timeout_s is None else float(timeout_s)
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout_s must be positive")
        deadline = time.monotonic() + timeout
        acquired_thread = False
        while not acquired_thread:
            if cancel_event is not None and cancel_event.is_set():
                raise InferenceLockTimeout("Inference wait was cancelled")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            acquired_thread = self._state.lock.acquire(
                timeout=min(self.poll_interval_s, remaining)
            )
        if not acquired_thread:
            raise InferenceLockTimeout(
                f"Timed out waiting for in-process inference lock after {timeout:.1f}s"
            )

        handle: Any | None = None
        acquired_file = False
        try:
            depth = int(getattr(self._state.local, "depth", 0))
            if depth > 0:
                self._state.local.depth = depth + 1
                try:
                    yield
                finally:
                    self._state.local.depth -= 1
                return

            self.lock_path.parent.mkdir(parents=True, exist_ok=True)
            handle = self.lock_path.open("a+b")
            _prepare_lock_file(handle)
            while not acquired_file:
                if cancel_event is not None and cancel_event.is_set():
                    raise InferenceLockTimeout("Inference wait was cancelled")
                try:
                    acquired_file = _try_lock(handle)
                except OSError as exc:
                    raise OSError(
                        f"Could not acquire inference lock {self.lock_path}: {exc}"
                    ) from exc
                if acquired_file:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise InferenceLockTimeout(
                        f"Timed out waiting for shared inference lock after {timeout:.1f}s"
                    )
                time.sleep(min(self.poll_interval_s, remaining))

            self._state.local.depth = 1
            try:
                yield
            finally:
                self._state.local.depth = 0
        finally:
            if acquired_file and handle is not None:
                try:
                    _unlock(handle)
                except OSError:
                    # Closing the descriptor also releases the advisory lock.
                    pass
            if handle is not None:
                handle.close()
            self._state.lock.release()

    def run(
        self,
        operation: Callable[..., T],
        /,
        *args: Any,
        lease_timeout_s: float | None = None,
        cancel_event: threading.Event | None = None,
        **kwargs: Any,
    ) -> T:
        with self.lease(timeout_s=lease_timeout_s, cancel_event=cancel_event):
            return operation(*args, **kwargs)

    @staticmethod
    def _prompt_text(messages: Sequence[Mapping[str, Any]]) -> str:
        pieces: list[str] = []
        for item in messages:
            role = str(item.get("role", ""))
            content = item.get("content", "")
            pieces.append(
                f"[{role}]\n{content if isinstance(content, str) else repr(content)}"
            )
        return "\n\n".join(pieces)

    def chat(
        self,
        *,
        role: str,
        messages: Sequence[Mapping[str, Any]],
        prompt_version: str = "v1",
        format: Mapping[str, Any] | str | None = None,
        context_tokens: int | None = None,
        num_predict: int | None = None,
        reasoning: str | None = None,
        temperature: float = 0.0,
        seed: int = 7,
        timeout_s: float | None = None,
        stream: bool = False,
        cancel_event: threading.Event | None = None,
    ) -> BrokerResult:
        """Run one role call and return its generation plus trace record.

        ``timeout_s`` bounds lock waiting and generation together. Failures are
        raised as :class:`BrokerCallError` with the failed call record attached.
        """

        normalized = self._role(role)
        model = self.model_for(normalized)
        role_ceiling = self.output_tokens_for(normalized)
        output = role_ceiling if num_predict is None else num_predict
        if isinstance(output, bool) or not isinstance(output, int):
            raise ValueError("num_predict must be an integer")
        if not 1 <= output <= role_ceiling:
            raise ValueError(
                f"num_predict for {normalized} must be between 1 and {role_ceiling}"
            )
        request_budget = (
            self.config.request_timeout_s if timeout_s is None else float(timeout_s)
        )
        if not math.isfinite(request_budget) or request_budget <= 0:
            raise ValueError("timeout_s must be a positive finite number")
        call_id = f"call_{uuid.uuid4().hex}"
        started = time.perf_counter()
        prompt = self._prompt_text(messages)
        account = self.current_answer_budget
        admission = None
        budget_metrics = {}
        try:
            if account is not None:
                admission = account.admit(role=normalized, prompt_version=prompt_version, requested_tokens=output)
                output = admission.output_ceiling
            with self.lease(
                timeout_s=min(self.config.inference_lock_timeout_s, request_budget),
                cancel_event=cancel_event,
            ):
                remaining = request_budget - (time.perf_counter() - started)
                if remaining <= 0:
                    raise InferenceLockTimeout(
                        "Inference lease consumed the complete request deadline"
                    )
                generation = self._client(model).chat(
                    messages=messages,
                    format=format,
                    context_tokens=context_tokens or self.config.context_tokens,
                    num_predict=output,
                    reasoning=self.config.reasoning if reasoning is None else reasoning,
                    temperature=temperature,
                    seed=seed,
                    keep_alive=self.config.keep_alive,
                    timeout_s=remaining,
                    stream=stream,
                )
        except Exception as exc:
            if admission is not None:
                charged = account.settle(admission, failed=True)
                budget_metrics = {"budget_admitted": True, "budget_output_ceiling": output, "budget_charged_output_tokens": charged}
            elif isinstance(exc, AnswerBudgetExceeded):
                budget_metrics = {"budget_admitted": False, "budget_charged_output_tokens": 0}
            record = ModelCallRecord(
                call_id=call_id,
                role=normalized,
                model=model,
                prompt_version=prompt_version,
                elapsed_s=time.perf_counter() - started,
                status="budget_denied" if isinstance(exc, AnswerBudgetExceeded) else "error",
                metrics=budget_metrics,
                prompt=prompt,
                error=f"{type(exc).__name__}: {exc}",
            )
            raise BrokerCallError(str(exc), record=record) from exc

        if admission is not None:
            charged = account.settle(admission, metrics=generation.metrics)
            budget_metrics = {"budget_admitted": True, "budget_output_ceiling": output, "budget_charged_output_tokens": charged}
        record = ModelCallRecord(
            call_id=call_id,
            role=normalized,
            model=model,
            prompt_version=prompt_version,
            elapsed_s=time.perf_counter() - started,
            status=(
                "partial"
                if (
                    int(generation.metrics.get("stream_incomplete", 0)) == 1
                    or str(generation.metrics.get("done_reason", "")).casefold()
                    in {"length", "max_tokens", "token_limit"}
                )
                else "ok"
            ),
            metrics={**dict(generation.metrics), **budget_metrics},
            reasoning=generation.reasoning,
            raw_output=generation.content,
            prompt=prompt,
        )
        return BrokerResult(generation=generation, call=record)


__all__ = [
    "BrokerCallError",
    "BrokerResult",
    "InferenceBroker",
    "InferenceLockTimeout",
    "RoleName",
]
