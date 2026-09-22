from __future__ import annotations

import ipaddress
import json
import math
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .config import Config
from .models import GenerationResult


class OllamaError(RuntimeError):
    """Base error for local Ollama transport and protocol failures."""


class OllamaHTTPError(OllamaError):
    """An HTTP response from Ollama was not successful."""

    def __init__(self, status: int, body: str, *, path: str) -> None:
        self.status = int(status)
        self.body = body[:8192]
        self.path = path
        suffix = f": {self.body}" if self.body else ""
        super().__init__(f"Ollama HTTP {self.status} for {path}{suffix}")


class OllamaProtocolError(OllamaError):
    """Ollama returned data that did not match its native protocol."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: Any,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        # A loopback endpoint must not be able to forward document text elsewhere.
        return None


@dataclass(frozen=True, slots=True)
class OllamaStatus:
    version: str
    model: str


_REASONING_LEVELS = {"off": 0, "low": 1, "medium": 2, "high": 3}
_DURATION_FIELDS = (
    "total_duration",
    "load_duration",
    "prompt_eval_duration",
    "eval_duration",
)
_COUNT_FIELDS = ("prompt_eval_count", "eval_count")
_MAX_RESPONSE_BYTES = 64 * 1024 * 1024


def _canonical_model(name: str) -> str:
    clean = name.strip()
    return clean[:-7] if clean.endswith(":latest") else clean


def _is_gpt_oss_model(name: str) -> bool:
    leaf = _canonical_model(name).rsplit("/", 1)[-1]
    return leaf.split(":", 1)[0].casefold() == "gpt-oss"


def _validated_loopback_url(value: str) -> str:
    """Return a normalized loopback-only Ollama origin or fail closed."""

    if not isinstance(value, str) or not value.strip():
        raise OllamaError("Ollama URL cannot be empty")
    try:
        parsed = urllib.parse.urlsplit(value.strip())
        port = parsed.port
        hostname = parsed.hostname
    except (UnicodeError, ValueError) as exc:
        raise OllamaError("Ollama URL is malformed") from exc
    if parsed.scheme.casefold() not in {"http", "https"}:
        raise OllamaError("Ollama URL must use http or https")
    if not parsed.netloc or not hostname:
        raise OllamaError("Ollama URL must include a loopback host")
    if parsed.username is not None or parsed.password is not None:
        raise OllamaError("Ollama URL must not contain credentials")
    if parsed.query or parsed.fragment:
        raise OllamaError("Ollama URL must not contain a query or fragment")
    if parsed.path not in {"", "/"}:
        raise OllamaError("Ollama URL must be an origin without an API path")

    host = hostname.casefold()
    local = host == "localhost"
    if not local:
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            address = None
        local = bool(address is not None and address.is_loopback)
    if not local:
        raise OllamaError(
            "Refusing non-local Ollama URL; use localhost or a loopback IP address"
        )

    rendered_host = f"[{host}]" if ":" in host else host
    rendered_port = f":{port}" if port is not None else ""
    return f"{parsed.scheme.casefold()}://{rendered_host}{rendered_port}"


def _json_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(value)


def _reasoning_text(message: Mapping[str, Any]) -> str:
    pieces: list[str] = []
    for key in ("thinking", "reasoning"):
        value = message.get(key)
        if isinstance(value, str) and value and value not in pieces:
            pieces.append(value)
    return "\n".join(pieces)


def _metrics(
    response: Mapping[str, Any], *, wall_s: float
) -> dict[str, int | float | str]:
    metrics: dict[str, int | float | str] = {"wall_s": float(wall_s)}
    for key in _DURATION_FIELDS:
        value = response.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            metrics[f"{key}_ns"] = value
            metrics[f"{key.removesuffix('_duration')}_s"] = value / 1_000_000_000
    for key in _COUNT_FIELDS:
        value = response.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            metrics[key] = value
    for key in ("model", "created_at", "done_reason"):
        value = response.get(key)
        if isinstance(value, str):
            metrics[key] = value
    if isinstance(response.get("done"), bool):
        metrics["done"] = int(response["done"])
    eval_count = metrics.get("eval_count")
    eval_s = metrics.get("eval_s")
    if isinstance(eval_count, int) and isinstance(eval_s, float) and eval_s > 0:
        metrics["eval_tokens_per_s"] = eval_count / eval_s
    return metrics


class OllamaClient:
    """Bounded native Ollama client with a loopback-only privacy boundary.

    The client has no writer/verifier concept. The broker chooses a model and
    output budget for each role. Streaming is supported because record-based
    roles can safely salvage complete JSONL records from a partial response.
    """

    def __init__(
        self, config: Config | None = None, *, model: str | None = None
    ) -> None:
        self.config = config or Config.load()
        self.model = (model or self.config.model).strip()
        if not self.model:
            raise ValueError("Ollama model cannot be empty")
        self._base_url = _validated_loopback_url(self.config.ollama_url)
        # Do not inherit HTTP(S)_PROXY, and never follow a redirect.
        self._opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}),
            _NoRedirect(),
        )
        self._ready: OllamaStatus | None = None
        self._ready_lock = threading.Lock()

    _is_gpt_oss_model = staticmethod(_is_gpt_oss_model)

    @property
    def num_predict_ceiling(self) -> int:
        configured = (
            getattr(self.config, "screen_output_tokens", 0),
            getattr(self.config, "reader_output_tokens", 0),
            getattr(self.config, "synthesis_output_tokens", 0),
            getattr(self.config, "review_output_tokens", 0),
            # Legacy configurations used this name as a per-call cap.
            # The bounded-answer contract instead uses it across all calls.
            getattr(self.config, "max_output_tokens", 0) if not hasattr(self.config, "max_model_calls") else 0,
        )
        return max((int(value) for value in configured if int(value) > 0), default=4096)

    def _url(self, path: str) -> str:
        if not path.startswith("/"):
            raise ValueError("Ollama API path must start with '/'")
        return f"{self._base_url}{path}"

    @staticmethod
    def _read_bounded(response: Any) -> bytes:
        declared = response.headers.get("Content-Length")
        if declared:
            try:
                if int(declared) > _MAX_RESPONSE_BYTES:
                    raise OllamaProtocolError("Ollama response exceeded the size limit")
            except ValueError:
                pass
        data = response.read(_MAX_RESPONSE_BYTES + 1)
        if len(data) > _MAX_RESPONSE_BYTES:
            raise OllamaProtocolError("Ollama response exceeded the size limit")
        return data

    def _open(
        self,
        method: str,
        path: str,
        payload: Mapping[str, Any] | None,
        *,
        timeout_s: float,
    ) -> Any:
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            try:
                data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Ollama request payload is not JSON serializable"
                ) from exc
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            self._url(path), data=data, headers=headers, method=method
        )
        try:
            return self._opener.open(request, timeout=timeout_s)
        except urllib.error.HTTPError as exc:
            try:
                body = exc.read(8193).decode("utf-8", errors="replace")[:8192]
            except OSError:
                body = ""
            raise OllamaHTTPError(exc.code, body, path=path) from exc
        except (urllib.error.URLError, TimeoutError, socket.timeout, OSError) as exc:
            raise OllamaError(f"Ollama request failed for {path}: {exc}") from exc

    def _request_json(
        self,
        method: str,
        path: str,
        payload: Mapping[str, Any] | None = None,
        *,
        timeout_s: float | None = None,
    ) -> Mapping[str, Any]:
        timeout = (
            self.config.request_timeout_s if timeout_s is None else float(timeout_s)
        )
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout_s must be a positive finite number")
        try:
            with self._open(method, path, payload, timeout_s=timeout) as response:
                raw = self._read_bounded(response)
        except OllamaError:
            raise
        except (urllib.error.URLError, TimeoutError, socket.timeout, OSError) as exc:
            raise OllamaError(f"Ollama response failed for {path}: {exc}") from exc
        try:
            decoded = json.loads(raw.decode("utf-8", errors="strict"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise OllamaProtocolError("Ollama returned malformed JSON") from exc
        if not isinstance(decoded, Mapping):
            raise OllamaProtocolError("Ollama returned a non-object response")
        return decoded

    def health(self, *, timeout_s: float | None = None) -> str:
        response = self._request_json("GET", "/api/version", timeout_s=timeout_s)
        version = response.get("version")
        if not isinstance(version, str) or not version.strip():
            raise OllamaProtocolError("Ollama health response omitted its version")
        return version.strip()

    def installed_models(self, *, timeout_s: float | None = None) -> tuple[str, ...]:
        response = self._request_json("GET", "/api/tags", timeout_s=timeout_s)
        models = response.get("models")
        if not isinstance(models, list):
            raise OllamaProtocolError("Ollama model response omitted its model list")
        names: list[str] = []
        for item in models:
            if not isinstance(item, Mapping):
                continue
            value = item.get("name") or item.get("model")
            if isinstance(value, str) and value.strip():
                names.append(value.strip())
        return tuple(names)

    def ensure_ready(self, *, timeout_s: float | None = None) -> OllamaStatus:
        if self._ready is not None:
            return self._ready
        with self._ready_lock:
            if self._ready is not None:
                return self._ready
            budget = self.config.request_timeout_s if timeout_s is None else float(timeout_s)
            if not math.isfinite(budget) or budget <= 0:
                raise ValueError("timeout_s must be a positive finite number")
            deadline = time.monotonic() + budget
            version = self.health(timeout_s=max(0.001, deadline - time.monotonic()))
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise OllamaError("Ollama readiness check exceeded its deadline")
            installed = self.installed_models(timeout_s=remaining)
            wanted = _canonical_model(self.model)
            if wanted not in {_canonical_model(item) for item in installed}:
                available = ", ".join(installed[:30]) or "none"
                if len(installed) > 30:
                    available += f", ... ({len(installed) - 30} more)"
                raise OllamaError(
                    f"Ollama model {self.model!r} is not installed (available: {available})"
                )
            self._ready = OllamaStatus(version=version, model=self.model)
            return self._ready

    def _budgets(
        self,
        context_tokens: int | None,
        num_predict: int | None,
        reasoning: str | None,
    ) -> tuple[int, int, str]:
        context = (
            self.config.context_tokens if context_tokens is None else context_tokens
        )
        output = self.num_predict_ceiling if num_predict is None else num_predict
        configured_reasoning = str(getattr(self.config, "reasoning", "off")).casefold()
        level = (
            configured_reasoning if reasoning is None else reasoning.strip().casefold()
        )
        if isinstance(context, bool) or not isinstance(context, int):
            raise ValueError("context_tokens must be an integer")
        if context < 2048 or context > self.config.context_tokens:
            raise ValueError(
                f"context_tokens must be between 2048 and {self.config.context_tokens}"
            )
        if isinstance(output, bool) or not isinstance(output, int):
            raise ValueError("num_predict must be an integer")
        if output < 1 or output > self.num_predict_ceiling:
            raise ValueError(
                f"num_predict must be between 1 and {self.num_predict_ceiling}"
            )
        if level not in _REASONING_LEVELS:
            raise ValueError("reasoning must be off, low, medium, or high")
        if configured_reasoning not in _REASONING_LEVELS:
            raise ValueError("configured reasoning must be off, low, medium, or high")
        if _REASONING_LEVELS[level] > _REASONING_LEVELS[configured_reasoning]:
            raise ValueError(
                f"reasoning {level!r} exceeds configured ceiling {configured_reasoning!r}"
            )
        return context, output, level

    @staticmethod
    def _messages(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        if not messages:
            raise ValueError("messages cannot be empty")
        result: list[dict[str, Any]] = []
        for message in messages:
            if not isinstance(message, Mapping):
                raise TypeError("every message must be a mapping")
            role = message.get("role")
            if not isinstance(role, str) or not role.strip():
                raise ValueError("every message needs a non-empty role")
            item = dict(message)
            item["role"] = role.strip()
            result.append(item)
        return result

    def _stream_chat(
        self, payload: Mapping[str, Any], *, timeout_s: float
    ) -> tuple[dict[str, Any], Mapping[str, Any], str | None]:
        combined: dict[str, Any] = {}
        content_parts: list[str] = []
        thinking_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: list[Any] = []
        final: Mapping[str, Any] = {}
        stream_error: str | None = None
        consumed = 0
        deadline = time.monotonic() + timeout_s
        try:
            with self._open(
                "POST", "/api/chat", payload, timeout_s=timeout_s
            ) as response:
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        stream_error = "deadline_exceeded"
                        break
                    try:
                        response.fp.raw._sock.settimeout(max(0.001, remaining))
                    except (AttributeError, OSError):
                        pass
                    raw_line = response.readline(_MAX_RESPONSE_BYTES + 1)
                    if not raw_line:
                        break
                    consumed += len(raw_line)
                    if consumed > _MAX_RESPONSE_BYTES:
                        raise OllamaProtocolError(
                            "Ollama response exceeded the size limit"
                        )
                    if not raw_line.strip():
                        continue
                    try:
                        chunk = json.loads(raw_line.decode("utf-8", errors="strict"))
                    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                        if content_parts:
                            stream_error = type(exc).__name__
                            break
                        raise OllamaProtocolError(
                            "Ollama returned malformed stream JSON"
                        ) from exc
                    if not isinstance(chunk, Mapping):
                        if content_parts:
                            stream_error = "non_object_chunk"
                            break
                        raise OllamaProtocolError(
                            "Ollama returned a non-object stream chunk"
                        )
                    error_value = chunk.get("error")
                    if isinstance(error_value, str) and error_value:
                        if content_parts:
                            stream_error = "ollama_error_chunk"
                            break
                        raise OllamaError(
                            f"Ollama streaming generation failed: {error_value[:2000]}"
                        )
                    final = chunk
                    message = chunk.get("message")
                    if isinstance(message, Mapping):
                        for key, value in message.items():
                            if key not in {
                                "content",
                                "thinking",
                                "reasoning",
                                "tool_calls",
                            }:
                                combined[key] = value
                        if isinstance(message.get("content"), str):
                            content_parts.append(message["content"])
                        if isinstance(message.get("thinking"), str):
                            thinking_parts.append(message["thinking"])
                        if isinstance(message.get("reasoning"), str):
                            reasoning_parts.append(message["reasoning"])
                        raw_tools = message.get("tool_calls")
                        if isinstance(raw_tools, list):
                            tool_calls.extend(raw_tools)
                    if chunk.get("done") is True:
                        break
        except (urllib.error.URLError, TimeoutError, socket.timeout, OSError) as exc:
            if not (content_parts or thinking_parts or reasoning_parts):
                raise OllamaError(f"Ollama streaming request failed: {exc}") from exc
            stream_error = type(exc).__name__

        if stream_error is not None and not (
            content_parts or thinking_parts or reasoning_parts
        ):
            raise OllamaError(
                f"Ollama streaming generation ended before producing content: {stream_error}"
            )
        combined["content"] = "".join(content_parts)
        if thinking_parts:
            combined["thinking"] = "".join(thinking_parts)
        if reasoning_parts:
            combined["reasoning"] = "".join(reasoning_parts)
        if tool_calls:
            combined["tool_calls"] = tool_calls
        if final.get("done") is not True and stream_error is None:
            stream_error = "stream_ended_before_done"
        return combined, final, stream_error

    def chat(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] = (),
        format: Mapping[str, Any] | str | None = None,
        context_tokens: int | None = None,
        num_predict: int | None = None,
        reasoning: str | None = None,
        temperature: float = 0.0,
        seed: int = 7,
        keep_alive: str | None = None,
        timeout_s: float | None = None,
        stream: bool = False,
    ) -> GenerationResult:
        """Run one bounded native chat request."""

        timeout = (
            self.config.request_timeout_s if timeout_s is None else float(timeout_s)
        )
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout_s must be a positive finite number")
        request_started = time.monotonic()
        self.ensure_ready(timeout_s=timeout)
        timeout -= time.monotonic() - request_started
        if timeout <= 0:
            raise OllamaError("Ollama readiness check consumed the request deadline")
        context, output, level = self._budgets(context_tokens, num_predict, reasoning)
        if not math.isfinite(float(temperature)) or temperature < 0:
            raise ValueError("temperature must be a non-negative finite number")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")

        gpt_oss = _is_gpt_oss_model(self.model)
        effective_reasoning = "low" if gpt_oss and level == "off" else level
        effective_temperature = 1.0 if gpt_oss else float(temperature)
        options: dict[str, Any] = {
            "num_ctx": context,
            "num_predict": output,
            "temperature": effective_temperature,
            "seed": seed,
        }
        if gpt_oss:
            options["top_p"] = 1.0

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._messages(messages),
            "stream": bool(stream),
            "truncate": False,
            "keep_alive": self.config.keep_alive if keep_alive is None else keep_alive,
            "think": effective_reasoning
            if gpt_oss
            else (False if level == "off" else level),
            "options": options,
        }
        if tools:
            payload["tools"] = [dict(item) for item in tools]
        if format is not None:
            payload["format"] = dict(format) if isinstance(format, Mapping) else format

        started = time.perf_counter()
        stream_error: str | None = None
        if stream:
            message, response, stream_error = self._stream_chat(
                payload, timeout_s=timeout
            )
        else:
            response = self._request_json(
                "POST", "/api/chat", payload, timeout_s=timeout
            )
            error_value = response.get("error")
            if isinstance(error_value, str) and error_value:
                raise OllamaError(f"Ollama generation failed: {error_value[:2000]}")
            raw_message = response.get("message")
            if not isinstance(raw_message, Mapping):
                raise OllamaProtocolError("Ollama chat response omitted its message")
            message = dict(raw_message)

        wall_s = time.perf_counter() - started
        metrics = _metrics(response, wall_s=wall_s)
        metrics.update(
            {
                "context_tokens": context,
                "num_predict": output,
                "reasoning": level,
                "effective_reasoning": effective_reasoning,
                "temperature": effective_temperature,
                "model_family": "gpt-oss" if gpt_oss else "other",
                "streamed": int(bool(stream)),
                "stream_incomplete": int(stream_error is not None),
            }
        )
        if stream_error is not None:
            metrics["stream_error"] = stream_error

        raw_tools = message.get("tool_calls")
        tool_calls = (
            tuple(dict(item) for item in raw_tools if isinstance(item, Mapping))
            if isinstance(raw_tools, list)
            else ()
        )
        return GenerationResult(
            content=_json_text(message.get("content")),
            reasoning=_reasoning_text(message),
            tool_calls=tool_calls,
            metrics=metrics,
            raw_message=dict(message),
        )


__all__ = [
    "OllamaClient",
    "OllamaError",
    "OllamaHTTPError",
    "OllamaProtocolError",
    "OllamaStatus",
]
