from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .config import Config
from .models import jsonable


_MAX_STRING = 600_000
_MAX_ITEMS = 20_000


def _bounded(value: Any, *, depth: int = 0) -> Any:
    if depth > 20:
        return "<depth-limit>"
    value = jsonable(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value if len(value) <= _MAX_STRING else value[:_MAX_STRING] + "\n<truncated>"
    if isinstance(value, dict):
        return {
            str(key)[:200]: _bounded(item, depth=depth + 1)
            for key, item in list(value.items())[:_MAX_ITEMS]
        }
    if isinstance(value, (list, tuple)):
        return [_bounded(item, depth=depth + 1) for item in value[:_MAX_ITEMS]]
    return str(value)[:_MAX_STRING]


def _without_private_call_text(payload: dict[str, Any], *, keep_answer: bool) -> dict[str, Any]:
    result = dict(payload)
    calls: list[dict[str, Any]] = []
    for raw in result.get("model_calls", ()):
        if not isinstance(raw, dict):
            continue
        call = dict(raw)
        for key in ("prompt", "raw_output", "reasoning", "raw_message"):
            call.pop(key, None)
        calls.append(call)
    result["model_calls"] = calls
    for key in ("reader_packets", "raw_sources", "raw_outputs", "reasoning"):
        result.pop(key, None)
    for key in (
        "screening",
        "reader_reports",
        "evidence_cards",
        "synthesis",
        "citation_checks",
        "exact_lane",
    ):
        result.pop(key, None)
    if not keep_answer:
        result.pop("request", None)
        result.pop("answer", None)
        result.pop("citations", None)
    return result


def _metrics_only(payload: dict[str, Any]) -> dict[str, Any]:
    """Retain operational numbers/codes without questions, answers, or source text."""

    stripped = _without_private_call_text(payload, keep_answer=False)
    for key in ("warnings", "errors"):
        values = stripped.get(key)
        if isinstance(values, (list, tuple)):
            stripped[key] = [str(item).split(":", 1)[0][:160] for item in values]
    calls = stripped.get("model_calls")
    if isinstance(calls, list):
        for call in calls:
            if isinstance(call, dict) and call.get("error"):
                call["error"] = str(call["error"]).split(":", 1)[0][:160]
    coverage = stripped.get("coverage")
    if isinstance(coverage, dict):
        stripped["coverage"] = {
            str(key): value
            for key, value in coverage.items()
            if isinstance(value, (bool, int, float)) or key in {"mode", "snapshot_id"}
        }
    timeline = stripped.get("timeline")
    if isinstance(timeline, (list, tuple)):
        stripped["timeline"] = [
            {
                key: item[key]
                for key in ("event_seq", "occurred_at", "event_type", "work_id")
                if isinstance(item, dict) and key in item
            }
            for item in timeline
            if isinstance(item, dict)
        ]
    stripped.pop("corpus", None)
    # Discovery queries are derived from the user's question and can contain
    # names, project codes, or other private text. Metrics mode retains only
    # numeric/boolean navigation telemetry.
    discovery = stripped.get("discovery")
    if isinstance(discovery, dict):
        stripped["discovery"] = {
            str(key): value
            for key, value in discovery.items()
            if isinstance(value, (bool, int, float))
        }
    return stripped


class TraceStore:
    """Writes immutable local traces with explicit privacy modes."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.last_error = ""

    def write(self, run_id: str, payload: dict[str, Any]) -> Path | None:
        mode = self.config.trace_mode
        if mode == "off":
            return None
        trace = dict(payload)
        trace.setdefault("schema_version", 1)
        trace.setdefault("pipeline", "authorize-route-screen-read-synthesize-v2")
        trace.setdefault("run_id", run_id)
        trace.setdefault("created_at", datetime.now(timezone.utc).isoformat())
        trace["trace_mode"] = mode
        if mode == "metrics":
            trace = _metrics_only(trace)
        elif mode == "answer":
            trace = _without_private_call_text(trace, keep_answer=True)
        elif not self.config.trace_reasoning:
            trace.pop("reasoning", None)
            calls = []
            for raw in trace.get("model_calls", ()):
                call = dict(raw) if isinstance(raw, dict) else {"value": raw}
                call.pop("reasoning", None)
                calls.append(call)
            trace["model_calls"] = calls

        safe = _bounded(trace)
        now = datetime.now(timezone.utc)
        day = self.config.trace_dir / now.strftime("%Y-%m-%d")
        day.mkdir(parents=True, exist_ok=True)
        safe_run = "".join(ch for ch in run_id if ch.isalnum() or ch in "_-")[:80] or "run"
        target = day / f"{now.strftime('%Y%m%dT%H%M%S.%fZ')}-{safe_run}.json"
        temporary = day / f".{target.name}.{os.getpid()}.tmp"
        try:
            encoded = json.dumps(safe, ensure_ascii=False, indent=2).encode("utf-8")
            with temporary.open("wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
            self.cleanup()
            self.last_error = ""
            return target
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            return None

    def cleanup(self) -> None:
        root = self.config.trace_dir
        if not root.exists():
            return
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.config.trace_retention_days)
        entries: list[tuple[Path, float, int]] = []
        for path in root.rglob("*.json"):
            try:
                stat = path.stat()
            except OSError:
                continue
            entries.append((path, stat.st_mtime, stat.st_size))
            if datetime.fromtimestamp(stat.st_mtime, timezone.utc) < cutoff:
                try:
                    path.unlink()
                except OSError:
                    pass
        entries = [item for item in entries if item[0].exists()]
        maximum = self.config.trace_max_total_mb * 1024 * 1024
        total = sum(size for _, _, size in entries)
        for path, _, size in sorted(entries, key=lambda item: item[1]):
            if total <= maximum:
                break
            try:
                path.unlink()
                total -= size
            except OSError:
                continue
        for directory in sorted(root.rglob("*"), reverse=True):
            if directory.is_dir():
                try:
                    directory.rmdir()
                except OSError:
                    pass


__all__ = ["TraceStore"]
