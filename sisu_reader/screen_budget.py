"""Conservative screen-prompt budgeting without a model-specific tokenizer.

This is a scheduling estimate, not an exact token count. Typed server overflow
errors can trigger bounded splitting when a tokenizer is less favorable.
"""
from __future__ import annotations

import json
import math
import re
from typing import Mapping, Sequence

from .ollama import OllamaHTTPError

_PIECES = re.compile(r"[A-Za-z0-9]+|\s+|[^A-Za-z0-9\s]")


def conservative_tokens(text: str) -> int:
    tokens = 0
    for piece in _PIECES.findall(str(text)):
        if piece.isascii() and piece.isalnum():
            # Random IDs/numeric literals often split much more than prose.
            tokens += len(piece) if any(c.isdigit() for c in piece) else math.ceil(len(piece) / 4)
        elif piece.isspace():
            tokens += max(1, math.ceil(len(piece.encode("utf-8")) / 4))
        else:
            # Count non-ASCII bytes and punctuation rather than assuming four
            # characters per token for CJK, emoji, combining marks or code.
            tokens += len(piece.encode("utf-8"))
    return tokens


def screen_prompt_cost(messages: Sequence[Mapping[str, str]]) -> int:
    # Roles/control tokens depend on the model's chat template. Keep an
    # explicit allowance in addition to the serialized content being counted.
    return 128 + sum(conservative_tokens(message.get("content", "")) for message in messages)


def screen_prompt_limit(config) -> int:
    return max(0, min(int(config.screen_batch_token_budget),
                      int(config.context_tokens) - int(config.screen_output_tokens) - 256))


def is_context_overflow(error: BaseException) -> bool:
    """Recognize a structured Ollama context rejection, never a generic 400."""
    seen = set()
    current = error
    for _ in range(5):
        if current is None or id(current) in seen:
            break
        seen.add(id(current))
        if isinstance(current, OllamaHTTPError) and current.status == 400:
            value = current.body
            for _ in range(5):
                if isinstance(value, str):
                    try:
                        value = json.loads(value)
                    except (ValueError, TypeError):
                        break
                if not isinstance(value, dict):
                    break
                if value.get("type") in {"exceed_context_size_error", "context_length_exceeded"}:
                    return True
                prompt, capacity = value.get("n_prompt_tokens"), value.get("n_ctx")
                if (type(prompt) is int and type(capacity) is int and prompt > capacity > 0):
                    return True
                value = value.get("error")
        current = current.__cause__
    return False
