"""Recover literal source evidence when a generated multi-span claim is rejected.

Each recovered card states one complete immutable source block. This does not
validate the discarded claim, infer a relation between spans, or establish that
the selected passage answers the question. Existing source scope/reference
checks and final evidence review still apply.
"""
from __future__ import annotations
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib

from .models import SourceBlock

MAX_RECOVERED_CARDS = 6
MAX_RECOVERED_CHARS = 8000
MAX_SOURCE_BLOCK_CHARS = 4000
MAX_SELECTED_BLOCKS = 4


@dataclass(frozen=True, slots=True)
class SourceExcerpt:
    block_id: str
    text: str
    role: str


def recover_source_excerpts(
    block_ids: Sequence[str], presented: Mapping[str, SourceBlock], *,
    document_revision_id: str, requested_role: str,
    already_recovered: frozenset[str] = frozenset(),
    remaining_cards: int = MAX_RECOVERED_CARDS,
    remaining_chars: int = MAX_RECOVERED_CHARS,
) -> tuple[tuple[SourceExcerpt, ...], tuple[str, ...]]:
    """Bounded whole-block recovery. Never trim a selected block to fit a limit.

    Unknown/mutated/cross-document selections reject the entire recovery request.
    Code, table, OCR, counter and operand selections remain context; a discarded
    model claim cannot turn an example into a rule or a quote into a calculation.
    A direct prose excerpt is literal source evidence only, not a relevance or
    entailment verdict. Source-authored commands remain untrusted data.
    """
    ids = tuple(dict.fromkeys(block_ids))
    if not ids or len(ids) > MAX_SELECTED_BLOCKS:
        return (), ("source_excerpt_selection_limit",)
    if remaining_cards < 0 or remaining_chars < 0:
        raise ValueError("Recovery budgets cannot be negative")
    if requested_role not in {"direct", "context", "counter", "operand"}:
        return (), ("source_excerpt_invalid_role",)
    blocks = [presented.get(bid) for bid in ids]
    if any(block is None or block.block_id != bid or block.document_revision_id != document_revision_id
           or not block.text.strip() or hashlib.sha256(block.text.encode("utf-8")).hexdigest() != block.text_sha256
           for bid, block in zip(ids, blocks)):
        return (), ("source_excerpt_unverified_selection",)
    output, warnings = [], []
    used = 0
    for block in sorted(blocks, key=lambda block: (block.ordinal, block.block_id)):
        if block.block_id in already_recovered:
            continue
        if len(block.text) > MAX_SOURCE_BLOCK_CHARS:
            warnings.append("source_excerpt_whole_block_too_large")
            continue
        if len(output) >= min(MAX_RECOVERED_CARDS, remaining_cards) or used + len(block.text) > min(MAX_RECOVERED_CHARS, remaining_chars):
            warnings.append("source_excerpt_recovery_budget_exhausted")
            continue
        fallible_or_example = any(flag in {"html_preformatted", "ocr_text_unverified"} for flag in block.extraction_flags)
        literal_direct = requested_role == "direct" and block.kind in {"prose", "paragraph", "list_item"} and not fallible_or_example
        output.append(SourceExcerpt(block.block_id, block.text, "direct" if literal_direct else "context"))
        used += len(block.text)
    return tuple(output), tuple(dict.fromkeys(warnings))
