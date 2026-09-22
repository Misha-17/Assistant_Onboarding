"""Keep attributed source conditions when a generated conclusion has reference debt.

No semantic entailment is inferred. Only complete hash-verified source blocks from
cards already cited in the affected unit may replace its unsafe generated prose.
"""
from __future__ import annotations

import hashlib
import html
import re
from collections.abc import Mapping, Sequence

from .models import EvidenceCard, SourceBlock

_MARKER = re.compile(r"\[E:([A-Za-z0-9_.:-]+)\]")
_SEPARATOR = re.compile(r"\n\s*\n|\n(?=\s*(?:[-*]|\d+\.)\s)")
_MAX_BLOCK_CHARS = 4000
_MAX_TOTAL_CHARS = 8000
_MAX_BLOCKS = 6
_NOTE = "Referenced details remain unverified; conclusions depending on them remain open."


def _quoted(text: str, *, prose: bool = False) -> str:
    # Source-authored markers are data, never new citation capabilities. Keep
    # embedded newlines visibly escaped so they cannot create new answer units.
    # Plain prose is commonly hard-wrapped by TXT/PDF extraction. Reflow only
    # whitespace in prose; code/table layout is retained as visible escapes.
    if prose:
        text = " ".join(text.split())
    value = html.escape(text, quote=False)
    # Escape every wrapper accepted by evidence-marker normalization. Source
    # text can pass through the final normalizer after this quotation is built;
    # its marker-like data must never become a new citation capability.
    for wrapper in "[]【】［］":
        value = value.replace(wrapper, f"&#{ord(wrapper)};")
    value = value.replace("\\", "\\\\").replace("\r", "\\r").replace("\n", "\\n")
    for symbol in ("`", "*", "_", "~"):
        value = value.replace(symbol, "\\" + symbol)
    return "The source states: “" + value + "”"


def qualify_reference_units(
    text: str,
    cards: Sequence[EvidenceCard],
    *,
    source_blocks: Mapping[str, SourceBlock] | None = None,
) -> tuple[str, tuple[str, ...]]:
    by_id = {card.card_id: card for card in cards}
    unresolved = {card.card_id for card in cards if card.reference_issues}
    if not unresolved:
        return text, ()
    sources = source_blocks or {}
    # Independent and unresolved cards can share a seed block but differ in
    # attached conditions. Do not let a closed card consume the unresolved
    # card's qualifier while deduplicating repeated source spans.
    emitted: set[tuple[str, str, bool]] = set()
    refused: set[str] = set()
    output: list[str] = []
    warnings: list[str] = []
    used_chars = 0
    for unit in _SEPARATOR.split(text):
        cited = tuple(dict.fromkeys(_MARKER.findall(unit)))
        affected = tuple(cid for cid in cited if cid in unresolved)
        if not affected:
            if unit.strip():
                output.append(unit.strip())
            continue
        warnings.append("reference_unresolved_card:answer_unit_qualified")
        for cid in cited:
            card = by_id.get(cid)
            if card is None:
                continue
            blocks = tuple(sources.get(bid) for bid in card.block_ids)
            valid = bool(blocks) and all(
                block is not None and block.block_id == bid
                and block.document_revision_id == card.document_revision_id
                and hashlib.sha256(block.text.encode("utf-8")).hexdigest() == block.text_sha256
                and 0 < len(block.text.strip()) <= _MAX_BLOCK_CHARS
                for bid, block in zip(card.block_ids, blocks)
            )
            fresh = tuple({block.block_id: block for block in blocks if block is not None
                           and (card.document_revision_id, block.block_id, cid in unresolved) not in emitted}.values())
            cost = sum(len(block.text) for block in fresh)
            if valid and used_chars + cost <= _MAX_TOTAL_CHARS and len(emitted) + len(fresh) <= _MAX_BLOCKS:
                for block in fresh:
                    quote = _quoted(block.text, prose=block.kind in {"prose", "paragraph"}) + f" [E:{cid}]"
                    if cid in unresolved:
                        quote += " " + _NOTE
                    output.append(quote)
                    emitted.add((card.document_revision_id, block.block_id, cid in unresolved))
                used_chars += cost
                warnings.append("reference_unresolved_card:complete_source_excerpt_retained")
            elif cid in unresolved and cid not in refused:
                output.append("A referenced condition needed for this point is unresolved; "
                              f"I cannot give a definite conclusion. [E:{cid}]")
                refused.add(cid)
                warnings.append("reference_unresolved_card:source_excerpt_unavailable")
    return "\n\n".join(output), tuple(dict.fromkeys(warnings))
