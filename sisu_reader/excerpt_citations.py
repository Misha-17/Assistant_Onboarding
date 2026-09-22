"""Bind an attributed whole-block quotation to its actual source span.

Reference closure is unchanged. A generated dependent conclusion still needs its
full context. Only an entire output unit equal to the controller's escaped source
quotation, original E-card marker, and required debt note receives a narrower
per-occurrence citation span. There is no new marker a model can forge.
"""
from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Mapping, Sequence

from .models import EvidenceCard, SourceBlock
from .reference_units import (
    _MARKER, _MAX_BLOCKS, _MAX_BLOCK_CHARS, _MAX_TOTAL_CHARS, _NOTE, _SEPARATOR, _quoted,
)


def source_excerpt_overrides(
    text: str,
    cards: Sequence[EvidenceCard],
    *,
    source_blocks: Mapping[str, SourceBlock],
) -> dict[int, tuple[str, ...]]:
    """Return source IDs for exact E-marker positions, never for a whole card.

    The caller supplies the authorized immutable view and still performs normal
    citation membership, revision, hash and final authorization checks. An
    absent/damaged source or unrecognized unit returns no override. At most the
    source-excerpt qualifier's existing six-block/eight-thousand-character
    envelope is recognized; ordinary citations are never truncated to that cap.
    """
    counts = Counter(card.card_id for card in cards)
    by_id = {card.card_id: card for card in cards if counts[card.card_id] == 1}
    overrides: dict[int, tuple[str, ...]] = {}
    recognized: set[tuple[str, str, bool]] = set()
    used_chars = 0
    boundaries = [(match.start(), match.end()) for match in _SEPARATOR.finditer(text)]
    begin = 0
    for end, next_begin in [*boundaries, (len(text), len(text))]:
        raw = text[begin:end]
        unit = raw.strip()
        unit_start = begin + len(raw) - len(raw.lstrip())
        begin = next_begin
        markers = list(_MARKER.finditer(unit))
        if len(markers) != 1:
            continue
        marker = markers[0]
        card = by_id.get(marker.group(1))
        if card is None or not card.block_ids:
            continue
        blocks = tuple(source_blocks.get(bid) for bid in card.block_ids)
        if not all(
            block is not None and block.block_id == bid
            and block.document_revision_id == card.document_revision_id
            and hashlib.sha256(block.text.encode("utf-8")).hexdigest() == block.text_sha256
            and 0 < len(block.text.strip()) <= _MAX_BLOCK_CHARS
            for bid, block in zip(card.block_ids, blocks)
        ):
            continue
        for block in blocks:
            assert block is not None  # all members were checked above
            expected = _quoted(block.text, prose=block.kind in {"prose", "paragraph"}) + f" [E:{card.card_id}]"
            if card.reference_issues:
                expected += " " + _NOTE
            if unit != expected:
                continue
            key = (card.document_revision_id, block.block_id, bool(card.reference_issues))
            if key not in recognized:
                if len(recognized) >= _MAX_BLOCKS or used_chars + len(block.text) > _MAX_TOTAL_CHARS:
                    continue
                recognized.add(key)
                used_chars += len(block.text)
            overrides[unit_start + marker.start()] = (block.block_id,)
            break
    return overrides
