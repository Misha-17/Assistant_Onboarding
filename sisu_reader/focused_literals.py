"""Bounded source-grounded literal navigation for verbose research queries.

Candidates are search strings, never approved identities or factual assertions.
Only already-read authorized blocks may ground a fallback literal. The helper
does not retrieve sources, invoke models, or consume any benchmark metadata.
"""
from __future__ import annotations
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import hashlib
import re
import unicodedata

from .identity import exact_surfaces
from .models import SourceBlock


@dataclass(frozen=True, slots=True)
class GroundedLiteral:
    query: str
    kind: str
    source_block_ids: tuple[str, ...]
    source_document_ids: tuple[str, ...]
    observed_document_frequency: int


@dataclass(frozen=True, slots=True)
class LiteralQueryPlan:
    candidates: tuple[GroundedLiteral, ...] = ()
    scanned_blocks: int = 0
    scanned_chars: int = 0
    warnings: tuple[str, ...] = ()


def _normalize(value: str) -> str:
    return ' '.join(unicodedata.normalize('NFKC', value).split()).casefold()


def grounded_literal_queries(
    query: str, source_blocks: Iterable[SourceBlock], *,
    allowed_document_ids: Sequence[str], limit: int = 2,
    max_blocks: int = 2048, max_chars: int = 500_000,
) -> LiteralQueryPlan:
    if not isinstance(query, str) or not query.strip() or len(query) > 4096:
        return LiteralQueryPlan()
    if not 1 <= limit <= 2 or not 1 <= max_blocks <= 2048 or not 1 <= max_chars <= 500_000:
        raise ValueError('Literal fallback limits exceed the bounded contract')
    literals = []
    original = _normalize(query)
    for position, surface in enumerate(exact_surfaces(query)):
        if not 2 <= len(surface) <= 100 or _normalize(surface) == original:
            continue
        has_letter = any(char.isalpha() for char in surface)
        has_digit = any(char.isdigit() for char in surface)
        if has_digit and (has_letter or len(surface) >= 4):
            kind, preference = 'structured_literal', 0
        elif has_letter and any(char in surface for char in '-_./'):
            kind, preference = 'structured_literal', 0
        elif ' ' in surface:
            kind, preference = 'quoted_phrase', 1
        elif has_letter and surface.isupper() and len(surface) <= 16:
            kind, preference = 'acronym_literal', 2
        else:
            continue
        normalized = _normalize(surface)
        literals.append((surface, kind, preference, position,
                         re.compile(r'(?<!\w)' + re.escape(normalized) + r'(?!\w)')))
    if not literals:
        return LiteralQueryPlan()
    allowed = set(allowed_document_ids)
    seen: dict[str, SourceBlock] = {}
    matches = {surface: [] for surface, *_ in literals}
    warnings = []
    scanned_chars = 0
    for block in source_blocks:
        if block.document_revision_id not in allowed:
            continue
        if block.block_id in seen:
            if seen[block.block_id] != block:
                raise ValueError('Conflicting immutable source blocks in literal fallback')
            continue
        if len(seen) >= max_blocks or scanned_chars + len(block.text) > max_chars:
            warnings.append('literal_source_scan_limited')
            break
        seen[block.block_id] = block
        scanned_chars += len(block.text)
        if hashlib.sha256(block.text.encode('utf-8')).hexdigest() != block.text_sha256:
            warnings.append('literal_source_hash_mismatch')
            continue
        text = _normalize(block.text)
        for surface, _, _, _, pattern in literals:
            if pattern.search(text):
                matches[surface].append(block)
    ranked = []
    for surface, kind, preference, position, _ in literals:
        values = matches[surface]
        documents = tuple(dict.fromkeys(block.document_revision_id for block in values))
        if not values:
            continue
        candidate = GroundedLiteral(surface, kind,
            tuple(dict.fromkeys(block.block_id for block in values))[:8], documents[:8], len(documents))
        ranked.append(((preference, len(documents), -len(surface), position), candidate))
    ranked.sort(key=lambda value: value[0])
    return LiteralQueryPlan(tuple(candidate for _, candidate in ranked[:limit]),
                            len(seen), scanned_chars, tuple(dict.fromkeys(warnings)))
