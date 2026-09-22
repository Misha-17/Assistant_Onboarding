"""Normalize bounded citation syntax against the current evidence-card vocabulary.

This recognizes formatting, not factual support. It never discovers a source,
guesses a similar ID, or grants authority to an ID the model was not supplied.
Source membership, hashes, authorization and reference qualification still apply.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
import re

_ID = re.compile(r"[A-Za-z0-9_.:-]{1,180}\Z")
_SHORT = re.compile(r"C[0-9]{3}\Z")
_PREFIX = re.compile(r"(.+-)C[0-9]{3}\Z")
_WRAPPED = re.compile(r"(?<!\\)(?:\[(E:[^\[\]【】［］\r\n]{1,2048})\]|【(E:[^\[\]【】［］\r\n]{1,2048})】|［(E:[^\[\]【】［］\r\n]{1,2048})］)")
_MAX_GROUP = 32


@dataclass(frozen=True, slots=True)
class MarkerNormalization:
    text: str
    warnings: tuple[str, ...] = ()


def normalize_evidence_markers(text: str, allowed_card_ids: Sequence[str]) -> MarkerNormalization:
    """Accept exact known IDs and one explicit, unambiguous group prefix.

    A shorthand C002 is accepted only inside a group that explicitly names a
    known full *-Cnnn ID, all full IDs share one prefix, and the expanded C002
    also exists in this prompt's unique vocabulary. Unknown or ambiguous members
    reject the entire conversion. Canonical unknown markers remain available to
    the existing binder's rejection/diagnostic path. No bare [C002] is inferred.
    """
    counts = Counter(allowed_card_ids)
    allowed = {cid for cid, count in counts.items() if count == 1 and _ID.fullmatch(cid)}
    warnings: list[str] = []

    def convert(match: re.Match[str]) -> str:
        original = match.group(0)
        body = next(value for value in match.groups() if value is not None)
        parts = [part.strip() for part in body.split(",")]
        if len(parts) > _MAX_GROUP or not all(parts):
            warnings.append("citation_marker_group_rejected:invalid_size")
            return original
        full_ids, shorthand = [], []
        for part in parts:
            if part.startswith("E:"):
                cid = part[2:]
                if not _ID.fullmatch(cid) or cid not in allowed:
                    warnings.append("citation_marker_group_rejected:unknown_or_duplicate_id")
                    return original
                full_ids.append(cid)
            elif _SHORT.fullmatch(part):
                shorthand.append(part)
            else:
                warnings.append("citation_marker_group_rejected:unsupported_member")
                return original
        prefix = None
        if shorthand:
            prefixes = {_PREFIX.fullmatch(cid).group(1) if _PREFIX.fullmatch(cid) else None for cid in full_ids}
            if len(prefixes) != 1 or None in prefixes:
                warnings.append("citation_marker_group_rejected:ambiguous_prefix")
                return original
            prefix = next(iter(prefixes))
        expanded = [part[2:] if part.startswith("E:") else prefix + part for part in parts]
        if any(cid not in allowed for cid in expanded):
            warnings.append("citation_marker_group_rejected:unavailable_expansion")
            return original
        canonical = "".join(f"[E:{cid}]" for cid in dict.fromkeys(expanded))
        if canonical != original:
            warnings.append("citation_marker_format_normalized")
        return canonical

    return MarkerNormalization(_WRAPPED.sub(convert, text), tuple(dict.fromkeys(warnings)))
