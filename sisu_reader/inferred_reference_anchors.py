"""Conservative numbered-heading anchors for coarsely sectioned source blocks.

This never mutates ingestion or fetches a source. Inference requires an authorized,
revision-bound block view and a coherent numbered heading family; ambiguous
duplicates stay separate so ReferenceIndex can refuse ambiguous resolution.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from bisect import bisect_left, bisect_right
from collections.abc import Sequence

from .models import Section, SourceBlock

_HEADING = re.compile(r"^(?P<label>\d{1,4}(?:\.\d{1,4}){0,5})\.?\s+(?P<title>[^\n\r]{1,160})$")
_TOC = re.compile(r"\.{2,}|\s\d+\s*$")


def infer_numbered_sections(
    blocks: Sequence[SourceBlock], sections: Sequence[Section], *,
    max_nodes: int, max_scan_chars: int,
) -> tuple[tuple[Section, ...], tuple[str, ...]]:
    by_section = {section.section_id: section for section in sections}
    candidates: list[tuple[SourceBlock, tuple[int, ...], str]] = []
    issues: list[str] = []
    scanned = 0
    scan_end = -1
    for number, block in enumerate(blocks):
        if number >= max_nodes or scanned + len(block.text) > max_scan_chars:
            issues.append("traversal_limit:inferred_heading_index")
            break
        scanned += len(block.text)
        scan_end = block.ordinal
        text = unicodedata.normalize("NFKC", block.text).strip()
        if len(text) > 180 or "\n" in text or "\r" in text:
            continue
        match = _HEADING.fullmatch(text)
        if not match:
            continue
        title = match["title"]
        if (_TOC.search(title) or len(title.split()) > 18
                or any(char in title for char in ";!?[]{}=")
                or title.endswith(".") or not any(char.isalpha() for char in title)):
            continue
        # A damaged or foreign source block cannot create an apparent target.
        if hashlib.sha256(block.text.encode("utf-8")).hexdigest() != block.text_sha256:
            issues.append(f"invalid_source_binding:inferred_heading:{block.block_id}")
            continue
        owner = by_section.get(block.section_id)
        if owner and _HEADING.fullmatch(unicodedata.normalize("NFKC", owner.heading).strip()):
            continue  # existing numbered structural metadata is authoritative
        candidates.append((block, tuple(int(part) for part in match["label"].split(".")), text))

    # Consecutive sibling headings with no intervening body are a catalogue
    # pattern even when extraction omitted dot leaders and page numbers.
    catalogue_ids: set[str] = set()
    run: list[tuple[SourceBlock, tuple[int, ...], str]] = []
    for candidate in candidates:
        if run and (candidate[0].ordinal != run[-1][0].ordinal + 1
                    or candidate[0].section_id != run[-1][0].section_id
                    or candidate[1][:-1] != run[-1][1][:-1]):
            if len(run) >= 3:
                catalogue_ids.update(value[0].block_id for value in run)
            run = []
        run.append(candidate)
    if len(run) >= 3:
        catalogue_ids.update(value[0].block_id for value in run)
    candidates = [value for value in candidates if value[0].block_id not in catalogue_ids]

    # A plain list of single-level numbered sentences is not section metadata.
    # Require a nested heading with a same-parent sibling, or an explicit parent
    # and child. The evidence must occur within the same stored coarse section.
    families: dict[str | None, set[tuple[int, ...]]] = {}
    for block, label, _ in candidates:
        families.setdefault(block.section_id, set()).add(label)
    coherent: set[str | None] = set()
    for owner_id, labels in families.items():
        parent_counts: dict[tuple[int, ...], int] = {}
        for label in labels:
            if len(label) > 1:
                parent_counts[label[:-1]] = parent_counts.get(label[:-1], 0) + 1
        if any(parent in labels or count > 1 for parent, count in parent_counts.items()):
            coherent.add(owner_id)
    accepted: list[tuple[SourceBlock, tuple[int, ...], str]] = []
    for candidate in candidates:
        block, label, _ = candidate
        if block.section_id in coherent:
            accepted.append(candidate)
    if not accepted:
        return (), tuple(dict.fromkeys(issues))
    ordinals = tuple(block.ordinal for block in blocks)
    invalid_prefix, token_prefix = [0], [0]
    for block in blocks:
        if block.ordinal > scan_end:
            break
        invalid_prefix.append(invalid_prefix[-1] + int(
            hashlib.sha256(block.text.encode('utf-8')).hexdigest() != block.text_sha256))
        token_prefix.append(token_prefix[-1] + block.token_estimate)
    inferred: list[Section] = []
    for i, (block, label, title) in enumerate(accepted):
        owner = by_section.get(block.section_id)
        end = min(scan_end, owner.last_block_ordinal if owner else scan_end)
        for following, next_label, _ in accepted[i + 1:]:
            if following.ordinal > end:
                break
            if (following.section_id != block.section_id or len(next_label) <= len(label)
                    or next_label[:len(label)] != label):
                end = following.ordinal - 1
                break
        # Verify the complete target interval, not only its heading. Invalid
        # body text must not be silently admitted as a closed dependency.
        first, last = bisect_left(ordinals, block.ordinal), bisect_right(ordinals, end)
        if invalid_prefix[last] != invalid_prefix[first]:
            issues.append(f"invalid_source_binding:inferred_target:{block.block_id}")
            continue
        inferred.append(Section(
            "inferred:" + block.block_id, block.document_revision_id,
            block.section_id, block.ordinal, len(label), title, title,
            block.locator, block.ordinal, end,
            token_prefix[last] - token_prefix[first],
        ))
    return tuple(inferred), tuple(dict.fromkeys(issues))
