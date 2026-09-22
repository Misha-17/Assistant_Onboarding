"""Bounded, deterministic closure of explicit references within one revision.

This module accepts only the caller's authorized source view. It never retrieves
documents, makes model calls, or treats source prose as instructions. Supported
references are Arabic numbered Section/Clause/section-sign labels, single-letter
or numbered Appendix/Annex labels, short integer lists/ranges, and Markdown
``[^label]`` notes whose definition is in a supplied block. A target section's
stored ordinal interval includes its complete subtree. Coarse sections also
admit hash-bound standalone numbered headings when a coherent hierarchy
supports them; multiline/leader/page-number TOC rows are excluded. Duplicate labels are
ambiguous, even if one happens to be nearer the referring block.

This is a deliberately limited grammar, not a semantic completeness guarantee.
Named/relative references, Roman numerals, implicit exceptions, backward external
qualifiers, HTML anchors, and footnote continuations across separate blocks are
not resolved. Recognized unsupported suffixes and external ``of/in/from``
qualifiers produce issues instead of silently claiming a closed dependency.
"""

from __future__ import annotations

import re
import unicodedata
from bisect import bisect_left, bisect_right
from collections import defaultdict, deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .models import Section, SourceBlock
from .inferred_reference_anchors import infer_numbered_sections


_NUMBER = r"\d{1,9}(?:\.\d{1,9}){0,8}[A-Za-z]?"
_LABEL = rf"(?:{_NUMBER}|[A-Za-z])"
_KIND = r"(?:Sections?\b|Clauses?\b|\u00a7{1,2}|Annex(?:es)?\b|Appendix\b|Appendices\b)"
_REFERENCE = re.compile(
    rf"(?<!\w)(?P<kind>{_KIND})\s*(?P<label>{_LABEL})(?!\w|\.\w)",
    re.IGNORECASE,
)
_LIST_ITEM = re.compile(
    rf"\s*(?P<join>,\s*(?:and\s+|or\s+)?|and\s+|or\s+|&\s*|[-\u2013\u2014]\s*|to\s+|through\s+)"
    rf"(?P<label>{_LABEL})(?!\w|\.\w)",
    re.IGNORECASE,
)
_QUALIFIER = re.compile(r"\s+(?:of|in|from)\s+([^.;\n]+)", re.IGNORECASE)
_LOCAL_QUALIFIER = re.compile(
    r"(?:this|the present)\s+(?:document|policy|handbook|agreement|manual|report)\b",
    re.IGNORECASE,
)
_FOOTNOTE = re.compile(r"\[\^(?P<label>[A-Za-z0-9_.-]{1,64})\]")
_FOOTNOTE_DEFINITION = re.compile(
    r"^\s{0,3}\[\^(?P<label>[A-Za-z0-9_.-]{1,64})\]:", re.MULTILINE
)
_BARE_HEADING = re.compile(rf"^(?P<label>{_NUMBER})\.?(?=\s|[:\-]|$)")
_UNSUPPORTED_SUFFIX = re.compile(r"\s*(?:\([A-Za-z0-9]+\)|\.[A-Za-z]|[-\u2013\u2014]\s*\w)")


@dataclass(frozen=True, slots=True)
class ClosureResult:
    required_block_ids: tuple[str, ...]
    issues: tuple[str, ...] = ()
    edges: tuple[tuple[str, str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class ClosedPacket:
    blocks: tuple[SourceBlock, ...]
    issues: tuple[str, ...] = ()


def _normalized(text: str) -> str:
    return unicodedata.normalize("NFKC", text).strip()


def _key(kind: str, label: str) -> str:
    kind = kind.casefold()
    if kind.startswith("clause"):
        family = "clause"
    elif kind.startswith("annex"):
        family = "annex"
    elif kind.startswith("append"):
        family = "appendix"
    else:
        family = "section"
    return f"{family}:{label.casefold()}"


def _heading_keys(heading: str) -> tuple[str, ...]:
    heading = _normalized(heading).lstrip("# ")
    explicit = _REFERENCE.match(heading)
    if explicit:
        return (_key(explicit["kind"], explicit["label"]),)
    bare = _BARE_HEADING.match(heading)
    if bare:
        # An unqualified numbered heading can be called a section or a clause.
        return (f"section:{bare['label'].casefold()}", f"clause:{bare['label'].casefold()}")
    return ()


def _unique(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


class ReferenceIndex:
    """Index immutable blocks already supplied by an authorized caller.

    Limits bound scanning and fixed-point expansion. Hitting any limit returns
    an explicit issue; a partial traversal must never be advertised as closed.
    Unknown seed IDs and inconsistent input fail immediately.
    """

    def __init__(
        self,
        blocks: Sequence[SourceBlock],
        sections: Sequence[Section],
        *,
        max_references: int = 2048,
        max_nodes: int = 8192,
        max_scan_chars: int = 2_000_000,
    ) -> None:
        if min(max_references, max_nodes, max_scan_chars) < 1:
            raise ValueError("Reference limits must be positive")
        self.max_references = max_references
        self.max_nodes = max_nodes
        self.max_scan_chars = max_scan_chars
        self.blocks = tuple(sorted(blocks, key=lambda item: (item.ordinal, item.block_id)))
        self.by_id = {block.block_id: block for block in self.blocks}
        self._ordinals = tuple(block.ordinal for block in self.blocks)
        self.sections = {section.section_id: section for section in sections}
        if len(self.by_id) != len(self.blocks):
            raise ValueError("Duplicate block IDs in reference input")
        if len(self.sections) != len(sections):
            raise ValueError("Duplicate section IDs in reference input")
        revisions = {item.document_revision_id for item in (*self.blocks, *sections)}
        if len(revisions) > 1:
            raise ValueError("Reference input must contain exactly one document revision")
        self._targets: dict[str, list[str]] = defaultdict(list)
        self._footnotes: dict[str, list[str]] = defaultdict(list)
        index_issues: list[str] = []
        for number, section in enumerate(sections):
            if number >= max_nodes:
                index_issues.append("traversal_limit:section_index")
                break
            if section.first_block_ordinal > section.last_block_ordinal:
                raise ValueError("Section ordinal interval is reversed")
            for key in _heading_keys(section.heading[:4096]):
                self._targets[key].append(section.section_id)
        scanned = 0
        note_count = 0
        for number, block in enumerate(self.blocks):
            if number >= max_nodes:
                index_issues.append("traversal_limit:footnote_index_nodes")
                break
            remaining = max_scan_chars - scanned
            if len(block.text) > remaining:
                index_issues.append("traversal_limit:footnote_index_chars")
            text = block.text[:max(0, remaining)]
            scanned += len(text)
            for match in _FOOTNOTE_DEFINITION.finditer(text):
                note_count += 1
                if note_count > max_references:
                    index_issues.append("traversal_limit:footnote_index_references")
                    break
                self._footnotes[match["label"].casefold()].append(block.block_id)
            if scanned >= max_scan_chars or note_count > max_references:
                if number + 1 < len(self.blocks):
                    index_issues.append("traversal_limit:footnote_index_chars" if scanned >= max_scan_chars else "traversal_limit:footnote_index_references")
                break
        inferred, anchor_issues = infer_numbered_sections(
            self.blocks, sections, max_nodes=max_nodes, max_scan_chars=max_scan_chars,
        )
        index_issues.extend(anchor_issues)
        for section in inferred:
            self.sections[section.section_id] = section
            for key in _heading_keys(section.heading):
                self._targets[key].append(section.section_id)
        self._index_issues = _unique(index_issues)

    def _block_references(
        self, block: SourceBlock, text: str
    ) -> tuple[list[tuple[str, str]], list[str]]:
        references: list[tuple[str, str]] = []
        issues: list[str] = []
        # A note definition is a target; its label is not a reference to itself.
        definitions = {match.start("label") - 2 for match in _FOOTNOTE_DEFINITION.finditer(text)}
        for match in _FOOTNOTE.finditer(text):
            if match.start() in definitions:
                continue
            references.append((f"footnote:{match['label'].casefold()}", match.group()))
            if len(references) > self.max_references:
                issues.append(f"traversal_limit:block_references:{block.block_id}")
                return references, issues
        consumed = -1
        for match in _REFERENCE.finditer(text):
            if match.start() < consumed:
                continue
            if len(references) + len(issues) >= self.max_references:
                issues.append(f"traversal_limit:block_references:{block.block_id}")
                break
            kind = match["kind"]
            labels = [match["label"]]
            end = match.end()
            unsupported = False
            while continuation := _LIST_ITEM.match(text, end):
                join = continuation["join"].strip().casefold()
                label = continuation["label"]
                if join in {"-", "\u2013", "\u2014", "to", "through"}:
                    first = labels[-1]
                    if first.isdecimal() and label.isdecimal() and 0 < int(label) - int(first) <= 32:
                        labels.extend(str(value) for value in range(int(first) + 1, int(label) + 1))
                    else:
                        unsupported = True
                else:
                    labels.append(label)
                end = continuation.end()
                if len(labels) > 64:
                    unsupported = True
                    break
            consumed = end
            label_text = text[match.start():end]
            suffix = text[end:]
            qualifier = _QUALIFIER.match(suffix)
            if qualifier and not _LOCAL_QUALIFIER.match(qualifier[1]):
                issues.append(f"external_reference:{block.block_id}:{label_text}")
                continue
            if _UNSUPPORTED_SUFFIX.match(suffix):
                unsupported = True
            if not (kind.casefold().startswith(("annex", "append"))):
                unsupported = unsupported or any(not label[0].isdigit() for label in labels)
            if unsupported:
                issues.append(f"unsupported_reference:{block.block_id}:{label_text}")
                continue
            for label in labels:
                references.append((_key(kind, label), f"{kind} {label}"))
                if len(references) > self.max_references:
                    issues.append(f"traversal_limit:block_references:{block.block_id}")
                    return references, issues
        return references, issues

    def closure(self, seed_ids: Sequence[str]) -> ClosureResult:
        seeds = _unique(seed_ids)
        unknown = [block_id for block_id in seeds if block_id not in self.by_id]
        if unknown:
            raise ValueError(f"Unknown reference seed ID: {unknown[0]}")
        required = set(seeds)
        frontier = deque(seeds)
        issues = list(self._index_issues)
        edges: list[tuple[str, str, str]] = []
        visited = 0
        reference_count = 0
        scanned = 0
        while frontier:
            if visited >= self.max_nodes:
                issues.append("traversal_limit:nodes")
                break
            block = self.by_id[frontier.popleft()]
            visited += 1
            remaining = self.max_scan_chars - scanned
            if len(block.text) > remaining:
                issues.append("traversal_limit:scan_chars")
            text = _normalized(block.text[:max(0, remaining)])
            scanned += len(block.text[:max(0, remaining)])
            references, local_issues = self._block_references(block, text)
            issues.extend(local_issues)
            reference_count += sum(
                issue.startswith(("external_reference:", "unsupported_reference:"))
                for issue in local_issues
            )
            if reference_count > self.max_references:
                issues.append("traversal_limit:references")
                break
            for key, label in references:
                reference_count += 1
                if reference_count > self.max_references:
                    issues.append("traversal_limit:references")
                    break
                if key.startswith("footnote:"):
                    candidates = self._footnotes.get(key.split(":", 1)[1], [])
                else:
                    candidates = self._targets.get(key, [])
                if len(candidates) != 1:
                    code = "ambiguous_reference" if candidates else "unresolved_reference"
                    issues.append(f"{code}:{block.block_id}:{label}")
                    continue
                target = candidates[0]
                if key.startswith("footnote:"):
                    values = (self.by_id[target],)
                    edge_target = self.by_id[target].section_id or key
                else:
                    section = self.sections[target]
                    first = bisect_left(self._ordinals, section.first_block_ordinal)
                    last = bisect_right(self._ordinals, section.last_block_ordinal)
                    if last - first > self.max_nodes:
                        issues.append("traversal_limit:target_nodes")
                    values = self.blocks[first:min(last, first + self.max_nodes)]
                    edge_target = target
                if not values:
                    issues.append(f"unresolved_reference:{block.block_id}:{label}:empty_target")
                    continue
                edges.append((block.block_id, label, edge_target))
                for value in values:
                    if value.block_id in required:
                        continue
                    if len(required) >= self.max_nodes:
                        issues.append("traversal_limit:nodes")
                        break
                    required.add(value.block_id)
                    frontier.append(value.block_id)
            if reference_count > self.max_references or scanned >= self.max_scan_chars:
                if frontier and scanned >= self.max_scan_chars:
                    issues.append("traversal_limit:scan_chars")
                break
        return ClosureResult(
            required_block_ids=tuple(block.block_id for block in self.blocks if block.block_id in required),
            issues=_unique(issues),
            edges=tuple(dict.fromkeys(edges)),
        )

    def audit(self, seed_ids: Sequence[str], presented_ids: Sequence[str]) -> ClosureResult:
        result = self.closure(seed_ids)
        presented = set(presented_ids)
        missing = tuple(f"missing_from_packet:{block_id}" for block_id in result.required_block_ids if block_id not in presented)
        return ClosureResult(result.required_block_ids, _unique((*result.issues, *missing)), result.edges)


def pack_closed(
    index: ReferenceIndex,
    seed_blocks: Sequence[SourceBlock],
    budget: int,
    cost: Callable[[Sequence[SourceBlock]], int],
) -> tuple[ClosedPacket, ...]:
    """Pack source-ordered seed/dependency unions without exceeding ``budget``.

    The cost function must be nonnegative and monotone as blocks are added.
    A seed whose closure cannot fit is retained with explicit missing/budget
    issues. An indivisible seed block that cannot fit raises ``ValueError``.
    """
    if budget < 1:
        raise ValueError("Packet budget must be positive")
    if not seed_blocks:
        return ()
    for block in seed_blocks:
        if index.by_id.get(block.block_id) != block:
            raise ValueError("Seed block does not match the indexed immutable source")
        if cost((block,)) > budget:
            raise ValueError(f"Indivisible source block exceeds packet budget: {block.block_id}")
    seeds = tuple(sorted({block.block_id: block for block in seed_blocks}.values(), key=lambda block: (block.ordinal, block.block_id)))

    def materialize(ids: Sequence[str]) -> tuple[SourceBlock, ...]:
        return tuple(index.by_id[block_id] for block_id in ids)

    whole = index.closure(tuple(block.block_id for block in seeds))
    whole_blocks = materialize(whole.required_block_ids)
    if cost(whole_blocks) <= budget:
        return (ClosedPacket(whole_blocks, whole.issues),)
    packets: list[ClosedPacket] = []
    pending: dict[str, SourceBlock] = {}
    pending_seed_ids: list[str] = []
    pending_issues: list[str] = []

    def flush() -> None:
        if not pending:
            return
        blocks = tuple(sorted(pending.values(), key=lambda block: (block.ordinal, block.block_id)))
        audit = index.audit(pending_seed_ids, tuple(block.block_id for block in blocks))
        packets.append(ClosedPacket(blocks, _unique((*pending_issues, *audit.issues))))
        pending.clear()
        pending_seed_ids.clear()
        pending_issues.clear()

    for seed in seeds:
        closed = index.closure((seed.block_id,))
        bundle = materialize(closed.required_block_ids)
        bundle_issues = list(closed.issues)
        if cost(bundle) > budget:
            # Keep the exact seed so the caller can report an honest unresolved
            # answer, never a deceptively complete truncated dependency.
            bundle = (seed,)
            bundle_issues.append(f"budget_exceeded:{seed.block_id}")
        proposed = {**pending, **{block.block_id: block for block in bundle}}
        ordered = tuple(sorted(proposed.values(), key=lambda block: (block.ordinal, block.block_id)))
        if pending and cost(ordered) > budget:
            flush()
        pending.update((block.block_id, block) for block in bundle)
        pending_seed_ids.append(seed.block_id)
        pending_issues.extend(bundle_issues)
    flush()
    return tuple(packets)
