"""Bounded, domain-independent scheduling primitives. No model or corpus facts."""
from __future__ import annotations

import hashlib
import re
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable

from .roles import ReaderPacket

if TYPE_CHECKING:
    from .evidence_review import EvidenceAssessment


def packet_key(packet: ReaderPacket) -> tuple[str, tuple[str, ...]]:
    # Packet IDs may change when repacked; immutable source blocks define work.
    return packet.document_revision_id, tuple(sorted(b.block_id for b in packet.blocks))


def focused_queries(values: Iterable[str], *, limit: int = 3) -> tuple[str, ...]:
    result: dict[str, str] = {}
    for value in values:
        query = " ".join(str(value).split())[:240]
        if query and len(re.findall(r"\w+", query)) <= 32:
            result.setdefault(query.casefold(), query)
        if len(result) >= limit:
            break
    return tuple(result.values())


class FairPacketQueue:
    """Lazy round robin: one packet per document before any second packet.

    A document is installed once. A duplicate discovery result cannot cause
    rereading, and packet identifiers cannot bypass immutable-block dedup.
    """

    def __init__(self, document_ids: Iterable[str] = ()) -> None:
        self.pending: deque[str] = deque()
        self.known: set[str] = set()
        self.packets: dict[str, deque[ReaderPacket]] = {}
        self.seen_packets: set[tuple[str, tuple[str, ...]]] = set()
        for document_id in document_ids:
            self.add(document_id)

    def add(self, document_id: str) -> bool:
        if document_id in self.known:
            return False
        self.known.add(document_id)
        self.pending.append(document_id)
        return True

    def install(self, document_id: str, packets: Iterable[ReaderPacket]) -> int:
        accepted: deque[ReaderPacket] = deque()
        for packet in packets:
            if packet.document_revision_id != document_id:
                raise ValueError("packet crosses its scheduled document revision")
            key = packet_key(packet)
            if key[1] and key not in self.seen_packets:
                self.seen_packets.add(key)
                accepted.append(packet)
        self.packets[document_id] = accepted
        return len(accepted)

    def pop_document(self) -> str:
        return self.pending.popleft()

    def take(self, document_id: str) -> ReaderPacket | None:
        packets = self.packets[document_id]
        if not packets:
            return None
        packet = packets.popleft()
        if packets:
            self.pending.append(document_id)
        return packet

    def discard(self, document_id: str) -> None:
        self.pending = deque(item for item in self.pending if item != document_id)

    def prioritize(self, document_ids: Iterable[str], *, unread: set[str], limit: int) -> tuple[str, ...]:
        """Move a bounded useful subset of first packets ahead of the backlog.

        Completed documents cannot be reopened here. A query matching every
        unread pending document supplies no discrimination, so it keeps order.
        Already read documents retain their round-robin positions.
        """
        eligible = set(self.pending) & unread
        matches = tuple(dict.fromkeys(item for item in document_ids if item in eligible))
        if not matches or set(matches) == eligible:
            return ()
        priority = matches[:max(0, limit)]
        selected = set(priority)
        self.pending = deque((*priority, *(item for item in self.pending if item not in selected)))
        return priority

    def __bool__(self) -> bool:
        return bool(self.pending)


@dataclass
class AdaptiveProgress:
    documents_read: set[str] = field(default_factory=set)
    documents_fully_read: set[str] = field(default_factory=set)
    completed_work_documents: set[str] = field(default_factory=set)
    terminal_work_documents: set[str] = field(default_factory=set)
    seen_section_ids: set[str] = field(default_factory=set)
    sections_total: int = 0
    research_window_exhausted: bool = False
    semantic_queue_stop: bool = False
    manifests_presented: int = 0
    added_manifests: list = field(default_factory=list)
    catalogue_unqueued_ids: tuple[str, ...] = ()
    last_assessment: EvidenceAssessment | None = None
    evidence_at_last_assessment: tuple[int, int] = (0, 0)
    debug: dict = field(default_factory=lambda: {"enabled": True, "waves": [], "stop_reason": ""})


def evidence_progress(artifacts) -> tuple[int, int]:
    valid = {block.block_id: block for artifact in artifacts for block in artifact.blocks
             if block.document_revision_id == artifact.report.document_revision_id
             and hashlib.sha256(block.text.encode("utf-8")).hexdigest() == block.text_sha256}
    blocks = set(valid)
    # Equivalent claims with new model IDs do not manufacture progress.
    claims = {
        hashlib.sha256((card.document_revision_id + "\0" +
                        " ".join(card.claim.casefold().split()) + "\0" +
                        "|".join(sorted(card.block_ids))).encode("utf-8")).hexdigest()
        for artifact in artifacts for card in artifact.report.cards
        if card.block_ids and all(bid in valid and valid[bid].document_revision_id == card.document_revision_id
                                  for bid in card.block_ids)
    }
    return len(blocks), len(claims)


def decision_state(*, remaining_s: float, research_budget_s: float, document_count: int,
                   documents_read: int, catalogue_remaining: int, unique_blocks: int,
                   source_bound_cards: int, missing_obligations: int, scope_issues: int,
                   pending_packets: int, wave: int, maximum_waves: int) -> dict[str, float]:
    """Public normalization shared by runtime and offline policy replay.

    Counts describe observed source work, not factual correctness. Saturation
    keeps features bounded and comparable across differently sized corpora.
    """
    def fraction(value, denominator):
        return min(1.0, max(0.0, float(value) / max(1.0, float(denominator))))
    return {
        "remaining_time_fraction": fraction(remaining_s, research_budget_s),
        "read_document_fraction": fraction(documents_read, document_count),
        "remaining_catalogue_fraction": fraction(catalogue_remaining, document_count),
        "unique_block_fraction": fraction(unique_blocks, 64),
        "unique_card_fraction": fraction(source_bound_cards, 32),
        "missing_obligation_fraction": fraction(missing_obligations, 16),
        "scope_issue_fraction": fraction(scope_issues, 8),
        "pending_packet_fraction": fraction(pending_packets, 64),
        "wave_fraction": fraction(wave, maximum_waves),
    }


def research_features(question: str, *, mode: str, document_count: int, full_scan_threshold: int,
                      has_exact_surfaces: bool = False) -> tuple[str, ...]:
    """Small query-shape features; no source facts, titles, or entity names."""
    result = ["corpus:large" if document_count > full_scan_threshold else "corpus:small"]
    if mode in {"targeted", "exhaustive"}:
        result.append("mode:" + mode)
    shapes = {
        "comparison": (
            r"\b(?:compare|comparison|differ(?:ence|ent)?|versus|vs|longer|shorter|higher|lower|"
            r"greater|fewer|larger|smaller|faster|slower|cheaper|costlier)\b|"
            r"\b(?:which|what)\b[^?.!\n]{0,160}\b(?:more|less|most|least)\b|\bsame\s+as\b"
        ),
        "temporal": r"\b(?:when|before|after|latest|current|date|year|version|updated)\b|\bas\s+of\b",
        "procedure": r"\b(?:how|steps?|procedure|troubleshoot|configure)\b",
    }
    for shape, pattern in shapes.items():
        if re.search(pattern, question, re.IGNORECASE):
            result.append("question:" + shape)
    if has_exact_surfaces:
        result.append("question:exact")
    if not any(item.startswith("question:") for item in result):
        result.append("question:general")
    return tuple(result)


def operational_event(value, *, key=""):
    """Metrics/off event logs contain numbers and finite controller codes only."""
    codes = {
        "balance_documents", "broaden_search", "focused_gap_search", "default",
        "research_deadline", "wave_limit", "queue_exhausted", "no_progress",
        "review_sufficient_advisory", "advisory", "all", "any", "phrase",
        "no_cited_draft_or_review_budget",
    }
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value if value in codes else None
    if isinstance(value, dict):
        return {k: clean for k, v in value.items() if (clean := operational_event(v, key=k)) is not None}
    if isinstance(value, (list, tuple)):
        return [clean for item in value if (clean := operational_event(item)) is not None]
    return None
