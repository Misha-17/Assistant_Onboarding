"""Gap-directed revisitation of immutable, not-yet-read source blocks.

No model calls, source facts, entity lists, filename rules or training rewards.
Navigation utility is lexical and structural, not a correctness judgment.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
import hashlib
import math
import re
import time
import unicodedata
from typing import Callable, Iterable, Sequence

from sisu_reader.models import SourceBlock
from sisu_reader.roles import ReaderPacket
from sisu_reader.screen_budget import conservative_tokens


from .source_priority import RevisitIntegrityError, words, block_cost, lexical_block_scores


@dataclass(frozen=True)
class RevisitPlan:
    document_revision_id: str
    query_sha256: str
    blocks: tuple[SourceBlock, ...]
    anchor_block_id: str
    section_ids: tuple[str, ...]
    lexical_score: float
    candidate_blocks: int
    estimated_tokens: int
    elapsed_s: float
    boundary_limited: bool

    def packet(self, *, run_id: str, document_title: str, source_path: str,
               entity_scope: str = 'DOCUMENT', sections_total: int = 0) -> ReaderPacket:
        identity = hashlib.sha256('|'.join(b.block_id for b in self.blocks).encode()).hexdigest()[:16]
        return ReaderPacket(
            packet_id=f'{run_id}-{self.document_revision_id}-revisit-{identity}',
            document_revision_id=self.document_revision_id, document_title=document_title,
            source_path=source_path, blocks=self.blocks, entity_scope=entity_scope,
            sections_seen=len(self.section_ids), sections_total=sections_total,
            section_label='New source window selected for an unresolved question facet',
            complete_document_read=False,
            reference_issues=('revisit_section_boundary_unresolved',) if self.boundary_limited else (),
        )


class RevisitPlanner:
    """One run's bounded planning ledger. It is never persisted as learned truth."""

    def __init__(self, *, maximum_plans: int = 2, maximum_queries_per_document: int = 2,
                 maximum_blocks: int = 8, maximum_scanned_blocks: int = 50000):
        if not 1 <= maximum_plans <= 2 or not 1 <= maximum_queries_per_document <= 2:
            raise ValueError('Revisitation allows at most two plans/queries per document per run')
        if not 1 <= maximum_blocks <= 8 or not 1 <= maximum_scanned_blocks <= 50000:
            raise ValueError('Invalid bounded block allowance')
        self.maximum_plans = maximum_plans
        self.maximum_queries_per_document = maximum_queries_per_document
        self.maximum_blocks = maximum_blocks
        self.maximum_scanned_blocks = maximum_scanned_blocks
        self.attempted: set[tuple[str, str, str]] = set()
        self.reserved: set[str] = set()
        self.accepted_plans = 0
        self.telemetry: list[dict] = []

    def plan(self, query: str, document_id: str, *, store, snapshot_id: str,
             allowed_document_ids: Sequence[str], already_read_block_ids: Iterable[str],
             token_budget: int, deadline: float, guard: Callable[[Sequence[str]], None],
             clock: Callable[[], float] = time.perf_counter) -> RevisitPlan | None:
        started = clock()
        def finish(reason, **counts):
            self.telemetry.append({'outcome': reason, 'elapsed_s': max(0.0, clock() - started), **counts})
            return None
        if document_id not in frozenset(allowed_document_ids):
            raise PermissionError('Revisit document is outside the authorized run scope')
        if not isinstance(query, str) or not 0 < len(query) <= 240:
            return finish('invalid_query')
        terms = tuple(dict.fromkeys(words(query)))
        if not terms or len(terms) > 32 or type(token_budget) is not int or token_budget < 128:
            return finish('invalid_query_or_budget')
        query_hash = hashlib.sha256(' '.join(terms).encode()).hexdigest()
        key = (snapshot_id, document_id, query_hash)
        if key in self.attempted:
            return finish('duplicate_query')
        if self.accepted_plans >= self.maximum_plans or sum(k[:2] == key[:2] for k in self.attempted) >= self.maximum_queries_per_document:
            return finish('revisit_limit')
        if clock() >= deadline:
            return finish('deadline')
        self.attempted.add(key)

        def validate_scope():
            guard((document_id,))
            if store.snapshot().snapshot_id != snapshot_id:
                raise RuntimeError('Corpus snapshot changed during block revisitation')

        validate_scope()
        blocks = tuple(store.blocks_for_document(document_id, allowed_document_revision_ids=allowed_document_ids))
        sections = tuple(store.sections_for_document(document_id, allowed_document_revision_ids=allowed_document_ids))
        if len(blocks) > self.maximum_scanned_blocks:
            return finish('scan_limit', scanned_blocks=0)
        if sum(len(b.text) for b in blocks) > 5_000_000:
            return finish('scan_character_limit', scanned_blocks=0)
        if len({b.block_id for b in blocks}) != len(blocks) or len({b.ordinal for b in blocks}) != len(blocks):
            raise RevisitIntegrityError('Duplicate immutable block identity or document ordinal')
        if any(s.document_revision_id != document_id for s in sections):
            raise RevisitIntegrityError('Section crosses document revision')
        excluded = set(already_read_block_ids) | self.reserved
        for index, block in enumerate(blocks):
            if index % 32 == 0 and clock() >= deadline:
                return finish('deadline', scanned_blocks=index)
            if (block.document_revision_id != document_id or
                    hashlib.sha256(block.text.encode('utf-8')).hexdigest() != block.text_sha256):
                raise RevisitIntegrityError('Source block hash or document revision mismatch')
        scores = lexical_block_scores(query, blocks)
        scored = []
        for block in blocks:
            if block.block_id in excluded:
                continue
            score = scores[block.block_id]
            if score <= 0:
                continue
            scored.append((score, -block.ordinal, block.block_id, block))
        if not scored:
            return finish('no_new_match', scanned_blocks=len(blocks))
        anchor = max(scored)[-1]
        ordered = sorted(blocks, key=lambda b: b.ordinal)
        group = [b for b in ordered if b.section_id == anchor.section_id]
        children = any(s.parent_section_id == anchor.section_id for s in sections) if anchor.section_id else False
        complete_group = (anchor.section_id in {s.section_id for s in sections} and not children and
                          all(b.block_id not in excluded for b in group) and
                          len(group) <= self.maximum_blocks and block_cost(group) <= token_budget)
        if complete_group:
            selected = group
        else:
            selected = [anchor]
            lookup = {b.ordinal: b for b in ordered}
            # Contiguous same-section context only; never bridge previously
            # read/excluded blocks or silently cross into another entity scope.
            for direction, depth in ((-1, 2), (1, 2)):
                for step in range(1, depth + 1):
                    neighbor = lookup.get(anchor.ordinal + direction * step)
                    if (neighbor is None or neighbor.section_id != anchor.section_id or
                            neighbor.block_id in excluded or len(selected) >= self.maximum_blocks):
                        break
                    if block_cost((*selected, neighbor)) > token_budget:
                        break
                    selected.append(neighbor)
        selected = tuple(sorted(selected, key=lambda b: b.ordinal))
        if block_cost(selected) > token_budget:
            return finish('atomic_block_exceeds_budget', scanned_blocks=len(blocks))
        if clock() >= deadline:
            return finish('deadline', scanned_blocks=len(blocks))
        validate_scope()
        fresh = tuple(store.blocks_by_ids(tuple(b.block_id for b in selected),
                                        allowed_document_revision_ids=allowed_document_ids))
        if {b.block_id: b for b in fresh} != {b.block_id: b for b in selected}:
            raise RevisitIntegrityError('Selected immutable source blocks changed before scheduling')
        validate_scope()
        if clock() >= deadline:
            return finish('deadline', scanned_blocks=len(blocks))
        plan = RevisitPlan(document_id, query_hash, selected, anchor.block_id,
                           tuple(dict.fromkeys(b.section_id for b in selected if b.section_id)),
                           max(scored)[0], len(scored), block_cost(selected),
                           max(0.0, clock() - started), not complete_group)
        self.reserved.update(b.block_id for b in selected)
        self.accepted_plans += 1
        self.telemetry.append({'outcome': 'planned', 'elapsed_s': plan.elapsed_s,
                               'planned_new_blocks': len(selected), 'estimated_tokens': plan.estimated_tokens})
        return plan


def merge_revisit_packet(packet: ReaderPacket, pending: Sequence[ReaderPacket], *,
                         already_read_block_ids: Iterable[str]) -> tuple[ReaderPacket, ...]:
    """Put the new window first and remove its blocks from old pending work.

    A changed old packet explicitly loses any claim of structural completeness.
    The engine must still apply its usual reference audit and hash/ACL guards
    immediately before invoking the reader. No consumed packet is replayed.
    """
    consumed = set(already_read_block_ids)
    selected = {b.block_id for b in packet.blocks}
    if not selected or len(selected) != len(packet.blocks) or selected & consumed:
        raise ValueError('A revisit packet must contain distinct unread blocks')
    result = [packet]
    excluded = consumed | selected
    for previous in pending:
        if previous.document_revision_id != packet.document_revision_id:
            raise ValueError('Pending packet crosses the reopened document')
        remaining = tuple(b for b in previous.blocks if b.block_id not in excluded)
        if not remaining:
            continue
        excluded.update(b.block_id for b in remaining)
        if remaining != previous.blocks:
            previous = replace(previous, blocks=remaining, complete_document_read=False,
                               reference_issues=tuple(dict.fromkeys((*previous.reference_issues,
                                                                   'revisit_pending_boundary_changed'))))
        result.append(previous)
    return tuple(result)


def qualify_revisit_artifact(artifact, packet: ReaderPacket):
    """Keep exact claims while exposing a source window's incomplete boundary."""
    issues = tuple(i for i in packet.reference_issues if i in {
        'revisit_section_boundary_unresolved', 'revisit_pending_boundary_changed'})
    if not issues:
        return artifact
    report = artifact.report
    return replace(artifact, report=replace(
        report,
        cards=tuple(replace(card, role='context', reference_issues=tuple(dict.fromkeys(
            (*card.reference_issues, *issues)))) for card in report.cards),
        reference_issues=tuple(dict.fromkeys((*report.reference_issues, *issues))),
    ))
