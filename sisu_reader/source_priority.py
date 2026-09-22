"""Source-bound lexical navigation priorities for complete reader packets.

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


class RevisitIntegrityError(ValueError):
    pass


def words(value: str) -> tuple[str, ...]:
    return tuple(re.findall(r'\w+', unicodedata.normalize('NFKC', value).casefold()))


def block_cost(blocks: Sequence[SourceBlock]) -> int:
    return 128 + sum(conservative_tokens(b.text) + conservative_tokens(b.locator) + 32 for b in blocks)


def lexical_block_scores(query: str, blocks: Sequence[SourceBlock]) -> dict[str, float]:
    """Presence-capped query coverage with block-frequency and length weighting.

    Scores nominate source locations only. They cannot determine that a block
    answers the question, entails a claim or carries the correct entity scope.
    """
    terms = set(words(query))
    if not terms:
        return {}
    tokens = {b.block_id: words(' '.join((b.text, *b.headers))) for b in blocks}
    signatures = {values: set(values) & terms for values in tokens.values()}
    frequency = Counter(t for present in signatures.values() for t in present)
    weights = {t: math.log1p(len(signatures) / (1 + frequency[t])) for t in terms}
    return {bid: sum(weights[t] for t in set(values) & terms) / math.sqrt(1 + len(values) / 80)
            for bid, values in tokens.items()}


def prioritize_reader_packets(query: str, packets: Sequence[ReaderPacket]) -> tuple[tuple[ReaderPacket, ...], dict]:
    """Reorder complete packets without changing their blocks or discarding work.

    Max block score prevents long packets from accumulating many scattered
    partial matches. Stable ties preserve the original coherent source order.
    """
    original = tuple(packets)
    if not original or not isinstance(query, str) or len(query) > 4096:
        return original, {'packets': len(original), 'reordered': False}
    documents = {p.document_revision_id for p in original}
    if len(documents) != 1:
        raise RevisitIntegrityError('Packet priority must stay inside one document revision')
    lookup = {}
    for packet in original:
        for block in packet.blocks:
            if (block.document_revision_id != packet.document_revision_id or
                    hashlib.sha256(block.text.encode('utf-8')).hexdigest() != block.text_sha256):
                raise RevisitIntegrityError('Packet priority source hash or document mismatch')
            if block.block_id in lookup and lookup[block.block_id] != block:
                raise RevisitIntegrityError('An immutable source block has conflicting values')
            lookup[block.block_id] = block
    scores = lexical_block_scores(query, tuple(lookup.values()))
    ranked = sorted(enumerate(original), key=lambda pair: (
        -max((scores.get(b.block_id, 0.0) for b in pair[1].blocks), default=0.0), pair[0]))
    result = tuple(packet for _, packet in ranked)
    return result, {'packets': len(original), 'reordered': result != original,
                    'original_positions': [index for index, _ in ranked],
                    'maximum_block_scores': [max((scores.get(b.block_id, 0.0) for b in p.blocks), default=0.0)
                                             for p in result]}


