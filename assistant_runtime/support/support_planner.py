"""Advisory post-binding audit planning; preserves every answer byte.

This isolated prototype accepts trusted parent-supplied index rows and the
actual displayed Citation list. It never imports SISU, reads source files,
discovers evidence, launches a model, rewrites an answer, or grants access.
"""
from __future__ import annotations
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import html
import math
import re

from support_contract import (CONTRACT_SHA256, build_request, canonical_sha256,
                              text_sha256, validate_response)

_CITATION = re.compile(r'\[S[0-9]+\]')
_HEX = re.compile(r'[0-9a-f]{64}\Z')


@dataclass(frozen=True, slots=True)
class AuditScope:
    run_id: str
    principal_id: str
    authorization_scope: str
    snapshot_id: str
    runtime_epoch_sha256: str
    authorized_document_revision_ids: frozenset[str]

    def __post_init__(self):
        for name in ('run_id','principal_id','authorization_scope','snapshot_id'):
            if not isinstance(getattr(self, name), str) or not getattr(self, name): raise ValueError('Missing trusted scope')
        if not _HEX.fullmatch(self.runtime_epoch_sha256): raise ValueError('Invalid runtime epoch')
        if not isinstance(self.authorized_document_revision_ids, frozenset): raise ValueError('Authorization inventory must be immutable')


@dataclass(frozen=True, slots=True)
class BoundEvidence:
    source_id: str
    card_id: str
    block_id: str
    document_revision_id: str
    source_sha256: str
    block_text_sha256: str
    quote_sha256: str
    quote: str
    locator: str
    extraction_flags: tuple[str, ...]

    def binding(self):
        return {name: list(value) if name == 'extraction_flags' else value
                for name in self.__dataclass_fields__ if name != 'quote'
                for value in [getattr(self, name)]}


@dataclass(frozen=True, slots=True)
class AuditUnit:
    unit_id: str
    start: int
    end: int
    exact_text: str
    span_sha256: str
    claim_text: str
    evidence: tuple[BoundEvidence, ...]
    status: str
    unchecked_reason: str | None
    # Serialized request is immutable; a caller obtains a fresh dict via request().
    request_json: str | None = None

    def request(self):
        if self.request_json is None: return None
        import json
        return json.loads(self.request_json)


@dataclass(frozen=True, slots=True)
class AuditPlan:
    answer: str
    answer_sha256: str
    scope: AuditScope
    units: tuple[AuditUnit, ...]
    advisory: bool = True
    can_modify_answer: bool = False
    learning_truth: bool = False


def _records(rows, key):
    result = {}
    for value in rows:
        if not isinstance(value, Mapping) or not isinstance(value.get(key), str): raise ValueError('Invalid trusted index row')
        if value[key] in result: raise ValueError('Duplicate trusted index identity')
        result[value[key]] = value
    return result


def bind_evidence(citations: Sequence[Mapping], *, documents: Sequence[Mapping],
                  blocks: Sequence[Mapping], scope: AuditScope):
    """Validate already admitted citation provenance; this is not authorization.

    The parent supplies a freshly authorized inventory and verifies original
    source hashes at ingestion/publication. This function verifies its internal
    relationships and exact quoted bytes, without opening any source path.
    """
    docs = _records(documents, 'document_revision_id'); source_blocks = _records(blocks, 'block_id')
    result = {}
    for citation in citations:
        sid = citation.get('source_id')
        if not isinstance(sid, str) or not re.fullmatch(r'S[1-9][0-9]*', sid): raise ValueError('Invalid bound source ID')
        if sid in result: raise ValueError('Duplicate bound citation ID')
        revision = citation.get('document_revision_id')
        if revision not in scope.authorized_document_revision_ids: raise ValueError('Source is not in authorized inventory')
        doc = docs.get(revision); block = source_blocks.get(citation.get('block_id'))
        if doc is None or block is None or block.get('document_revision_id') != revision:
            raise ValueError('Source block/revision binding differs')
        for key in ('source_path',):
            if citation.get(key) != doc.get(key): raise ValueError('Source identity differs')
        text = block.get('text'); quote = citation.get('quote')
        if not isinstance(text, str) or text_sha256(text) != block.get('text_sha256'): raise ValueError('Source block mutated')
        if not isinstance(quote, str) or not quote.strip() or quote not in text:
            raise ValueError('Quote is not contained in the bound block')
        if text_sha256(quote) != citation.get('quote_sha256'): raise ValueError('Quote hash changed')
        if citation.get('locator') != block.get('locator'): raise ValueError('Quote locator changed')
        if not isinstance(doc.get('source_sha256'), str) or not _HEX.fullmatch(doc['source_sha256']): raise ValueError('Missing source hash')
        card_id = citation.get('card_id')
        if not isinstance(card_id, str) or not card_id: raise ValueError('Missing source-bound card identity')
        flags = block.get('extraction_flags', ())
        if not isinstance(flags, (list, tuple)) or any(not isinstance(flag, str) for flag in flags): raise ValueError('Invalid source flags')
        result[sid] = BoundEvidence(sid, card_id, block['block_id'], revision, doc['source_sha256'],
            block['text_sha256'], citation['quote_sha256'], quote, citation['locator'], tuple(flags))
    return result


def _protected(text):
    """Literal quote/inline-code ranges; markers inside them are data, not links."""
    protected = [False] * len(text)
    i = 0; ambiguous = False
    while i < len(text):
        char = text[i]
        if char == '`':
            width = len(text[i:]) - len(text[i:].lstrip('`')); end = text.find('`' * width, i + width)
            if end < 0: ambiguous = True; protected[i:] = [True] * (len(text) - i); break
            end += width; protected[i:end] = [True] * (end-i); i = end; continue
        if char in {'"', '“', '‘'}:
            closer = {'"': '"', '“': '”', '‘': '’'}[char]
            end = text.find(closer, i+1)
            if end < 0: ambiguous = True; i += 1; continue
            protected[i:end+1] = [True] * (end+1-i); i = end+1; continue
        i += 1
    return protected, ambiguous


def _trim_span(answer, start, end):
    while start < end and answer[start].isspace(): start += 1
    while end > start and answer[end-1].isspace(): end -= 1
    return start, end


def _paragraphs(answer):
    start = 0
    for separator in re.finditer(r'\n[ \t]*\n', answer):
        if separator.start() > start: yield _trim_span(answer, start, separator.start())
        start = separator.end()
    if start < len(answer): yield _trim_span(answer, start, len(answer))


def _sentences(answer, start, end):
    text = answer[start:end]; protected, ambiguous = _protected(text)
    if ambiguous: return [(start, end, 'ambiguous_span_or_attribution')]
    result = []; cursor = 0; i = 0
    while i < len(text):
        if text[i] not in '.!?' or protected[i]: i += 1; continue
        # Decimal/member-access dots and dotted abbreviations do not safely end a sentence.
        if text[i] == '.' and i and i+1 < len(text) and text[i-1].isalnum() and text[i+1].isalnum(): i += 1; continue
        token = re.search(r'(\S+)$', text[:i])
        if text[i] == '.' and token and (len(token[1]) == 1 or '.' in token[1]): i += 1; continue
        j = i+1
        while j < len(text) and text[j] in ' \t': j += 1
        # Consume trailing displayed citations, preserving offsets and their order.
        while True:
            marker = _CITATION.match(text, j)
            if not marker or protected[j]: break
            j = marker.end()
            while j < len(text) and text[j] in ' \t': j += 1
        if j < len(text) and not text[j].isspace() and not text[j].isupper(): i += 1; continue
        k = j
        while k < len(text) and text[k].isspace(): k += 1
        if k < len(text) and text[k].islower(): i += 1; continue
        a, b = _trim_span(answer, start+cursor, start+j)
        if b > a: result.append((a, b, None))
        cursor = j; i = max(i+1, j)
    a, b = _trim_span(answer, start+cursor, end)
    if b > a: result.append((a, b, None))
    return result or [(start, end, None)]


def _spans(answer):
    fenced = []; opening = None; fence_char = None; fence_width = 0
    for line in re.finditer(r'(?m)^.*(?:\n|$)', answer):
        marker = re.match(r'\s*(`{3,}|~{3,})', line.group())
        if marker:
            if opening is None:
                opening = line.start(); fence_char = marker[1][0]; fence_width = len(marker[1])
            elif marker[1][0] == fence_char and len(marker[1]) >= fence_width:
                fenced.append((opening, line.end())); opening = None
    if opening is not None: fenced.append((opening, len(answer)))
    for start, end in _paragraphs(answer):
        text = answer[start:end]
        line_start = answer.rfind('\n', 0, start) + 1
        indented_code = re.search(r'(?m)^(?: {4}|\t)', answer[line_start:end]) is not None
        if indented_code or any(a < end and b > start for a,b in fenced) or re.search(r'^\s*\|', text, re.M) or re.search(r'^\s*[-:]+\s*\|', text, re.M):
            yield start, end, 'unsupported_representation'; continue
        if text.startswith('#') or re.fullmatch(r'[A-Za-z_][\w]*\s*=\s*\S+(?:\s+[A-Za-z_][\w]*\s*=\s*\S+)*', text):
            yield start, end, 'unsupported_representation'; continue
        yield from _sentences(answer, start, end)


def _claim_and_ids(text):
    mask, _ = _protected(text); matches = [m for m in _CITATION.finditer(text) if not mask[m.start()]]
    ids = tuple(dict.fromkeys(m.group()[1:-1] for m in matches))
    pieces = []; start = 0
    for marker in matches: pieces.append(text[start:marker.start()]); start = marker.end()
    pieces.append(text[start:])
    return ''.join(pieces).strip(), ids


def _literal_quote(claim, evidence):
    # Only an entire visible quoted unit, never a model declaration or substring.
    value = claim.strip()
    if value.startswith('>'): value = value[1:].strip()
    for opening, closing in [('“','”'), ('"','"'), ('‘','’')]:
        if value.startswith(opening) and value.endswith(closing):
            inner = html.unescape(value[1:-1])
            return any(inner == item.quote for item in evidence)
    return False


def _binding(scope, answer_hash, start, end, span_hash, evidence):
    return {'phase': 'final', 'run_id': scope.run_id, 'principal_id': scope.principal_id,
            'authorization_scope': scope.authorization_scope, 'snapshot_id': scope.snapshot_id,
            'runtime_epoch_sha256': scope.runtime_epoch_sha256, 'answer_sha256': answer_hash,
            'span_start': start, 'span_end': end, 'span_sha256': span_hash,
            'source_bindings': [item.binding() for item in evidence]}


def plan_audit(answer: str, citations: Sequence[Mapping], *, documents: Sequence[Mapping],
               blocks: Sequence[Mapping], scope: AuditScope, deadline_monotonic: float,
               contract_sha256: str = CONTRACT_SHA256, maximum_claims: int = 6,
               maximum_pair_chars: int = 16000, now: float = 0,
               premise_spans: Sequence[tuple[int,int]] = (),
               unresolved_citation_ids: frozenset[str] = frozenset()):
    import json
    if not isinstance(answer, str) or len(answer) > 100000: raise ValueError('Answer transport budget exceeded')
    if type(maximum_claims) is not int or not 0 <= maximum_claims <= 32: raise ValueError('Invalid claim budget')
    if type(maximum_pair_chars) is not int or not 1 <= maximum_pair_chars <= 100000: raise ValueError('Invalid pair budget')
    for value in (now, deadline_monotonic):
        if isinstance(value, bool) or not isinstance(value, (int,float)) or not math.isfinite(value) or value < 0:
            raise ValueError('Invalid monotonic time')
    for a, b in premise_spans:
        if type(a) is not int or type(b) is not int or not 0 <= a < b <= len(answer): raise ValueError('Invalid question-premise span')
    inventory = bind_evidence(citations, documents=documents, blocks=blocks, scope=scope)
    answer_hash = text_sha256(answer); units = []; planned = 0
    for start, end, reason in _spans(answer):
        text = answer[start:end]; span_hash = text_sha256(text); claim, ids = _claim_and_ids(text)
        evidence = tuple(inventory[sid] for sid in ids if sid in inventory)
        if reason is None and len(evidence) != len(ids): reason = 'invalid_citation'
        if reason is None and not ids: reason = 'no_citation'
        if reason is None and any(a < end and b > start for a, b in premise_spans): reason = 'mixed_question_premise'
        if reason is None and any(sid in unresolved_citation_ids for sid in ids): reason = 'unresolved_required_context'
        status = 'unchecked' if reason else 'planned'
        if reason is None and _literal_quote(claim, evidence): status = 'literal_quote_verified'
        if status == 'planned':
            if not claim or len(claim) > 12000 or len(claim) + sum(len(e.quote) for e in evidence) > maximum_pair_chars:
                reason = 'input_too_long'
            elif now >= deadline_monotonic: reason = 'budget_exhausted'
            elif planned >= maximum_claims: reason = 'claim_limit'
            if reason: status = 'unchecked'
        uid = 'unit_' + canonical_sha256({'answer': answer_hash, 'start': start, 'end': end})[:20]
        request = None
        if status == 'planned':
            request = build_request(request_id=uid, contract_sha256=contract_sha256, claim=claim,
                sources=[{'text': item.quote, 'sha256': item.quote_sha256} for item in evidence],
                binding=_binding(scope, answer_hash, start, end, span_hash, evidence), deadline_monotonic=deadline_monotonic)
            planned += 1
        units.append(AuditUnit(uid, start, end, text, span_hash, claim, evidence, status, reason,
                               json.dumps(request, ensure_ascii=False, sort_keys=True) if request else None))
    return AuditPlan(answer, answer_hash, scope, tuple(units))


def validate_unit_response(plan: AuditPlan, unit_id: str, response: Mapping, *, current_answer: str,
                           current_scope: AuditScope, citations: Sequence[Mapping],
                           documents: Sequence[Mapping], blocks: Sequence[Mapping], now: float,
                           expected_worker_generation: str, expected_model_identity_sha256: str):
    if current_scope != plan.scope: raise ValueError('Trusted authority/snapshot changed')
    if text_sha256(current_answer) != plan.answer_sha256: raise ValueError('Answer changed after audit planning')
    unit = next((unit for unit in plan.units if unit.unit_id == unit_id), None)
    if unit is None or unit.status != 'planned': raise ValueError('Unknown or unscored audit unit')
    if current_answer[unit.start:unit.end] != unit.exact_text: raise ValueError('Claim span mutated')
    inventory = bind_evidence(citations, documents=documents, blocks=blocks, scope=current_scope)
    if any(item.source_id not in inventory for item in unit.evidence): raise ValueError('Bound citation removed')
    evidence = tuple(inventory[item.source_id] for item in unit.evidence)
    binding = _binding(current_scope, plan.answer_sha256, unit.start, unit.end, unit.span_sha256, evidence)
    return validate_response(response, unit.request(), expected_worker_generation=expected_worker_generation,
        expected_model_identity_sha256=expected_model_identity_sha256, current_binding=binding, now=now)
