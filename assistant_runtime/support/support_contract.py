"""Strict, label-free protocol for an isolated advisory support scorer.

No inference, document discovery, authority decisions, rewriting or learning.
The caller must supply already authorized source bindings and reauthorize before
publishing. Hash validation detects mutation; it does not grant access.
"""
from __future__ import annotations
from collections.abc import Mapping, Sequence
import copy
import hashlib
import json
import math
import re

SCHEMA_VERSION = 1
UNCHECKED_REASONS = frozenset({
    'no_citation', 'ambiguous_span_or_attribution', 'mixed_question_premise',
    'input_too_long', 'unresolved_required_context', 'budget_exhausted',
    'worker_unavailable', 'worker_failed', 'cancelled',
    'unsupported_representation', 'invalid_citation', 'claim_limit',
    'deadline_exceeded', 'queue_full', 'model_unavailable',
})
_HASH = re.compile(r'[0-9a-f]{64}\Z')
CONTRACT = {
    'schema_version': 1, 'purpose': 'advisory_exact_citation_support',
    'phases': ['final', 'diagnostic'], 'input_limit_tokens': 2048,
    'source_join': 'ordered_exact_quotes_separated_by_two_newlines',
    'claim_preprocessing': 'parent_exact_span_remove_only_valid_attached_citation_tokens',
    'truncation': False, 'scores_are_truth_probabilities': False,
    'answer_mutation': False, 'training_truth': False,
}


def canonical_sha256(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()


def text_sha256(text):
    if not isinstance(text, str): raise ValueError('Text must be a string')
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


CONTRACT_SHA256 = canonical_sha256(CONTRACT)


def _keys(value, names, label):
    if not isinstance(value, Mapping) or set(value) != set(names):
        raise ValueError(label + ' fields differ from the protocol')


def _string(value, label, maximum=4096):
    if not isinstance(value, str) or not value or len(value) > maximum or '\x00' in value:
        raise ValueError(label + ' must be bounded nonempty text')


def _hash(value, label):
    if not isinstance(value, str) or not _HASH.fullmatch(value): raise ValueError(label + ' is not SHA256')


def _number(value, label, minimum=0):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < minimum:
        raise ValueError(label + ' must be a finite nonnegative number')


def _integer(value, label, minimum=0):
    if type(value) is not int or value < minimum: raise ValueError(label + ' must be an integer')


def input_fingerprint(request):
    return canonical_sha256({key: value for key, value in request.items()
                             if key not in {'request_id', 'deadline_monotonic', 'input_fingerprint'}})


def validate_request(request, expected_contract_sha256=None):
    _keys(request, {'schema_version','request_id','contract_sha256','input_fingerprint',
                    'deadline_monotonic','claim','sources','binding'}, 'request')
    if type(request['schema_version']) is not int or request['schema_version'] != SCHEMA_VERSION:
        raise ValueError('Unsupported protocol version')
    _string(request['request_id'], 'request_id', 256)
    for key in ('contract_sha256','input_fingerprint'): _hash(request[key], key)
    if expected_contract_sha256 is not None and request['contract_sha256'] != expected_contract_sha256:
        raise ValueError('Scorer contract changed')
    _number(request['deadline_monotonic'], 'deadline_monotonic')
    _keys(request['claim'], {'text','sha256'}, 'claim')
    _string(request['claim']['text'], 'claim', 12000)
    if text_sha256(request['claim']['text']) != request['claim']['sha256']: raise ValueError('Claim text mutated')
    sources = request['sources']
    if not isinstance(sources, list) or not 1 <= len(sources) <= 32: raise ValueError('Source count is invalid')
    for source in sources:
        _keys(source, {'text','sha256'}, 'source')
        _string(source['text'], 'source text', 100000)
        if text_sha256(source['text']) != source['sha256']: raise ValueError('Source text mutated')
    if sum(len(s['text']) for s in sources) > 100000: raise ValueError('Source transport budget exceeded')
    binding = request['binding']
    _keys(binding, {'phase','run_id','principal_id','authorization_scope','snapshot_id','runtime_epoch_sha256',
                    'answer_sha256','span_start','span_end','span_sha256','source_bindings'}, 'binding')
    if binding['phase'] not in {'final','diagnostic'}: raise ValueError('Unknown audit phase')
    for key in ('run_id','principal_id','authorization_scope','snapshot_id'): _string(binding[key], key)
    for key in ('runtime_epoch_sha256','answer_sha256','span_sha256'): _hash(binding[key], key)
    _integer(binding['span_start'], 'span_start'); _integer(binding['span_end'], 'span_end', 1)
    if binding['span_end'] <= binding['span_start']: raise ValueError('Empty or reversed span')
    bound = binding['source_bindings']
    if not isinstance(bound, list) or len(bound) != len(sources): raise ValueError('Source binding count differs')
    seen = set()
    for source, item in zip(sources, bound):
        _keys(item, {'source_id','card_id','block_id','document_revision_id','source_sha256',
                    'block_text_sha256','quote_sha256','locator','extraction_flags'}, 'source binding')
        for key in ('source_id','card_id','block_id','document_revision_id','locator'): _string(item[key], key)
        for key in ('source_sha256','block_text_sha256','quote_sha256'): _hash(item[key], key)
        if item['quote_sha256'] != source['sha256']: raise ValueError('Quote and source binding differ')
        if item['source_id'] in seen: raise ValueError('Duplicate citation binding')
        seen.add(item['source_id'])
        flags = item['extraction_flags']
        if not isinstance(flags, list) or len(flags) > 32: raise ValueError('Invalid extraction flags')
        for flag in flags: _string(flag, 'extraction flag', 256)
    if input_fingerprint(request) != request['input_fingerprint']: raise ValueError('Request fingerprint changed')
    return copy.deepcopy(dict(request))


def build_request(*, request_id: str, contract_sha256: str, claim: str,
                  sources: Sequence[Mapping], binding: Mapping, deadline_monotonic: float):
    request = {'schema_version': SCHEMA_VERSION, 'request_id': request_id,
               'contract_sha256': contract_sha256, 'deadline_monotonic': deadline_monotonic,
               'claim': {'text': claim, 'sha256': text_sha256(claim)},
               'sources': copy.deepcopy(list(sources)), 'binding': copy.deepcopy(dict(binding))}
    request['input_fingerprint'] = input_fingerprint(request)
    return validate_request(request)


def validate_response(response, request, *, expected_worker_generation=None,
                      expected_model_identity_sha256=None, current_binding=None, now=None):
    request = validate_request(request)
    _keys(response, {'schema_version','request_id','contract_sha256','input_fingerprint','status',
                     'support_score','input_tokens','elapsed_s','unchecked_reason','truncated','usage',
                     'worker_generation','model_identity_sha256'}, 'response')
    for key in ('schema_version','request_id','contract_sha256','input_fingerprint'):
        if response[key] != request[key]: raise ValueError('Stale or mismatched response: ' + key)
    if type(response['schema_version']) is not int: raise ValueError('Invalid response version')
    _string(response['worker_generation'], 'worker_generation', 256)
    _hash(response['model_identity_sha256'], 'model_identity_sha256')
    if expected_worker_generation is not None and response['worker_generation'] != expected_worker_generation:
        raise ValueError('Stale worker generation')
    if expected_model_identity_sha256 is not None and response['model_identity_sha256'] != expected_model_identity_sha256:
        raise ValueError('Model identity changed')
    if current_binding is not None and dict(current_binding) != request['binding']:
        raise ValueError('Current answer/source/authorization binding changed')
    if now is not None:
        _number(now, 'current monotonic time')
        if now >= request['deadline_monotonic'] and response['status'] == 'scored':
            raise ValueError('Late worker response')
    if response['truncated'] is not False: raise ValueError('Truncated evidence cannot be scored')
    _number(response['elapsed_s'], 'elapsed_s')
    usage = response['usage']
    _keys(usage, {'queue_s','tokenize_s','forward_s','total_s','forward_calls','input_tokens',
                  'decoder_steps','generated_output_tokens'}, 'usage')
    _number(usage['total_s'], 'total_s')
    for key in ('queue_s','tokenize_s','forward_s'):
        if usage[key] is not None: _number(usage[key], key)
    for key in ('forward_calls','decoder_steps'):
        if usage[key] is not None: _integer(usage[key], key)
    _integer(usage['generated_output_tokens'], 'generated_output_tokens')
    if usage['generated_output_tokens'] != 0: raise ValueError('Scorer must not generate output text')
    if any(usage[key] is not None and usage[key] > 1 for key in ('forward_calls','decoder_steps')):
        raise ValueError('Unbounded scorer execution')
    if usage['total_s'] + 1e-6 < sum(usage[k] or 0 for k in ('queue_s','tokenize_s','forward_s')):
        raise ValueError('Reported total omits measured cost')
    if abs(response['elapsed_s'] - usage['total_s']) > 1e-6: raise ValueError('Elapsed and total cost disagree')
    if response['input_tokens'] != usage['input_tokens']: raise ValueError('Token counts disagree')
    if response['input_tokens'] is not None: _integer(response['input_tokens'], 'input_tokens', 1)
    if response['status'] == 'scored':
        if any(usage[key] is None for key in ('queue_s','tokenize_s','forward_s')):
            raise ValueError('Scored result lacks measured cost')
        _number(response['support_score'], 'support_score')
        if response['support_score'] > 1: raise ValueError('Support score outside [0,1]')
        if response['unchecked_reason'] is not None: raise ValueError('Scored result carries unchecked reason')
        if response['input_tokens'] is None or response['input_tokens'] > 2048: raise ValueError('Scored input exceeds contract')
        if usage['forward_calls'] != 1 or usage['decoder_steps'] != 1: raise ValueError('Scored result lacks actual inference')
    elif response['status'] == 'unchecked':
        if response['support_score'] is not None: raise ValueError('Unchecked result invented score')
        if response['unchecked_reason'] not in UNCHECKED_REASONS: raise ValueError('Unknown unchecked status')
    else: raise ValueError('Worker returns scored or unchecked, never factual verdicts')
    return copy.deepcopy(dict(response))
