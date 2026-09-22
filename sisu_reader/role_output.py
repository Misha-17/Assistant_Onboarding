"""Bounded structural decoding only; existing role/source validators remain authoritative."""
from __future__ import annotations

import json
import re
from collections.abc import Mapping

MAX_OUTPUT_CHARS = 240_000
MAX_RECORDS = 128
_ENVELOPE_START = re.compile(r'^\s*\{\s*"records"\s*:\s*\[')


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _constant(value):
    raise ValueError("nonfinite JSON number")


_DECODER = json.JSONDecoder(object_pairs_hook=_object, parse_constant=_constant)


def _valid_record(record):
    if not isinstance(record, Mapping):
        return False
    # The exact source allowlist, atomicity, absence and role guards run later.
    string_fields = {"type", "record_type", "manifest_id", "id", "document_revision_id", "decision", "label", "reason",
                     "subject", "claim", "text", "role", "answerability"}
    array_fields = {"block_ids", "source_ids", "relevant_entities"}
    if any(key not in string_fields | array_fields for key in record):
        return False
    return all(isinstance(value, str) if key in string_fields else
               isinstance(value, list) and all(isinstance(x, str) for x in value)
               for key, value in record.items())


def parse_record_objects(raw):
    """Unwrap records; salvage only completely decoded prefix items after truncation.

    No guessed quotes/braces, string reparsing, inner-object regex extraction or
    prose interpretation. Legacy physical-line JSONL remains supported.
    """
    if not isinstance(raw, str) or len(raw) > MAX_OUTPUT_CHARS:
        return [], ["structured_output_size_rejected"]
    clean = raw.strip()
    if clean.startswith("```"):
        lines = clean.splitlines()
        if len(lines) > 1:
            clean = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:]).strip()
    warnings = []
    decoded = None
    try:
        decoded = _DECODER.decode(clean)
    except json.JSONDecodeError:
        pass
    except (ValueError, RecursionError):
        return [], ["invalid_structured_json"]
    if isinstance(decoded, Mapping) and "records" in decoded:
        if set(decoded) != {"records"} or not isinstance(decoded["records"], list):
            return [], ["invalid_records_envelope"]
        candidates = decoded["records"]
    elif _ENVELOPE_START.match(clean):
        # Incrementally decode a truncated *known* envelope from its beginning.
        # Once a record is incomplete, nothing after it is trusted/salvaged.
        position = _ENVELOPE_START.match(clean).end()
        candidates = []
        while len(candidates) < MAX_RECORDS:
            while position < len(clean) and clean[position].isspace():
                position += 1
            if position >= len(clean) or clean[position] == "]":
                break
            try:
                value, end = _DECODER.raw_decode(clean, position)
            except (ValueError, RecursionError):
                break
            candidates.append(value)
            position = end
            while position < len(clean) and clean[position].isspace():
                position += 1
            if position >= len(clean) or clean[position] != ",":
                break
            position += 1
        warnings.append("incomplete_records_envelope_complete_prefix_salvaged")
    else:
        objects = []
        for number, line in enumerate(raw.splitlines(), 1):
            line = line.strip()
            if not line or line.startswith("```"):
                continue
            try:
                value = _DECODER.decode(line)
            except (ValueError, RecursionError):
                warnings.append(f"malformed_jsonl_line:{number}"); continue
            if not isinstance(value, Mapping):
                warnings.append(f"non_object_jsonl_line:{number}"); continue
            if len(objects) == MAX_RECORDS:
                warnings.append("structured_records_limit_reached"); break
            # Keep the legacy role validators' established compatibility behavior.
            objects.append((number, value))
        return objects, warnings
    if len(candidates) > MAX_RECORDS:
        warnings.append("structured_records_limit_reached")
    objects = []
    for number, record in enumerate(candidates[:MAX_RECORDS], 1):
        if not _valid_record(record):
            warnings.append(f"invalid_structured_record:{number}"); continue
        objects.append((number, record))
    return objects, warnings


def _shape(properties, required):
    return {"type": "object", "properties": properties, "required": required, "additionalProperties": False}


def screen_output_schema(manifest_ids):
    record = _shape({"manifest_id": {"type": "string", "enum": list(manifest_ids)},
                     "decision": {"type": "string", "enum": ["read", "maybe", "unlikely"]},
                     "reason": {"type": "string"},
                     "relevant_entities": {"type": "array", "items": {"type": "string"}}},
                    ["manifest_id", "decision", "reason", "relevant_entities"])
    return _shape({"records": {"type": "array", "items": record,
                                "minItems": len(manifest_ids), "maxItems": len(manifest_ids)}}, ["records"])


def reader_output_schema(block_ids):
    ids = {"type": "array", "items": {"type": "string", "enum": list(block_ids)}, "maxItems": 4}
    evidence = _shape({"type": {"type": "string", "enum": ["evidence"]},
                       "subject": {"type": "string"}, "claim": {"type": "string"}, "block_ids": {**ids, "minItems": 1},
                       "role": {"type": "string", "enum": ["direct", "context", "counter", "operand"]}},
                      ["type", "subject", "claim", "block_ids", "role"])
    gap = _shape({"type": {"type": "string", "enum": ["conflict", "unresolved"]},
                  "subject": {"type": "string"}, "text": {"type": "string"}, "block_ids": ids},
                 ["type", "subject", "text", "block_ids"])
    answer = _shape({"type": {"type": "string", "enum": ["answer"]},
                     "answerability": {"type": "string", "enum": ["answered", "partial", "not_found"]},
                     "text": {"type": "string"}}, ["type", "answerability", "text"])
    return _shape({"records": {"type": "array", "items": {"anyOf": [evidence, gap, answer]},
                                "minItems": 1, "maxItems": 24}}, ["records"])
