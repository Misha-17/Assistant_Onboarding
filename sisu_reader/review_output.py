"""Bounded structured advisory assessment; legacy JSONL remains auditable."""
from __future__ import annotations
import json
from collections.abc import Mapping

KINDS = ("answer", "entity", "temporal", "condition", "comparison", "procedure", "scope", "reference")
ISSUES = ("scope", "entity", "temporal", "condition", "unsupported", "contradicted", "reference")


def _shape(properties):
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


def assessment_output_schema(card_ids, block_ids):
    text = lambda maximum: {"type": "string", "minLength": 1, "maxLength": maximum}
    ids = lambda values: {"type": "array", "items": {"type": "string", "enum": list(values)},
                          "minItems": 1, "maxItems": 8}
    coverage = _shape({"kind": {"type": "string", "enum": list(KINDS)}, "obligation": text(400),
                       "card_ids": ids(card_ids), "block_ids": ids(block_ids)})
    gap = _shape({"kind": {"type": "string", "enum": list(KINDS)}, "obligation": text(400), "query": text(240)})
    issue = _shape({"kind": {"type": "string", "enum": list(ISSUES)}, "description": text(600),
                    "card_ids": ids(card_ids), "block_ids": ids(block_ids)})
    return _shape({"covered": {"type": "array", "items": coverage, "maxItems": 6},
                   "gaps": {"type": "array", "items": gap, "maxItems": 3},
                   "issues": {"type": "array", "items": issue, "maxItems": 6},
                   "sufficient": {"type": "boolean"}})


def assessment_objects(raw, legacy_parser):
    """Decode complete objects only; a truncated positive verdict is never used."""
    if not isinstance(raw, str) or len(raw) > 32000:
        return [], ["evidence_review_output_too_large"], False
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = value
        return result
    clean = raw.strip()
    if clean.startswith("```"):
        lines = clean.splitlines()
        if len(lines) > 2 and lines[-1].strip() == "```":
            clean = "\n".join(lines[1:-1])
    try:
        value = json.loads(clean, object_pairs_hook=pairs,
                           parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except json.JSONDecodeError:
        # Preserve legacy physical-line parsing. Its warnings prohibit a stop
        # when a structured object is truncated or mixed with invalid prose.
        values, warnings = legacy_parser(raw)
        return values, warnings, False
    except (ValueError, RecursionError):
        return [], ["evidence_review_invalid_json"], False
    if (not isinstance(value, Mapping) or not set(value).intersection({"covered", "gaps", "issues", "sufficient"})
            or "type" in value and not set(value).intersection({"covered", "gaps", "issues"})):
        values, warnings = legacy_parser(raw)
        return values, warnings, False
    if set(value) != {"covered", "gaps", "issues", "sufficient"} or type(value["sufficient"]) is not bool:
        return [], ["evidence_review_invalid_envelope"], True
    limits = {"covered": 6, "gaps": 3, "issues": 6}
    if any(not isinstance(value[key], list) or len(value[key]) > limit or
           any(not isinstance(row, Mapping) for row in value[key]) for key, limit in limits.items()):
        return [], ["evidence_review_invalid_envelope"], True
    fields = {"covered": {"kind", "obligation", "card_ids", "block_ids"},
              "gaps": {"kind", "obligation", "query"},
              "issues": {"kind", "description", "card_ids", "block_ids"}}
    if any(set(row) != fields[key] for key in limits for row in value[key]):
        return [], ["evidence_review_invalid_envelope_record"], True
    records = [{**row, "type": kind} for key, kind in (("covered", "covered"), ("gaps", "gap"), ("issues", "issue"))
               for row in value[key]]
    records.append({"type": "coverage", "sufficient": value["sufficient"]})
    return records, [], True
