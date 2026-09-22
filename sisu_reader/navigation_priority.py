"""Bounded query-word priorities for navigation, never source evidence."""
from __future__ import annotations

import re
from typing import Iterable


def navigation_priorities(question: str, store, allowed_document_ids: Iterable[str], *,
                          maximum_queries: int = 24, maximum_terms: int = 8):
    allowed = frozenset(allowed_document_ids)
    if not allowed:
        return {}, {"queries": 0, "terms_used": 0, "query_errors": 0}
    allowed_sequence = tuple(sorted(allowed))
    # Whole alphabetic Unicode words only. Entity/code tokens retain their
    # separate exact-phrase path; we do not infer new entity identities here.
    words = tuple(dict.fromkeys(word.casefold() for word in
                  re.findall(r"(?u)\b[^\W\d_]{3,}\b", question)))[:max(0, min(24, maximum_queries))]
    candidates = []
    errors = 0
    for order, word in enumerate(words):
        try:
            matched = frozenset(store.exact_document_matches(
                word, allowed_document_revision_ids=allowed_sequence)) & allowed
        except PermissionError:
            raise
        except Exception:
            errors += 1
            continue
        if matched and len(matched) < len(allowed):
            candidates.append((len(matched), order, matched))
    scores = {}
    selected = sorted(candidates, key=lambda item: (item[0], item[1]))[:max(0, min(8, maximum_terms))]
    for frequency, _, matched in selected:
        weight = 1.0 / frequency - 1.0 / len(allowed)
        for document_id in matched:
            scores[document_id] = scores.get(document_id, 0.0) + weight
    return scores, {"queries": len(words), "terms_used": len(selected), "query_errors": errors}
