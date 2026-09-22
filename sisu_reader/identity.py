from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .config import Config
from .models import EntityTarget


_TOKEN = re.compile(r"(?u)(?<!\w)[\w][\w.&/-]{1,80}(?!\w)")
_QUOTED = re.compile(r"[\"'“”‘’]([^\"'“”‘’]{2,100})[\"'“”‘’]")
_POSSESSIVE = re.compile(r"(?iu)(?<!\w)([\w][\w.&/-]{1,80})(?:['’]s)(?!\w)")
_STOP = frozenset({
    "a", "about", "all", "an", "and", "are", "as", "at", "be", "by", "can",
    "did", "do", "does", "for", "from", "give", "how", "i", "in", "is", "it",
    "its", "job", "me", "of", "on", "or", "our", "please", "project", "role",
    "tell", "that", "the", "their", "them", "these", "this", "those", "to", "us",
    "was", "were", "what", "when", "where", "which", "who", "why", "with", "you",
})


def normalize_surface(value: str) -> str:
    text = unicodedata.normalize("NFKC", value).replace("’", "'")
    return " ".join(text.strip().split())


def exact_surfaces(question: str) -> tuple[str, ...]:
    """Extract literal navigation anchors without claiming they are entities."""

    ordered: list[str] = []
    for match in _QUOTED.finditer(question):
        ordered.append(normalize_surface(match.group(1)))
    for match in _POSSESSIVE.finditer(question):
        ordered.append(normalize_surface(match.group(1)))
    tokens = [normalize_surface(item) for item in _TOKEN.findall(question)]
    ordered.extend(
        token for token in tokens
        if any(ch.isdigit() for ch in token)
        or (any(ch.isalpha() for ch in token) and token.upper() == token and len(token) <= 16)
    )
    ordered.extend(
        token for token in tokens
        if len(token) >= 3 and token.casefold() not in _STOP
    )
    result: list[str] = []
    seen: set[str] = set()
    for value in ordered:
        key = value.casefold()
        if key and key not in seen:
            seen.add(key)
            result.append(value)
    return tuple(result[:32])



_APPROVED_IDENTITY_BASES = frozenset({
    "approved_exact_alias", "approved_distinct_identity", "approved_session_alias",
})


def identity_target_rows(entities):
    """Preserve origin: an attention label does not establish a real identity."""
    if isinstance(entities, Mapping):
        values = [{"entity_id": key, "surface": value} for key, value in entities.items()]
    else:
        values = [dict(item) if isinstance(item, Mapping) else {
            key: getattr(item, key, ()) if key == "distinct_from" else getattr(item, key, "")
            for key in ("entity_id", "surface", "canonical_name", "basis", "distinct_from")}
            for item in entities]
    approved_ids = {str(item.get("entity_id", "")) for item in values
                    if item.get("basis") in _APPROVED_IDENTITY_BASES}
    rows = []
    for item in values:
        target_id = str(item.get("entity_id", "")).strip()
        if not target_id:
            continue
        approved = item.get("basis") in _APPROVED_IDENTITY_BASES
        distinct = item.get("distinct_from", ())
        if not isinstance(distinct, (tuple, list)):
            distinct = ()
        rows.append({"entity_id": target_id, "surface": str(item.get("surface", "")),
            "canonical_name": str(item.get("canonical_name", "")) if approved else "",
            "basis": str(item.get("basis") or "unspecified_lexical_hint"),
            "identity_status": "registered_identity" if approved else "lexical_hint",
            "distinct_from": [str(other) for other in distinct
                              if approved and str(other) in approved_ids and str(other) != target_id]})
    return rows


def identity_target_rules(entities):
    rows = identity_target_rows(entities)
    return (
        "IDENTITY MAPPINGS AND LEXICAL HINTS (controller-owned): "
        + json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
        + "\nOnly registered_identity rows provide approved name/alias mappings. "
        "Different lexical hint IDs are attention labels, not proof of different real "
        "entities. A hint can be a sentence fragment, instruction word, acronym, "
        "overlapping name or another surface for the same entity. Do not infer "
        "identity, non-identity, ownership or a canonical name from a lexical label. "
        "Resolve such references from source context; preserve unresolved ambiguity. "
        "Use approved mappings to recognize aliases of a registered identity. "
        "Honor explicitly declared distinct_from relationships between registered "
        "identities. Do not merge distinct named products, people or organizations "
        "when the sources distinguish them, or transfer facts between them merely "
        "because names overlap. Different registry IDs alone do not create an "
        "undeclared distinctness relationship. Preserve product, version and task "
        "scope from the source. Bind continuation lines such as 'Responsible:' or "
        "'Owner:' only to the associated preceding item."
    )


@dataclass(frozen=True, slots=True)
class RegistryResolution:
    targets: tuple[EntityTarget, ...]
    ambiguities: tuple[str, ...]
    exact_surfaces: tuple[str, ...]


class EntityRegistry:
    """Editable local identity hints; facts always remain in source documents."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.version = 0
        self._entities: tuple[dict[str, Any], ...] = ()
        self.reload()

    def reload(self) -> None:
        path = self.config.entity_registry_path
        if not path.is_file():
            self.version = 0
            self._entities = ()
            return
        if path.stat().st_size > 2 * 1024 * 1024:
            raise ValueError("entities.json is too large")
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or not isinstance(raw.get("entities", []), list):
            raise ValueError("entities.json must contain an entities list")
        entities: list[dict[str, Any]] = []
        for item in raw.get("entities", []):
            if not isinstance(item, dict):
                continue
            entity_id = str(item.get("id", "")).strip()
            name = str(item.get("name", "")).strip()
            if not entity_id or not name:
                continue
            aliases: list[dict[str, str]] = []
            for alias in item.get("aliases", ()):
                if isinstance(alias, str):
                    aliases.append({"text": alias, "mode": "exact_case"})
                elif isinstance(alias, dict) and str(alias.get("text", "")).strip():
                    aliases.append({
                        "text": str(alias["text"]).strip(),
                        "mode": str(alias.get("mode", "exact_case")),
                    })
            aliases.append({"text": name, "mode": "casefold"})
            entities.append({
                "id": entity_id,
                "name": name,
                "aliases": aliases,
                "distinct_from": tuple(str(x) for x in item.get("distinct_from", ()) if str(x)),
            })
        self.version = int(raw.get("version", 1))
        self._entities = tuple(entities)

    @staticmethod
    def _occurs(question: str, alias: str, *, casefold: bool) -> bool:
        haystack = normalize_surface(question)
        needle = normalize_surface(alias)
        if casefold:
            haystack, needle = haystack.casefold(), needle.casefold()
        # Possessive suffix is outside the boundary and therefore accepted.
        return re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", haystack) is not None

    def resolve(
        self,
        question: str,
        *,
        inherited: Iterable[EntityTarget] = (),
    ) -> RegistryResolution:
        matches: dict[str, EntityTarget] = {}
        alias_to_ids: dict[str, set[str]] = {}
        for entity in self._entities:
            for alias in entity["aliases"]:
                mode = alias.get("mode", "exact_case")
                if mode not in {"exact_case", "casefold"}:
                    continue
                if not self._occurs(question, alias["text"], casefold=mode == "casefold"):
                    continue
                normalized = normalize_surface(alias["text"]).casefold()
                alias_to_ids.setdefault(normalized, set()).add(entity["id"])
                matches[entity["id"]] = EntityTarget(
                    entity_id=entity["id"],
                    surface=alias["text"],
                    canonical_name=entity["name"],
                    basis="approved_exact_alias",
                    distinct_from=entity["distinct_from"],
                    inherited=False,
                )
        ambiguous_ids = {item for ids in alias_to_ids.values() if len(ids) > 1 for item in ids}
        for entity_id in ambiguous_ids:
            matches.pop(entity_id, None)
        targets = list(matches.values())
        if not targets:
            registry_by_id = {entity["id"]: entity for entity in self._entities}
            for item in inherited:
                registered = registry_by_id.get(item.entity_id)
                alias_valid = registered is not None and any(
                    normalize_surface(item.surface).casefold() == normalize_surface(alias["text"]).casefold()
                    for alias in registered["aliases"])
                targets.append(EntityTarget(
                    entity_id=item.entity_id, surface=item.surface,
                    canonical_name=registered["name"] if alias_valid else "",
                    basis="approved_session_alias" if alias_valid else "session_lexical_surface",
                    distinct_from=registered["distinct_from"] if alias_valid else (), inherited=True))
        surfaces = list(exact_surfaces(question))
        surfaces.extend(item.surface for item in targets)
        unique = tuple(dict.fromkeys(item for item in surfaces if item))
        ambiguities = tuple(sorted(
            alias for alias, ids in alias_to_ids.items() if len(ids) > 1
        ))
        return RegistryResolution(tuple(targets), ambiguities, unique)

    def prompt_partition(self, targets: Iterable[EntityTarget]) -> str:
        return identity_target_rules(tuple(targets))

    def expand_prompt_partition(
        self,
        targets: Iterable[EntityTarget],
    ) -> tuple[EntityTarget, ...]:
        """Add approved ``distinct_from`` identities for model separation only.

        The returned identities must never be fed into query forcing: an
        unmentioned company is context for attribution, not a new search
        target. Facts and relationships still come exclusively from sources.
        """

        selected = list(targets)
        seen = {item.entity_id for item in selected}
        registry_by_id = {item["id"]: item for item in self._entities}
        pending = [
            entity_id
            for item in selected
            for entity_id in item.distinct_from
            if item.basis in _APPROVED_IDENTITY_BASES
            if entity_id in registry_by_id and entity_id not in seen
        ]
        for entity_id in dict.fromkeys(pending):
            entity = registry_by_id[entity_id]
            selected.append(
                EntityTarget(
                    entity_id=entity_id,
                    surface=entity["name"],
                    canonical_name=entity["name"],
                    basis="approved_distinct_identity",
                    distinct_from=entity["distinct_from"],
                )
            )
            seen.add(entity_id)
        return tuple(selected)


def load_profile(config: Config) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "answer_style": "Direct, natural, and concise; add detail when the question needs it.",
        "language": "Match the user's language.",
        "custom_instructions": [],
        "glossary": {},
    }
    path: Path = config.profile_path
    if not path.is_file():
        return defaults
    if path.stat().st_size > 1024 * 1024:
        raise ValueError("profile.json is too large")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("profile.json must be a JSON object")
    result = dict(defaults)
    for key in defaults:
        if key in raw and isinstance(raw[key], type(defaults[key])):
            result[key] = raw[key]
    return result


__all__ = [
    "EntityRegistry",
    "RegistryResolution",
    "exact_surfaces",
    "load_profile",
    "normalize_surface",
]
