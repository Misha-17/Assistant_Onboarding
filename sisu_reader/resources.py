"""Persistent organizational roles and metadata-only video resources.

This module deliberately does not import media, networking, subprocess, audio,
image, or transcription libraries. Video records are searched only through
metadata supplied by an authorized administrator.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import unicodedata
import uuid
from datetime import date, datetime, timezone
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import parse_qsl, urlsplit

from .config import Config
from .models import (
    Answer,
    Citation,
    Coverage,
    OrganizationalRole,
    PersonRecord,
    VideoChapter,
    VideoRecord,
)


_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_WORD = re.compile(r"[^\W_]+", re.UNICODE)
_VIDEO_INTENT = re.compile(
    r"\b(?:video|recording|watch|chapter|timestamp|speaker|webinar)\b",
    re.IGNORECASE,
)
_ROLE_INTENT = re.compile(
    r"(?:\bwho\b.{0,80}\b(?:handles?|owns?|manages?|leads?|responsib\w*|"
    r"manager|supervisor|hr)\b|\bwhat\s+is\b.{0,80}\b(?:job|role)\b|"
    r"\b(?:registered|organizational|company)\s+(?:role|directory)\b)",
    re.IGNORECASE,
)
_DOCUMENT_INTENT = re.compile(r"\b(?:document|file|report|source|according to)\b", re.IGNORECASE)
_STOP = frozenset({
    "a", "about", "and", "are", "at", "by", "does", "for", "from", "in",
    "is", "it", "job", "of", "on", "or", "please", "role", "the", "this",
    "to", "what", "when", "where", "which", "who", "whose",
    # Resource-intent words route a question but should not make unrelated
    # metadata look relevant. If these are the only words, an empty token set
    # intentionally means "list the visible video catalogue".
    "chapter", "discuss", "discusses", "recording", "speaker", "timestamp",
    "video", "watch", "webinar",
})


class ResourceValidationError(ValueError):
    """Raised when supplied directory or video metadata is invalid."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _clean(value: Any, *, maximum: int = 4_000, required: bool = False) -> str:
    if value is None:
        result = ""
    elif isinstance(value, str):
        result = " ".join(value.split()).strip()
    else:
        raise ResourceValidationError("value must be text")
    if len(result) > maximum:
        raise ResourceValidationError(f"value exceeds {maximum} characters")
    if required and not result:
        raise ResourceValidationError("value is required")
    return result


def _identifier(value: Any, prefix: str) -> str:
    clean = _clean(value, maximum=128) if value else f"{prefix}_{uuid.uuid4().hex}"
    if not _ID.fullmatch(clean):
        raise ResourceValidationError(
            "identifiers may contain only letters, numbers, dot, underscore, colon, and hyphen"
        )
    return clean


def _normalized(value: str) -> str:
    return " ".join(
        unicodedata.normalize("NFKC", value).casefold().split()
    )


def _tokens(value: str) -> tuple[str, ...]:
    return tuple(
        token
        for token in (_normalized(item) for item in _WORD.findall(value))
        if len(token) > 1 and token not in _STOP
    )


def _iso_date(value: Any, *, field: str) -> str:
    clean = _clean(value, maximum=64)
    if not clean:
        return ""
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", clean):
        try:
            return date.fromisoformat(clean).isoformat()
        except ValueError as exc:
            raise ResourceValidationError(f"{field} must be a real calendar date") from exc
    try:
        parsed = datetime.fromisoformat(clean.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ResourceValidationError(
            f"{field} must use YYYY-MM-DD or an ISO date-time with an offset"
        ) from exc
    if parsed.tzinfo is None:
        raise ResourceValidationError(f"{field} date-time must include a UTC offset")
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _recorded_at(value: Any, *, field: str) -> str:
    return _iso_date(value, field=field)


def parse_timestamp(value: Any, *, field: str = "timestamp") -> int:
    """Parse seconds, MM:SS, or HH:MM:SS into whole seconds."""

    if isinstance(value, bool):
        raise ResourceValidationError(f"{field} must be a timestamp")
    if isinstance(value, int):
        if value < 0:
            raise ResourceValidationError(f"{field} cannot be negative")
        return value
    clean = _clean(value, maximum=32, required=True)
    if clean.isdigit():
        return int(clean)
    parts = clean.split(":")
    if len(parts) not in {2, 3} or any(not item.isdigit() for item in parts):
        raise ResourceValidationError(f"{field} must be seconds, MM:SS, or HH:MM:SS")
    numbers = [int(item) for item in parts]
    if any(item > 59 for item in numbers[-2:]):
        raise ResourceValidationError(f"{field} has an invalid minute or second")
    if len(numbers) == 2:
        return numbers[0] * 60 + numbers[1]
    return numbers[0] * 3600 + numbers[1] * 60 + numbers[2]


def format_timestamp(seconds: int) -> str:
    hours, remainder = divmod(max(0, int(seconds)), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def _locator(value: Any) -> tuple[str, str]:
    if value is None or value == "":
        raise ResourceValidationError("video locator is required")
    if not isinstance(value, str):
        raise ResourceValidationError("video locator must be text")
    clean = value.strip()
    if not clean:
        raise ResourceValidationError("video locator is required")
    if len(clean) > 4_000:
        raise ResourceValidationError("video locator exceeds 4000 characters")
    if any(char in clean for char in ("\x00", "\r", "\n")):
        raise ResourceValidationError("video locator contains invalid characters")
    if re.match(r"^[A-Za-z]:[\\/]", clean) or clean.startswith(("\\\\", "//")):
        return "local_path", clean
    parsed = urlsplit(clean)
    if parsed.scheme:
        if parsed.scheme.casefold() != "https" or not parsed.netloc:
            raise ResourceValidationError("video URL must use https")
        if parsed.username or parsed.password:
            raise ResourceValidationError("video URL must not contain credentials")
        sensitive = re.compile(
            r"(?:^|[_-])(?:api[_-]?key|auth|credential|secret|sig|signature|token)(?:$|[_-])",
            re.IGNORECASE,
        )
        if any(sensitive.search(key) for key, _ in parse_qsl(parsed.query, keep_blank_values=True)):
            raise ResourceValidationError("video URL query must not contain credentials")
        return "url", clean
    return "local_path", clean


def _json_strings(value: Any, *, field: str, maximum: int = 64) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Iterable):
        raise ResourceValidationError(f"{field} must be a list of strings")
    result: list[str] = []
    for item in value:
        clean = _clean(item, maximum=200, required=True)
        if clean not in result:
            result.append(clean)
        if len(result) > maximum:
            raise ResourceValidationError(f"{field} contains too many values")
    return tuple(result)


class ResourceRegistry:
    """CRUD, search, and deterministic answers for protected local metadata."""

    def __init__(self, config: Config, access: Any) -> None:
        self.config = config
        self.access = access

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.config.db_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def _require_manage(self, principal: Any, resource_type: str, action: str) -> None:
        del resource_type
        if not self.access.has_permission(principal, action):
            raise PermissionError("resource is unavailable")

    def _snapshot(self, principal: Any) -> Any:
        if hasattr(principal, "user_id") and hasattr(principal, "revision"):
            return principal
        return self.access.principal_snapshot(str(principal))

    def _user_id(self, principal: Any) -> str:
        return str(self._snapshot(principal).user_id)

    def _touch_resource(
        self,
        principal: Any,
        resource_type: str,
        resource_id: str,
        *,
        classification: str | None = None,
        active: bool | None = None,
    ) -> None:
        """Invalidate sessions and cached answers after protected data changes."""

        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT stable_key, classification, owner_user_id, active, metadata_json
                FROM protected_resources
                WHERE resource_type = ? AND resource_id = ?
                """,
                (resource_type, resource_id),
            ).fetchone()
        if row is None:
            raise RuntimeError("protected resource registration is missing")
        try:
            metadata = json.loads(row["metadata_json"] or "{}")
        except (TypeError, ValueError, json.JSONDecodeError):
            metadata = {}
        if not isinstance(metadata, dict):
            metadata = {}
        metadata["content_updated_at"] = _now()
        self.access.register_resource(
            resource_type,
            resource_id,
            stable_key=row["stable_key"],
            classification=(
                self._classification(classification)
                if classification is not None
                else row["classification"]
            ),
            owner_user_id=row["owner_user_id"],
            active=bool(row["active"]) if active is None else bool(active),
            metadata=metadata,
            actor_user_id=self._user_id(principal),
        )

    @staticmethod
    def _classification(value: Any) -> str:
        clean = _clean(value or "internal", maximum=40).casefold()
        if clean not in {"public", "internal", "confidential", "restricted"}:
            raise ResourceValidationError(
                "classification must be public, internal, confidential, or restricted"
            )
        return clean

    def create_person(
        self,
        principal: Any,
        name: str,
        *,
        person_id: str = "",
        aliases: Sequence[str] = (),
        organization: str = "",
        email: str = "",
        metadata: Mapping[str, Any] | None = None,
        classification: str = "internal",
    ) -> PersonRecord:
        self._require_manage(principal, "org_role", "org_role.manage")
        person_id = _identifier(person_id, "person")
        canonical_name = _clean(name, maximum=300, required=True)
        organization = _clean(organization, maximum=300)
        email = _clean(email, maximum=320)
        alias_values = _json_strings(aliases, field="aliases")
        classification = self._classification(classification)
        now = _now()
        actor = self._user_id(principal)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO people(
                    person_id, canonical_name, organization, email, metadata_json,
                    active, created_by_user_id, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, 1, ?, ?, ?)
                """,
                (
                    person_id,
                    canonical_name,
                    organization,
                    email,
                    json.dumps(dict(metadata or {}), ensure_ascii=False, sort_keys=True),
                    actor,
                    now,
                    now,
                ),
            )
            connection.executemany(
                """
                INSERT INTO person_aliases(
                    alias_id, person_id, alias, normalized_alias, match_mode, created_at
                ) VALUES (?, ?, ?, ?, 'casefold', ?)
                """,
                (
                    (f"alias_{uuid.uuid4().hex}", person_id, item, _normalized(item), now)
                    for item in alias_values
                ),
            )
        try:
            self.access.register_resource(
                "person",
                person_id,
                stable_key=person_id,
                classification=classification,
                owner_user_id=actor,
                actor_user_id=actor,
            )
        except Exception:
            with self._connect() as connection:
                connection.execute("DELETE FROM people WHERE person_id = ?", (person_id,))
            raise
        return self.person(principal, person_id, include_inactive=True)

    def person(
        self,
        principal: Any,
        person_id: str,
        *,
        include_inactive: bool = False,
    ) -> PersonRecord:
        # A role grant exposes only the name carried by that role. Full
        # directory fields and aliases require access to the protected person
        # record so aliases from a restricted context cannot leak sideways.
        manager = self.access.has_permission(principal, "org_role.manage")
        person_id = _identifier(person_id, "person")
        if not manager and not self.access.can(
            principal, "person", person_id, "org_role.read"
        ):
            raise KeyError("person not found")
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM people WHERE person_id = ?",
                (person_id,),
            ).fetchone()
            if row is None or (not include_inactive and not bool(row["active"])):
                raise KeyError("person not found")
            aliases = tuple(
                item["alias"]
                for item in connection.execute(
                    "SELECT alias FROM person_aliases WHERE person_id = ? ORDER BY created_at, alias_id",
                    (person_id,),
                ).fetchall()
            )
        return PersonRecord(
            person_id=row["person_id"],
            display_name=row["canonical_name"],
            aliases=aliases,
            organization=row["organization"],
            active=bool(row["active"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def update_person(
        self,
        principal: Any,
        person_id: str,
        *,
        name: str | None = None,
        aliases: Sequence[str] | None = None,
        organization: str | None = None,
        email: str | None = None,
        classification: str | None = None,
    ) -> PersonRecord:
        self._require_manage(principal, "org_role", "org_role.manage")
        person_id = _identifier(person_id, "person")
        updates: dict[str, Any] = {"updated_at": _now()}
        if name is not None:
            updates["canonical_name"] = _clean(name, maximum=300, required=True)
        if organization is not None:
            updates["organization"] = _clean(organization, maximum=300)
        if email is not None:
            updates["email"] = _clean(email, maximum=320)
        with self._connect() as connection:
            cursor = connection.execute(
                f"UPDATE people SET {', '.join(f'{key} = ?' for key in updates)} WHERE person_id = ?",
                (*updates.values(), person_id),
            )
            if cursor.rowcount != 1:
                raise KeyError("person not found")
            if aliases is not None:
                values = _json_strings(aliases, field="aliases")
                connection.execute("DELETE FROM person_aliases WHERE person_id = ?", (person_id,))
                connection.executemany(
                    """
                    INSERT INTO person_aliases(
                        alias_id, person_id, alias, normalized_alias, match_mode, created_at
                    ) VALUES (?, ?, ?, ?, 'casefold', ?)
                    """,
                    (
                        (f"alias_{uuid.uuid4().hex}", person_id, item, _normalized(item), _now())
                        for item in values
                    ),
                )
        self._touch_resource(
            principal,
            "person",
            person_id,
            classification=classification,
        )
        return self.person(principal, person_id, include_inactive=True)

    def list_people(
        self,
        principal: Any,
        query: str = "",
        *,
        include_inactive: bool = False,
    ) -> tuple[PersonRecord, ...]:
        # A concrete role-read grant permits access to that role, not directory
        # enumeration. Searching people therefore needs the same explicit
        # capability as searching organizational roles.
        if not self.access.has_permission(principal, "org_role.search"):
            return ()
        needle = _normalized(query)
        with self._connect() as connection:
            identifiers = tuple(
                row["person_id"]
                for row in connection.execute(
                    "SELECT person_id FROM people ORDER BY canonical_name, person_id"
                ).fetchall()
            )
        result: list[PersonRecord] = []
        for person_id in identifiers:
            try:
                item = self.person(
                    principal, person_id, include_inactive=include_inactive
                )
            except KeyError:
                continue
            haystack = _normalized(
                " ".join((item.display_name, item.organization, *item.aliases))
            )
            if needle and needle not in haystack:
                continue
            result.append(item)
        return tuple(result)

    def deactivate_person(self, principal: Any, person_id: str) -> None:
        self._require_manage(principal, "org_role", "org_role.manage")
        person_id = _identifier(person_id, "person")
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE people SET active = 0, updated_at = ? WHERE person_id = ?",
                (_now(), person_id),
            )
            if cursor.rowcount != 1:
                raise KeyError("person not found")
        self._touch_resource(principal, "person", person_id, active=False)

    def add_role(
        self,
        principal: Any,
        person_id: str,
        role_name: str,
        *,
        role_id: str = "",
        organization: str = "",
        scope: str = "",
        responsibility: str = "",
        target_person_id: str = "",
        valid_from: str = "",
        valid_until: str = "",
        status: str = "asserted",
        provenance_kind: str = "manual",
        source_document_revision_id: str = "",
        source_block_id: str = "",
        provenance_note: str = "",
        classification: str = "internal",
        supersedes_role_id: str = "",
    ) -> OrganizationalRole:
        self._require_manage(principal, "org_role", "org_role.manage")
        role_id = _identifier(role_id, "role")
        person_id = _identifier(person_id, "person")
        role_name = _clean(role_name, maximum=300, required=True)
        organization = _clean(organization, maximum=300)
        scope = _clean(scope, maximum=500)
        responsibility = _clean(responsibility, maximum=2_000)
        target_person_id = _identifier(target_person_id, "person") if target_person_id else ""
        start = _iso_date(valid_from, field="valid_from")
        end = _iso_date(valid_until, field="valid_until")
        if start and end and start >= end:
            raise ResourceValidationError("valid_until must be later than valid_from")
        status = _clean(status, maximum=40).casefold() or "asserted"
        if status not in {"asserted", "disputed", "superseded", "retracted"}:
            raise ResourceValidationError("invalid role status")
        provenance_kind = _clean(provenance_kind, maximum=40).casefold() or "manual"
        if provenance_kind == "import":
            provenance_kind = "imported"
        if provenance_kind not in {"manual", "document", "imported"}:
            raise ResourceValidationError("provenance_kind must be manual, document, or imported")
        source_document_revision_id = _clean(source_document_revision_id, maximum=160)
        source_block_id = _clean(source_block_id, maximum=160)
        if provenance_kind == "document" and not (
            source_document_revision_id and source_block_id
        ):
            raise ResourceValidationError(
                "document-derived roles require source document revision and block IDs"
            )
        classification = self._classification(classification)
        now = _now()
        actor = self._user_id(principal)
        with self._connect() as connection:
            person = connection.execute(
                "SELECT 1 FROM people WHERE person_id = ? AND active = 1", (person_id,)
            ).fetchone()
            if person is None:
                raise ResourceValidationError("person does not exist or is inactive")
            if target_person_id and connection.execute(
                "SELECT 1 FROM people WHERE person_id = ? AND active = 1",
                (target_person_id,),
            ).fetchone() is None:
                raise ResourceValidationError("target person does not exist or is inactive")
            if provenance_kind == "document":
                source = connection.execute(
                    """
                    SELECT 1 FROM source_blocks
                    WHERE block_id = ? AND document_revision_id = ?
                    """,
                    (source_block_id, source_document_revision_id),
                ).fetchone()
                if source is None:
                    raise ResourceValidationError("document provenance does not identify a valid block")
            connection.execute(
                """
                INSERT INTO organizational_roles(
                    role_id, person_id, role_name, organization, scope,
                    responsibility, target_person_id, valid_from, valid_until,
                    status, provenance_kind, source_document_revision_id,
                    source_block_id, provenance_note, supersedes_role_id, active,
                    created_by_user_id, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?)
                """,
                (
                    role_id,
                    person_id,
                    role_name,
                    organization,
                    scope,
                    responsibility,
                    target_person_id or None,
                    start or None,
                    end or None,
                    status,
                    provenance_kind,
                    source_document_revision_id or None,
                    source_block_id or None,
                    _clean(provenance_note, maximum=2_000),
                    _identifier(supersedes_role_id, "role") if supersedes_role_id else None,
                    actor,
                    now,
                    now,
                ),
            )
        try:
            self.access.register_resource(
                "org_role",
                role_id,
                stable_key=role_id,
                classification=classification,
                owner_user_id=actor,
                actor_user_id=actor,
            )
        except Exception:
            with self._connect() as connection:
                connection.execute(
                    "DELETE FROM organizational_roles WHERE role_id = ?", (role_id,)
                )
            raise
        return self.role(principal, role_id, include_inactive=True)

    def role(
        self,
        principal: Any,
        role_id: str,
        *,
        include_inactive: bool = False,
    ) -> OrganizationalRole:
        role_id = _identifier(role_id, "role")
        if not self.access.can(principal, "org_role", role_id, "org_role.read"):
            raise KeyError("role not found")
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT r.*, p.canonical_name AS person_name,
                       p.active AS person_active,
                       d.logical_document_id AS source_logical_document_id,
                       pr.classification AS access_classification
                FROM organizational_roles AS r
                JOIN people AS p ON p.person_id = r.person_id
                LEFT JOIN document_revisions AS d
                  ON d.document_revision_id = r.source_document_revision_id
                JOIN protected_resources AS pr
                  ON pr.resource_type = 'org_role' AND pr.resource_id = r.role_id
                WHERE r.role_id = ?
                """,
                (role_id,),
            ).fetchone()
        if row is None or (not include_inactive and not bool(row["active"])):
            raise KeyError("role not found")
        return self._role_record(principal, row)

    def _role_record(self, principal: Any, row: sqlite3.Row) -> OrganizationalRole:
        provenance_ref = row["provenance_note"]
        if row["provenance_kind"] == "document":
            logical_id = str(row["source_logical_document_id"] or "")
            source_visible = bool(logical_id) and all(
                self.access.can(principal, "document", logical_id, action)
                for action in ("document.read", "document.cite")
            )
            provenance_ref = (
                f"{row['source_document_revision_id']}:{row['source_block_id']}"
                if source_visible
                else ""
            )
        return OrganizationalRole(
            role_id=row["role_id"],
            person_id=row["person_id"],
            person_name=row["person_name"],
            role_name=row["role_name"],
            responsibility=row["responsibility"],
            organization=row["organization"],
            valid_from=row["valid_from"] or "",
            valid_to=row["valid_until"] or "",
            status=row["status"],
            provenance_type=row["provenance_kind"],
            provenance_ref=provenance_ref,
            access_classification=row["access_classification"],
            active=bool(row["active"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def search_roles(
        self,
        principal: Any,
        query: str = "",
        *,
        include_history: bool = False,
        as_of: str = "",
        limit: int = 50,
    ) -> tuple[OrganizationalRole, ...]:
        if not self.access.has_permission(principal, "org_role.search"):
            return ()
        limit = max(1, min(int(limit), 200))
        now = _iso_date(as_of, field="as_of") or _now()
        query_tokens = _tokens(query)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT r.*, p.canonical_name AS person_name,
                       GROUP_CONCAT(a.alias, ' | ') AS aliases,
                       p.active AS person_active,
                       d.logical_document_id AS source_logical_document_id,
                       pr.classification AS access_classification
                FROM organizational_roles AS r
                JOIN people AS p ON p.person_id = r.person_id
                LEFT JOIN person_aliases AS a ON a.person_id = p.person_id
                LEFT JOIN document_revisions AS d
                  ON d.document_revision_id = r.source_document_revision_id
                JOIN protected_resources AS pr
                  ON pr.resource_type = 'org_role' AND pr.resource_id = r.role_id
                WHERE pr.active = 1
                GROUP BY r.role_id
                ORDER BY r.updated_at DESC, r.role_id
                """
            ).fetchall()
        scored: list[tuple[float, sqlite3.Row]] = []
        directory_manager = self.access.has_permission(principal, "org_role.manage")
        for row in rows:
            if not self.access.can(principal, "org_role", row["role_id"], "org_role.read"):
                continue
            if not include_history:
                if (
                    not bool(row["active"])
                    or not bool(row["person_active"])
                    or row["status"] not in {"asserted", "disputed"}
                ):
                    continue
                if row["valid_from"] and row["valid_from"] > now:
                    continue
                if row["valid_until"] and now >= row["valid_until"]:
                    continue
            alias_text = ""
            if directory_manager or self.access.can(
                principal, "person", row["person_id"], "org_role.read"
            ):
                alias_text = str(row["aliases"] or "")
            haystack = " ".join((
                str(row["person_name"] or ""), alias_text,
                str(row["role_name"] or ""), str(row["organization"] or ""),
                str(row["scope"] or ""), str(row["responsibility"] or ""),
                str(row["provenance_note"] or ""),
            ))
            hay_tokens = set(_tokens(haystack))
            if query_tokens:
                matched = sum(1 for token in query_tokens if token in hay_tokens)
                phrase = 2.0 if _normalized(query) in _normalized(haystack) else 0.0
                if not matched and not phrase:
                    continue
                score = phrase + matched / max(1, len(set(query_tokens)))
            else:
                score = 0.0
            scored.append((score, row))
        scored.sort(key=lambda item: (-item[0], item[1]["person_name"].casefold(), item[1]["role_id"]))
        return tuple(self._role_record(principal, row) for _, row in scored[:limit])

    def update_role(self, principal: Any, role_id: str, **changes: Any) -> OrganizationalRole:
        """Create a successor role and preserve the prior assignment as history."""

        self._require_manage(principal, "org_role", "org_role.manage")
        allowed_changes = {
            "person_id", "role_name", "new_role_id", "organization", "scope",
            "responsibility", "target_person_id", "valid_from", "valid_until",
            "status", "provenance_kind", "source_document_revision_id",
            "source_block_id", "provenance_note", "classification",
        }
        unknown = set(changes) - allowed_changes
        if unknown:
            raise ResourceValidationError(
                f"unknown role fields: {', '.join(sorted(unknown))}"
            )
        old = self.role(principal, role_id, include_inactive=True)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM organizational_roles WHERE role_id = ?", (old.role_id,)
            ).fetchone()
            classification_row = connection.execute(
                """
                SELECT classification FROM protected_resources
                WHERE resource_type = 'org_role' AND resource_id = ?
                """,
                (old.role_id,),
            ).fetchone()
        if row is None:
            raise KeyError("role not found")
        successor = self.add_role(
            principal,
            changes.pop("person_id", row["person_id"]),
            changes.pop("role_name", row["role_name"]),
            role_id=changes.pop("new_role_id", ""),
            organization=changes.pop("organization", row["organization"]),
            scope=changes.pop("scope", row["scope"]),
            responsibility=changes.pop("responsibility", row["responsibility"]),
            target_person_id=changes.pop("target_person_id", row["target_person_id"] or ""),
            valid_from=changes.pop("valid_from", _now()),
            valid_until=changes.pop("valid_until", row["valid_until"] or ""),
            status=changes.pop("status", "asserted"),
            provenance_kind=changes.pop("provenance_kind", row["provenance_kind"]),
            source_document_revision_id=changes.pop(
                "source_document_revision_id", row["source_document_revision_id"] or ""
            ),
            source_block_id=changes.pop("source_block_id", row["source_block_id"] or ""),
            provenance_note=changes.pop("provenance_note", row["provenance_note"]),
            classification=changes.pop(
                "classification", classification_row["classification"] if classification_row else "internal"
            ),
            supersedes_role_id=old.role_id,
        )
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE organizational_roles
                SET status = 'superseded', active = 0,
                    valid_until = COALESCE(valid_until, ?), updated_at = ?
                WHERE role_id = ?
                """,
                (successor.valid_from or _now(), _now(), old.role_id),
            )
        self._touch_resource(principal, "org_role", old.role_id)
        return successor

    def end_role(self, principal: Any, role_id: str, *, ended_at: str = "") -> OrganizationalRole:
        self._require_manage(principal, "org_role", "org_role.manage")
        role_id = _identifier(role_id, "role")
        ended = _iso_date(ended_at, field="ended_at") or _now()
        with self._connect() as connection:
            current = connection.execute(
                "SELECT valid_from FROM organizational_roles WHERE role_id = ? AND active = 1",
                (role_id,),
            ).fetchone()
            if current is None:
                raise KeyError("role not found")
            if current["valid_from"] and ended <= current["valid_from"]:
                raise ResourceValidationError("ended_at must be later than valid_from")
            cursor = connection.execute(
                """
                UPDATE organizational_roles
                SET valid_until = ?, updated_at = ?
                WHERE role_id = ? AND active = 1
                """,
                (ended, _now(), role_id),
            )
            if cursor.rowcount != 1:
                raise KeyError("role not found")
        self._touch_resource(principal, "org_role", role_id)
        return self.role(principal, role_id, include_inactive=True)

    def deactivate_role(self, principal: Any, role_id: str) -> None:
        self._require_manage(principal, "org_role", "org_role.manage")
        role_id = _identifier(role_id, "role")
        self.role(principal, role_id, include_inactive=True)
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE organizational_roles
                SET active = 0, status = 'retracted', updated_at = ?
                WHERE role_id = ?
                """,
                (_now(), role_id),
            )
            if cursor.rowcount != 1:
                raise KeyError("role not found")
        # The assignment is inactive for current queries, while its protected
        # resource remains readable to authorized users as historical data.
        self._touch_resource(principal, "org_role", role_id)

    def import_video(
        self,
        principal: Any,
        metadata: Mapping[str, Any],
    ) -> VideoRecord:
        """Validate and persist one supplied metadata record without opening its locator."""

        self._require_manage(principal, "video", "video.manage")
        if not isinstance(metadata, Mapping):
            raise ResourceValidationError("video metadata must be a JSON object")
        video_id = _identifier(metadata.get("video_id") or metadata.get("id"), "video")
        external_id = _clean(metadata.get("external_id") or video_id, maximum=300, required=True)
        title = _clean(metadata.get("title"), maximum=500, required=True)
        description = _clean(metadata.get("description"), maximum=8_000)
        raw_locator = metadata.get("url") or metadata.get("path") or metadata.get("locator")
        locator_kind, locator = _locator(raw_locator)
        speaker = _clean(metadata.get("speaker"), maximum=500)
        recorded_at = _recorded_at(
            metadata.get("date") or metadata.get("recorded_at"), field="recorded_at"
        )
        raw_duration = metadata.get("duration_seconds", metadata.get("duration"))
        duration = (
            parse_timestamp(raw_duration, field="duration")
            if raw_duration is not None and raw_duration != ""
            else None
        )
        tags = _json_strings(metadata.get("tags", ()), field="tags")
        project = _clean(metadata.get("project"), maximum=500)
        topic_keywords = _json_strings(
            metadata.get("topic_keywords", metadata.get("keywords", ())),
            field="topic_keywords",
        )
        summary = _clean(
            metadata.get("summary", metadata.get("optional_summary")), maximum=12_000
        )
        classification = self._classification(
            metadata.get("access_classification", metadata.get("classification", "internal"))
        )
        raw_chapters = metadata.get("chapters", ())
        if isinstance(raw_chapters, (str, bytes, Mapping)) or not isinstance(raw_chapters, Iterable):
            raise ResourceValidationError("chapters must be a list")
        chapters: list[dict[str, Any]] = []
        previous_start = -1
        for ordinal, raw in enumerate(raw_chapters):
            if not isinstance(raw, Mapping):
                raise ResourceValidationError("each chapter must be an object")
            start = parse_timestamp(
                raw.get("start_seconds", raw.get("start")), field=f"chapter {ordinal + 1} start"
            )
            raw_end = raw.get("end_seconds", raw.get("end"))
            end = (
                parse_timestamp(raw_end, field=f"chapter {ordinal + 1} end")
                if raw_end is not None and raw_end != ""
                else None
            )
            if start < previous_start:
                raise ResourceValidationError("chapters must be ordered by start time")
            if end is not None and end <= start:
                raise ResourceValidationError("chapter end must be later than its start")
            if duration is not None and (start > duration or (end is not None and end > duration)):
                raise ResourceValidationError("chapter timestamp is outside video duration")
            previous_start = start
            chapters.append({
                "chapter_id": _identifier(raw.get("chapter_id") or raw.get("id"), "chapter"),
                "ordinal": ordinal,
                "title": _clean(raw.get("title"), maximum=500, required=True),
                "description": _clean(raw.get("description"), maximum=4_000),
                "start_seconds": start,
                "end_seconds": end,
                "topic_keywords": _json_strings(
                    raw.get("topic_keywords", raw.get("keywords", ())),
                    field="chapter keywords",
                ),
            })
        fingerprint_payload = {
            "external_id": external_id,
            "title": title,
            "locator_kind": locator_kind,
            "locator": locator,
            "speaker": speaker,
            "recorded_at": recorded_at,
            "duration_seconds": duration,
            "project": project,
            "chapters": chapters,
        }
        fingerprint = hashlib.sha256(
            json.dumps(fingerprint_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        now = _now()
        actor = self._user_id(principal)
        try:
            with self._connect() as connection:
                duplicate = connection.execute(
                    "SELECT video_id FROM videos WHERE external_id = ? OR metadata_sha256 = ?",
                    (external_id, fingerprint),
                ).fetchone()
                if duplicate is not None:
                    raise ResourceValidationError("duplicate video metadata record")
                connection.execute(
                    """
                    INSERT INTO videos(
                        video_id, external_id, title, description, locator_kind,
                        locator, speaker, recorded_at, duration_seconds, tags_json,
                        project, topic_keywords_json, optional_summary,
                        metadata_sha256, active, created_by_user_id, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?)
                    """,
                    (
                        video_id,
                        external_id,
                        title,
                        description,
                        locator_kind,
                        locator,
                        speaker,
                        recorded_at or None,
                        duration,
                        json.dumps(tags, ensure_ascii=False),
                        project,
                        json.dumps(topic_keywords, ensure_ascii=False),
                        summary,
                        fingerprint,
                        actor,
                        now,
                        now,
                    ),
                )
                connection.executemany(
                    """
                    INSERT INTO video_chapters(
                        chapter_id, video_id, ordinal, title, description,
                        start_seconds, end_seconds, topic_keywords_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        (
                            item["chapter_id"],
                            video_id,
                            item["ordinal"],
                            item["title"],
                            item["description"],
                            item["start_seconds"],
                            item["end_seconds"],
                            json.dumps(item["topic_keywords"], ensure_ascii=False),
                        )
                        for item in chapters
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise ResourceValidationError("duplicate or invalid video metadata") from exc
        try:
            self.access.register_resource(
                "video",
                video_id,
                stable_key=video_id,
                classification=classification,
                owner_user_id=actor,
                actor_user_id=actor,
            )
        except Exception:
            with self._connect() as connection:
                connection.execute("DELETE FROM videos WHERE video_id = ?", (video_id,))
            raise
        return self.video(principal, video_id, include_inactive=True)

    def video(
        self,
        principal: Any,
        video_id: str,
        *,
        include_inactive: bool = False,
    ) -> VideoRecord:
        video_id = _identifier(video_id, "video")
        if not self.access.can(principal, "video", video_id, "video.metadata.read"):
            raise KeyError("video not found")
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT v.*, pr.classification AS access_classification
                FROM videos AS v
                JOIN protected_resources AS pr
                  ON pr.resource_type = 'video' AND pr.resource_id = v.video_id
                WHERE v.video_id = ?
                """,
                (video_id,),
            ).fetchone()
            chapter_rows = connection.execute(
                "SELECT * FROM video_chapters WHERE video_id = ? ORDER BY ordinal",
                (video_id,),
            ).fetchall()
        if row is None or (not include_inactive and not bool(row["active"])):
            raise KeyError("video not found")
        return self._video_record(row, chapter_rows, include_locator=self.access.can(
            principal, "video", video_id, "video.open"
        ))

    @staticmethod
    def _video_record(
        row: sqlite3.Row,
        chapters: Sequence[sqlite3.Row],
        *,
        include_locator: bool,
    ) -> VideoRecord:
        return VideoRecord(
            video_id=row["video_id"],
            title=row["title"],
            description=row["description"],
            uri=row["locator"] if include_locator else "",
            speaker=row["speaker"],
            recorded_date=row["recorded_at"] or "",
            duration_seconds=row["duration_seconds"],
            tags=tuple(json.loads(row["tags_json"] or "[]")),
            project=row["project"],
            summary=row["optional_summary"],
            access_classification=row["access_classification"],
            chapters=tuple(
                VideoChapter(
                    chapter_id=item["chapter_id"],
                    video_id=item["video_id"],
                    title=item["title"],
                    description=item["description"],
                    start_seconds=int(item["start_seconds"]),
                    end_seconds=item["end_seconds"],
                    keywords=tuple(json.loads(item["topic_keywords_json"] or "[]")),
                )
                for item in chapters
            ),
            active=bool(row["active"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def search_videos(
        self,
        principal: Any,
        query: str = "",
        *,
        limit: int = 20,
    ) -> tuple[tuple[VideoRecord, VideoChapter | None, float], ...]:
        if not self.access.has_permission(principal, "video.search"):
            return ()
        limit = max(1, min(int(limit), 100))
        query_tokens = _tokens(query)
        with self._connect() as connection:
            videos = connection.execute(
                """
                SELECT v.*, pr.classification AS access_classification
                FROM videos AS v
                JOIN protected_resources AS pr
                  ON pr.resource_type = 'video' AND pr.resource_id = v.video_id
                WHERE v.active = 1 AND pr.active = 1
                ORDER BY v.recorded_at DESC, v.video_id
                """
            ).fetchall()
            chapters_by_video: dict[str, list[sqlite3.Row]] = {}
            for chapter in connection.execute(
                "SELECT * FROM video_chapters ORDER BY video_id, ordinal"
            ).fetchall():
                chapters_by_video.setdefault(chapter["video_id"], []).append(chapter)
        result: list[tuple[VideoRecord, VideoChapter | None, float]] = []
        for row in videos:
            video_id = row["video_id"]
            if not self.access.can(principal, "video", video_id, "video.metadata.read"):
                continue
            chapter_rows = chapters_by_video.get(video_id, [])
            base_text = " ".join(str(row[key] or "") for key in (
                "title", "description", "speaker", "project", "optional_summary",
                "tags_json", "topic_keywords_json",
            ))
            base_tokens = set(_tokens(base_text))
            base_match = sum(1 for token in query_tokens if token in base_tokens)
            best_chapter: sqlite3.Row | None = None
            best_chapter_match = 0
            for chapter in chapter_rows:
                chapter_text = " ".join(str(chapter[key] or "") for key in (
                    "title", "description", "topic_keywords_json",
                ))
                match = sum(1 for token in query_tokens if token in set(_tokens(chapter_text)))
                if match > best_chapter_match:
                    best_chapter, best_chapter_match = chapter, match
            phrase = 2.0 if query and _normalized(query) in _normalized(base_text) else 0.0
            if query_tokens and not (base_match or best_chapter_match or phrase):
                continue
            score = phrase + (base_match + 1.5 * best_chapter_match) / max(1, len(set(query_tokens)))
            record = self._video_record(
                row,
                chapter_rows,
                include_locator=self.access.can(principal, "video", video_id, "video.open"),
            )
            chapter_record = None
            if best_chapter is not None:
                chapter_record = next(
                    item for item in record.chapters if item.chapter_id == best_chapter["chapter_id"]
                )
            result.append((record, chapter_record, score))
        result.sort(key=lambda item: (-item[2], item[0].title.casefold(), item[0].video_id))
        return tuple(result[:limit])

    def deactivate_video(self, principal: Any, video_id: str) -> None:
        self._require_manage(principal, "video", "video.manage")
        current = self.video(principal, video_id, include_inactive=True)
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE videos SET active = 0, updated_at = ? WHERE video_id = ?",
                (_now(), current.video_id),
            )
            if cursor.rowcount != 1:
                raise KeyError("video not found")
        self.access.register_resource(
            "video",
            current.video_id,
            classification=current.access_classification,
            active=False,
            actor_user_id=self._user_id(principal),
        )

    @staticmethod
    def _coverage(
        snapshot_id: str,
        *,
        sources: int,
        complete: bool = True,
        incomplete_reason: str = "",
    ) -> Coverage:
        return Coverage(
            snapshot_id=snapshot_id,
            mode="targeted",
            authorized_documents=0,
            manifests_screened=0,
            documents_queued=0,
            documents_read=0,
            documents_fully_read=0,
            documents_remaining=0,
            sections_seen=0,
            sections_total_for_opened_documents=0,
            exact_match_documents=0,
            evidence_cards=sources,
            exhaustive=False,
            complete=complete,
            provisional=not complete,
            incomplete_reasons=(incomplete_reason,) if incomplete_reason else (),
        )

    def answer_roles(
        self,
        principal: Any,
        question: str,
        *,
        snapshot_id: str,
    ) -> Answer | None:
        if not _ROLE_INTENT.search(question) or _VIDEO_INTENT.search(question):
            return None
        if _DOCUMENT_INTENT.search(question):
            return None
        matches = self.search_roles(principal, question, limit=12)
        if not matches:
            # A role-shaped question can still be asking what source documents
            # say about a project participant. Fall through to document
            # research unless an authorized registry record actually matches.
            return None
        # Preserve competing assignments instead of silently selecting one.
        citations: list[Citation] = []
        lines: list[str] = []
        grouped: dict[tuple[str, str], set[str]] = {}
        truncated = len(matches) > 8
        for index, item in enumerate(matches[:8], 1):
            details = item.responsibility or item.role_name
            scope = f" for {item.organization}" if item.organization else ""
            validity = ""
            if item.valid_from or item.valid_to:
                validity = f" (valid {item.valid_from or 'until further notice'}"
                validity += f" to {item.valid_to})" if item.valid_to else ")"
            status_label = "Disputed assignment: " if item.status == "disputed" else ""
            lines.append(
                f"- {status_label}{item.person_name}: {details}{scope}{validity}. [S{index}]"
            )
            provenance = (
                "manually registered"
                if item.provenance_type == "manual"
                else "imported"
                if item.provenance_type == "imported"
                else "document-derived"
            )
            quote = (
                f"Registered role: {item.person_name} | {item.role_name}"
                + (f" | {item.responsibility}" if item.responsibility else "")
                + (f" | {item.organization}" if item.organization else "")
                + f" | Status: {item.status}"
                + (f" | Valid from: {item.valid_from}" if item.valid_from else "")
                + (f" | Valid until: {item.valid_to}" if item.valid_to else "")
                + f" | Provenance: {provenance}"
            )
            citations.append(Citation(
                source_id=f"S{index}",
                card_id=item.role_id,
                block_id=item.role_id,
                document_revision_id="",
                title="Organizational role registry",
                locator=f"role {item.role_id}",
                quote=quote,
                source_path="",
                quote_sha256=hashlib.sha256(quote.encode("utf-8")).hexdigest(),
                resource_type="org_role",
                resource_id=item.role_id,
                provenance=provenance,
            ))
            grouped.setdefault(
                (_normalized(item.role_name), _normalized(item.organization)), set()
            ).add(item.person_id)
        warnings: list[str] = []
        if any(len(values) > 1 for values in grouped.values()):
            warnings.append(
                "More than one current assignment matches this role; competing visible records are shown."
            )
        if any(item.status == "disputed" for item in matches[:8]):
            warnings.append("One or more displayed assignments are marked as disputed.")
        if truncated:
            warnings.append("More matching role records exist than this answer displays.")
        return Answer(
            status="partial" if truncated else "answer",
            text="\n".join(lines),
            sources=tuple(citations),
            warnings=tuple(warnings),
            coverage=self._coverage(
                snapshot_id,
                sources=len(citations),
                complete=not truncated,
                incomplete_reason="role_result_limit" if truncated else "",
            ),
            debug={
                "answer_route": "organizational_roles",
                "authorization_revision": self._snapshot(principal).revision,
            },
        )

    def answer_videos(
        self,
        principal: Any,
        question: str,
        *,
        snapshot_id: str,
    ) -> Answer | None:
        if not _VIDEO_INTENT.search(question):
            return None
        if _DOCUMENT_INTENT.search(question):
            return None
        matches = self.search_videos(principal, question, limit=8)
        if not matches:
            # A video-shaped phrase may still refer to information contained
            # in documents. Only take the metadata route on a positive match.
            return None
        lines: list[str] = []
        citations: list[Citation] = []
        truncated = len(matches) > 5
        for index, (video, chapter, _) in enumerate(matches[:5], 1):
            timestamp = chapter.start_seconds if chapter is not None else 0
            chapter_text = f', chapter "{chapter.title}"' if chapter is not None else ""
            pointer = f" at {format_timestamp(timestamp)}" if chapter is not None else ""
            access_text = " Open the registered link." if video.uri else " The link is restricted."
            lines.append(
                f'- "{video.title}"{chapter_text}{pointer}.{access_text} [S{index}]'
            )
            quote = " | ".join(filter(None, (
                f"Title: {video.title}",
                f"Speaker: {video.speaker}" if video.speaker else "",
                f"Project: {video.project}" if video.project else "",
                f"Chapter: {chapter.title}" if chapter is not None else "",
                f"Chapter description: {chapter.description}" if chapter is not None and chapter.description else "",
                f"Start: {format_timestamp(timestamp)}" if chapter is not None else "",
                f"Summary: {video.summary}" if video.summary else "",
            )))
            citations.append(Citation(
                source_id=f"S{index}",
                card_id=chapter.chapter_id if chapter is not None else video.video_id,
                block_id=chapter.chapter_id if chapter is not None else video.video_id,
                document_revision_id="",
                title=video.title,
                locator=(
                    f"chapter {chapter.title}, {format_timestamp(timestamp)}"
                    if chapter is not None
                    else "video metadata"
                ),
                quote=quote,
                source_path="",
                quote_sha256=hashlib.sha256(quote.encode("utf-8")).hexdigest(),
                resource_type="video",
                resource_id=video.video_id,
                uri=video.uri,
                timestamp_seconds=timestamp if chapter is not None else None,
                provenance="supplied video metadata",
            ))
        return Answer(
            status="partial" if truncated else "answer",
            text="\n".join(lines),
            sources=tuple(citations),
            warnings=(
                ("More matching video records exist than this answer displays.",)
                if truncated
                else ()
            ),
            coverage=self._coverage(
                snapshot_id,
                sources=len(citations),
                complete=not truncated,
                incomplete_reason="video_result_limit" if truncated else "",
            ),
            debug={
                "answer_route": "video_metadata",
                "authorization_revision": self._snapshot(principal).revision,
            },
        )


__all__ = [
    "ResourceRegistry",
    "ResourceValidationError",
    "format_timestamp",
    "parse_timestamp",
]
