from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import uuid
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .config import Config


ACCESS_SCHEMA_VERSION = 2
LOCAL_ADMIN_USER_ID = "local-admin"

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}$")
_RESOURCE_TYPE = re.compile(r"^(?:\*|[a-z][a-z0-9_.-]{0,79})$")
_ACTION = re.compile(r"^(?:\*|[a-z][a-z0-9_.:-]{0,119})$")
_CLASSIFICATION = re.compile(r"^[a-z][a-z0-9_.-]{0,79}$")
_PRINCIPAL_TYPES = frozenset({"user", "group"})
_EFFECTS = frozenset({"allow", "deny"})

_SYSTEM_ROLE_PERMISSIONS: dict[str, tuple[str, ...]] = {
    "administrator": ("*",),
    "corpus_manager": (
        "question.ask",
        "document.search",
        "document.read",
        "document.cite",
        "document.manage",
        "video.search",
        "video.metadata.read",
        "video.open",
        "video.manage",
        "org_role.search",
        "org_role.read",
        "org_role.manage",
        "identity.manage",
        "corpus.manage",
        "trace.read_own",
    ),
    "researcher": (
        "question.ask",
        "document.search",
        "document.read",
        "document.cite",
        "video.search",
        "video.metadata.read",
        "video.open",
        "org_role.search",
        "org_role.read",
        "trace.read_own",
    ),
    "viewer": (
        "question.ask",
        "document.search",
        "document.read",
        "document.cite",
        "video.search",
        "video.metadata.read",
        "org_role.search",
        "org_role.read",
        "trace.read_own",
    ),
    "guest": (
        "question.ask",
        "document.search",
        "document.read",
        "document.cite",
        "video.search",
        "video.metadata.read",
        "org_role.search",
        "org_role.read",
    ),
    "trace_auditor": ("trace.read_any", "audit.read"),
}


@dataclass(frozen=True, slots=True)
class PrincipalSnapshot:
    user_id: str
    display_name: str
    known: bool
    active: bool
    revision: int
    evaluated_at: str
    group_ids: tuple[str, ...] = ()
    system_roles: tuple[str, ...] = ()
    permissions: tuple[str, ...] = ()

    @property
    def principal_keys(self) -> tuple[tuple[str, str], ...]:
        if not self.known or not self.active:
            return ()
        return (("user", self.user_id),) + tuple(
            ("group", group_id) for group_id in self.group_ids
        )


def _now(clock: Callable[[], datetime] | None = None) -> datetime:
    value = clock() if clock is not None else datetime.now(timezone.utc)
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _timestamp(
    value: datetime | str | None = None,
    *,
    clock: Callable[[], datetime] | None = None,
) -> str:
    if value is None:
        parsed = _now(clock)
    elif isinstance(value, datetime):
        parsed = value
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        parsed = parsed.astimezone(timezone.utc)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("timestamp cannot be empty")
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"invalid ISO-8601 timestamp: {value}") from exc
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        parsed = parsed.astimezone(timezone.utc)
    else:
        raise TypeError("timestamp must be a datetime, string, or None")
    return parsed.isoformat(timespec="microseconds")


def _optional_timestamp(value: datetime | str | None) -> str | None:
    return None if value is None else _timestamp(value)


def _identifier(value: str, name: str) -> str:
    clean = str(value or "").strip()
    if not _IDENTIFIER.fullmatch(clean):
        raise ValueError(f"{name} must be a safe identifier")
    return clean


def _resource_type(value: str) -> str:
    clean = str(value or "").strip().casefold()
    if not _RESOURCE_TYPE.fullmatch(clean):
        raise ValueError("resource_type is invalid")
    return clean


def _action(value: str) -> str:
    clean = str(value or "").strip().casefold()
    if not _ACTION.fullmatch(clean):
        raise ValueError("action is invalid")
    return clean


def _classification(value: str) -> str:
    clean = str(value or "internal").strip().casefold()
    if not _CLASSIFICATION.fullmatch(clean):
        raise ValueError("classification is invalid")
    return clean


def _text(value: str, name: str, maximum: int = 300, *, allow_empty: bool = False) -> str:
    clean = " ".join(str(value or "").split()).strip()
    if not clean and not allow_empty:
        raise ValueError(f"{name} cannot be empty")
    if len(clean) > maximum:
        raise ValueError(f"{name} cannot exceed {maximum} characters")
    return clean


def _json(value: Mapping[str, Any] | None) -> str:
    if value is None:
        return "{}"
    encoded = json.dumps(dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if len(encoded) > 32_000:
        raise ValueError("metadata cannot exceed 32,000 characters")
    return encoded


def _columns(connection: sqlite3.Connection, table: str) -> frozenset[str]:
    return frozenset(str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})"))


def _table_exists(connection: sqlite3.Connection, table: str) -> bool:
    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone()
    return row is not None


def _create_access_tables(connection: sqlite3.Connection) -> None:
    statements = (
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            checksum TEXT NOT NULL,
            applied_at TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS access_state (
            singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
            revision INTEGER NOT NULL CHECK(revision >= 1),
            updated_at TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            username TEXT NOT NULL UNIQUE COLLATE NOCASE,
            display_name TEXT NOT NULL,
            active INTEGER NOT NULL CHECK(active IN (0, 1)),
            credential_scheme TEXT NOT NULL DEFAULT '',
            credential_hash TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS access_groups (
            group_id TEXT PRIMARY KEY,
            display_name TEXT NOT NULL,
            active INTEGER NOT NULL CHECK(active IN (0, 1)),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS group_memberships (
            group_id TEXT NOT NULL REFERENCES access_groups(group_id) ON DELETE CASCADE,
            user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
            valid_from TEXT,
            valid_until TEXT,
            created_by_user_id TEXT REFERENCES users(user_id),
            created_at TEXT NOT NULL,
            PRIMARY KEY(group_id, user_id),
            CHECK(valid_until IS NULL OR valid_from IS NULL OR valid_until > valid_from)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS system_roles (
            role_name TEXT PRIMARY KEY,
            description TEXT NOT NULL,
            built_in INTEGER NOT NULL CHECK(built_in IN (0, 1))
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS system_role_permissions (
            role_name TEXT NOT NULL REFERENCES system_roles(role_name) ON DELETE CASCADE,
            action TEXT NOT NULL,
            PRIMARY KEY(role_name, action)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS principal_system_roles (
            principal_type TEXT NOT NULL CHECK(principal_type IN ('user', 'group')),
            principal_id TEXT NOT NULL,
            role_name TEXT NOT NULL REFERENCES system_roles(role_name) ON DELETE CASCADE,
            valid_from TEXT,
            valid_until TEXT,
            created_by_user_id TEXT REFERENCES users(user_id),
            created_at TEXT NOT NULL,
            PRIMARY KEY(principal_type, principal_id, role_name),
            CHECK(valid_until IS NULL OR valid_from IS NULL OR valid_until > valid_from)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS protected_resources (
            resource_type TEXT NOT NULL,
            resource_id TEXT NOT NULL,
            stable_key TEXT NOT NULL,
            classification TEXT NOT NULL,
            owner_user_id TEXT REFERENCES users(user_id),
            active INTEGER NOT NULL CHECK(active IN (0, 1)),
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY(resource_type, resource_id),
            UNIQUE(resource_type, stable_key)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS access_grants (
            grant_id TEXT PRIMARY KEY,
            principal_type TEXT NOT NULL CHECK(principal_type IN ('user', 'group')),
            principal_id TEXT NOT NULL,
            resource_type TEXT NOT NULL,
            resource_id TEXT NOT NULL,
            action TEXT NOT NULL,
            effect TEXT NOT NULL CHECK(effect IN ('allow', 'deny')),
            valid_from TEXT,
            valid_until TEXT,
            created_by_user_id TEXT REFERENCES users(user_id),
            created_at TEXT NOT NULL,
            revoked_at TEXT,
            CHECK(valid_until IS NULL OR valid_from IS NULL OR valid_until > valid_from)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS audit_events (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            actor_user_id TEXT REFERENCES users(user_id),
            action TEXT NOT NULL,
            target_type TEXT NOT NULL,
            target_id TEXT NOT NULL,
            outcome TEXT NOT NULL,
            details_json TEXT NOT NULL,
            access_revision INTEGER NOT NULL,
            occurred_at TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS people (
            person_id TEXT PRIMARY KEY,
            canonical_name TEXT NOT NULL,
            organization TEXT NOT NULL DEFAULT '',
            email TEXT NOT NULL DEFAULT '',
            metadata_json TEXT NOT NULL DEFAULT '{}',
            active INTEGER NOT NULL CHECK(active IN (0, 1)),
            created_by_user_id TEXT REFERENCES users(user_id),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS person_aliases (
            alias_id TEXT PRIMARY KEY,
            person_id TEXT NOT NULL REFERENCES people(person_id) ON DELETE CASCADE,
            alias TEXT NOT NULL,
            normalized_alias TEXT NOT NULL,
            match_mode TEXT NOT NULL CHECK(match_mode IN ('exact_case', 'casefold')),
            created_at TEXT NOT NULL,
            UNIQUE(person_id, normalized_alias, match_mode)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS organizational_roles (
            role_id TEXT PRIMARY KEY,
            person_id TEXT NOT NULL REFERENCES people(person_id),
            role_name TEXT NOT NULL,
            organization TEXT NOT NULL DEFAULT '',
            scope TEXT NOT NULL DEFAULT '',
            responsibility TEXT NOT NULL DEFAULT '',
            target_person_id TEXT REFERENCES people(person_id),
            valid_from TEXT,
            valid_until TEXT,
            status TEXT NOT NULL CHECK(status IN ('asserted', 'disputed', 'superseded', 'retracted')),
            provenance_kind TEXT NOT NULL CHECK(provenance_kind IN ('manual', 'document', 'imported')),
            source_document_revision_id TEXT REFERENCES document_revisions(document_revision_id),
            source_block_id TEXT REFERENCES source_blocks(block_id),
            provenance_note TEXT NOT NULL DEFAULT '',
            supersedes_role_id TEXT REFERENCES organizational_roles(role_id),
            active INTEGER NOT NULL CHECK(active IN (0, 1)),
            created_by_user_id TEXT REFERENCES users(user_id),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            CHECK(valid_until IS NULL OR valid_from IS NULL OR valid_until > valid_from),
            CHECK(
                provenance_kind != 'document'
                OR (source_document_revision_id IS NOT NULL AND source_block_id IS NOT NULL)
            )
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS videos (
            video_id TEXT PRIMARY KEY,
            external_id TEXT NOT NULL UNIQUE,
            title TEXT NOT NULL,
            description TEXT NOT NULL DEFAULT '',
            locator_kind TEXT NOT NULL CHECK(locator_kind IN ('local_path', 'url')),
            locator TEXT NOT NULL,
            speaker TEXT NOT NULL DEFAULT '',
            recorded_at TEXT,
            duration_seconds INTEGER CHECK(duration_seconds IS NULL OR duration_seconds >= 0),
            tags_json TEXT NOT NULL DEFAULT '[]',
            project TEXT NOT NULL DEFAULT '',
            topic_keywords_json TEXT NOT NULL DEFAULT '[]',
            optional_summary TEXT NOT NULL DEFAULT '',
            metadata_sha256 TEXT NOT NULL,
            active INTEGER NOT NULL CHECK(active IN (0, 1)),
            created_by_user_id TEXT REFERENCES users(user_id),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS video_chapters (
            chapter_id TEXT PRIMARY KEY,
            video_id TEXT NOT NULL REFERENCES videos(video_id) ON DELETE CASCADE,
            ordinal INTEGER NOT NULL CHECK(ordinal >= 0),
            title TEXT NOT NULL,
            description TEXT NOT NULL DEFAULT '',
            start_seconds INTEGER NOT NULL CHECK(start_seconds >= 0),
            end_seconds INTEGER CHECK(end_seconds IS NULL OR end_seconds > start_seconds),
            topic_keywords_json TEXT NOT NULL DEFAULT '[]',
            UNIQUE(video_id, ordinal)
        )
        """,
    )
    for statement in statements:
        connection.execute(statement)
    indexes = (
        "CREATE INDEX IF NOT EXISTS group_memberships_user ON group_memberships(user_id, group_id)",
        "CREATE INDEX IF NOT EXISTS principal_system_roles_lookup ON principal_system_roles(principal_type, principal_id, role_name)",
        "CREATE INDEX IF NOT EXISTS access_grants_principal ON access_grants(principal_type, principal_id, resource_type, resource_id, action, effect)",
        "CREATE INDEX IF NOT EXISTS access_grants_resource ON access_grants(resource_type, resource_id, action, effect)",
        "CREATE INDEX IF NOT EXISTS protected_resources_owner ON protected_resources(owner_user_id, resource_type)",
        "CREATE INDEX IF NOT EXISTS audit_events_order ON audit_events(event_id, occurred_at)",
        "CREATE INDEX IF NOT EXISTS organizational_roles_person ON organizational_roles(person_id, active, valid_from, valid_until)",
        "CREATE INDEX IF NOT EXISTS organizational_roles_name ON organizational_roles(role_name, organization, active)",
        "CREATE INDEX IF NOT EXISTS video_chapters_video ON video_chapters(video_id, ordinal)",
    )
    for statement in indexes:
        connection.execute(statement)


def _add_run_access_columns(connection: sqlite3.Connection) -> None:
    if not _table_exists(connection, "research_runs"):
        return
    columns = _columns(connection, "research_runs")
    if "principal_id" not in columns:
        connection.execute(
            "ALTER TABLE research_runs ADD COLUMN principal_id TEXT NOT NULL DEFAULT 'local-admin'"
        )
    if "authorization_revision" not in columns:
        connection.execute(
            "ALTER TABLE research_runs ADD COLUMN authorization_revision INTEGER NOT NULL DEFAULT 0"
        )


def sync_document_resources(
    connection: sqlite3.Connection,
    *,
    occurred_at: str | None = None,
    bump_revision: bool = False,
) -> int:
    """Register every logical document without changing immutable corpus IDs."""

    if not _table_exists(connection, "protected_resources") or not _table_exists(
        connection, "document_revisions"
    ):
        return 0
    now = occurred_at or _timestamp()
    before = connection.total_changes
    connection.execute(
        """
        INSERT OR IGNORE INTO protected_resources(
            resource_type, resource_id, stable_key, classification,
            owner_user_id, active, metadata_json, created_at, updated_at
        )
        SELECT 'document', logical_document_id, logical_document_id, 'internal',
               NULL, 1, '{}', ?, ?
        FROM document_revisions
        GROUP BY logical_document_id
        """,
        (now, now),
    )
    inserted = connection.total_changes - before
    if inserted and bump_revision:
        connection.execute(
            "UPDATE access_state SET revision = revision + 1, updated_at = ? WHERE singleton = 1",
            (now,),
        )
        revision_row = connection.execute(
            "SELECT revision FROM access_state WHERE singleton = 1"
        ).fetchone()
        if revision_row is not None and _table_exists(connection, "audit_events"):
            connection.execute(
                """
                INSERT INTO audit_events(
                    actor_user_id, action, target_type, target_id, outcome,
                    details_json, access_revision, occurred_at
                ) VALUES (?, 'document_resources.sync', 'document', '*',
                          'success', ?, ?, ?)
                """,
                (
                    LOCAL_ADMIN_USER_ID,
                    _json({"registered": int(inserted)}),
                    int(revision_row[0]),
                    now,
                ),
            )
    return int(inserted)


def sync_person_resources(
    connection: sqlite3.Connection,
    *,
    occurred_at: str | None = None,
) -> int:
    """Backfill protected directory identities created by older v2 builds."""

    if not _table_exists(connection, "protected_resources") or not _table_exists(
        connection, "people"
    ):
        return 0
    now = occurred_at or _timestamp()
    before = connection.total_changes
    connection.execute(
        """
        INSERT OR IGNORE INTO protected_resources(
            resource_type, resource_id, stable_key, classification,
            owner_user_id, active, metadata_json, created_at, updated_at
        )
        SELECT 'person', person_id, person_id, 'internal',
               created_by_user_id, active, '{}', ?, ?
        FROM people
        """,
        (now, now),
    )
    return int(connection.total_changes - before)


def migrate_connection(connection: sqlite3.Connection) -> int:
    """Transactionally migrate a legacy corpus database to access schema v2."""

    if not _table_exists(connection, "metadata"):
        raise RuntimeError("database has no corpus metadata table")
    row = connection.execute(
        "SELECT value FROM metadata WHERE key = 'schema_version'"
    ).fetchone()
    if row is None:
        raise RuntimeError("database has no schema_version metadata")
    try:
        found = int(str(row[0]))
    except ValueError as exc:
        raise RuntimeError(f"unsupported corpus schema {row[0]}") from exc
    if found not in {1, ACCESS_SCHEMA_VERSION}:
        raise RuntimeError(f"unsupported corpus schema {found}")

    required_tables = (
        "schema_migrations",
        "access_state",
        "users",
        "access_groups",
        "group_memberships",
        "system_roles",
        "system_role_permissions",
        "principal_system_roles",
        "protected_resources",
        "access_grants",
        "audit_events",
        "people",
        "person_aliases",
        "organizational_roles",
        "videos",
        "video_chapters",
    )
    run_columns = _columns(connection, "research_runs")
    missing_people = False
    if _table_exists(connection, "people") and _table_exists(
        connection, "protected_resources"
    ):
        missing_people = connection.execute(
            """
            SELECT 1
            FROM people AS p
            LEFT JOIN protected_resources AS pr
              ON pr.resource_type = 'person' AND pr.resource_id = p.person_id
            WHERE pr.resource_id IS NULL
            LIMIT 1
            """
        ).fetchone() is not None
    if (
        found == ACCESS_SCHEMA_VERSION
        and all(_table_exists(connection, table) for table in required_tables)
        and {"principal_id", "authorization_revision"}.issubset(run_columns)
        and not missing_people
        and connection.execute(
            "SELECT 1 FROM schema_migrations WHERE version = ?",
            (ACCESS_SCHEMA_VERSION,),
        ).fetchone()
        is not None
    ):
        return ACCESS_SCHEMA_VERSION

    now = _timestamp()
    connection.execute("BEGIN IMMEDIATE")
    try:
        _create_access_tables(connection)
        connection.execute(
            """
            INSERT OR IGNORE INTO schema_migrations(version, name, checksum, applied_at)
            VALUES (1, 'legacy-corpus-schema', 'sisu-reader-corpus-v1', ?)
            """,
            (now,),
        )
        connection.execute(
            """
            INSERT OR IGNORE INTO access_state(singleton, revision, updated_at)
            VALUES (1, 1, ?)
            """,
            (now,),
        )
        connection.execute(
            """
            INSERT OR IGNORE INTO users(
                user_id, username, display_name, active,
                credential_scheme, credential_hash, created_at, updated_at
            ) VALUES (?, ?, ?, 1, '', '', ?, ?)
            """,
            (LOCAL_ADMIN_USER_ID, LOCAL_ADMIN_USER_ID, "Local administrator", now, now),
        )
        for role_name, permissions in _SYSTEM_ROLE_PERMISSIONS.items():
            connection.execute(
                """
                INSERT OR IGNORE INTO system_roles(role_name, description, built_in)
                VALUES (?, ?, 1)
                """,
                (role_name, f"Built-in {role_name.replace('_', ' ')} role"),
            )
            connection.executemany(
                """
                INSERT OR IGNORE INTO system_role_permissions(role_name, action)
                VALUES (?, ?)
                """,
                ((role_name, permission) for permission in permissions),
            )
        connection.execute(
            """
            INSERT OR IGNORE INTO principal_system_roles(
                principal_type, principal_id, role_name, valid_from, valid_until,
                created_by_user_id, created_at
            ) VALUES ('user', ?, 'administrator', NULL, NULL, ?, ?)
            """,
            (LOCAL_ADMIN_USER_ID, LOCAL_ADMIN_USER_ID, now),
        )
        connection.execute(
            """
            INSERT OR IGNORE INTO access_grants(
                grant_id, principal_type, principal_id, resource_type, resource_id,
                action, effect, valid_from, valid_until, created_by_user_id,
                created_at, revoked_at
            ) VALUES (
                'grant:local-admin:global', 'user', ?, '*', '*', '*', 'allow',
                NULL, NULL, ?, ?, NULL
            )
            """,
            (LOCAL_ADMIN_USER_ID, LOCAL_ADMIN_USER_ID, now),
        )
        _add_run_access_columns(connection)
        sync_document_resources(connection, occurred_at=now)
        sync_person_resources(connection, occurred_at=now)
        connection.execute(
            """
            INSERT OR IGNORE INTO schema_migrations(version, name, checksum, applied_at)
            VALUES (?, 'identity-access-resources', 'sisu-reader-access-v2', ?)
            """,
            (ACCESS_SCHEMA_VERSION, now),
        )
        connection.execute(
            """
            INSERT INTO metadata(key, value) VALUES ('schema_version', ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (str(ACCESS_SCHEMA_VERSION),),
        )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    return ACCESS_SCHEMA_VERSION


class AccessManager:
    """Local authorization service with default-deny resource grants.

    Organizational roles are deliberately absent from permission resolution.
    Only system-role permissions, direct group memberships, and explicit access
    grants participate in :meth:`can`.
    """

    def __init__(
        self,
        source: Config | str | Path | Any,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if isinstance(source, Config):
            path = source.db_path
        elif isinstance(source, (str, Path)):
            path = Path(source)
        elif hasattr(source, "config") and hasattr(source.config, "db_path"):
            path = Path(source.config.db_path)
        else:
            raise TypeError("source must be Config, database path, or CorpusStore")
        self.db_path = Path(path).resolve()
        self.clock = clock

    def _connect(self) -> sqlite3.Connection:
        if not self.db_path.is_file():
            raise FileNotFoundError(self.db_path)
        connection = sqlite3.connect(
            self.db_path,
            timeout=30.0,
            check_same_thread=False,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=30000")
        migrate_connection(connection)
        return connection

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _at(self, value: datetime | str | None = None) -> str:
        return _timestamp(value, clock=self.clock) if value is not None else _timestamp(clock=self.clock)

    @staticmethod
    def _revision(connection: sqlite3.Connection) -> int:
        row = connection.execute(
            "SELECT revision FROM access_state WHERE singleton = 1"
        ).fetchone()
        if row is None:
            raise RuntimeError("access_state is missing")
        return int(row[0])

    @staticmethod
    def _bump(connection: sqlite3.Connection, occurred_at: str) -> int:
        connection.execute(
            "UPDATE access_state SET revision = revision + 1, updated_at = ? WHERE singleton = 1",
            (occurred_at,),
        )
        return AccessManager._revision(connection)

    @staticmethod
    def _audit(
        connection: sqlite3.Connection,
        *,
        actor_user_id: str | None,
        action: str,
        target_type: str,
        target_id: str,
        outcome: str,
        details: Mapping[str, Any] | None,
        revision: int,
        occurred_at: str,
    ) -> None:
        connection.execute(
            """
            INSERT INTO audit_events(
                actor_user_id, action, target_type, target_id, outcome,
                details_json, access_revision, occurred_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                actor_user_id,
                action,
                target_type,
                target_id,
                outcome,
                _json(details),
                revision,
                occurred_at,
            ),
        )

    def authorization_revision(self) -> int:
        connection = self._connect()
        try:
            return self._revision(connection)
        finally:
            connection.close()

    def principal_snapshot(
        self,
        user_id: str,
        *,
        at: datetime | str | None = None,
    ) -> PrincipalSnapshot:
        clean_user = _identifier(user_id, "user_id")
        instant = self._at(at)
        connection = self._connect()
        try:
            revision = self._revision(connection)
            user = connection.execute(
                "SELECT user_id, display_name, active FROM users WHERE user_id = ?",
                (clean_user,),
            ).fetchone()
            if user is None:
                return PrincipalSnapshot(
                    user_id=clean_user,
                    display_name="",
                    known=False,
                    active=False,
                    revision=revision,
                    evaluated_at=instant,
                )
            active = bool(int(user["active"]))
            if not active:
                return PrincipalSnapshot(
                    user_id=clean_user,
                    display_name=str(user["display_name"]),
                    known=True,
                    active=False,
                    revision=revision,
                    evaluated_at=instant,
                )
            group_rows = connection.execute(
                """
                SELECT g.group_id
                FROM group_memberships AS gm
                JOIN access_groups AS g ON g.group_id = gm.group_id
                WHERE gm.user_id = ? AND g.active = 1
                  AND (gm.valid_from IS NULL OR gm.valid_from <= ?)
                  AND (gm.valid_until IS NULL OR gm.valid_until > ?)
                ORDER BY g.group_id
                """,
                (clean_user, instant, instant),
            ).fetchall()
            groups = tuple(str(row[0]) for row in group_rows)
            principal_terms: list[tuple[str, str]] = [("user", clean_user)]
            principal_terms.extend(("group", item) for item in groups)
            role_names: set[str] = set()
            for principal_type, principal_id in principal_terms:
                rows = connection.execute(
                    """
                    SELECT role_name FROM principal_system_roles
                    WHERE principal_type = ? AND principal_id = ?
                      AND (valid_from IS NULL OR valid_from <= ?)
                      AND (valid_until IS NULL OR valid_until > ?)
                    """,
                    (principal_type, principal_id, instant, instant),
                ).fetchall()
                role_names.update(str(row[0]) for row in rows)
            permissions: set[str] = set()
            if role_names:
                placeholders = ",".join("?" for _ in role_names)
                rows = connection.execute(
                    f"SELECT action FROM system_role_permissions WHERE role_name IN ({placeholders})",
                    tuple(sorted(role_names)),
                ).fetchall()
                permissions.update(str(row[0]) for row in rows)
            return PrincipalSnapshot(
                user_id=clean_user,
                display_name=str(user["display_name"]),
                known=True,
                active=True,
                revision=revision,
                evaluated_at=instant,
                group_ids=groups,
                system_roles=tuple(sorted(role_names)),
                permissions=tuple(sorted(permissions)),
            )
        finally:
            connection.close()

    def authorization_scope_hash(
        self,
        principal: str | PrincipalSnapshot,
        *,
        at: datetime | str | None = None,
    ) -> str:
        snapshot = self._fresh_snapshot(principal, at=at)
        payload = {
            "user_id": snapshot.user_id,
            "known": snapshot.known,
            "active": snapshot.active,
            "revision": snapshot.revision,
            "groups": snapshot.group_ids,
            "roles": snapshot.system_roles,
            "permissions": snapshot.permissions,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _fresh_snapshot(
        self,
        principal: str | PrincipalSnapshot,
        *,
        at: datetime | str | None = None,
    ) -> PrincipalSnapshot:
        # PrincipalSnapshot is a convenient cache/display value, not a bearer
        # credential.  Always resolve it again so a caller cannot forge roles
        # or retain permissions after an access-state revision changes.
        user_id = principal.user_id if isinstance(principal, PrincipalSnapshot) else str(principal)
        return self.principal_snapshot(user_id, at=at)

    def create_user(
        self,
        user_id: str,
        display_name: str,
        *,
        username: str | None = None,
        active: bool = True,
        actor_user_id: str = LOCAL_ADMIN_USER_ID,
    ) -> str:
        clean_id = _identifier(user_id, "user_id")
        clean_username = _identifier(username or user_id, "username")
        clean_name = _text(display_name, "display_name")
        actor = _identifier(actor_user_id, "actor_user_id")
        now = self._at()
        with self._transaction() as connection:
            connection.execute(
                """
                INSERT INTO users(
                    user_id, username, display_name, active,
                    credential_scheme, credential_hash, created_at, updated_at
                ) VALUES (?, ?, ?, ?, '', '', ?, ?)
                """,
                (clean_id, clean_username, clean_name, int(bool(active)), now, now),
            )
            revision = self._bump(connection, now)
            self._audit(
                connection,
                actor_user_id=actor,
                action="user.create",
                target_type="user",
                target_id=clean_id,
                outcome="success",
                details={"active": bool(active)},
                revision=revision,
                occurred_at=now,
            )
        return clean_id

    def deactivate_user(
        self,
        user_id: str,
        *,
        actor_user_id: str = LOCAL_ADMIN_USER_ID,
    ) -> None:
        clean_id = _identifier(user_id, "user_id")
        if clean_id == LOCAL_ADMIN_USER_ID:
            raise ValueError("the compatibility local administrator cannot be deactivated")
        actor = _identifier(actor_user_id, "actor_user_id")
        now = self._at()
        with self._transaction() as connection:
            cursor = connection.execute(
                "UPDATE users SET active = 0, updated_at = ? WHERE user_id = ?",
                (now, clean_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"unknown user: {clean_id}")
            revision = self._bump(connection, now)
            self._audit(
                connection,
                actor_user_id=actor,
                action="user.deactivate",
                target_type="user",
                target_id=clean_id,
                outcome="success",
                details=None,
                revision=revision,
                occurred_at=now,
            )

    def create_group(
        self,
        group_id: str,
        display_name: str,
        *,
        active: bool = True,
        actor_user_id: str = LOCAL_ADMIN_USER_ID,
    ) -> str:
        clean_id = _identifier(group_id, "group_id")
        clean_name = _text(display_name, "display_name")
        actor = _identifier(actor_user_id, "actor_user_id")
        now = self._at()
        with self._transaction() as connection:
            connection.execute(
                """
                INSERT INTO access_groups(group_id, display_name, active, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (clean_id, clean_name, int(bool(active)), now, now),
            )
            revision = self._bump(connection, now)
            self._audit(
                connection,
                actor_user_id=actor,
                action="group.create",
                target_type="group",
                target_id=clean_id,
                outcome="success",
                details={"active": bool(active)},
                revision=revision,
                occurred_at=now,
            )
        return clean_id

    @staticmethod
    def _principal_exists(
        connection: sqlite3.Connection,
        principal_type: str,
        principal_id: str,
    ) -> bool:
        table, column = (
            ("users", "user_id")
            if principal_type == "user"
            else ("access_groups", "group_id")
        )
        row = connection.execute(
            f"SELECT active FROM {table} WHERE {column} = ?",
            (principal_id,),
        ).fetchone()
        return row is not None

    def add_group_member(
        self,
        group_id: str,
        user_id: str,
        *,
        valid_from: datetime | str | None = None,
        valid_until: datetime | str | None = None,
        actor_user_id: str = LOCAL_ADMIN_USER_ID,
    ) -> None:
        group = _identifier(group_id, "group_id")
        user = _identifier(user_id, "user_id")
        actor = _identifier(actor_user_id, "actor_user_id")
        start = _optional_timestamp(valid_from)
        end = _optional_timestamp(valid_until)
        if start is not None and end is not None and end <= start:
            raise ValueError("valid_until must be later than valid_from")
        now = self._at()
        with self._transaction() as connection:
            if not self._principal_exists(connection, "group", group):
                raise KeyError(f"unknown group: {group}")
            if not self._principal_exists(connection, "user", user):
                raise KeyError(f"unknown user: {user}")
            connection.execute(
                """
                INSERT INTO group_memberships(
                    group_id, user_id, valid_from, valid_until,
                    created_by_user_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(group_id, user_id) DO UPDATE SET
                    valid_from = excluded.valid_from,
                    valid_until = excluded.valid_until,
                    created_by_user_id = excluded.created_by_user_id,
                    created_at = excluded.created_at
                """,
                (group, user, start, end, actor, now),
            )
            revision = self._bump(connection, now)
            self._audit(
                connection,
                actor_user_id=actor,
                action="group.member.add",
                target_type="group",
                target_id=group,
                outcome="success",
                details={"user_id": user, "valid_from": start, "valid_until": end},
                revision=revision,
                occurred_at=now,
            )

    def remove_group_member(
        self,
        group_id: str,
        user_id: str,
        *,
        actor_user_id: str = LOCAL_ADMIN_USER_ID,
    ) -> None:
        group = _identifier(group_id, "group_id")
        user = _identifier(user_id, "user_id")
        actor = _identifier(actor_user_id, "actor_user_id")
        now = self._at()
        with self._transaction() as connection:
            cursor = connection.execute(
                "DELETE FROM group_memberships WHERE group_id = ? AND user_id = ?",
                (group, user),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"membership does not exist: {group}/{user}")
            revision = self._bump(connection, now)
            self._audit(
                connection,
                actor_user_id=actor,
                action="group.member.remove",
                target_type="group",
                target_id=group,
                outcome="success",
                details={"user_id": user},
                revision=revision,
                occurred_at=now,
            )

    def assign_system_role(
        self,
        principal_id: str,
        role_name: str,
        *,
        principal_type: str = "user",
        valid_from: datetime | str | None = None,
        valid_until: datetime | str | None = None,
        actor_user_id: str = LOCAL_ADMIN_USER_ID,
    ) -> None:
        kind = str(principal_type).strip().casefold()
        if kind not in _PRINCIPAL_TYPES:
            raise ValueError("principal_type must be user or group")
        principal = _identifier(principal_id, "principal_id")
        role = _identifier(role_name, "role_name")
        actor = _identifier(actor_user_id, "actor_user_id")
        start = _optional_timestamp(valid_from)
        end = _optional_timestamp(valid_until)
        if start is not None and end is not None and end <= start:
            raise ValueError("valid_until must be later than valid_from")
        now = self._at()
        with self._transaction() as connection:
            if not self._principal_exists(connection, kind, principal):
                raise KeyError(f"unknown {kind}: {principal}")
            if connection.execute(
                "SELECT 1 FROM system_roles WHERE role_name = ?", (role,)
            ).fetchone() is None:
                raise KeyError(f"unknown system role: {role}")
            connection.execute(
                """
                INSERT INTO principal_system_roles(
                    principal_type, principal_id, role_name, valid_from, valid_until,
                    created_by_user_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(principal_type, principal_id, role_name) DO UPDATE SET
                    valid_from = excluded.valid_from,
                    valid_until = excluded.valid_until,
                    created_by_user_id = excluded.created_by_user_id,
                    created_at = excluded.created_at
                """,
                (kind, principal, role, start, end, actor, now),
            )
            revision = self._bump(connection, now)
            self._audit(
                connection,
                actor_user_id=actor,
                action="system_role.assign",
                target_type=kind,
                target_id=principal,
                outcome="success",
                details={"role_name": role, "valid_from": start, "valid_until": end},
                revision=revision,
                occurred_at=now,
            )

    def register_resource(
        self,
        resource_type: str,
        resource_id: str,
        *,
        stable_key: str | None = None,
        classification: str = "internal",
        owner_user_id: str | None = None,
        active: bool = True,
        metadata: Mapping[str, Any] | None = None,
        affects_authorization: bool = True,
        actor_user_id: str = LOCAL_ADMIN_USER_ID,
    ) -> str:
        kind = _resource_type(resource_type)
        if kind == "*":
            raise ValueError("a concrete resource_type is required")
        identifier = _identifier(resource_id, "resource_id")
        stable = _identifier(stable_key or resource_id, "stable_key")
        level = _classification(classification)
        owner = None if owner_user_id is None else _identifier(owner_user_id, "owner_user_id")
        actor = _identifier(actor_user_id, "actor_user_id")
        metadata_json = _json(metadata)
        if not affects_authorization and kind != "trace":
            raise ValueError(
                "only trace registration may opt out of authorization revision changes"
            )
        now = self._at()
        with self._transaction() as connection:
            if owner is not None and not self._principal_exists(connection, "user", owner):
                raise KeyError(f"unknown owner user: {owner}")
            existing = connection.execute(
                """
                SELECT classification, owner_user_id, active
                FROM protected_resources
                WHERE resource_type = ? AND resource_id = ?
                """,
                (kind, identifier),
            ).fetchone()
            effective_affects_authorization = bool(affects_authorization)
            if existing is not None and not effective_affects_authorization:
                effective_affects_authorization = (
                    str(existing["classification"]) != level
                    or str(existing["owner_user_id"] or "") != str(owner or "")
                    or bool(int(existing["active"])) != bool(active)
                )
            connection.execute(
                """
                INSERT INTO protected_resources(
                    resource_type, resource_id, stable_key, classification,
                    owner_user_id, active, metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(resource_type, resource_id) DO UPDATE SET
                    stable_key = excluded.stable_key,
                    classification = excluded.classification,
                    owner_user_id = excluded.owner_user_id,
                    active = excluded.active,
                    metadata_json = excluded.metadata_json,
                    updated_at = excluded.updated_at
                """,
                (
                    kind,
                    identifier,
                    stable,
                    level,
                    owner,
                    int(bool(active)),
                    metadata_json,
                    now,
                    now,
                ),
            )
            revision = (
                self._bump(connection, now)
                if effective_affects_authorization
                else self._revision(connection)
            )
            self._audit(
                connection,
                actor_user_id=actor,
                action="resource.register",
                target_type=kind,
                target_id=identifier,
                outcome="success",
                details={
                    "classification": level,
                    "active": bool(active),
                    "affects_authorization": effective_affects_authorization,
                },
                revision=revision,
                occurred_at=now,
            )
        return identifier

    def grant(
        self,
        principal_id: str,
        resource_type: str,
        resource_id: str,
        action: str,
        *,
        principal_type: str = "user",
        effect: str = "allow",
        valid_from: datetime | str | None = None,
        valid_until: datetime | str | None = None,
        grant_id: str | None = None,
        actor_user_id: str = LOCAL_ADMIN_USER_ID,
    ) -> str:
        kind = str(principal_type).strip().casefold()
        if kind not in _PRINCIPAL_TYPES:
            raise ValueError("principal_type must be user or group")
        principal = _identifier(principal_id, "principal_id")
        resource_kind = _resource_type(resource_type)
        identifier = "*" if resource_id == "*" else _identifier(resource_id, "resource_id")
        permission = _action(action)
        selected_effect = str(effect).strip().casefold()
        if selected_effect not in _EFFECTS:
            raise ValueError("effect must be allow or deny")
        actor = _identifier(actor_user_id, "actor_user_id")
        start = _optional_timestamp(valid_from)
        end = _optional_timestamp(valid_until)
        if start is not None and end is not None and end <= start:
            raise ValueError("valid_until must be later than valid_from")
        identifier_grant = (
            _identifier(grant_id, "grant_id")
            if grant_id is not None
            else f"grant:{uuid.uuid4().hex}"
        )
        now = self._at()
        with self._transaction() as connection:
            if not self._principal_exists(connection, kind, principal):
                raise KeyError(f"unknown {kind}: {principal}")
            if resource_kind != "*" and identifier != "*" and connection.execute(
                """
                SELECT 1 FROM protected_resources
                WHERE resource_type = ? AND resource_id = ?
                """,
                (resource_kind, identifier),
            ).fetchone() is None:
                raise KeyError(f"unknown resource: {resource_kind}/{identifier}")
            connection.execute(
                """
                INSERT INTO access_grants(
                    grant_id, principal_type, principal_id, resource_type, resource_id,
                    action, effect, valid_from, valid_until, created_by_user_id,
                    created_at, revoked_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
                """,
                (
                    identifier_grant,
                    kind,
                    principal,
                    resource_kind,
                    identifier,
                    permission,
                    selected_effect,
                    start,
                    end,
                    actor,
                    now,
                ),
            )
            revision = self._bump(connection, now)
            self._audit(
                connection,
                actor_user_id=actor,
                action="access.grant",
                target_type=resource_kind,
                target_id=identifier,
                outcome="success",
                details={
                    "grant_id": identifier_grant,
                    "principal_type": kind,
                    "principal_id": principal,
                    "action": permission,
                    "effect": selected_effect,
                    "valid_from": start,
                    "valid_until": end,
                },
                revision=revision,
                occurred_at=now,
            )
        return identifier_grant

    def has_permission(
        self,
        principal: str | PrincipalSnapshot,
        action: str,
        *,
        at: datetime | str | None = None,
    ) -> bool:
        """Check a system capability backed by a wildcard resource grant.

        This is the creation/control-plane check used before a concrete
        protected resource exists.  A system role supplies the capability;
        an active ``resource_id='*'`` grant supplies its scope.  Any matching
        deny wins over every allow.
        """

        try:
            permission = _action(action)
            snapshot = self._fresh_snapshot(principal, at=at)
        except (KeyError, TypeError, ValueError):
            return False
        if not snapshot.known or not snapshot.active:
            return False
        if "*" not in snapshot.permissions and permission not in snapshot.permissions:
            return False
        instant = snapshot.evaluated_at if at is None else self._at(at)
        capability_resource_type = permission.split(".", 1)[0]
        connection = self._connect()
        try:
            matches = tuple(
                row
                for row in self._grant_rows(connection, snapshot, instant)
                if str(row["resource_id"]) == "*"
                and str(row["resource_type"]) in {"*", capability_resource_type}
                and str(row["action"]) in {"*", permission}
            )
            if any(str(row["effect"]) == "deny" for row in matches):
                return False
            return any(str(row["effect"]) == "allow" for row in matches)
        finally:
            connection.close()

    def revoke_grant(
        self,
        grant_id: str,
        *,
        actor_user_id: str = LOCAL_ADMIN_USER_ID,
    ) -> None:
        identifier = _identifier(grant_id, "grant_id")
        if identifier == "grant:local-admin:global":
            raise ValueError("the compatibility administrator grant cannot be revoked")
        actor = _identifier(actor_user_id, "actor_user_id")
        now = self._at()
        with self._transaction() as connection:
            cursor = connection.execute(
                "UPDATE access_grants SET revoked_at = ? WHERE grant_id = ? AND revoked_at IS NULL",
                (now, identifier),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"unknown or already revoked grant: {identifier}")
            revision = self._bump(connection, now)
            self._audit(
                connection,
                actor_user_id=actor,
                action="access.revoke",
                target_type="grant",
                target_id=identifier,
                outcome="success",
                details=None,
                revision=revision,
                occurred_at=now,
            )

    @staticmethod
    def _grant_rows(
        connection: sqlite3.Connection,
        snapshot: PrincipalSnapshot,
        instant: str,
    ) -> tuple[sqlite3.Row, ...]:
        rows: list[sqlite3.Row] = []
        for principal_type, principal_id in snapshot.principal_keys:
            rows.extend(
                connection.execute(
                    """
                    SELECT * FROM access_grants
                    WHERE principal_type = ? AND principal_id = ?
                      AND revoked_at IS NULL
                      AND (valid_from IS NULL OR valid_from <= ?)
                      AND (valid_until IS NULL OR valid_until > ?)
                    """,
                    (principal_type, principal_id, instant, instant),
                ).fetchall()
            )
        return tuple(rows)

    @staticmethod
    def _grant_matches(
        row: sqlite3.Row,
        resource_type: str,
        resource_id: str,
        action: str,
    ) -> bool:
        return (
            str(row["resource_type"]) in {"*", resource_type}
            and str(row["resource_id"]) in {"*", resource_id}
            and str(row["action"]) in {"*", action}
        )

    def can(
        self,
        principal: str | PrincipalSnapshot,
        resource_type: str,
        resource_id: str,
        action: str,
        *,
        at: datetime | str | None = None,
    ) -> bool:
        kind = _resource_type(resource_type)
        if kind == "*":
            return False
        try:
            identifier = _identifier(resource_id, "resource_id")
            permission = _action(action)
            snapshot = self._fresh_snapshot(principal, at=at)
        except (KeyError, TypeError, ValueError):
            return False
        if not snapshot.known or not snapshot.active:
            return False
        if "*" not in snapshot.permissions and permission not in snapshot.permissions:
            return False
        instant = snapshot.evaluated_at if at is None else self._at(at)
        connection = self._connect()
        try:
            resource = connection.execute(
                """
                SELECT owner_user_id, active FROM protected_resources
                WHERE resource_type = ? AND resource_id = ?
                """,
                (kind, identifier),
            ).fetchone()
            if resource is None or not bool(int(resource["active"])):
                return False
            matching = tuple(
                row
                for row in self._grant_rows(connection, snapshot, instant)
                if self._grant_matches(row, kind, identifier, permission)
            )
            if any(str(row["effect"]) == "deny" for row in matching):
                return False
            if any(str(row["effect"]) == "allow" for row in matching):
                return True
            return bool(
                kind == "trace"
                and permission == "trace.read_own"
                and str(resource["owner_user_id"] or "") == snapshot.user_id
            )
        finally:
            connection.close()

    def allowed_document_revision_ids(
        self,
        principal: str | PrincipalSnapshot,
        action: str = "document.read",
        *,
        at: datetime | str | None = None,
    ) -> tuple[str, ...]:
        snapshot = self._fresh_snapshot(principal, at=at)
        if not snapshot.known or not snapshot.active:
            return ()
        permission = _action(action)
        instant = snapshot.evaluated_at if at is None else self._at(at)
        connection = self._connect()
        try:
            snapshot_row = connection.execute(
                "SELECT value FROM metadata WHERE key = 'active_snapshot_id'"
            ).fetchone()
            if snapshot_row is None:
                return ()
            if "*" not in snapshot.permissions and permission not in snapshot.permissions:
                return ()
            grants = tuple(
                grant
                for grant in self._grant_rows(connection, snapshot, instant)
                if str(grant["resource_type"]) in {"*", "document"}
                and str(grant["action"]) in {"*", permission}
            )
            denied = {
                str(grant["resource_id"])
                for grant in grants
                if str(grant["effect"]) == "deny"
            }
            if "*" in denied:
                return ()
            allow_all = any(
                str(grant["effect"]) == "allow"
                and str(grant["resource_id"]) == "*"
                for grant in grants
            )
            explicitly_allowed = {
                str(grant["resource_id"])
                for grant in grants
                if str(grant["effect"]) == "allow"
                and str(grant["resource_id"]) != "*"
            }
            if not allow_all and not explicitly_allowed:
                return ()

            base = """
                SELECT d.document_revision_id, d.logical_document_id,
                       sd.corpus_ordinal
                FROM snapshot_documents AS sd
                JOIN document_revisions AS d
                  ON d.document_revision_id = sd.document_revision_id
                JOIN protected_resources AS r
                  ON r.resource_type = 'document'
                 AND r.resource_id = d.logical_document_id
                 AND r.active = 1
                WHERE sd.snapshot_id = ?
            """
            rows: list[sqlite3.Row] = []
            if allow_all:
                rows.extend(connection.execute(
                    base + " ORDER BY sd.corpus_ordinal, d.document_revision_id",
                    (str(snapshot_row[0]),),
                ).fetchall())
            else:
                logical_ids = tuple(sorted(explicitly_allowed))
                for start in range(0, len(logical_ids), 800):
                    chunk = logical_ids[start:start + 800]
                    placeholders = ",".join("?" for _ in chunk)
                    rows.extend(connection.execute(
                        base + f" AND d.logical_document_id IN ({placeholders})",
                        (str(snapshot_row[0]), *chunk),
                    ).fetchall())
                rows.sort(key=lambda row: (
                    int(row["corpus_ordinal"]),
                    str(row["document_revision_id"]),
                ))
            return tuple(
                str(row["document_revision_id"])
                for row in rows
                if str(row["logical_document_id"]) not in denied
            )
        finally:
            connection.close()

    def resource_listing(
        self,
        principal: str | PrincipalSnapshot,
        *,
        resource_type: str = "",
        limit: int = 200,
    ) -> tuple[dict[str, Any], ...]:
        """List protected IDs for an authorized local administrator."""

        current = self._fresh_snapshot(principal)
        if not self.has_permission(current, "access.manage"):
            raise PermissionError("access administration is unavailable")
        selected_type = _resource_type(resource_type) if resource_type else ""
        if selected_type == "*":
            selected_type = ""
        bounded_limit = max(1, min(int(limit), 1_000))
        connection = self._connect()
        try:
            rows = connection.execute(
                """
                SELECT pr.*,
                       CASE pr.resource_type
                         WHEN 'document' THEN COALESCE((
                           SELECT MAX(d.title) FROM document_revisions AS d
                           WHERE d.logical_document_id = pr.resource_id
                         ), '')
                         WHEN 'person' THEN COALESCE((
                           SELECT p.canonical_name FROM people AS p
                           WHERE p.person_id = pr.resource_id
                         ), '')
                         WHEN 'org_role' THEN COALESCE((
                           SELECT r.role_name FROM organizational_roles AS r
                           WHERE r.role_id = pr.resource_id
                         ), '')
                         WHEN 'video' THEN COALESCE((
                           SELECT v.title FROM videos AS v
                           WHERE v.video_id = pr.resource_id
                         ), '')
                         ELSE ''
                       END AS display_name
                FROM protected_resources AS pr
                WHERE (? = '' OR pr.resource_type = ?)
                ORDER BY pr.resource_type, display_name, pr.resource_id
                LIMIT ?
                """,
                (selected_type, selected_type, bounded_limit),
            ).fetchall()
            return tuple(
                {
                    "resource_type": row["resource_type"],
                    "resource_id": row["resource_id"],
                    "display_name": row["display_name"],
                    "classification": row["classification"],
                    "active": bool(row["active"]),
                    "owner_user_id": row["owner_user_id"] or "",
                }
                for row in rows
            )
        finally:
            connection.close()

    def audit_listing(
        self,
        *,
        limit: int = 200,
        after_event_id: int = 0,
    ) -> tuple[dict[str, Any], ...]:
        if isinstance(limit, bool) or not 1 <= int(limit) <= 1_000:
            raise ValueError("limit must be between 1 and 1000")
        if isinstance(after_event_id, bool) or int(after_event_id) < 0:
            raise ValueError("after_event_id must be non-negative")
        connection = self._connect()
        try:
            rows = connection.execute(
                """
                SELECT * FROM audit_events
                WHERE event_id > ?
                ORDER BY event_id
                LIMIT ?
                """,
                (int(after_event_id), int(limit)),
            ).fetchall()
            result: list[dict[str, Any]] = []
            for row in rows:
                item = dict(row)
                try:
                    item["details"] = json.loads(str(item.pop("details_json")))
                except json.JSONDecodeError:
                    item["details"] = {}
                result.append(item)
            return tuple(result)
        finally:
            connection.close()


__all__ = [
    "ACCESS_SCHEMA_VERSION",
    "AccessManager",
    "LOCAL_ADMIN_USER_ID",
    "PrincipalSnapshot",
    "migrate_connection",
    "sync_document_resources",
]
