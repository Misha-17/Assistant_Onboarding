"""Versioned local storage for model-led document research.

SQLite is used here as a durable corpus catalogue and exact enumeration tool,
not as a relevance oracle.  There are deliberately no similarity scores,
ranked result limits, or semantic release gates in this module.  The model can
screen every authorized manifest and request whole documents or sections; the
literal helpers only add documents whose supplied surface forms occur.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
import time
import unicodedata
import uuid
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .access import (
    ACCESS_SCHEMA_VERSION,
    LOCAL_ADMIN_USER_ID,
    migrate_connection,
    sync_document_resources,
)
from .config import Config
from .corpus import ExtractedDocument, extract_corpus, normalize_surface
from .models import (
    BuildReport,
    CorpusSnapshot,
    DocumentManifest,
    DocumentRevision,
    Section,
    SourceBlock,
)


_CORPUS_SCHEMA_VERSION = "1"
_SCHEMA_VERSION = str(ACCESS_SCHEMA_VERSION)
_REBUILD_LOCK = threading.RLock()
_SQL_VARIABLE_CHUNK = 800
_FTS_TOKEN = re.compile(r"[^\W_]+", flags=re.UNICODE)
_DISCOVERY_MAX_PAGE_SIZE = 1_000
_DISCOVERY_MAX_QUERY_TOKENS = 64
_DISCOVERY_MAX_PRINCIPALS = 256


@dataclass(frozen=True, slots=True)
class DiscoveryPage:
    """One stable, unscored page of lexical document navigation results."""

    snapshot_id: str
    query: str
    match_mode: str
    document_revision_ids: tuple[str, ...]
    total: int
    offset: int
    page_size: int
    returned: int
    next_offset: int | None
    exhausted: bool
    acl_filtered: bool


class CorpusStore:
    """Own the immutable corpus generations and lightweight run ledger."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self._connection: sqlite3.Connection | None = None
        self._snapshot: CorpusSnapshot | None = None
        self._lock = threading.RLock()

    def load(self, *, warm: bool = False) -> "CorpusStore":
        """Open and validate the active corpus snapshot.

        ``warm`` is accepted for a uniform engine-facing API.  There are no
        retrieval models to warm in this store.
        """

        del warm
        with self._lock:
            self.close()
            if not self.config.db_path.is_file():
                raise FileNotFoundError(self.config.db_path)
            connection = _open_connection(self.config.db_path)
            try:
                migrate_connection(connection)
                version_row = connection.execute(
                    "SELECT value FROM metadata WHERE key = 'schema_version'"
                ).fetchone()
                if version_row is None or str(version_row[0]) != _SCHEMA_VERSION:
                    found = "missing" if version_row is None else str(version_row[0])
                    raise RuntimeError(
                        f"unsupported corpus schema {found}; rebuild the corpus"
                    )
                active_row = connection.execute(
                    "SELECT value FROM metadata WHERE key = 'active_snapshot_id'"
                ).fetchone()
                if active_row is None or not str(active_row[0]):
                    raise RuntimeError("corpus database has no active snapshot")
                snapshot_row = connection.execute(
                    "SELECT * FROM corpus_snapshots WHERE snapshot_id = ?",
                    (str(active_row[0]),),
                ).fetchone()
                if snapshot_row is None:
                    raise RuntimeError("active corpus snapshot is missing")
                snapshot = _row_to_snapshot(snapshot_row)
                _validate_snapshot_counts(connection, snapshot)
            except Exception:
                connection.close()
                raise
            self._connection = connection
            self._snapshot = snapshot
        return self

    def close(self) -> None:
        with self._lock:
            connection, self._connection = self._connection, None
            self._snapshot = None
            if connection is not None:
                connection.close()

    def rebuild(
        self,
        paths: str | Path | Iterable[str | Path],
    ) -> BuildReport:
        """Extract, persist, validate, and atomically activate one snapshot."""

        started = time.perf_counter()
        extracted = extract_corpus(
            paths,
            hard_block_tokens=self.config.source_block_hard_tokens,
        )
        manifest_sha256 = _snapshot_fingerprint(extracted)
        snapshot = CorpusSnapshot(
            snapshot_id=f"snapshot_{manifest_sha256[:32]}",
            created_at=_utc_now(),
            manifest_sha256=manifest_sha256,
            document_count=len(extracted),
            section_count=sum(len(item.sections) for item in extracted),
            block_count=sum(len(item.blocks) for item in extracted),
        )

        self.config.workspace_dir.mkdir(parents=True, exist_ok=True)
        with _REBUILD_LOCK, _exclusive_rebuild_lock(
            self.config.workspace_dir / ".reader-rebuild.lock"
        ):
            # Prevent this instance's readers from reopening the database in
            # the short interval between close and atomic replacement.
            with self._lock:
                self._replace_generation(extracted, snapshot)

        warnings = tuple(
            warning
            for item in extracted
            for warning in item.document.warnings
        )
        return BuildReport(
            documents=snapshot.document_count,
            sections=snapshot.section_count,
            blocks=snapshot.block_count,
            warnings=warnings,
            elapsed_s=time.perf_counter() - started,
            snapshot_id=snapshot.snapshot_id,
        )

    def snapshot(self) -> CorpusSnapshot:
        self._ensure_loaded()
        assert self._snapshot is not None
        return self._snapshot

    def status(self) -> dict[str, Any]:
        snapshot = self._snapshot
        return {
            "loaded": self._connection is not None,
            "database": str(self.config.db_path),
            "schema_version": _SCHEMA_VERSION if snapshot is not None else "",
            "snapshot_id": snapshot.snapshot_id if snapshot else "",
            "documents": snapshot.document_count if snapshot else 0,
            "sections": snapshot.section_count if snapshot else 0,
            "blocks": snapshot.block_count if snapshot else 0,
        }

    def documents(
        self,
        *,
        allowed_document_revision_ids: Sequence[str] | None = None,
    ) -> tuple[DocumentRevision, ...]:
        snapshot = self.snapshot()
        allowed = _allowed_document_ids(allowed_document_revision_ids)
        if allowed == frozenset():
            return ()
        if allowed is None:
            rows = self._query_all(
                """
                SELECT d.*, sd.corpus_ordinal AS authorization_corpus_ordinal
                FROM snapshot_documents AS sd
                JOIN document_revisions AS d
                  ON d.document_revision_id = sd.document_revision_id
                WHERE sd.snapshot_id = ?
                ORDER BY sd.corpus_ordinal
                """,
                (snapshot.snapshot_id,),
            )
        else:
            rows = []
            for chunk in _chunks(tuple(allowed)):
                placeholders = ",".join("?" for _ in chunk)
                rows.extend(self._query_all(
                    f"""
                    SELECT d.*, sd.corpus_ordinal AS authorization_corpus_ordinal
                    FROM snapshot_documents AS sd
                    JOIN document_revisions AS d
                      ON d.document_revision_id = sd.document_revision_id
                    WHERE sd.snapshot_id = ?
                      AND d.document_revision_id IN ({placeholders})
                    """,
                    (snapshot.snapshot_id, *chunk),
                ))
            rows.sort(key=lambda row: (
                int(row["authorization_corpus_ordinal"]),
                str(row["document_revision_id"]),
            ))
        return tuple(
            _row_to_document(row)
            for row in rows
        )

    def document(
        self,
        document_revision_id: str,
        *,
        allowed_document_revision_ids: Sequence[str] | None = None,
    ) -> DocumentRevision | None:
        allowed = _allowed_document_ids(allowed_document_revision_ids)
        if allowed is not None and document_revision_id not in allowed:
            return None
        row = self._query_one(
            "SELECT * FROM document_revisions WHERE document_revision_id = ?",
            (document_revision_id,),
        )
        return _row_to_document(row) if row is not None else None

    def manifests(
        self,
        document_revision_ids: Sequence[str] = (),
        *,
        allowed_document_revision_ids: Sequence[str] | None = None,
    ) -> tuple[DocumentManifest, ...]:
        snapshot = self.snapshot()
        allowed = _allowed_document_ids(allowed_document_revision_ids)
        if allowed == frozenset():
            return ()
        requested = tuple(dict.fromkeys(str(item) for item in document_revision_ids if item))
        selection_was_explicit = bool(requested)
        if allowed is not None:
            requested = tuple(item for item in requested if item in allowed)
        if selection_was_explicit and not requested:
            return ()
        if requested:
            rows_by_id: dict[str, sqlite3.Row] = {}
            for chunk in _chunks(requested):
                placeholders = ",".join("?" for _ in chunk)
                rows = self._query_all(
                    f"""
                    SELECT m.*
                    FROM document_manifests AS m
                    JOIN snapshot_documents AS sd
                      ON sd.document_revision_id = m.document_revision_id
                    WHERE sd.snapshot_id = ?
                      AND m.document_revision_id IN ({placeholders})
                    """,
                    (snapshot.snapshot_id, *chunk),
                )
                rows_by_id.update(
                    (str(row["document_revision_id"]), row) for row in rows
                )
            return tuple(
                _row_to_manifest(rows_by_id[item])
                for item in requested
                if item in rows_by_id
            )
        if allowed is None:
            rows = self._query_all(
                """
                SELECT m.*, sd.corpus_ordinal AS authorization_corpus_ordinal
                FROM snapshot_documents AS sd
                JOIN document_manifests AS m
                  ON m.document_revision_id = sd.document_revision_id
                WHERE sd.snapshot_id = ?
                ORDER BY sd.corpus_ordinal
                """,
                (snapshot.snapshot_id,),
            )
        else:
            rows = []
            for chunk in _chunks(tuple(allowed)):
                placeholders = ",".join("?" for _ in chunk)
                rows.extend(self._query_all(
                    f"""
                    SELECT m.*, sd.corpus_ordinal AS authorization_corpus_ordinal
                    FROM snapshot_documents AS sd
                    JOIN document_manifests AS m
                      ON m.document_revision_id = sd.document_revision_id
                    WHERE sd.snapshot_id = ?
                      AND m.document_revision_id IN ({placeholders})
                    """,
                    (snapshot.snapshot_id, *chunk),
                ))
            rows.sort(key=lambda row: (
                int(row["authorization_corpus_ordinal"]),
                str(row["document_revision_id"]),
            ))
        return tuple(
            _row_to_manifest(row)
            for row in rows
        )

    def manifest(
        self,
        document_revision_id: str,
        *,
        allowed_document_revision_ids: Sequence[str] | None = None,
    ) -> DocumentManifest | None:
        allowed = _allowed_document_ids(allowed_document_revision_ids)
        if allowed is not None and document_revision_id not in allowed:
            return None
        row = self._query_one(
            "SELECT * FROM document_manifests WHERE document_revision_id = ?",
            (document_revision_id,),
        )
        return _row_to_manifest(row) if row is not None else None

    def sections_for_document(
        self,
        document_revision_id: str,
        *,
        allowed_document_revision_ids: Sequence[str] | None = None,
    ) -> tuple[Section, ...]:
        allowed = _allowed_document_ids(allowed_document_revision_ids)
        if allowed is not None and document_revision_id not in allowed:
            return ()
        rows = self._query_all(
            """
            SELECT * FROM sections
            WHERE document_revision_id = ?
            ORDER BY ordinal
            """,
            (document_revision_id,),
        )
        return tuple(_row_to_section(row) for row in rows)

    def blocks_for_document(
        self,
        document_revision_id: str,
        *,
        allowed_document_revision_ids: Sequence[str] | None = None,
    ) -> tuple[SourceBlock, ...]:
        allowed = _allowed_document_ids(allowed_document_revision_ids)
        if allowed is not None and document_revision_id not in allowed:
            return ()
        rows = self._query_all(
            """
            SELECT * FROM source_blocks
            WHERE document_revision_id = ?
            ORDER BY ordinal
            """,
            (document_revision_id,),
        )
        return tuple(_row_to_block(row) for row in rows)

    def blocks_for_section(
        self,
        section_id: str,
        *,
        include_descendants: bool = True,
        allowed_document_revision_ids: Sequence[str] | None = None,
    ) -> tuple[SourceBlock, ...]:
        """Return a coherent section range, including structural children."""

        section_row = self._query_one(
            "SELECT * FROM sections WHERE section_id = ?",
            (section_id,),
        )
        if section_row is None:
            return ()
        allowed = _allowed_document_ids(allowed_document_revision_ids)
        if (
            allowed is not None
            and str(section_row["document_revision_id"]) not in allowed
        ):
            return ()
        first = int(section_row["first_block_ordinal"])
        last = int(section_row["last_block_ordinal"])
        if first < 0 or last < first:
            return ()
        if include_descendants:
            rows = self._query_all(
                """
                SELECT * FROM source_blocks
                WHERE document_revision_id = ? AND ordinal BETWEEN ? AND ?
                ORDER BY ordinal
                """,
                (str(section_row["document_revision_id"]), first, last),
            )
        else:
            rows = self._query_all(
                """
                SELECT * FROM source_blocks
                WHERE document_revision_id = ? AND section_id = ?
                ORDER BY ordinal
                """,
                (str(section_row["document_revision_id"]), section_id),
            )
        return tuple(_row_to_block(row) for row in rows)

    def blocks_by_ids(
        self,
        block_ids: Sequence[str],
        *,
        allowed_document_revision_ids: Sequence[str] | None = None,
    ) -> tuple[SourceBlock, ...]:
        """Fetch immutable citation blocks while preserving caller order."""

        allowed = _allowed_document_ids(allowed_document_revision_ids)
        if allowed == frozenset():
            return ()
        ordered = tuple(str(item) for item in block_ids if item)
        rows_by_id: dict[str, sqlite3.Row] = {}
        for chunk in _chunks(tuple(dict.fromkeys(ordered))):
            placeholders = ",".join("?" for _ in chunk)
            rows_by_id.update(
                (str(row["block_id"]), row)
                for row in self._query_all(
                    f"SELECT * FROM source_blocks WHERE block_id IN ({placeholders})",
                    chunk,
                )
            )
        return tuple(
            _row_to_block(rows_by_id[item])
            for item in ordered
            if item in rows_by_id
            and (
                allowed is None
                or str(rows_by_id[item]["document_revision_id"]) in allowed
            )
        )

    def exact_document_matches(
        self,
        surfaces: str | Sequence[str],
        *,
        allowed_document_revision_ids: Sequence[str] | None = None,
    ) -> tuple[str, ...]:
        """Enumerate every active document containing supplied literal forms.

        This is intentionally an unranked, untruncated union.  Structured
        surfaces captured during ingestion use normalized equality; arbitrary
        supplied phrases also use FTS5 solely as a literal occurrence index.
        """

        snapshot = self.snapshot()
        supplied = _surface_tuple(surfaces)
        if not supplied:
            return ()
        allowed = _allowed_document_ids(allowed_document_revision_ids)
        if allowed == frozenset():
            return ()
        ordinals: dict[str, int] = {}
        normalized = tuple(dict.fromkeys(
            value for value in (normalize_surface(item) for item in supplied) if value
        ))
        for chunk in _chunks(normalized):
            placeholders = ",".join("?" for _ in chunk)
            rows = self._query_all(
                f"""
                SELECT DISTINCT et.document_revision_id, sd.corpus_ordinal
                FROM exact_terms AS et
                JOIN snapshot_documents AS sd
                  ON sd.document_revision_id = et.document_revision_id
                WHERE sd.snapshot_id = ?
                  AND et.term_normalized IN ({placeholders})
                """,
                (snapshot.snapshot_id, *chunk),
            )
            for row in rows:
                ordinals[str(row["document_revision_id"])] = int(row["corpus_ordinal"])

        for surface in supplied:
            expression = _fts_phrase(surface)
            if not expression:
                continue
            rows = self._query_all(
                """
                SELECT DISTINCT f.document_revision_id, sd.corpus_ordinal
                FROM block_fts AS f
                JOIN snapshot_documents AS sd
                  ON sd.document_revision_id = f.document_revision_id
                WHERE sd.snapshot_id = ? AND block_fts MATCH ?
                """,
                (snapshot.snapshot_id, expression),
            )
            for row in rows:
                ordinals[str(row["document_revision_id"])] = int(row["corpus_ordinal"])
        return tuple(
            item
            for item in sorted(ordinals, key=lambda item: (ordinals[item], item))
            if allowed is None or item in allowed
        )

    def exact_block_matches(
        self,
        surfaces: str | Sequence[str],
        *,
        allowed_document_revision_ids: Sequence[str] | None = None,
    ) -> tuple[SourceBlock, ...]:
        """Enumerate all active source blocks with a literal body occurrence."""

        snapshot = self.snapshot()
        supplied = _surface_tuple(surfaces)
        if not supplied:
            return ()
        allowed = _allowed_document_ids(allowed_document_revision_ids)
        if allowed == frozenset():
            return ()
        block_order: dict[str, tuple[int, int]] = {}
        normalized = tuple(dict.fromkeys(
            value for value in (normalize_surface(item) for item in supplied) if value
        ))
        for chunk in _chunks(normalized):
            placeholders = ",".join("?" for _ in chunk)
            rows = self._query_all(
                f"""
                SELECT DISTINCT et.block_id, sd.corpus_ordinal, b.ordinal
                FROM exact_terms AS et
                JOIN snapshot_documents AS sd
                  ON sd.document_revision_id = et.document_revision_id
                JOIN source_blocks AS b ON b.block_id = et.block_id
                WHERE sd.snapshot_id = ?
                  AND et.block_id IS NOT NULL
                  AND et.term_normalized IN ({placeholders})
                """,
                (snapshot.snapshot_id, *chunk),
            )
            for row in rows:
                block_order[str(row["block_id"])] = (
                    int(row["corpus_ordinal"]), int(row["ordinal"])
                )
        for surface in supplied:
            phrase = _fts_phrase(surface)
            if not phrase:
                continue
            rows = self._query_all(
                """
                SELECT f.block_id, sd.corpus_ordinal, b.ordinal
                FROM block_fts AS f
                JOIN snapshot_documents AS sd
                  ON sd.document_revision_id = f.document_revision_id
                JOIN source_blocks AS b ON b.block_id = f.block_id
                WHERE sd.snapshot_id = ? AND block_fts MATCH ?
                """,
                (snapshot.snapshot_id, f"body : {phrase}"),
            )
            for row in rows:
                block_order[str(row["block_id"])] = (
                    int(row["corpus_ordinal"]), int(row["ordinal"])
                )
        ordered_ids = tuple(sorted(
            block_order,
            key=lambda item: (*block_order[item], item),
        ))
        return self.blocks_by_ids(
            ordered_ids,
            allowed_document_revision_ids=allowed_document_revision_ids,
        )

    def discover_documents(
        self,
        query: str,
        *,
        offset: int = 0,
        page_size: int = 100,
        match_mode: str = "all",
        authorization_principals: Sequence[str] | None = None,
        allowed_document_revision_ids: Sequence[str] | None = None,
        expected_snapshot_id: str = "",
    ) -> DiscoveryPage:
        """Page through lexical document matches without assigning relevance.

        Results are ordered only by immutable corpus order and carry no score.
        ``authorization_principals=None`` denotes the local single-user corpus;
        a supplied sequence requires a matching ``document_acl`` read/owner
        entry, and an empty sequence authorizes nothing.  Callers must union
        :meth:`exact_document_matches` IDs separately before scheduling reads.

        ``expected_snapshot_id`` lets a caller pin a multi-page traversal.  A
        changed active snapshot raises instead of silently mixing generations.
        """

        if isinstance(offset, bool) or int(offset) != offset or int(offset) < 0:
            raise ValueError("offset must be a non-negative integer")
        if (
            isinstance(page_size, bool)
            or int(page_size) != page_size
            or not 1 <= int(page_size) <= _DISCOVERY_MAX_PAGE_SIZE
        ):
            raise ValueError(
                f"page_size must be between 1 and {_DISCOVERY_MAX_PAGE_SIZE}"
            )
        selected_mode = str(match_mode).strip().casefold()
        if selected_mode not in {"all", "any", "phrase"}:
            raise ValueError("match_mode must be all, any, or phrase")
        clean_query = unicodedata.normalize("NFKC", str(query or "")).strip()
        expression = _fts_discovery_expression(clean_query, selected_mode)
        snapshot = self.snapshot()
        if expected_snapshot_id and expected_snapshot_id != snapshot.snapshot_id:
            raise RuntimeError(
                "active corpus snapshot changed during discovery pagination"
            )

        principals: tuple[str, ...] | None
        if authorization_principals is None:
            principals = None
        else:
            if isinstance(authorization_principals, (str, bytes)):
                raise TypeError(
                    "authorization_principals must be a sequence, not one string"
                )
            principals = tuple(dict.fromkeys(
                str(item).strip()
                for item in authorization_principals
                if item is not None and str(item).strip()
            ))
            if len(principals) > _DISCOVERY_MAX_PRINCIPALS:
                raise ValueError(
                    "authorization_principals exceeds the supported scope size"
                )
            if not principals:
                self._ensure_loaded()
                with self._lock:
                    if (
                        self._snapshot is None
                        or self._snapshot.snapshot_id != snapshot.snapshot_id
                    ):
                        raise RuntimeError(
                            "active corpus snapshot changed during discovery"
                        )
                    return DiscoveryPage(
                        snapshot_id=snapshot.snapshot_id,
                        query=clean_query,
                        match_mode=selected_mode,
                        document_revision_ids=(),
                        total=0,
                        offset=int(offset),
                        page_size=int(page_size),
                        returned=0,
                        next_offset=None,
                        exhausted=True,
                        acl_filtered=True,
                    )

        allowed = _allowed_document_ids(allowed_document_revision_ids)
        if allowed == frozenset():
            return DiscoveryPage(
                snapshot_id=snapshot.snapshot_id,
                query=clean_query,
                match_mode=selected_mode,
                document_revision_ids=(),
                total=0,
                offset=int(offset),
                page_size=int(page_size),
                returned=0,
                next_offset=None,
                exhausted=True,
                acl_filtered=True,
            )

        acl_clause = ""
        parameters: list[Any] = [snapshot.snapshot_id, expression]
        if principals is not None:
            placeholders = ",".join("?" for _ in principals)
            acl_clause = f"""
                AND EXISTS (
                    SELECT 1 FROM document_acl AS acl
                    WHERE acl.document_revision_id = f.document_revision_id
                      AND acl.principal IN ({placeholders})
                      AND acl.permission IN ('read', 'owner')
                )
            """
            parameters.extend(principals)
        allowed_clause = ""
        if allowed is not None:
            allowed_clause = """
                AND EXISTS (
                    SELECT 1 FROM temp.sisu_allowed_documents AS permitted
                    WHERE permitted.document_revision_id = f.document_revision_id
                )
            """
        matched_cte = f"""
            WITH matched AS (
                SELECT DISTINCT
                    f.document_revision_id AS document_revision_id,
                    sd.corpus_ordinal AS corpus_ordinal
                FROM block_fts AS f
                JOIN snapshot_documents AS sd
                  ON sd.document_revision_id = f.document_revision_id
                WHERE sd.snapshot_id = ?
                  AND block_fts MATCH ?
                  {acl_clause}
                  {allowed_clause}
            )
        """

        self._ensure_loaded()
        with self._lock:
            assert self._connection is not None
            if self._snapshot is None or self._snapshot.snapshot_id != snapshot.snapshot_id:
                raise RuntimeError("active corpus snapshot changed during discovery")
            connection = self._connection
            connection.execute("BEGIN")
            try:
                if allowed is not None:
                    connection.execute(
                        """
                        CREATE TEMP TABLE IF NOT EXISTS sisu_allowed_documents (
                            document_revision_id TEXT PRIMARY KEY
                        ) WITHOUT ROWID
                        """
                    )
                    connection.execute("DELETE FROM temp.sisu_allowed_documents")
                    connection.executemany(
                        """
                        INSERT OR IGNORE INTO temp.sisu_allowed_documents(
                            document_revision_id
                        ) VALUES (?)
                        """,
                        ((item,) for item in allowed),
                    )
                total_row = connection.execute(
                    matched_cte + "SELECT COUNT(*) FROM matched",
                    tuple(parameters),
                ).fetchone()
                total = int(total_row[0]) if total_row is not None else 0
                rows = connection.execute(
                    matched_cte
                    + """
                    SELECT document_revision_id
                    FROM matched
                    ORDER BY corpus_ordinal, document_revision_id
                    LIMIT ? OFFSET ?
                    """,
                    (*parameters, int(page_size), int(offset)),
                ).fetchall()
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        identifiers = tuple(str(row[0]) for row in rows)
        consumed = int(offset) + len(identifiers)
        exhausted = consumed >= total
        return DiscoveryPage(
            snapshot_id=snapshot.snapshot_id,
            query=clean_query,
            match_mode=selected_mode,
            document_revision_ids=identifiers,
            total=total,
            offset=int(offset),
            page_size=int(page_size),
            returned=len(identifiers),
            next_offset=None if exhausted else consumed,
            exhausted=exhausted,
            acl_filtered=principals is not None or allowed is not None,
        )

    def create_run(
        self,
        question: str,
        *,
        effective_question: str = "",
        session_id: str = "",
        run_id: str = "",
        snapshot_id: str = "",
        authorization_scope_hash: str = "local-single-user",
        principal_id: str = LOCAL_ADMIN_USER_ID,
        authorization_revision: int = 0,
        route: str = "screen-read-synthesize",
        completeness_mode: str = "bounded",
        config: Mapping[str, Any] | None = None,
    ) -> str:
        """Create durable run state before screening or reading begins."""

        active = self.snapshot()
        selected_snapshot = snapshot_id or active.snapshot_id
        identifier = run_id or f"run_{uuid.uuid4().hex}"
        created_at = _utc_now()
        deadline = _add_seconds(created_at, self.config.total_deadline_s)
        synthesis_cutoff = _add_seconds(
            created_at,
            max(
                0.0,
                self.config.total_deadline_s
                - self.config.synthesis_reserve_s
                - self.config.finalize_reserve_s,
            ),
        )
        values = (
            identifier,
            session_id,
            selected_snapshot,
            authorization_scope_hash,
            str(principal_id or LOCAL_ADMIN_USER_ID),
            int(authorization_revision),
            str(question),
            str(effective_question or question),
            route,
            completeness_mode,
            "created",
            created_at,
            deadline,
            synthesis_cutoff,
            _json_dump(config or {}),
        )
        with self._transaction(immediate=True) as connection:
            connection.execute(
                """
                INSERT INTO research_runs(
                    run_id, session_id, snapshot_id, authorization_scope_hash,
                    principal_id, authorization_revision,
                    question, effective_question, route, completeness_mode,
                    state, created_at, deadline_at, synthesis_cutoff_at,
                    config_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                values,
            )
        return identifier

    def set_run_state(
        self,
        run_id: str,
        state: str,
        *,
        finalized: bool = False,
    ) -> None:
        finalized_at = _utc_now() if finalized else None
        with self._transaction(immediate=True) as connection:
            cursor = connection.execute(
                """
                UPDATE research_runs
                SET state = ?, finalized_at = COALESCE(?, finalized_at)
                WHERE run_id = ?
                """,
                (state, finalized_at, run_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"unknown run: {run_id}")

    def record_event(
        self,
        run_id: str,
        event_type: str,
        payload: Mapping[str, Any] | None = None,
        *,
        work_id: str | None = None,
        occurred_at: str | None = None,
    ) -> int:
        """Append one ordered event; reasoning text belongs in trace storage."""

        with self._transaction(immediate=True) as connection:
            cursor = connection.execute(
                """
                INSERT INTO run_events(
                    run_id, occurred_at, event_type, work_id, payload_json
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    occurred_at or _utc_now(),
                    event_type,
                    work_id,
                    _json_dump(payload or {}),
                ),
            )
            return int(cursor.lastrowid)

    def events(self, run_id: str) -> tuple[dict[str, Any], ...]:
        rows = self._query_all(
            "SELECT * FROM run_events WHERE run_id = ? ORDER BY event_seq",
            (run_id,),
        )
        return tuple({
            "event_seq": int(row["event_seq"]),
            "run_id": str(row["run_id"]),
            "occurred_at": str(row["occurred_at"]),
            "event_type": str(row["event_type"]),
            "work_id": str(row["work_id"]) if row["work_id"] is not None else None,
            "payload": _json_load(str(row["payload_json"]), {}),
        } for row in rows)

    def enqueue_work(
        self,
        run_id: str,
        kind: str,
        target_kind: str,
        target_id: str,
        *,
        entity_scope_id: str = "",
        source_lane: str = "semantic",
        wave: int = 0,
        queue_ordinal: int = 0,
        parent_work_id: str | None = None,
        payload: Mapping[str, Any] | None = None,
        work_id: str = "",
    ) -> str:
        """Idempotently enqueue a screening/reading/synthesis work item."""

        identifier = work_id or f"work_{uuid.uuid4().hex}"
        natural_key = (
            run_id,
            kind,
            target_kind,
            target_id,
            entity_scope_id,
            int(wave),
        )
        with self._transaction(immediate=True) as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO work_items(
                    work_id, run_id, parent_work_id, kind, target_kind,
                    target_id, entity_scope_id, source_lane, wave,
                    queue_ordinal, state, attempts, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued', 0, ?)
                """,
                (
                    identifier,
                    run_id,
                    parent_work_id,
                    kind,
                    target_kind,
                    target_id,
                    entity_scope_id,
                    source_lane,
                    int(wave),
                    int(queue_ordinal),
                    _json_dump(payload or {}),
                ),
            )
            row = connection.execute(
                """
                SELECT work_id FROM work_items
                WHERE run_id = ? AND kind = ? AND target_kind = ?
                  AND target_id = ? AND entity_scope_id = ? AND wave = ?
                """,
                natural_key,
            ).fetchone()
            if row is None:
                raise RuntimeError("work item could not be enqueued")
            return str(row["work_id"])

    def claim_next_work(
        self,
        run_id: str,
        worker_id: str,
        *,
        kinds: Sequence[str] = (),
        lease_s: float = 120.0,
    ) -> dict[str, Any] | None:
        """Atomically claim the next queued item, recovering expired leases."""

        now = _utc_now()
        lease_until = _add_seconds(now, max(1.0, float(lease_s)))
        with self._transaction(immediate=True) as connection:
            connection.execute(
                """
                UPDATE work_items
                SET state = 'queued', lease_owner = NULL, lease_until = NULL
                WHERE run_id = ? AND state = 'running'
                  AND lease_until IS NOT NULL AND lease_until < ?
                """,
                (run_id, now),
            )
            params: list[Any] = [run_id]
            kind_clause = ""
            selected_kinds = tuple(dict.fromkeys(str(item) for item in kinds if item))
            if selected_kinds:
                placeholders = ",".join("?" for _ in selected_kinds)
                kind_clause = f" AND kind IN ({placeholders})"
                params.extend(selected_kinds)
            row = connection.execute(
                f"""
                SELECT * FROM work_items
                WHERE run_id = ? AND state = 'queued'{kind_clause}
                ORDER BY wave, queue_ordinal, work_id
                LIMIT 1
                """,
                tuple(params),
            ).fetchone()
            if row is None:
                return None
            cursor = connection.execute(
                """
                UPDATE work_items
                SET state = 'running', attempts = attempts + 1,
                    lease_owner = ?, lease_until = ?,
                    started_at = COALESCE(started_at, ?)
                WHERE work_id = ? AND state = 'queued'
                """,
                (worker_id, lease_until, now, str(row["work_id"])),
            )
            if cursor.rowcount != 1:
                return None
            claimed = connection.execute(
                "SELECT * FROM work_items WHERE work_id = ?",
                (str(row["work_id"]),),
            ).fetchone()
            return _work_row(claimed) if claimed is not None else None

    def finish_work(
        self,
        work_id: str,
        *,
        state: str = "completed",
        result: Mapping[str, Any] | None = None,
        error_code: str = "",
    ) -> None:
        if state not in {"completed", "failed", "deferred", "cancelled"}:
            raise ValueError("invalid terminal work state")
        with self._transaction(immediate=True) as connection:
            cursor = connection.execute(
                """
                UPDATE work_items
                SET state = ?, result_json = ?, error_code = ?,
                    finished_at = ?, lease_owner = NULL, lease_until = NULL
                WHERE work_id = ?
                """,
                (state, _json_dump(result or {}), error_code, _utc_now(), work_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"unknown work item: {work_id}")

    def work_items(self, run_id: str) -> tuple[dict[str, Any], ...]:
        rows = self._query_all(
            """
            SELECT * FROM work_items
            WHERE run_id = ?
            ORDER BY wave, queue_ordinal, work_id
            """,
            (run_id,),
        )
        return tuple(_work_row(row) for row in rows)

    def _replace_generation(
        self,
        extracted: Sequence[ExtractedDocument],
        snapshot: CorpusSnapshot,
    ) -> None:
        token = uuid.uuid4().hex
        temporary = self.config.workspace_dir / f".reader-{token}.sqlite3"
        backup = self.config.workspace_dir / f".reader-{token}.backup.sqlite3"
        was_loaded = self._connection is not None
        backed_up = False
        installed = False
        try:
            self.close()
            _prepare_temporary_database(self.config.db_path, temporary)
            connection = _open_connection(temporary)
            try:
                _create_schema(connection)
                _insert_snapshot(connection, extracted, snapshot)
                integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
                if integrity.casefold() != "ok":
                    raise RuntimeError(f"corpus integrity check failed: {integrity}")
                foreign_rows = connection.execute("PRAGMA foreign_key_check").fetchall()
                if foreign_rows:
                    raise RuntimeError("corpus foreign-key validation failed")
            finally:
                connection.close()

            if self.config.db_path.exists():
                os.replace(self.config.db_path, backup)
                backed_up = True
            os.replace(temporary, self.config.db_path)
            installed = True
            self.load()
        except Exception as exc:
            self.close()
            restore_error: Exception | None = None
            try:
                if installed:
                    self.config.db_path.unlink(missing_ok=True)
                if backed_up and backup.exists():
                    os.replace(backup, self.config.db_path)
                if was_loaded and self.config.db_path.exists():
                    self.load()
            except Exception as restore_exc:
                restore_error = restore_exc
            if restore_error is not None:
                raise RuntimeError(
                    "corpus replacement failed and rollback was incomplete"
                ) from exc
            raise
        finally:
            temporary.unlink(missing_ok=True)
        try:
            backup.unlink(missing_ok=True)
        except OSError:
            pass

    def _ensure_loaded(self) -> None:
        if self._connection is None:
            self.load()

    def _query_one(
        self,
        statement: str,
        parameters: Sequence[Any] = (),
    ) -> sqlite3.Row | None:
        self._ensure_loaded()
        with self._lock:
            assert self._connection is not None
            return self._connection.execute(statement, tuple(parameters)).fetchone()

    def _query_all(
        self,
        statement: str,
        parameters: Sequence[Any] = (),
    ) -> list[sqlite3.Row]:
        self._ensure_loaded()
        with self._lock:
            assert self._connection is not None
            return self._connection.execute(statement, tuple(parameters)).fetchall()

    @contextmanager
    def _transaction(self, *, immediate: bool = False) -> Iterator[sqlite3.Connection]:
        self._ensure_loaded()
        with self._lock:
            assert self._connection is not None
            connection = self._connection
            connection.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
            try:
                yield connection
            except Exception:
                connection.rollback()
                raise
            else:
                connection.commit()


def _open_connection(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path, timeout=30.0, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA busy_timeout=30000")
    return connection


def _prepare_temporary_database(current: Path, temporary: Path) -> None:
    if current.is_file():
        if not _has_supported_schema(current):
            raise RuntimeError(
                "existing corpus database has an unsupported schema; "
                "refusing to discard its control-plane data during rebuild"
            )
        source = sqlite3.connect(current, timeout=30.0)
        target = sqlite3.connect(temporary, timeout=30.0)
        try:
            source.backup(target)
        finally:
            target.close()
            source.close()
    else:
        connection = sqlite3.connect(temporary)
        connection.close()


def _has_supported_schema(path: Path) -> bool:
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(path, timeout=5.0)
        row = connection.execute(
            "SELECT value FROM metadata WHERE key = 'schema_version'"
        ).fetchone()
        return row is not None and str(row[0]) in {
            _CORPUS_SCHEMA_VERSION,
            _SCHEMA_VERSION,
        }
    except sqlite3.Error:
        return False
    finally:
        if connection is not None:
            connection.close()


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA journal_mode=DELETE;
        PRAGMA foreign_keys=ON;

        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS corpus_snapshots (
            snapshot_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            manifest_sha256 TEXT NOT NULL UNIQUE,
            document_count INTEGER NOT NULL,
            section_count INTEGER NOT NULL,
            block_count INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS document_revisions (
            document_revision_id TEXT PRIMARY KEY,
            logical_document_id TEXT NOT NULL,
            title TEXT NOT NULL,
            source_path TEXT NOT NULL,
            source_sha256 TEXT NOT NULL,
            file_type TEXT NOT NULL,
            extraction_coverage TEXT NOT NULL,
            warnings_json TEXT NOT NULL,
            token_estimate INTEGER NOT NULL,
            body_sha256 TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS document_revisions_logical
            ON document_revisions(logical_document_id, document_revision_id);
        CREATE TABLE IF NOT EXISTS snapshot_documents (
            snapshot_id TEXT NOT NULL
                REFERENCES corpus_snapshots(snapshot_id) ON DELETE CASCADE,
            document_revision_id TEXT NOT NULL
                REFERENCES document_revisions(document_revision_id),
            corpus_ordinal INTEGER NOT NULL,
            PRIMARY KEY(snapshot_id, document_revision_id),
            UNIQUE(snapshot_id, corpus_ordinal)
        );
        CREATE TABLE IF NOT EXISTS document_acl (
            document_revision_id TEXT NOT NULL
                REFERENCES document_revisions(document_revision_id) ON DELETE CASCADE,
            principal TEXT NOT NULL,
            permission TEXT NOT NULL,
            PRIMARY KEY(document_revision_id, principal, permission)
        );
        CREATE INDEX IF NOT EXISTS document_acl_principal
            ON document_acl(principal, permission, document_revision_id);
        CREATE TABLE IF NOT EXISTS sections (
            section_id TEXT PRIMARY KEY,
            document_revision_id TEXT NOT NULL
                REFERENCES document_revisions(document_revision_id) ON DELETE CASCADE,
            parent_section_id TEXT
                REFERENCES sections(section_id) DEFERRABLE INITIALLY DEFERRED,
            ordinal INTEGER NOT NULL,
            depth INTEGER NOT NULL,
            heading TEXT NOT NULL,
            section_path TEXT NOT NULL,
            locator TEXT NOT NULL,
            first_block_ordinal INTEGER NOT NULL,
            last_block_ordinal INTEGER NOT NULL,
            token_estimate INTEGER NOT NULL,
            UNIQUE(document_revision_id, ordinal)
        );
        CREATE INDEX IF NOT EXISTS sections_document_order
            ON sections(document_revision_id, ordinal);
        CREATE TABLE IF NOT EXISTS source_blocks (
            block_id TEXT PRIMARY KEY,
            document_revision_id TEXT NOT NULL
                REFERENCES document_revisions(document_revision_id) ON DELETE CASCADE,
            section_id TEXT REFERENCES sections(section_id)
                DEFERRABLE INITIALLY DEFERRED,
            ordinal INTEGER NOT NULL,
            kind TEXT NOT NULL,
            locator TEXT NOT NULL,
            text TEXT NOT NULL,
            text_sha256 TEXT NOT NULL,
            canonical_char_start INTEGER NOT NULL,
            canonical_char_end INTEGER NOT NULL,
            previous_block_id TEXT REFERENCES source_blocks(block_id)
                DEFERRABLE INITIALLY DEFERRED,
            next_block_id TEXT REFERENCES source_blocks(block_id)
                DEFERRABLE INITIALLY DEFERRED,
            table_id TEXT,
            row_id TEXT,
            headers_json TEXT NOT NULL,
            token_estimate INTEGER NOT NULL,
            extraction_flags_json TEXT NOT NULL,
            UNIQUE(document_revision_id, ordinal)
        );
        CREATE INDEX IF NOT EXISTS source_blocks_document_order
            ON source_blocks(document_revision_id, ordinal);
        CREATE INDEX IF NOT EXISTS source_blocks_section_order
            ON source_blocks(section_id, ordinal);
        CREATE INDEX IF NOT EXISTS source_blocks_table_order
            ON source_blocks(document_revision_id, table_id, ordinal);
        CREATE INDEX IF NOT EXISTS source_blocks_previous_link
            ON source_blocks(previous_block_id);
        CREATE INDEX IF NOT EXISTS source_blocks_next_link
            ON source_blocks(next_block_id);
        CREATE TABLE IF NOT EXISTS document_manifests (
            manifest_id TEXT PRIMARY KEY,
            document_revision_id TEXT NOT NULL UNIQUE
                REFERENCES document_revisions(document_revision_id) ON DELETE CASCADE,
            title TEXT NOT NULL,
            source_path TEXT NOT NULL,
            file_type TEXT NOT NULL,
            extraction_coverage TEXT NOT NULL,
            outline TEXT NOT NULL,
            lead_text TEXT NOT NULL,
            exact_surfaces_json TEXT NOT NULL,
            token_estimate INTEGER NOT NULL,
            warnings_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS exact_terms (
            term_id TEXT PRIMARY KEY,
            term_normalized TEXT NOT NULL,
            term_kind TEXT NOT NULL,
            surface TEXT NOT NULL,
            document_revision_id TEXT NOT NULL
                REFERENCES document_revisions(document_revision_id) ON DELETE CASCADE,
            block_id TEXT REFERENCES source_blocks(block_id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS exact_terms_lookup
            ON exact_terms(term_normalized, document_revision_id, block_id);
        CREATE VIRTUAL TABLE IF NOT EXISTS block_fts USING fts5(
            block_id UNINDEXED,
            document_revision_id UNINDEXED,
            title,
            section_path,
            locator,
            body,
            headers,
            tokenize='unicode61 remove_diacritics 2'
        );
        CREATE TABLE IF NOT EXISTS catalog_nodes (
            node_id TEXT PRIMARY KEY,
            snapshot_id TEXT NOT NULL
                REFERENCES corpus_snapshots(snapshot_id) ON DELETE CASCADE,
            parent_node_id TEXT REFERENCES catalog_nodes(node_id)
                DEFERRABLE INITIALLY DEFERRED,
            kind TEXT NOT NULL,
            title TEXT NOT NULL,
            summary TEXT NOT NULL,
            ordinal INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS catalog_nodes_parent
            ON catalog_nodes(snapshot_id, parent_node_id, ordinal);
        CREATE TABLE IF NOT EXISTS catalog_members (
            node_id TEXT NOT NULL REFERENCES catalog_nodes(node_id) ON DELETE CASCADE,
            document_revision_id TEXT NOT NULL
                REFERENCES document_revisions(document_revision_id),
            PRIMARY KEY(node_id, document_revision_id)
        );
        CREATE TABLE IF NOT EXISTS research_runs (
            run_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            snapshot_id TEXT NOT NULL REFERENCES corpus_snapshots(snapshot_id),
            authorization_scope_hash TEXT NOT NULL,
            question TEXT NOT NULL,
            effective_question TEXT NOT NULL,
            route TEXT NOT NULL,
            completeness_mode TEXT NOT NULL,
            state TEXT NOT NULL,
            created_at TEXT NOT NULL,
            deadline_at TEXT NOT NULL,
            synthesis_cutoff_at TEXT NOT NULL,
            finalized_at TEXT,
            continuation_of_run_id TEXT REFERENCES research_runs(run_id),
            config_json TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS research_runs_session
            ON research_runs(session_id, created_at);
        CREATE TABLE IF NOT EXISTS work_items (
            work_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL REFERENCES research_runs(run_id) ON DELETE CASCADE,
            parent_work_id TEXT REFERENCES work_items(work_id),
            kind TEXT NOT NULL,
            target_kind TEXT NOT NULL,
            target_id TEXT NOT NULL,
            entity_scope_id TEXT NOT NULL DEFAULT '',
            source_lane TEXT NOT NULL,
            wave INTEGER NOT NULL,
            queue_ordinal INTEGER NOT NULL,
            state TEXT NOT NULL,
            attempts INTEGER NOT NULL,
            lease_owner TEXT,
            lease_until TEXT,
            started_at TEXT,
            finished_at TEXT,
            error_code TEXT NOT NULL DEFAULT '',
            payload_json TEXT NOT NULL,
            result_json TEXT NOT NULL DEFAULT '{}',
            UNIQUE(
                run_id, kind, target_kind, target_id, entity_scope_id, wave
            )
        );
        CREATE INDEX IF NOT EXISTS work_items_queue
            ON work_items(run_id, state, wave, queue_ordinal);
        CREATE TABLE IF NOT EXISTS run_events (
            event_seq INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL REFERENCES research_runs(run_id) ON DELETE CASCADE,
            occurred_at TEXT NOT NULL,
            event_type TEXT NOT NULL,
            work_id TEXT REFERENCES work_items(work_id) ON DELETE SET NULL,
            payload_json TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS run_events_run_order
            ON run_events(run_id, event_seq);
        """
    )
    connection.execute(
        """
        INSERT OR IGNORE INTO metadata(key, value) VALUES ('schema_version', ?)
        """,
        (_CORPUS_SCHEMA_VERSION,),
    )
    connection.commit()
    migrate_connection(connection)


def _insert_snapshot(
    connection: sqlite3.Connection,
    extracted: Sequence[ExtractedDocument],
    snapshot: CorpusSnapshot,
) -> None:
    connection.execute("BEGIN IMMEDIATE")
    try:
        connection.execute(
            """
            INSERT OR IGNORE INTO corpus_snapshots(
                snapshot_id, created_at, manifest_sha256,
                document_count, section_count, block_count
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                snapshot.snapshot_id,
                snapshot.created_at,
                snapshot.manifest_sha256,
                snapshot.document_count,
                snapshot.section_count,
                snapshot.block_count,
            ),
        )
        connection.execute(
            "DELETE FROM snapshot_documents WHERE snapshot_id = ?",
            (snapshot.snapshot_id,),
        )
        connection.execute(
            "DELETE FROM catalog_nodes WHERE snapshot_id = ?",
            (snapshot.snapshot_id,),
        )

        # Stage the same immutable IDs previously deleted per document. FTS
        # identifiers are unindexed, so batch membership avoids rescanning the
        # entire payload corpus for every document or 800-block chunk.
        connection.execute(
            "CREATE TEMP TABLE rebuild_document_ids "
            "(document_revision_id TEXT PRIMARY KEY) WITHOUT ROWID"
        )
        connection.execute(
            "CREATE TEMP TABLE rebuild_block_ids "
            "(block_id TEXT PRIMARY KEY) WITHOUT ROWID"
        )
        connection.executemany(
            "INSERT OR IGNORE INTO rebuild_document_ids VALUES (?)",
            ((item.document.document_revision_id,) for item in extracted),
        )
        connection.executemany(
            "INSERT OR IGNORE INTO rebuild_block_ids VALUES (?)",
            ((block.block_id,) for item in extracted for block in item.blocks),
        )
        connection.execute(
            "DELETE FROM block_fts WHERE block_id IN "
            "(SELECT block_id FROM rebuild_block_ids)"
        )
        connection.execute(
            "DELETE FROM exact_terms WHERE document_revision_id IN "
            "(SELECT document_revision_id FROM rebuild_document_ids)"
        )
        connection.execute("DROP TABLE rebuild_block_ids")
        connection.execute("DROP TABLE rebuild_document_ids")

        for corpus_ordinal, extracted_document in enumerate(extracted):
            document = extracted_document.document
            manifest = extracted_document.manifest
            _verify_existing_artifacts(connection, extracted_document)
            connection.execute(
                """
                INSERT OR IGNORE INTO document_revisions VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                (
                    document.document_revision_id,
                    document.logical_document_id,
                    document.title,
                    document.source_path,
                    document.source_sha256,
                    document.file_type,
                    document.extraction_coverage,
                    _json_dump(document.warnings),
                    document.token_estimate,
                    document.body_sha256,
                ),
            )
            connection.execute(
                """
                INSERT OR IGNORE INTO document_manifests VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                (
                    manifest.manifest_id,
                    manifest.document_revision_id,
                    manifest.title,
                    manifest.source_path,
                    manifest.file_type,
                    manifest.extraction_coverage,
                    manifest.outline,
                    manifest.lead_text,
                    _json_dump(manifest.exact_surfaces),
                    manifest.token_estimate,
                    _json_dump(manifest.warnings),
                ),
            )
            connection.executemany(
                """
                INSERT OR IGNORE INTO sections VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                (
                    (
                        section.section_id,
                        section.document_revision_id,
                        section.parent_section_id,
                        section.ordinal,
                        section.depth,
                        section.heading,
                        section.section_path,
                        section.locator,
                        section.first_block_ordinal,
                        section.last_block_ordinal,
                        section.token_estimate,
                    )
                    for section in extracted_document.sections
                ),
            )
            connection.executemany(
                """
                INSERT OR IGNORE INTO source_blocks VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                (
                    (
                        block.block_id,
                        block.document_revision_id,
                        block.section_id,
                        block.ordinal,
                        block.kind,
                        block.locator,
                        block.text,
                        block.text_sha256,
                        block.canonical_char_start,
                        block.canonical_char_end,
                        block.previous_block_id,
                        block.next_block_id,
                        block.table_id,
                        block.row_id,
                        _json_dump(block.headers),
                        block.token_estimate,
                        _json_dump(block.extraction_flags),
                    )
                    for block in extracted_document.blocks
                ),
            )

            section_paths = {
                section.section_id: section.section_path
                for section in extracted_document.sections
            }
            connection.executemany(
                """
                INSERT INTO block_fts(
                    block_id, document_revision_id, title, section_path,
                    locator, body, headers
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    (
                        block.block_id,
                        block.document_revision_id,
                        document.title,
                        section_paths.get(block.section_id or "", ""),
                        block.locator,
                        block.text,
                        " | ".join(block.headers),
                    )
                    for block in extracted_document.blocks
                ),
            )

            exact_rows: list[tuple[str, str, str, str, str, str | None]] = []
            block_term_normalized: set[str] = set()
            for term in extracted_document.exact_terms:
                block_term_normalized.add(term.term_normalized)
                term_digest = _digest(
                    term.document_revision_id,
                    term.term_kind,
                    term.term_normalized,
                    term.block_id,
                )
                exact_rows.append((
                    f"term_{term_digest[:32]}",
                    term.term_normalized,
                    term.term_kind,
                    term.surface,
                    term.document_revision_id,
                    term.block_id,
                ))
            for surface in manifest.exact_surfaces:
                normalized = normalize_surface(surface)
                if not normalized or normalized in block_term_normalized:
                    continue
                term_digest = _digest(
                    document.document_revision_id,
                    "manifest",
                    normalized,
                    "",
                )
                exact_rows.append((
                    f"term_{term_digest[:32]}",
                    normalized,
                    "manifest",
                    surface,
                    document.document_revision_id,
                    None,
                ))
            connection.executemany(
                """
                INSERT OR IGNORE INTO exact_terms(
                    term_id, term_normalized, term_kind, surface,
                    document_revision_id, block_id
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                exact_rows,
            )
            connection.execute(
                """
                INSERT INTO snapshot_documents(
                    snapshot_id, document_revision_id, corpus_ordinal
                ) VALUES (?, ?, ?)
                """,
                (snapshot.snapshot_id, document.document_revision_id, corpus_ordinal),
            )

        _insert_catalog(connection, extracted, snapshot)
        sync_document_resources(connection, bump_revision=True)
        connection.execute(
            """
            INSERT INTO metadata(key, value) VALUES ('active_snapshot_id', ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (snapshot.snapshot_id,),
        )
        connection.commit()
    except Exception:
        connection.rollback()
        raise


def _insert_catalog(
    connection: sqlite3.Connection,
    extracted: Sequence[ExtractedDocument],
    snapshot: CorpusSnapshot,
) -> None:
    root_id = f"catalog_{_digest(snapshot.snapshot_id, 'root')[:24]}"
    connection.execute(
        """
        INSERT INTO catalog_nodes(
            node_id, snapshot_id, parent_node_id, kind, title, summary, ordinal
        ) VALUES (?, ?, NULL, 'corpus', 'Corpus', ?, 0)
        """,
        (
            root_id,
            snapshot.snapshot_id,
            f"{snapshot.document_count} documents",
        ),
    )
    for ordinal, item in enumerate(extracted):
        node_id = f"catalog_{_digest(snapshot.snapshot_id, item.document.document_revision_id)[:24]}"
        connection.execute(
            """
            INSERT INTO catalog_nodes(
                node_id, snapshot_id, parent_node_id, kind, title, summary, ordinal
            ) VALUES (?, ?, ?, 'document', ?, ?, ?)
            """,
            (
                node_id,
                snapshot.snapshot_id,
                root_id,
                item.manifest.title,
                item.manifest.lead_text,
                ordinal,
            ),
        )
        connection.execute(
            "INSERT INTO catalog_members(node_id, document_revision_id) VALUES (?, ?)",
            (node_id, item.document.document_revision_id),
        )


def _validate_snapshot_counts(
    connection: sqlite3.Connection,
    snapshot: CorpusSnapshot,
) -> None:
    document_row = connection.execute(
        """
        SELECT COUNT(*)
        FROM snapshot_documents AS sd
        WHERE sd.snapshot_id = ?
        """,
        (snapshot.snapshot_id,),
    ).fetchone()
    section_row = connection.execute(
        """
        SELECT COUNT(*)
        FROM sections AS s
        JOIN snapshot_documents AS sd
          ON sd.document_revision_id = s.document_revision_id
        WHERE sd.snapshot_id = ?
        """,
        (snapshot.snapshot_id,),
    ).fetchone()
    block_row = connection.execute(
        """
        SELECT COUNT(*)
        FROM source_blocks AS b
        JOIN snapshot_documents AS sd
          ON sd.document_revision_id = b.document_revision_id
        WHERE sd.snapshot_id = ?
        """,
        (snapshot.snapshot_id,),
    ).fetchone()
    actual = (
        int(document_row[0]) if document_row else 0,
        int(section_row[0]) if section_row else 0,
        int(block_row[0]) if block_row else 0,
    )
    expected = (
        snapshot.document_count,
        snapshot.section_count,
        snapshot.block_count,
    )
    if actual != expected:
        raise RuntimeError(
            f"active snapshot counts do not match: expected {expected}, found {actual}"
        )


def _verify_existing_artifacts(
    connection: sqlite3.Connection,
    extracted: ExtractedDocument,
) -> None:
    """Refuse to mutate an already-addressed immutable representation."""

    document = extracted.document
    row = connection.execute(
        "SELECT * FROM document_revisions WHERE document_revision_id = ?",
        (document.document_revision_id,),
    ).fetchone()
    if row is None:
        return
    stored_document = _row_to_document(row)
    same_source_path = os.path.normcase(stored_document.source_path) == os.path.normcase(
        document.source_path
    )
    document_matches = (
        stored_document.document_revision_id == document.document_revision_id
        and stored_document.logical_document_id == document.logical_document_id
        and stored_document.title == document.title
        and same_source_path
        and stored_document.source_sha256 == document.source_sha256
        and stored_document.file_type == document.file_type
        and stored_document.extraction_coverage == document.extraction_coverage
        and stored_document.warnings == document.warnings
        and stored_document.token_estimate == document.token_estimate
        and stored_document.body_sha256 == document.body_sha256
    )
    stored_sections = tuple(
        _row_to_section(item)
        for item in connection.execute(
            """
            SELECT * FROM sections
            WHERE document_revision_id = ? ORDER BY ordinal
            """,
            (document.document_revision_id,),
        ).fetchall()
    )
    stored_blocks = tuple(
        _row_to_block(item)
        for item in connection.execute(
            """
            SELECT * FROM source_blocks
            WHERE document_revision_id = ? ORDER BY ordinal
            """,
            (document.document_revision_id,),
        ).fetchall()
    )
    manifest_row = connection.execute(
        "SELECT * FROM document_manifests WHERE document_revision_id = ?",
        (document.document_revision_id,),
    ).fetchone()
    stored_manifest = _row_to_manifest(manifest_row) if manifest_row is not None else None
    manifest_matches = stored_manifest == extracted.manifest
    if stored_manifest is not None:
        manifest_matches = (
            stored_manifest.manifest_id == extracted.manifest.manifest_id
            and stored_manifest.document_revision_id
                == extracted.manifest.document_revision_id
            and stored_manifest.title == extracted.manifest.title
            and os.path.normcase(stored_manifest.source_path)
                == os.path.normcase(extracted.manifest.source_path)
            and stored_manifest.file_type == extracted.manifest.file_type
            and stored_manifest.extraction_coverage
                == extracted.manifest.extraction_coverage
            and stored_manifest.outline == extracted.manifest.outline
            and stored_manifest.lead_text == extracted.manifest.lead_text
            and stored_manifest.exact_surfaces == extracted.manifest.exact_surfaces
            and stored_manifest.token_estimate == extracted.manifest.token_estimate
            and stored_manifest.warnings == extracted.manifest.warnings
        )
    if not (
        document_matches
        and stored_sections == extracted.sections
        and stored_blocks == extracted.blocks
        and manifest_matches
    ):
        raise RuntimeError(
            "immutable corpus artifact collision; increment the ingestion version"
        )


def _snapshot_fingerprint(extracted: Sequence[ExtractedDocument]) -> str:
    digest = hashlib.sha256()
    digest.update(f"sisu-reader-schema:{_CORPUS_SCHEMA_VERSION}\n".encode("utf-8"))
    for item in extracted:
        document = item.document
        _hash_record(digest, (
            "document",
            document.document_revision_id,
            document.logical_document_id,
            document.source_sha256,
            document.body_sha256,
            document.source_path,
            document.extraction_coverage,
        ))
        for section in item.sections:
            _hash_record(digest, (
                "section",
                section.section_id,
                section.parent_section_id or "",
                section.ordinal,
                section.depth,
                section.heading,
                section.section_path,
                section.locator,
                section.first_block_ordinal,
                section.last_block_ordinal,
            ))
        for block in item.blocks:
            _hash_record(digest, (
                "block",
                block.block_id,
                block.section_id or "",
                block.ordinal,
                block.kind,
                block.locator,
                block.text_sha256,
                block.canonical_char_start,
                block.canonical_char_end,
                block.previous_block_id or "",
                block.next_block_id or "",
                block.table_id or "",
                block.row_id or "",
                *block.headers,
                *block.extraction_flags,
            ))
        _hash_record(digest, (
            "manifest",
            item.manifest.manifest_id,
            item.manifest.outline,
            item.manifest.lead_text,
            *item.manifest.exact_surfaces,
        ))
    return digest.hexdigest()


def _hash_record(digest: Any, values: Sequence[Any]) -> None:
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    digest.update(b"\n")


def _row_to_snapshot(row: sqlite3.Row) -> CorpusSnapshot:
    return CorpusSnapshot(
        snapshot_id=str(row["snapshot_id"]),
        created_at=str(row["created_at"]),
        manifest_sha256=str(row["manifest_sha256"]),
        document_count=int(row["document_count"]),
        section_count=int(row["section_count"]),
        block_count=int(row["block_count"]),
    )


def _row_to_document(row: sqlite3.Row) -> DocumentRevision:
    return DocumentRevision(
        document_revision_id=str(row["document_revision_id"]),
        logical_document_id=str(row["logical_document_id"]),
        title=str(row["title"]),
        source_path=str(row["source_path"]),
        source_sha256=str(row["source_sha256"]),
        file_type=str(row["file_type"]),
        extraction_coverage=str(row["extraction_coverage"]),
        warnings=tuple(_json_load(str(row["warnings_json"]), [])),
        token_estimate=int(row["token_estimate"]),
        body_sha256=str(row["body_sha256"]),
    )


def _row_to_manifest(row: sqlite3.Row) -> DocumentManifest:
    return DocumentManifest(
        manifest_id=str(row["manifest_id"]),
        document_revision_id=str(row["document_revision_id"]),
        title=str(row["title"]),
        source_path=str(row["source_path"]),
        file_type=str(row["file_type"]),
        extraction_coverage=str(row["extraction_coverage"]),
        outline=str(row["outline"]),
        lead_text=str(row["lead_text"]),
        exact_surfaces=tuple(_json_load(str(row["exact_surfaces_json"]), [])),
        token_estimate=int(row["token_estimate"]),
        warnings=tuple(_json_load(str(row["warnings_json"]), [])),
    )


def _row_to_section(row: sqlite3.Row) -> Section:
    return Section(
        section_id=str(row["section_id"]),
        document_revision_id=str(row["document_revision_id"]),
        parent_section_id=(
            str(row["parent_section_id"])
            if row["parent_section_id"] is not None
            else None
        ),
        ordinal=int(row["ordinal"]),
        depth=int(row["depth"]),
        heading=str(row["heading"]),
        section_path=str(row["section_path"]),
        locator=str(row["locator"]),
        first_block_ordinal=int(row["first_block_ordinal"]),
        last_block_ordinal=int(row["last_block_ordinal"]),
        token_estimate=int(row["token_estimate"]),
    )


def _row_to_block(row: sqlite3.Row) -> SourceBlock:
    return SourceBlock(
        block_id=str(row["block_id"]),
        document_revision_id=str(row["document_revision_id"]),
        section_id=str(row["section_id"]) if row["section_id"] is not None else None,
        ordinal=int(row["ordinal"]),
        kind=str(row["kind"]),
        locator=str(row["locator"]),
        text=str(row["text"]),
        text_sha256=str(row["text_sha256"]),
        canonical_char_start=int(row["canonical_char_start"]),
        canonical_char_end=int(row["canonical_char_end"]),
        previous_block_id=(
            str(row["previous_block_id"])
            if row["previous_block_id"] is not None
            else None
        ),
        next_block_id=(
            str(row["next_block_id"])
            if row["next_block_id"] is not None
            else None
        ),
        table_id=str(row["table_id"]) if row["table_id"] is not None else None,
        row_id=str(row["row_id"]) if row["row_id"] is not None else None,
        headers=tuple(_json_load(str(row["headers_json"]), [])),
        token_estimate=int(row["token_estimate"]),
        extraction_flags=tuple(_json_load(str(row["extraction_flags_json"]), [])),
    )


def _work_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "work_id": str(row["work_id"]),
        "run_id": str(row["run_id"]),
        "parent_work_id": (
            str(row["parent_work_id"]) if row["parent_work_id"] is not None else None
        ),
        "kind": str(row["kind"]),
        "target_kind": str(row["target_kind"]),
        "target_id": str(row["target_id"]),
        "entity_scope_id": str(row["entity_scope_id"]),
        "source_lane": str(row["source_lane"]),
        "wave": int(row["wave"]),
        "queue_ordinal": int(row["queue_ordinal"]),
        "state": str(row["state"]),
        "attempts": int(row["attempts"]),
        "lease_owner": str(row["lease_owner"]) if row["lease_owner"] is not None else None,
        "lease_until": str(row["lease_until"]) if row["lease_until"] is not None else None,
        "started_at": str(row["started_at"]) if row["started_at"] is not None else None,
        "finished_at": str(row["finished_at"]) if row["finished_at"] is not None else None,
        "error_code": str(row["error_code"]),
        "payload": _json_load(str(row["payload_json"]), {}),
        "result": _json_load(str(row["result_json"]), {}),
    }


def _surface_tuple(surfaces: str | Sequence[str]) -> tuple[str, ...]:
    values = (surfaces,) if isinstance(surfaces, str) else tuple(surfaces)
    return tuple(dict.fromkeys(
        normalized
        for value in values
        if (normalized := unicodedata.normalize("NFKC", str(value or "")).strip())
    ))


def _allowed_document_ids(
    values: Sequence[str] | None,
) -> frozenset[str] | None:
    """Normalize an optional authorization boundary.

    ``None`` deliberately preserves legacy local single-user behaviour; an
    explicit empty sequence means that the caller may read no documents.
    """

    if values is None:
        return None
    if isinstance(values, (str, bytes)):
        raise TypeError(
            "allowed_document_revision_ids must be a sequence, not one string"
        )
    return frozenset(str(item).strip() for item in values if str(item).strip())


def _fts_phrase(surface: str) -> str:
    tokens = _FTS_TOKEN.findall(unicodedata.normalize("NFKC", surface))
    if not tokens:
        return ""
    phrase = " ".join(tokens).replace('"', '""')
    return f'"{phrase}"'


def _fts_discovery_expression(query: str, match_mode: str) -> str:
    tokens = _FTS_TOKEN.findall(query)
    if not tokens:
        raise ValueError("query must contain at least one searchable token")
    if len(tokens) > _DISCOVERY_MAX_QUERY_TOKENS:
        raise ValueError(
            f"query contains more than {_DISCOVERY_MAX_QUERY_TOKENS} searchable tokens"
        )
    if match_mode == "phrase":
        return _fts_phrase(query)
    unique: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        key = token.casefold()
        if key not in seen:
            seen.add(key)
            unique.append(token)
    operator = " AND " if match_mode == "all" else " OR "
    return operator.join(f'"{token.replace(chr(34), chr(34) * 2)}"' for token in unique)


def _chunks(values: Sequence[str]) -> Iterator[tuple[str, ...]]:
    for start in range(0, len(values), _SQL_VARIABLE_CHUNK):
        yield tuple(values[start:start + _SQL_VARIABLE_CHUNK])


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _add_seconds(value: str, seconds: float) -> str:
    parsed = datetime.fromisoformat(value)
    return (parsed + timedelta(seconds=float(seconds))).isoformat(timespec="microseconds")


def _json_dump(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        default=str,
    )


def _json_load(value: str, fallback: Any) -> Any:
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return fallback


def _digest(*values: str) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


@contextmanager
def _exclusive_rebuild_lock(path: Path) -> Iterator[None]:
    """Hold a one-byte advisory lock across local rebuild processes."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


__all__ = ["CorpusStore", "DiscoveryPage"]
