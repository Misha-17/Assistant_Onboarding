from __future__ import annotations

import hashlib
import heapq
import math
import re
import threading
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .access import AccessManager, PrincipalSnapshot
from .adaptive_controller import AdaptiveProgress, FairPacketQueue, decision_state, evidence_progress, focused_queries, operational_event, research_features
from .broker import BrokerCallError, InferenceBroker
from .config import Config
from .adaptive_controller import packet_key
from .revisit_blocks import RevisitPlanner, merge_revisit_packet, qualify_revisit_artifact
from .source_priority import prioritize_reader_packets
from .research_budgets import reader_window, screening_cutoff
from .evidence_review import build_synthesis_guidance
from .screen_budget import is_context_overflow, screen_prompt_cost, screen_prompt_limit
from .runtime_compatibility import current_runtime_compatibility
from .identity import EntityRegistry, exact_surfaces, load_profile
from .models import (
    Answer,
    Citation,
    Coverage,
    DocumentManifest,
    DocumentRevision,
    EntityTarget,
    EvidenceCard,
    ModelCallRecord,
    ProgressEvent,
    ScreenDecision,
    Section,
    SourceBlock,
    jsonable,
)
from .ollama import OllamaClient
from .roles import (
    IsolatedReader,
    ReaderArtifact,
    ReaderPacket,
    SemanticScreener,
    Synthesizer,
    _uncited_answer_units,
    qualify_reference_units,
)
from .resources import ResourceRegistry
from .reference_closure import ReferenceIndex, pack_closed
from .excerpt_citations import source_excerpt_overrides
from .evidence_markers import normalize_evidence_markers
from .session import Session
from .store import CorpusStore
from .trace_store import TraceStore


_CARD_MARKER = re.compile(r"\[E:([A-Za-z0-9_.:-]+)\]")
_PUBLIC_SOURCE_MARKER = re.compile(r"\[S\d+\]")
_EXHAUSTIVE = re.compile(
    r"\b(?:all|every|entire|complete list|exhaustive|each document|kaikki|jokainen)\b",
    re.IGNORECASE,
)
_ALL_ABOUT_OVERVIEW = re.compile(
    r"\bwhat\s+(?:is|are)\s+.{1,80}?\s+all\s+about\b",
    re.IGNORECASE,
)
_OVERVIEW = re.compile(
    r"\b(?:what is .{0,80} about|overview|summari[sz]e|trying to accomplish|"
    r"mikä .{0,80} on|mitä .* tekee|yhteenveto)\b",
    re.IGNORECASE,
)
_ACRONYM = re.compile(r"(?u)(?<!\w)[A-ZÅÄÖ][A-ZÅÄÖ0-9_-]{1,15}(?!\w)")
_PROPER_SURFACE = re.compile(
    r"(?u)(?<!\w)([A-ZÅÄÖ][\w.&/-]{1,80}(?:\s+[A-ZÅÄÖ][\w.&/-]{1,80}){0,3})(?!\w)"
)
_QUOTED_SURFACE = re.compile(r"[\"“”]([^\"“”]{2,100})[\"“”]")
_POSSESSIVE_SURFACE = re.compile(r"(?iu)(?<!\w)([\w][\w.&/-]{1,80})(?:['’]s)(?!\w)")
_STRUCTURED_SURFACE = re.compile(r"(?u)(?<!\w)[\w./-]*\d[\w./-]*(?!\w)")
_OWNERSHIP_CONTINUATION = re.compile(
    r"^\s*(?:(?:responsible|owner|owned\s+by|task\s+leader|lead)\s*:)",
    re.IGNORECASE,
)
_PROPER_SURFACE_STOP = frozenset({
    "a", "about", "also", "an", "and", "are", "can", "compare", "could",
    "did", "do", "does", "explain", "for", "from", "give", "how", "in",
    "is", "list", "me", "of", "on", "or", "please", "summarize", "tell",
    "the", "this", "project", "initiative", "funding", "budget", "date", "dates",
    "goal", "goals", "role", "job", "to", "what", "when", "where", "which",
    "who", "why", "would", "management", "software", "development", "security",
    "research", "document", "report", "plan", "work", "task", "team", "partner",
    "company", "organization", "organisation", "quality", "assurance", "deliverable",
    "figure", "amount", "it", "its", "they", "them", "their", "he", "him", "his",
    "she", "her", "hers",
})
_DEICTIC_SURFACE_START = frozenset({
    "it", "its", "they", "them", "their", "theirs", "he", "him", "his",
    "she", "her", "hers", "this", "that", "these", "those",
})
_ROLE_ENTITY_PATTERNS = (
    re.compile(r"(?iu)\bwhat\s+does\s+(.{1,80}?)\s+do\b"),
    re.compile(r"(?iu)\bwhat\s+is\s+(.{1,80}?)\s+about\b"),
    re.compile(r"(?iu)\b(?:what|how)\s+about\s+(.{1,80}?)(?:[?!.;,]|$)"),
    re.compile(
        r"(?iu)\bwhat\s+is\s+(.{1,80}?)(?:['’]s)?\s+"
        r"(?:job|role|responsibilit(?:y|ies))\b"
    ),
)


ProgressCallback = Callable[[ProgressEvent], None]


def _unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value))


def _exact_match_specificities(
    exact_hits_by_doc: Mapping[str, Sequence[str]],
    allowed_document_ids: Iterable[str],
) -> dict[str, float]:
    """Count each literal form once per authorized document, then weight by 1/df.

    This is a navigation priority, never an evidence or answerability score.
    Computing frequencies before capacity limits prevents early catalogue pages
    from crowding out a document containing a discriminative question phrase.
    """
    allowed = frozenset(allowed_document_ids)
    normalized = {
        document_id: frozenset(surface.casefold() for surface in hits if surface)
        for document_id, hits in exact_hits_by_doc.items()
        if document_id in allowed
    }
    frequencies: dict[str, int] = defaultdict(int)
    for hits in normalized.values():
        for surface in hits:
            frequencies[surface] += 1
    return {
        document_id: sum(1.0 / frequencies[surface] for surface in sorted(hits))
        for document_id, hits in normalized.items()
        if hits
    }


def _admit_document_candidates(
    prior_ids: Iterable[str],
    catalogue_ids: Iterable[str],
    exact_specificities: Mapping[str, float],
    *,
    allowed_document_ids: Iterable[str],
    corpus_order: Mapping[str, int],
    capacity: int,
) -> tuple[str, ...]:
    """Preserve session documents and select specific exact hits before slicing.

    Frequencies are computed only once by the caller. The bounded heap selects
    at most capacity entries; later catalogue pages remain resumable as before.
    """
    capacity = max(0, int(capacity))
    if not capacity:
        return ()
    allowed = frozenset(allowed_document_ids)
    selected: dict[str, None] = {}
    for document_id in prior_ids:
        if document_id in allowed:
            selected.setdefault(document_id, None)
            if len(selected) == capacity:
                return tuple(selected)
    ranked = heapq.nsmallest(
        capacity - len(selected),
        (document_id for document_id in exact_specificities
         if document_id in allowed and document_id not in selected),
        key=lambda document_id: (
            -exact_specificities[document_id],
            corpus_order.get(document_id, 10**9),
            document_id,
        ),
    )
    for document_id in (*ranked, *catalogue_ids):
        if document_id in allowed:
            selected.setdefault(document_id, None)
            if len(selected) == capacity:
                break
    return tuple(selected)


def _order_screen_manifests(
    manifests: Sequence[DocumentManifest],
    prior_document_ids: Iterable[str],
    candidate_priorities: Mapping[str, float],
    corpus_order: Mapping[str, int],
) -> tuple[DocumentManifest, ...]:
    """Present admitted manifests by existing navigation priority before cutoff.

    All input objects survive, unchanged and exactly once per input occurrence.
    Prior-session documents keep their supplied order; equal priorities retain
    authorized corpus order and then input order. These are attention priorities,
    not source evidence or a semantic decision to exclude a document.
    """
    prior_order = {document_id: index for index, document_id in enumerate(_unique(prior_document_ids))}

    def key(item: tuple[int, DocumentManifest]) -> tuple[int, int, float, int, int]:
        input_index, manifest = item
        document_id = manifest.document_revision_id
        score = candidate_priorities.get(document_id, 0.0)
        if not isinstance(score, (int, float)) or isinstance(score, bool) or not math.isfinite(score):
            score = 0.0
        return (
            0 if document_id in prior_order else 1,
            prior_order.get(document_id, 0),
            -max(0.0, score),
            corpus_order.get(document_id, input_index),
            input_index,
        )

    return tuple(manifest for _, manifest in sorted(enumerate(manifests), key=key))


def _mode(question: str, requested: str) -> str:
    clean = requested.strip().casefold()
    if clean not in {"auto", "targeted", "overview", "exhaustive"}:
        raise ValueError("mode must be auto, targeted, overview, or exhaustive")
    if clean != "auto":
        return clean
    if _ALL_ABOUT_OVERVIEW.search(question):
        return "overview"
    if _EXHAUSTIVE.search(question):
        return "exhaustive"
    if _OVERVIEW.search(question):
        return "overview"
    return "targeted"


def _manifest_cost(manifest: DocumentManifest) -> int:
    chars = len(manifest.title) + len(manifest.outline) + len(manifest.lead_text)
    chars += sum(len(item) for item in manifest.exact_surfaces)
    return max(80, chars // 4 + 80)


def _block_packet_cost(blocks: Sequence[SourceBlock]) -> int:
    # JSON field names, IDs, locators, and escaping add real prompt cost. This
    # conservative surcharge prevents Ollama from silently truncating a view.
    return 1_200 + sum(block.token_estimate + 48 for block in blocks)


def _has_direct_evidence(artifacts: Sequence[ReaderArtifact]) -> bool:
    return any(
        card.role == "direct"
        for artifact in artifacts
        for card in artifact.report.cards
    )


def _should_stop_secondary_queue(
    *,
    selected_mode: str,
    is_secondary: bool,
    artifacts: Sequence[ReaderArtifact],
    probes_completed: int,
    probe_limit: int,
) -> bool:
    return (
        selected_mode != "exhaustive"
        and is_secondary
        and _has_direct_evidence(artifacts)
        and probes_completed >= max(0, int(probe_limit))
    )


def _all_candidates_screened_unlikely(
    candidate_ids: Sequence[str],
    decisions: Mapping[str, ScreenDecision],
    *,
    exact_ids: Iterable[str] = (),
    prior_ids: Iterable[str] = (),
) -> bool:
    candidates = tuple(candidate_ids)
    forced = set(exact_ids) | set(prior_ids)
    return (
        bool(candidates)
        and not forced.intersection(candidates)
        and all(
            (decision := decisions.get(document_id)) is not None
            and decision.decision == "unlikely"
            for document_id in candidates
        )
    )


def _research_complete(
    *,
    selected_mode: str,
    exhaustive_complete: bool,
    planned_document_ids: Sequence[str],
    completed_document_ids: Iterable[str],
    catalogue_unqueued_ids: Sequence[str],
    semantic_queue_stop: bool,
    research_window_exhausted: bool,
    errors: Sequence[str],
) -> bool:
    if selected_mode == "exhaustive":
        return exhaustive_complete
    completed = set(completed_document_ids)
    return (
        all(document_id in completed for document_id in planned_document_ids)
        and not catalogue_unqueued_ids
        and not semantic_queue_stop
        and not research_window_exhausted
        and not errors
    )


def _read_extraction_gaps(
    documents: Sequence[DocumentRevision],
    read_document_ids: Iterable[str],
) -> tuple[str, ...]:
    read_ids = set(read_document_ids)
    return tuple(
        f"{document.title}: {warning}"
        for document in documents
        if document.document_revision_id in read_ids
        for warning in document.warnings
        if warning
    )


def _has_source_backed_cards(reports: Sequence[ReaderArtifact]) -> bool:
    """An exact, presented source span is the minimum for final synthesis.

    Report prose, titles, and reference flags are not source evidence. Keep all
    card roles eligible: even a qualified context card can support a useful
    exact excerpt. This checks existence, not factual correctness or scope.
    The synthesizer retains its own per-card and context-budget checks.
    """
    for artifact in reports:
        report = artifact.report
        blocks = {block.block_id: block for block in artifact.blocks}
        for card in report.cards:
            if card.document_revision_id != report.document_revision_id:
                continue
            for block_id in card.block_ids:
                block = blocks.get(block_id)
                if (block is not None and block.text.strip()
                        and block.document_revision_id == card.document_revision_id
                        and hashlib.sha256(block.text.encode("utf-8")).hexdigest()
                        == block.text_sha256):
                    return True
    return False


def _no_source_evidence_text(*, packing_limited: bool = False) -> str:
    if packing_limited:
        return (
            "I could not establish an answer from the source evidence available "
            "to the final answer step. No usable source passages remained after "
            "its source validation and context budget checks."
        )
    return (
        "I could not establish an answer from the source evidence successfully "
        "extracted during this reading run. The available document reports did "
        "not retain usable source passages to support an answer."
    )


def _bounded_not_found_text(
    *,
    content_was_read: bool,
    manifests_were_screened: bool = True,
) -> str:
    if content_was_read:
        return "I did not find an answer in the documents that were actually read."
    if not manifests_were_screened:
        return "I did not find an answer in the sources available for this question."
    return (
        "I did not find a likely answer in the document manifests screened for "
        "this question; no document content was read."
    )


def _effective_question(question: str, session: Session | None) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    if session is None:
        return question, (), ()
    context = session.context(question)
    if not context.is_follow_up or not context.prior_user_questions:
        return question, (), ()
    prior = context.prior_user_questions[-1]
    rendered = (
        f"CURRENT USER QUESTION:\n{question}\n\n"
        "PRIOR USER QUESTION (conversation context only; it is not evidence):\n"
        f"{prior}"
    )
    return rendered, context.cited_document_ids, context.prior_user_questions


def _inherited_targets(prior_questions: Sequence[str]) -> tuple[EntityTarget, ...]:
    if not prior_questions:
        return ()
    previous = prior_questions[-1]
    surfaces = list(_ACRONYM.findall(previous))
    if not surfaces:
        candidates = exact_surfaces(previous)
        surfaces = [item for item in candidates if len(item) >= 3][:1]
    return tuple(
        EntityTarget(
            entity_id=f"session:{surface.casefold()}",
            surface=surface,
            canonical_name=surface,
            basis="prior_user_question",
            inherited=True,
        )
        for surface in _unique(surfaces)[:3]
    )



def _question_targets(current_surfaces, resolution):
    """Join approved registry identities and non-authoritative lexical labels."""
    stale = bool(current_surfaces) and any(item.inherited for item in resolution.targets)
    result = [] if stale else list(resolution.targets)
    known = {item.surface.casefold() for item in result}
    ambiguous = {surface.casefold() for surface in resolution.ambiguities}
    for surface in current_surfaces:
        key = surface.casefold()
        if key in known or key in ambiguous:
            continue
        result.append(EntityTarget(entity_id=f"surface:{key}", surface=surface,
                                   basis="question_lexical_surface"))
        known.add(key)
    # Neither capitalization nor a generated label proves distinct identity.
    # Keep only distinctions that arrived from the approved identity registry.
    return tuple(result)


def _forced_surfaces(question: str, targets: Sequence[EntityTarget]) -> tuple[str, ...]:
    # Inherited targets exist only for turns Session classified as genuine
    # follow-ups. Keep their exact lane so "what does it own?" still searches
    # the resolved organization instead of relying on semantic luck.
    values: list[str] = [item.surface for item in targets]
    values.extend(match.group(1) for match in _QUOTED_SURFACE.finditer(question))
    values.extend(match.group(1) for match in _POSSESSIVE_SURFACE.finditer(question))
    values.extend(_ACRONYM.findall(question))
    values.extend(_STRUCTURED_SURFACE.findall(question))
    return _unique(" ".join(item.split()).strip() for item in values)


def _explicit_entity_surfaces(question: str) -> tuple[str, ...]:
    """Return conservative, non-authoritative identity candidates from the turn.

    These candidates partition model attention; they do not assert that a name
    is a particular real-world entity. Approved registry aliases remain the
    authoritative identity layer. This soft lane matters before an enterprise
    has curated its registry (for example, ``Haltian`` beside ``MAISA``).
    """

    values: list[str] = []
    for pattern in _ROLE_ENTITY_PATTERNS:
        for match in pattern.finditer(question):
            candidate = " ".join(match.group(1).strip(" \t\r\n,;:.!?()[]{}\"'’").split())
            words = candidate.split()
            if (
                candidate
                and len(candidate) <= 80
                and len(words) <= 5
                and words[0].casefold() not in _DEICTIC_SURFACE_START
                and any(word.casefold() not in _PROPER_SURFACE_STOP for word in words)
            ):
                values.append(candidate)
    values.extend(match.group(1) for match in _POSSESSIVE_SURFACE.finditer(question))
    values.extend(_ACRONYM.findall(question))
    for match in _PROPER_SURFACE.finditer(question):
        surface = " ".join(match.group(1).split()).strip()
        if not surface:
            continue
        words = surface.split()
        while words and words[0].casefold() in _PROPER_SURFACE_STOP:
            words.pop(0)
        while words and words[-1].casefold() in _PROPER_SURFACE_STOP:
            words.pop()
        if words:
            values.append(" ".join(words))
    return _unique(values)[:8]


def _public_warnings(
    internal: Sequence[str],
    errors: Sequence[str],
    coverage: Coverage,
) -> tuple[str, ...]:
    notes: list[str] = []
    mode = coverage.mode if coverage.mode in {"targeted", "overview", "exhaustive"} else "targeted"
    if coverage.extraction_gaps:
        notes.append("Some source material was not fully extractable; coverage details identify it.")
    if coverage.deadline_reached and coverage.documents_remaining:
        if mode == "exhaustive":
            notes.append(
                f"The interactive reading window ended with {coverage.documents_remaining} "
                "document(s) still queued or outside the first catalogue page; "
                "the requested exhaustive research is incomplete."
            )
        else:
            notes.append(
                f"The interactive reading window ended with {coverage.documents_remaining} "
                "document(s) still queued or outside the first catalogue page; "
                f"this {mode} answer has incomplete research coverage."
            )
    elif any("catalogue" in item for item in coverage.incomplete_reasons):
        if mode == "exhaustive":
            notes.append(
                "This exhaustive request used the current local catalogue page; "
                "additional authorized documents remain, so exhaustive research is incomplete."
            )
        else:
            notes.append(
                f"This {mode} answer used the current local catalogue page; additional "
                "authorized documents remain available for continued research."
            )
    if errors:
        notes.append("One or more research stages had a technical error; completed work was retained.")
    if any(item.startswith("synthesis_fallback_used") for item in internal):
        notes.append("Final synthesis was unavailable, so completed document-reader findings were retained.")
    if any(
        item.startswith((
            "unknown_citation_card_removed",
            "model_authored_public_citation_removed",
            "citation_source_missing",
            "citation_revision_mismatch",
            "citation_hash_mismatch",
        ))
        for item in internal
    ):
        notes.append("One or more invalid citation markers were removed.")
    if any(item in {"no_valid_citations_bound", "answer_has_no_bound_citations"} for item in internal):
        notes.append("The answer contains model text without a valid bound source marker.")
    if any(
        item.startswith(("uncited_answer_units:", "uncited_bound_answer_units:"))
        for item in internal
    ):
        notes.append("Some answer text was not paired with a source marker; inspect the cited passages.")
    if any(
        item.startswith((
            "reader_output_incomplete",
            "synthesis_stream_incomplete",
            "synthesis_incomplete",
        ))
        for item in internal
    ):
        notes.append("A local model hit its output limit; complete records and sentences were retained.")
    if coverage.reference_incomplete:
        if "synthesis_skipped_no_source_evidence" in internal:
            notes.append("Some read passages also had unresolved references; details remain in the research trace.")
        else:
            notes.append("Some referenced conditions could not be kept with their evidence. "
                         "Dependent conclusions remain provisional; inspect the cited passages.")
    return _unique(notes)


class SisuReader:
    """Private model-led document research without a vector-RAG release gate."""

    def __init__(
        self,
        config: Config | None = None,
        *,
        store: CorpusStore | None = None,
        broker: InferenceBroker | None = None,
        access: AccessManager | None = None,
        resources: ResourceRegistry | None = None,
    ) -> None:
        self.config = config or Config.load()
        self.store = store or CorpusStore(self.config)
        self.broker = broker or InferenceBroker(self.config)
        self.access = access or AccessManager(self.config)
        self.resources = resources or ResourceRegistry(self.config, self.access)
        self.screener = SemanticScreener(self.broker)
        self.reader = IsolatedReader(self.broker)
        self.synthesizer = Synthesizer(self.broker)
        from .evidence_review import EvidenceReviewer
        self.evidence_reviewer = EvidenceReviewer(self.broker)
        self.registry = EntityRegistry(self.config)
        self.traces = TraceStore(self.config)
        self._loaded = False
        self._status: dict[str, Any] = {}
        self._ask_lock = threading.RLock()

    def load(self) -> "SisuReader":
        self.store.load()
        model_status: dict[str, Any] = {}
        role_models = [
            ("screen", self.config.effective_screen_model),
            ("reader", self.config.effective_reader_model),
            ("synthesis", self.config.effective_synthesis_model),
        ]
        if self.config.adaptive_research or self.config.claim_review:
            role_models.append(("review", self.config.effective_review_model))
        for role, model in role_models:
            if model in {item.get("model") for item in model_status.values()}:
                existing = next(item for item in model_status.values() if item.get("model") == model)
                model_status[role] = dict(existing)
                continue
            ready = OllamaClient(self.config, model=model).ensure_ready()
            model_status[role] = {"model": ready.model, "ollama_version": ready.version}
        self.registry.reload()
        self._status = {
            **self.store.status(),
            "ready": True,
            "pipeline": "adaptive-evidence-research-v1" if self.config.adaptive_research else "authorize-route-screen-read-synthesize-v2",
            "adaptive_research": self.config.adaptive_research,
            "claim_review": self.config.claim_review,
            "strategy_learning": self.config.strategy_learning,
            "models": model_status,
            "reasoning": self.config.reasoning,
            "deadline_s": self.config.total_deadline_s,
            "minimum_screen_window_s": self.config.minimum_screen_window_s,
            "minimum_reader_window_s": self.config.minimum_reader_window_s,
            "local_only": True,
            "principal_id": self.config.principal_id,
            "authorization_revision": self.access.authorization_revision(),
        }
        self._loaded = True
        return self

    def rebuild(self, paths: str | Path | Iterable[str | Path]):
        report = self.store.rebuild(paths)
        self._loaded = False
        self.store.load()
        return report

    def status(self) -> dict[str, Any]:
        base = self.store.status()
        learning: dict[str, Any] = {"enabled": self.config.strategy_learning}
        if self.config.strategy_learning:
            try:
                from .strategy_learning import StrategyStore
                principal = self.access.principal_snapshot(self.config.principal_id)
                if not principal.known or not principal.active:
                    raise PermissionError("configured principal is inactive")
                with StrategyStore(self.config.workspace_dir / "strategy_learning.sqlite3",
                                   runtime_compatibility=current_runtime_compatibility()) as memory:
                    scoped = memory.status(
                        principal=principal.user_id,
                        authorization_scope=self.access.authorization_scope_hash(principal),
                    )
                learning.update({key: scoped[key] for key in (
                    "verified_replay_steps", "unverified_episodes", "active_policy_ids", "policy_checkpoints"
                )})
            except Exception as exc:
                learning["error"] = type(exc).__name__
        return {
            **base,
            **self._status,
            "ready": self._loaded,
            "model": self.config.model,
            "local_only": True,
            "strategy_learning": learning,
        }

    def close(self) -> None:
        self.store.close()
        self._loaded = False

    def _reauthorize_answer_dependencies(
        self,
        auth: PrincipalSnapshot,
        *,
        document_revision_ids: Iterable[str] = (),
        sources: Sequence[Citation] = (),
    ) -> PrincipalSnapshot:
        """Recheck revision, expiry, and every released dependency.

        Access snapshots are not bearer credentials. A grant or membership can
        expire while a local model is running without changing the database
        revision, so finalization must perform fresh concrete checks.
        """

        current = self.access.principal_snapshot(auth.user_id)
        if (
            current.revision != auth.revision
            or not current.known
            or not current.active
            or not self.access.has_permission(current, "question.ask")
        ):
            raise PermissionError(
                "Access changed while the answer was being prepared. Please retry."
            )

        dependencies = {str(item) for item in document_revision_ids if item}
        if dependencies:
            allowed = set(
                self.access.allowed_document_revision_ids(current, "document.search")
            )
            allowed.intersection_update(
                self.access.allowed_document_revision_ids(current, "document.read")
            )
            allowed.intersection_update(
                self.access.allowed_document_revision_ids(current, "document.cite")
            )
            if not dependencies.issubset(allowed):
                raise PermissionError(
                    "Access changed while the answer was being prepared. Please retry."
                )

        for source in sources:
            resource_id = str(source.resource_id or "")
            if source.resource_type == "document":
                if not resource_id or not all(
                    self.access.can(current, "document", resource_id, action)
                    for action in (
                        "document.search",
                        "document.read",
                        "document.cite",
                    )
                ):
                    raise PermissionError(
                        "Access changed while the answer was being prepared. Please retry."
                    )
            elif source.resource_type == "org_role":
                if (
                    not self.access.has_permission(current, "org_role.search")
                    or not resource_id
                    or not self.access.can(
                        current, "org_role", resource_id, "org_role.read"
                    )
                ):
                    raise PermissionError(
                        "Access changed while the answer was being prepared. Please retry."
                    )
            elif source.resource_type == "video":
                if (
                    not self.access.has_permission(current, "video.search")
                    or not resource_id
                    or not self.access.can(
                        current, "video", resource_id, "video.metadata.read"
                    )
                    or (
                        bool(source.uri)
                        and not self.access.can(
                            current, "video", resource_id, "video.open"
                        )
                    )
                ):
                    raise PermissionError(
                        "Access changed while the answer was being prepared. Please retry."
                    )
            else:
                raise PermissionError(
                    "Access changed while the answer was being prepared. Please retry."
                )
        return current

    @staticmethod
    def _emit(
        callback: ProgressCallback | None,
        started: float,
        stage: str,
        label: str,
        *,
        coverage: Mapping[str, Any] | None = None,
        current: Mapping[str, Any] | None = None,
    ) -> None:
        if callback is None:
            return
        event = ProgressEvent(
            stage=stage,
            label=label,
            elapsed_s=time.perf_counter() - started,
            coverage=dict(coverage or {}),
            current=dict(current or {}),
        )
        try:
            callback(event)
        except Exception:
            # A UI progress renderer is not allowed to break the research run.
            pass

    def _event(self, run_id: str, kind: str, payload: Mapping[str, Any] | None = None) -> None:
        try:
            self.store.record_event(run_id, kind, payload or {})
        except Exception:
            pass

    def _adaptive_event(self, run_id, kind, payload):
        if self.config.trace_mode in {"off", "metrics"}:
            payload = operational_event(payload)
        self._event(run_id, kind, payload)

    def _screen_batches(
        self,
        question: str,
        manifests: Sequence[DocumentManifest],
        targets: Sequence[EntityTarget],
        *,
        deadline: float,
        started: float,
        progress: ProgressCallback | None,
        calls: list[ModelCallRecord],
        warnings: list[str],
        stage_label: str,
    ) -> tuple[tuple[ScreenDecision, ...], int]:
        if self.config.adaptive_research:
            deadline = screening_cutoff(time.perf_counter(), deadline,
                                        minimum_screen_s=self.config.minimum_screen_window_s)
        decisions: list[ScreenDecision] = []
        presented_ids: set[str] = set()
        limit = screen_prompt_limit(self.config)
        batches: list[tuple[DocumentManifest, ...]] = []
        batch: list[DocumentManifest] = []

        def cost(values):
            return screen_prompt_cost(SemanticScreener.build_messages(
                question, values, entity_registry=targets
            ))

        def retain(values, reason):
            decisions.extend(ScreenDecision(
                manifest_id=item.manifest_id, document_revision_id=item.document_revision_id,
                decision="maybe", reason=reason,
            ) for item in values)

        for manifest in manifests:
            if batch and cost((*batch, manifest)) > limit:
                batches.append(tuple(batch))
                batch = []
            if cost((manifest,)) > limit:
                warnings.append(f"screen_manifest_exceeds_context:{manifest.manifest_id}")
                retain((manifest,), "Manifest or question exceeds the screen context allowance; retained for reading.")
            else:
                batch.append(manifest)
        if batch:
            batches.append(tuple(batch))

        index = 0
        context_splits = 0
        while index < len(batches):
            remaining = deadline - time.perf_counter()
            if remaining < self.config.minimum_screen_window_s:
                break
            values = batches[index]
            index += 1
            self._emit(progress, started, "screening", stage_label,
                       current={"batch": index, "batches": len(batches), "manifests": len(values)})
            presented_ids.update(item.manifest_id for item in values)
            try:
                artifact = self.screener.screen(
                    question, values, entity_registry=targets,
                    timeout_s=min(self.config.request_timeout_s, remaining),
                )
                calls.append(artifact.call)
                decisions.extend(artifact.decisions)
                warnings.extend(artifact.warnings)
            except Exception as exc:
                if isinstance(exc, BrokerCallError):
                    calls.append(exc.record)
                # At most two binary splits (four extra requests) per screen
                # stage, all inside the original deadline. Other failures have
                # the original fail-open behavior and are never retried.
                if is_context_overflow(exc) and len(values) > 1 and context_splits < 2:
                    middle = len(values) // 2
                    batches[index:index] = [values[:middle], values[middle:]]
                    context_splits += 1
                    warnings.append(f"screen_context_overflow_split:{len(values)}")
                    continue
                cause = type(exc.__cause__).__name__ if exc.__cause__ else type(exc).__name__
                warnings.append(f"screen_call_failed:{cause}")
                retain(values, "Screening failed; retained for consideration.")
        seen = {item.manifest_id for item in decisions}
        for manifest in manifests:
            if manifest.manifest_id not in seen:
                retain((manifest,), "Not screened before the interactive deadline; retained in the queue.")
        # Splitting must not reorder candidate admission or double-count a
        # manifest as coverage merely because the same metadata was retried.
        by_id = {item.manifest_id: item for item in decisions}
        return tuple(by_id[item.manifest_id] for item in manifests), len(presented_ids)


    def _section_manifests(
        self,
        document: DocumentRevision,
        sections: Sequence[Section],
        blocks: Sequence[SourceBlock],
        exact_block_ids: set[str],
        query_surfaces: Sequence[str],
    ) -> tuple[DocumentManifest, ...]:
        by_section: dict[str | None, list[SourceBlock]] = defaultdict(list)
        for block in blocks:
            by_section[block.section_id].append(block)
        result: list[DocumentManifest] = []
        for section in sections:
            values = by_section.get(section.section_id, [])
            if not values:
                continue
            lead = "\n".join(block.text for block in values[:3])[:1_600]
            hits = tuple(
                surface
                for surface in query_surfaces
                if any(
                    block.block_id in exact_block_ids
                    and surface.casefold() in block.text.casefold()
                    for block in values
                )
            )
            result.append(
                DocumentManifest(
                    manifest_id=section.section_id,
                    document_revision_id=document.document_revision_id,
                    title=f"{document.title} — {section.heading or section.section_path or 'document section'}",
                    source_path=document.source_path,
                    file_type=document.file_type,
                    extraction_coverage=document.extraction_coverage,
                    outline=section.section_path,
                    lead_text=lead,
                    exact_surfaces=_unique(hits),
                    token_estimate=section.token_estimate,
                    warnings=document.warnings,
                )
            )
        return tuple(result)

    def _pack_document(
        self,
        run_id: str,
        question: str,
        document: DocumentRevision,
        targets: Sequence[EntityTarget],
        role_targets: Sequence[EntityTarget],
        query_surfaces: Sequence[str],
        exact_blocks: Sequence[SourceBlock],
        *,
        allowed_document_ids: Sequence[str],
        exhaustive: bool = False,
        deadline: float,
        started: float,
        progress: ProgressCallback | None,
        calls: list[ModelCallRecord],
        warnings: list[str],
    ) -> tuple[ReaderPacket, ...]:
        blocks = self.store.blocks_for_document(
            document.document_revision_id,
            allowed_document_revision_ids=allowed_document_ids,
        )
        sections = self.store.sections_for_document(
            document.document_revision_id,
            allowed_document_revision_ids=allowed_document_ids,
        )
        packet_budget = max(2_000, int(self.config.reader_input_tokens * 0.78))
        entity_scope = ",".join(item.entity_id for item in targets) or "DOCUMENT"

        def close_references(values: tuple[ReaderPacket, ...]) -> tuple[ReaderPacket, ...]:
            if not self.config.reference_closure:
                return values
            index = ReferenceIndex(blocks, sections)
            result = []
            for value in values:
                for part in pack_closed(index, value.blocks, packet_budget, _block_packet_cost):
                    if part.issues:
                        warnings.append("reference_packet_observed_gap:" + document.document_revision_id)
                    section_ids = {b.section_id for b in part.blocks if b.section_id}
                    result.append(replace(
                        value, packet_id=f"{value.packet_id}-closed-{len(result) + 1}",
                        blocks=part.blocks, sections_seen=len(section_ids),
                        section_label=value.section_label + " (including detected reference context)",
                        complete_document_read={b.block_id for b in part.blocks} ==
                                               {b.block_id for b in blocks},
                        reference_issues=part.issues,
                    ))
            return tuple(result)

        if _block_packet_cost(blocks) <= packet_budget:
            return close_references((
                ReaderPacket(
                    packet_id=f"{run_id}-{document.document_revision_id}-whole",
                    document_revision_id=document.document_revision_id,
                    document_title=document.title,
                    source_path=document.source_path,
                    blocks=blocks,
                    section_label="complete document",
                    entity_scope=entity_scope,
                    sections_seen=len(sections),
                    sections_total=len(sections),
                    complete_document_read=True,
                ),
            ))

        exact_ids = {
            block.block_id
            for block in exact_blocks
            if block.document_revision_id == document.document_revision_id
        }
        section_manifests = self._section_manifests(
            document, sections, blocks, exact_ids, query_surfaces
        )
        section_decisions: tuple[ScreenDecision, ...] = ()
        section_screen_deadline = deadline - self.config.minimum_reader_window_s
        if (
            not exhaustive
            and section_manifests
            and section_screen_deadline - time.perf_counter()
            >= self.config.minimum_screen_window_s
        ):
            section_decisions, _ = self._screen_batches(
                question,
                section_manifests,
                role_targets,
                deadline=section_screen_deadline,
                started=started,
                progress=progress,
                calls=calls,
                warnings=warnings,
                stage_label=f"Screening coherent sections in {document.title}",
            )

        direct_block_sections = {
            block.section_id for block in blocks if block.block_id in exact_ids and block.section_id
        }
        read_ids = {
            decision.manifest_id for decision in section_decisions if decision.decision == "read"
        }
        maybe_ids = {
            decision.manifest_id for decision in section_decisions if decision.decision == "maybe"
        }
        if exhaustive:
            selected = {section.section_id for section in sections}
        else:
            semantic_ids = read_ids or maybe_ids
            # A selected parent heading represents its coherent subtree. The
            # section manifest's token estimate already includes descendants;
            # packet construction must honor the same boundary rather than
            # showing only the parent's few direct blocks.
            expanded_semantic = set(semantic_ids)
            frontier = list(semantic_ids)
            while frontier:
                parent_id = frontier.pop()
                for section in sections:
                    if (
                        section.parent_section_id == parent_id
                        and section.section_id not in expanded_semantic
                    ):
                        expanded_semantic.add(section.section_id)
                        frontier.append(section.section_id)
            selected = expanded_semantic | direct_block_sections
        if not selected and sections:
            # The model could not narrow this oversized document. Retaining all
            # sections lets the token/deadline scheduler expose an honest queue
            # instead of inventing a relevance cut-off.
            selected = {section.section_id for section in sections}

        section_by_id = {section.section_id: section for section in sections}
        include_unsectioned = (
            exhaustive
            or not sections
            or selected == {section.section_id for section in sections}
        )
        block_groups: list[tuple[Section | None, tuple[SourceBlock, ...]]] = []
        group_section_id: str | None = None
        group_blocks: list[SourceBlock] = []

        def flush_group() -> None:
            if not group_blocks:
                return
            block_groups.append((section_by_id.get(group_section_id), tuple(group_blocks)))
            group_blocks.clear()

        # Walk the source once so parent blocks that surround child sections,
        # plus unsectioned front matter, never get reordered by grouping.
        for block in blocks:
            include = (
                block.section_id in selected
                if block.section_id is not None
                else include_unsectioned
            )
            if not include:
                flush_group()
                group_section_id = None
                continue
            if group_blocks and block.section_id != group_section_id:
                flush_group()
            group_section_id = block.section_id
            group_blocks.append(block)
        flush_group()
        if not block_groups:
            block_groups = [(None, tuple(blocks))]

        packets: list[ReaderPacket] = []
        pending_blocks: list[SourceBlock] = []
        pending_sections: list[Section] = []

        def flush() -> None:
            if not pending_blocks:
                return
            index = len(packets) + 1
            labels = [item.section_path or item.heading for item in pending_sections]
            packets.append(
                ReaderPacket(
                    packet_id=f"{run_id}-{document.document_revision_id}-section-{index}",
                    document_revision_id=document.document_revision_id,
                    document_title=document.title,
                    source_path=document.source_path,
                    blocks=tuple(pending_blocks),
                    section_label="; ".join(filter(None, labels)) or "coherent document part",
                    entity_scope=entity_scope,
                    sections_seen=max(1, len(pending_sections)),
                    sections_total=len(sections),
                    complete_document_read=False,
                )
            )
            pending_blocks.clear()
            pending_sections.clear()

        for section, values in block_groups:
            proposed = (*pending_blocks, *values)
            if pending_blocks and _block_packet_cost(proposed) > packet_budget:
                flush()
            if _block_packet_cost(values) <= packet_budget:
                pending_blocks.extend(values)
                if section is not None and section not in pending_sections:
                    pending_sections.append(section)
                continue
            # Last-resort split for one pathological natural section. It is
            # explicit in coverage and never presented as a complete read.
            warnings.append(f"oversized_natural_section_split:{section.section_id if section else document.document_revision_id}")
            for block in values:
                if pending_blocks and _block_packet_cost((*pending_blocks, block)) > packet_budget:
                    if (
                        _OWNERSHIP_CONTINUATION.search(block.text)
                        and len(pending_blocks) > 1
                    ):
                        predecessor = pending_blocks.pop()
                        flush()
                        pending_blocks.append(predecessor)
                        if section is not None:
                            pending_sections.append(section)
                    else:
                        flush()
                pending_blocks.append(block)
                if section is not None and section not in pending_sections:
                    pending_sections.append(section)
        flush()
        return close_references(tuple(packets))

    @staticmethod
    def _with_reference_context(artifact: ReaderArtifact, index: ReferenceIndex) -> ReaderArtifact:
        """Bind dependency metadata to the actual packet after model-ID restoration."""
        presented = {b.block_id for b in artifact.blocks}
        cards = []
        warnings = list(artifact.report.warnings)
        for card in artifact.report.cards:
            closure = index.audit(card.block_ids, presented)
            context = tuple(bid for bid in closure.required_block_ids
                            if bid not in card.block_ids and bid in presented)
            if closure.issues:
                warnings.append(f"reference_unresolved_card:{card.card_id}")
            cards.append(replace(card, context_block_ids=context,
                                 reference_issues=closure.issues,
                                 reference_checked=True,
                                 role="context" if closure.issues and card.role == "direct" else card.role))
        packet_issues = index.audit(tuple(presented), presented).issues
        if packet_issues and not cards:
            warnings.append("reference_packet_incomplete:reader_report")
        return replace(artifact, report=replace(artifact.report, cards=tuple(cards),
                       reference_issues=packet_issues, warnings=tuple(_unique(warnings))))

    @staticmethod
    def _fallback_answer(reports: Sequence[ReaderArtifact]) -> tuple[str, tuple[EvidenceCard, ...]]:
        cards = tuple(card for artifact in reports for card in artifact.report.cards)
        if cards:
            lookup = {b.block_id: b for artifact in reports for b in artifact.blocks}
            def finding(card: EvidenceCard) -> str:
                if card.reference_issues:
                    return ("The cited passage has an unresolved referenced condition; "
                            "a definite conclusion from it is unavailable.")
                context = " ".join(lookup[bid].text for bid in card.context_block_ids if bid in lookup)
                return card.claim + (f" Referenced source context: {context}" if context else "")
            return "\n".join(
                f"- {card.subject} — {card.document_title}: "
                f"{finding(card)} [E:{card.card_id}]"
                for card in cards
            ), cards
        natural = [
            f"{artifact.report.document_title}: " +
            ("A referenced condition is unresolved; a definite conclusion is unavailable."
             if artifact.report.reference_issues else artifact.report.proposed_answer)
            for artifact in reports
            if artifact.report.proposed_answer.strip()
        ]
        return "\n\n".join(natural), ()

    @staticmethod
    def _alias_reader_packet(
        packet: ReaderPacket,
    ) -> tuple[ReaderPacket, dict[str, str], tuple[SourceBlock, ...]]:
        """Give the model short copyable IDs, then restore immutable IDs later."""

        original = packet.blocks
        alias_to_original = {
            f"B{index:04d}": block.block_id
            for index, block in enumerate(original, 1)
        }
        original_to_alias = {value: key for key, value in alias_to_original.items()}
        aliased = tuple(
            replace(
                block,
                block_id=original_to_alias[block.block_id],
                previous_block_id=original_to_alias.get(block.previous_block_id or ""),
                next_block_id=original_to_alias.get(block.next_block_id or ""),
            )
            for block in original
        )
        return replace(packet, blocks=aliased), alias_to_original, original

    @staticmethod
    def _restore_reader_artifact(
        artifact: ReaderArtifact,
        alias_to_original: Mapping[str, str],
        original_blocks: tuple[SourceBlock, ...],
    ) -> ReaderArtifact:
        report = artifact.report
        cards = tuple(
            replace(
                card,
                block_ids=tuple(
                    alias_to_original[item]
                    for item in card.block_ids
                    if item in alias_to_original
                ),
            )
            for card in report.cards
        )
        records = tuple(
            replace(
                record,
                block_ids=tuple(
                    alias_to_original[item]
                    for item in record.block_ids
                    if item in alias_to_original
                ),
            )
            for record in report.records
        )
        restored_report = replace(
            report,
            cards=cards,
            records=records,
            presented_block_ids=tuple(block.block_id for block in original_blocks),
        )
        return ReaderArtifact(
            report=restored_report,
            blocks=original_blocks,
            call=artifact.call,
        )

    def _bind_citations(
        self,
        text: str,
        cards: Sequence[EvidenceCard],
        block_lookup: Mapping[str, SourceBlock],
        document_lookup: Mapping[str, DocumentRevision],
    ) -> tuple[str, tuple[Citation, ...], tuple[str, ...]]:
        warnings: list[str] = []
        by_card = {card.card_id: card for card in cards}
        citations: list[Citation] = []
        rendered_by_card: dict[tuple[str, tuple[str, ...]], str] = {}
        source_id_by_block: dict[str, str] = {}
        excerpt_overrides: dict[int, tuple[str, ...]] = {}

        def marker(match: re.Match[str]) -> str:
            card_id = match.group(1)
            card = by_card.get(card_id)
            if card is None:
                warnings.append(f"unknown_citation_card_removed:{card_id}")
                return ""
            selected_ids = excerpt_overrides.get(match.start())
            if selected_ids is None:
                selected_ids = _unique((*card.block_ids, *card.context_block_ids))
            else:
                warnings.append(f"citation_complete_source_excerpt_bound:{card_id}")
            cache_key = (card_id, selected_ids)
            if cache_key in rendered_by_card:
                return rendered_by_card[cache_key]
            refs: list[str] = []
            for block_id in selected_ids:
                block = block_lookup.get(block_id)
                document = document_lookup.get(card.document_revision_id)
                if block is None or document is None:
                    warnings.append(f"citation_source_missing:{card_id}:{block_id}")
                    continue
                if block.document_revision_id != card.document_revision_id:
                    warnings.append(f"citation_revision_mismatch:{card_id}:{block_id}")
                    continue
                expected = hashlib.sha256(block.text.encode("utf-8")).hexdigest()
                if expected != block.text_sha256:
                    warnings.append(f"citation_hash_mismatch:{card_id}:{block_id}")
                    continue
                source_id = source_id_by_block.get(block.block_id)
                if source_id is None:
                    source_id = f"S{len(citations) + 1}"
                    source_id_by_block[block.block_id] = source_id
                    citations.append(
                        Citation(
                            source_id=source_id,
                            card_id=card_id,
                            block_id=block.block_id,
                            document_revision_id=block.document_revision_id,
                            title=document.title,
                            locator=block.locator,
                            quote=block.text,
                            source_path=document.source_path,
                            quote_sha256=block.text_sha256,
                            resource_type="document",
                            resource_id=document.logical_document_id,
                            provenance="immutable document block",
                        )
                    )
                reference = f"[{source_id}]"
                if reference not in refs:
                    refs.append(reference)
            rendered = "".join(refs)
            rendered_by_card[cache_key] = rendered
            return rendered

        # Public source numbers belong solely to this binder. Strip any S-tag
        # the model authored before assigning real numbers, otherwise a fake
        # [S999] could make an unsupported paragraph appear cited.
        def remove_public_marker(match: re.Match[str]) -> str:
            warnings.append(f"model_authored_public_citation_removed:{match.group(0)}")
            return ""

        sanitized_text = _PUBLIC_SOURCE_MARKER.sub(remove_public_marker, text)
        excerpt_overrides = source_excerpt_overrides(
            sanitized_text, cards, source_blocks=block_lookup)
        rendered_text = _CARD_MARKER.sub(marker, sanitized_text).strip()
        rendered_text = re.sub(r"(?:\[S\d+\])+", lambda match: "".join(dict.fromkeys(
            re.findall(r"\[S\d+\]", match.group(0)))), rendered_text)
        if cards and not citations:
            warnings.append("no_valid_citations_bound")
        return rendered_text, tuple(citations), tuple(_unique(warnings))

    def _finalize_resource_answer(
        self,
        question: str,
        answer: Answer,
        *,
        auth: PrincipalSnapshot,
        session: Session | None,
    ) -> Answer:
        """Persist a fast deterministic role/video answer and its safe trace."""

        started = time.perf_counter()
        route = str(answer.debug.get("answer_route") or "protected_resource")
        snapshot = self.store.snapshot()
        run_id = self.store.create_run(
            question,
            effective_question=question,
            session_id=f"session_{id(session):x}" if session is not None else "",
            snapshot_id=snapshot.snapshot_id,
            authorization_scope_hash=self.access.authorization_scope_hash(auth),
            principal_id=auth.user_id,
            authorization_revision=auth.revision,
            route=route,
            completeness_mode="targeted",
            config={"model_calls": 0, "local_only": True},
        )
        try:
            self._reauthorize_answer_dependencies(auth, sources=answer.sources)
        except PermissionError:
            self.store.set_run_state(run_id, "access_changed", finalized=True)
            raise
        answer.debug = {
            **answer.debug,
            "run_id": run_id,
            "principal_id": auth.user_id,
            "authorization_revision": auth.revision,
            "model_calls": 0,
            "pipeline": "protected-resource-direct-v1",
        }
        answer.timings = {
            "answer_ready_s": time.perf_counter() - started,
            "total_s": time.perf_counter() - started,
        }
        self.store.set_run_state(run_id, answer.status, finalized=True)
        trace_payload = {
            "request": {"question": question, "mode": "targeted"},
            "route": route,
            "principal_id": auth.user_id,
            "authorization_revision": auth.revision,
            "resource_dependencies": [
                {"type": item.resource_type, "id": item.resource_id}
                for item in answer.sources
            ],
            "citation_checks": [jsonable(item) for item in answer.sources],
            "coverage": jsonable(answer.coverage),
            "answer": jsonable(answer),
            "model_calls": [],
            "timings": dict(answer.timings),
        }
        trace_path = self.traces.write(run_id, trace_payload)
        if trace_path is not None:
            self.access.register_resource(
                "trace",
                run_id,
                stable_key=run_id,
                classification="restricted",
                owner_user_id=auth.user_id,
                metadata={
                    "resource_dependencies": [
                        f"{item.resource_type}:{item.resource_id}" for item in answer.sources
                    ]
                },
                actor_user_id=auth.user_id,
                affects_authorization=False,
            )
            if self.access.can(auth, "trace", run_id, "trace.read_own") or self.access.can(
                auth, "trace", run_id, "trace.read_any"
            ):
                answer.trace_path = str(trace_path)
        elif self.config.trace_mode != "off":
            answer.warnings = tuple(_unique((
                *answer.warnings,
                f"trace_write_failed:{self.traces.last_error}",
            )))
        if session is not None:
            session.record(question, answer)
        return answer

    def ask(
        self,
        question: str,
        *,
        session: Session | None = None,
        mode: str = "auto",
        progress: ProgressCallback | None = None,
        principal: str | PrincipalSnapshot | None = None,
    ) -> Answer:
        clean_question = " ".join(str(question).split()).strip()
        if not clean_question:
            raise ValueError("question cannot be empty")
        if len(clean_question) > 8_000:
            raise ValueError("question is too long")
        with self._ask_lock:
            if not self._loaded:
                self.load()
            auth = (
                principal
                if isinstance(principal, PrincipalSnapshot)
                else self.access.principal_snapshot(principal or self.config.principal_id)
            )
            if not auth.known or not auth.active or not self.access.has_permission(
                auth, "question.ask"
            ):
                raise PermissionError("The requested research workspace is unavailable.")
            if session is not None:
                session.bind_authorization(auth.user_id, auth.revision)
            snapshot = self.store.snapshot()
            resource_answer = self.resources.answer_videos(
                auth, clean_question, snapshot_id=snapshot.snapshot_id
            )
            if resource_answer is None:
                resource_answer = self.resources.answer_roles(
                    auth, clean_question, snapshot_id=snapshot.snapshot_id
                )
            if resource_answer is not None:
                return self._finalize_resource_answer(
                    clean_question,
                    resource_answer,
                    auth=auth,
                    session=session,
                )
            from .runtime_compatibility import runtime_epoch_scope
            with runtime_epoch_scope(self.config), self.broker.answer_budget(self.config):
                return self._ask(
                    clean_question,
                    session=session,
                    mode=mode,
                    progress=progress,
                    auth=auth,
                )

    def _rank_strategy_actions(self, context, *, principal, authorization_scope, snapshot_id):
        """Production rankings use promoted checkpoints only.

        Locked offline evaluations may override this method to shadow a
        candidate through the exact same scheduler. No runtime flag activates
        an unverified checkpoint.
        """
        if not self.config.strategy_learning:
            return ()
        from .strategy_learning import StrategyStore
        with StrategyStore(self.config.workspace_dir / "strategy_learning.sqlite3",
                                   runtime_compatibility=current_runtime_compatibility()) as memory:
            return memory.rank_actions(
                principal=principal, authorization_scope=authorization_scope,
                snapshot_id=snapshot_id, context=context,
                allowed_actions=("balance_documents", "broaden_search", "focused_gap_search"),
                exploration=0.0,
            )

    def _adaptive_read(
        self, *, run_id, question, selected_mode, auth, snapshot_id,
        document_lookup, allowed_document_ids, targets, role_targets,
        query_surfaces, exact_blocks, read_queue, queued_ids, work_ids,
        annotated_manifests, discovery_state, catalogue_unqueued_ids,
        screening, reader_artifacts, calls, warnings, errors,
        synthesis_cutoff, started, progress, strategy_action="", strategy_features=(), strategy_scope="",
    ) -> AdaptiveProgress:
        """Execute a bounded adaptive search; all models remain sequential.

        The reviewer supplies search suggestions, never a correctness oracle.
        Source IDs are pinned to this run, and every model-visible document is
        reauthorized before use. Exhaustiveness still depends on actual reads.
        """
        state = AdaptiveProgress(catalogue_unqueued_ids=catalogue_unqueued_ids)
        state.debug["strategy_action"] = strategy_action or "default"
        executed_actions: set[str] = set()
        queue = FairPacketQueue(read_queue)
        revisit_planner = RevisitPlanner()
        admitted = {item.document_revision_id for item in annotated_manifests}
        read_blocks: dict[str, set[str]] = defaultdict(set)
        document_data: dict[str, tuple] = {}
        packet_counts: dict[str, int] = defaultdict(int)
        packet_totals: dict[str, int] = {}
        cursors: list[dict[str, Any]] = []
        attempted_pages: set[tuple[str, str, int]] = set()
        page_work_ids: set[str] = set()
        completed_page_work_ids: set[str] = set()
        if discovery_state.get("next_offset") is not None:
            cursors.append({
                "query": discovery_state["query"], "match_mode": discovery_state["match_mode"],
                "offset": discovery_state["next_offset"], "work_id": discovery_state.get("work_id"),
            })
            if discovery_state.get("work_id"):
                page_work_ids.add(discovery_state["work_id"])
        attempted_queries = {str(discovery_state.get("query", "")).casefold()}
        latest_queries: tuple[str, ...] = ()
        literal_queries_used = 0
        missing_count = 0
        scope_count = 0
        trajectory: list[dict[str, Any]] = []
        research_budget = max(1.0, synthesis_cutoff - started)
        max_waves = 1 + max(0, min(2, int(self.config.adaptive_max_additional_waves)))
        base_quota = max(1, min(64, int(self.config.adaptive_packets_per_wave)))
        state.debug.update({"maximum_waves": max_waves, "packets_per_wave": base_quota})

        def context(wave):
            from .learning_policy import encode_context
            blocks, cards = evidence_progress(reader_artifacts)
            stats = decision_state(
                remaining_s=synthesis_cutoff - time.perf_counter(), research_budget_s=research_budget,
                document_count=len(document_lookup), documents_read=len(state.documents_read),
                catalogue_remaining=len(state.catalogue_unqueued_ids), unique_blocks=blocks,
                source_bound_cards=cards, missing_obligations=missing_count, scope_issues=scope_count,
                pending_packets=sum(len(queue.packets[i]) if i in queue.packets else 1 for i in queue.pending),
                wave=wave, maximum_waves=max_waves,
            )
            return list(encode_context(features=strategy_features, stats=stats))

        def guard(document_ids=()):
            if self.store.snapshot().snapshot_id != snapshot_id:
                raise RuntimeError("active corpus snapshot changed during adaptive research")
            self._reauthorize_answer_dependencies(auth, document_revision_ids=document_ids)

        def discover(wave):
            nonlocal latest_queries, literal_queries_used
            # One continuation and one new gap query per wave: no query explosion.
            requests = []
            pending = cursors.pop(0) if cursors else None
            fresh = next((q for q in latest_queries if q.casefold() not in attempted_queries), None)
            if fresh:
                attempted_queries.add(fresh.casefold())
                requests.append({"query": fresh, "match_mode": "all", "offset": 0, "gap_query": True})
            if pending:
                requests.insert(len(requests) if strategy_action == "focused_gap_search" else 0, pending)
            if not requests and state.catalogue_unqueued_ids:
                # An all-term query can miss distributed wording; a single
                # original-question union search is a bounded generic fallback.
                key = (question, "any", 0)
                if key not in attempted_pages:
                    requests.append({"query": question, "match_mode": "any", "offset": 0})
            new_ids = []
            gap_hits = []
            revisit_queries = {}
            discovery_records = []
            for request in requests[:2]:
                if synthesis_cutoff - time.perf_counter() < self.config.minimum_reader_window_s:
                    break
                key = (request["query"], request["match_mode"], request["offset"])
                if key in attempted_pages:
                    continue
                attempted_pages.add(key)
                guard()
                try:
                    page = self.store.discover_documents(
                        request["query"], offset=request["offset"],
                        page_size=min(1000, self.config.discovery_page_size *
                                      (2 if strategy_action == "broaden_search" else 1)),
                        match_mode=request["match_mode"], expected_snapshot_id=snapshot_id,
                        allowed_document_revision_ids=allowed_document_ids,
                    )
                    executed_actions.add("focused_gap_search" if request.get("gap_query") else "broaden_search")
                    # A nonempty page containing only completed work does not
                    # resolve a gap. Permit one bounded union fallback, while
                    # treating pending admitted first packets as useful hits.
                    if request.get("gap_query"):
                        revisit_queries.update((item, request["query"]) for item in page.document_revision_ids
                                               if item in document_lookup and item in state.documents_read)
                    useful_hits = any(item in document_lookup and (
                        item not in admitted or item in queue.pending and item not in state.documents_read
                    ) for item in page.document_revision_ids)
                    literal_attempts = []
                    if (request.get("gap_query") and page.match_mode == "all" and page.offset == 0
                            and not useful_hits and literal_queries_used < 2):
                        from .focused_literals import grounded_literal_queries
                        plan = grounded_literal_queries(
                            request["query"],
                            (block for artifact in reversed(reader_artifacts) for block in artifact.blocks),
                            allowed_document_ids=allowed_document_ids, limit=2 - literal_queries_used,
                        )
                        warnings.extend(plan.warnings)
                        for candidate in plan.candidates:
                            literal_key = (candidate.query, "phrase", 0)
                            if literal_key in attempted_pages:
                                continue
                            if synthesis_cutoff - time.perf_counter() < self.config.minimum_reader_window_s:
                                break
                            guard(candidate.source_document_ids)
                            attempted_pages.add(literal_key)
                            literal_queries_used += 1
                            candidate_page = self.store.discover_documents(
                                candidate.query, page_size=self.config.discovery_page_size,
                                match_mode="phrase", expected_snapshot_id=snapshot_id,
                                allowed_document_revision_ids=allowed_document_ids,
                            )
                            useful_candidate = any(item in document_lookup and (
                                item not in admitted or item in queue.pending and item not in state.documents_read
                            ) for item in candidate_page.document_revision_ids)
                            literal_attempts.append({"query": candidate.query, "mode": "phrase",
                                "kind": candidate.kind, "returned": candidate_page.returned,
                                "useful_pending_or_new_hit": useful_candidate,
                                "source_block_ids": candidate.source_block_ids,
                                "observed_document_frequency": candidate.observed_document_frequency})
                            if useful_candidate:
                                page = candidate_page
                                useful_hits = True
                                break
                    fallback_key = (request["query"], "any", 0)
                    if (page.match_mode == "all" and page.offset == 0
                            and (not page.document_revision_ids or request.get("gap_query") and not useful_hits)
                            and fallback_key not in attempted_pages):
                        attempted_pages.add(fallback_key)
                        page = self.store.discover_documents(
                            request["query"], page_size=self.config.discovery_page_size,
                            match_mode="any", expected_snapshot_id=snapshot_id,
                            allowed_document_revision_ids=allowed_document_ids,
                        )
                    # Defend the interface even when a custom store is supplied.
                    candidate_ids = [item for item in page.document_revision_ids
                                     if item in document_lookup and item not in admitted]
                    new_ids.extend(candidate_ids)
                    if request.get("gap_query"):
                        gap_hits.extend(item for item in page.document_revision_ids if item in document_lookup)
                        revisit_queries.update((item, request["query"]) for item in page.document_revision_ids
                                               if item in document_lookup and item in state.documents_read)
                    if request.get("work_id"):
                        self.store.finish_work(request["work_id"], state="completed",
                                               result={"returned": page.returned})
                        completed_page_work_ids.add(request["work_id"])
                    if page.next_offset is not None:
                        work_id = self.store.enqueue_work(
                            run_id, "discover_documents", "catalogue_page",
                            f"{hashlib.sha256(page.query.encode()).hexdigest()[:12]}:{page.next_offset}",
                            source_lane="adaptive_discovery", wave=wave + 1, queue_ordinal=len(cursors),
                            payload={"query": page.query, "match_mode": page.match_mode,
                                     "offset": page.next_offset, "snapshot_id": snapshot_id},
                        )
                        cursors.append({"query": page.query, "match_mode": page.match_mode,
                                        "offset": page.next_offset, "work_id": work_id})
                        page_work_ids.add(work_id)
                    discovery_records.append({"query": page.query, "mode": page.match_mode,
                                              "offset": page.offset, "returned": page.returned,
                                              "new_documents": len(candidate_ids),
                                              "literal_attempts": literal_attempts})
                except (PermissionError, RuntimeError):
                    raise
                except Exception as exc:
                    warnings.append(f"adaptive_discovery_failed:{type(exc).__name__}")
            new_ids = list(_unique(new_ids))
            if new_ids:
                guard(new_ids)
                manifests = self.store.manifests(new_ids, allowed_document_revision_ids=allowed_document_ids)
                decisions, presented = self._screen_batches(
                    question, manifests, role_targets,
                    deadline=synthesis_cutoff - self.config.minimum_reader_window_s,
                    started=started, progress=progress, calls=calls, warnings=warnings,
                    stage_label=f"Screening new documents in research wave {wave + 1}",
                )
                screening.extend(decisions)
                state.manifests_presented += presented
                state.added_manifests.extend(manifests)
                admitted.update(item.document_revision_id for item in manifests)
                decision_map = {item.document_revision_id: item for item in decisions}
                for document_id in new_ids:
                    if document_id not in queued_ids:
                        queued_ids.append(document_id)
                    decision = decision_map.get(document_id)
                    work_ids[document_id] = self.store.enqueue_work(
                        run_id, "read_document", "document_revision", document_id,
                        source_lane="adaptive_discovery", wave=wave, queue_ordinal=len(queued_ids),
                        payload={"decision": decision.decision if decision else "maybe"},
                    )
                    if selected_mode == "exhaustive" or not decision or decision.decision != "unlikely":
                        if queue.add(document_id):
                            read_queue.append(document_id)
                # New document first packets precede second packets of long docs.
                fresh_ids = [item for item in queue.pending if item not in state.documents_read]
                old_ids = [item for item in queue.pending if item in state.documents_read]
                queue.pending.clear()
                queue.pending.extend((*fresh_ids, *old_ids))
            if strategy_action == "focused_gap_search":
                prioritized = queue.prioritize(
                    gap_hits, unread=set(document_lookup) - state.documents_read, limit=base_quota,
                )
                state.debug.setdefault("focused_priority", []).append({
                    "wave": wave, "matched_documents": len(set(gap_hits)),
                    "promoted_pending_documents": list(prioritized),
                })
            if strategy_action == "focused_gap_search":
                for document_id, gap_query in list(revisit_queries.items())[:2]:
                    if document_id not in state.documents_read:
                        continue
                    plan = revisit_planner.plan(
                        gap_query, document_id, store=self.store, snapshot_id=snapshot_id,
                        allowed_document_ids=allowed_document_ids,
                        already_read_block_ids=read_blocks[document_id],
                        token_budget=max(128, int(self.config.reader_input_tokens * 0.78)),
                        deadline=synthesis_cutoff - self.config.minimum_reader_window_s,
                        guard=guard, clock=time.perf_counter,
                    )
                    if plan is None:
                        continue
                    document = document_lookup[document_id]
                    packet = plan.packet(
                        run_id=run_id, document_title=document.title, source_path=document.source_path,
                        entity_scope=",".join(item.entity_id for item in targets) or "DOCUMENT",
                        sections_total=len(document_data[document_id][0]),
                    )
                    revised = merge_revisit_packet(
                        packet, tuple(queue.packets.get(document_id, ())),
                        already_read_block_ids=read_blocks[document_id],
                    )
                    queue.packets[document_id] = type(queue.pending)(revised)
                    queue.seen_packets.update(packet_key(item) for item in revised)
                    queue.discard(document_id)
                    queue.pending.appendleft(document_id)
                    packet_totals[document_id] = packet_counts[document_id] + len(revised)
                    previous_work_id = work_ids[document_id]
                    if (document_id not in state.completed_work_documents
                            and document_id not in state.terminal_work_documents):
                        self.store.finish_work(previous_work_id, state="deferred",
                                               result={"reason": "refined_source_window"})
                    state.completed_work_documents.discard(document_id)
                    state.terminal_work_documents.discard(document_id)
                    work_ids[document_id] = self.store.enqueue_work(
                        run_id, "read_document", "document_revision", document_id,
                        source_lane="adaptive_revisit", wave=wave, queue_ordinal=0,
                        parent_work_id=previous_work_id,
                        payload={"query_sha256": plan.query_sha256,
                                 "new_block_ids": [b.block_id for b in plan.blocks]},
                    )
                    state.debug.setdefault("block_revisits", []).append({
                        "wave": wave, "document_revision_id": document_id,
                        "query_sha256": plan.query_sha256, "planned_new_blocks": len(plan.blocks),
                        "estimated_tokens": plan.estimated_tokens, "elapsed_s": plan.elapsed_s,
                        "boundary_limited": plan.boundary_limited,
                    })
                    # One focused reopening per wave; the enclosing wave quota
                    # and global synthesis cutoff still govern actual reading.
                    break
            state.catalogue_unqueued_ids = tuple(item for item in document_lookup if item not in admitted)
            self._adaptive_event(run_id, "adaptive_discovery", {"wave": wave, "pages": discovery_records})
            return discovery_records

        for wave in range(max_waves):
            if synthesis_cutoff - time.perf_counter() < self.config.minimum_reader_window_s:
                state.research_window_exhausted = True
                state.debug["stop_reason"] = "research_deadline"
                break
            before = evidence_progress(reader_artifacts)
            wave_started = time.perf_counter()
            calls_before = len(calls)
            documents_before = len(state.documents_read)
            decision_context = context(wave)
            if trajectory:
                # Decision latency is part of the transition. Reuse the actual
                # next decision's state, not a previous timestamp approximation.
                trajectory[-1]["next_context"] = decision_context
            allowed_actions = {"balance_documents"}
            if wave and (cursors or state.catalogue_unqueued_ids):
                allowed_actions.add("broaden_search")
            if wave and any(q.casefold() not in attempted_queries for q in latest_queries):
                allowed_actions.add("focused_gap_search")
            default_action = ("broaden_search" if "broaden_search" in allowed_actions else
                              "focused_gap_search" if "focused_gap_search" in allowed_actions else "balance_documents")
            rankings = ()
            if self.config.strategy_learning:
                try:
                    proposed_rankings = self._rank_strategy_actions(
                        decision_context, principal=auth.user_id, authorization_scope=strategy_scope,
                        snapshot_id=snapshot_id,
                    )
                    if not isinstance(proposed_rankings, (list, tuple)):
                        raise ValueError("policy rankings must be a sequence")
                    safe_rankings = []
                    for item in proposed_rankings:
                        if not isinstance(item, Mapping) or item.get("action") not in {
                            "balance_documents", "broaden_search", "focused_gap_search"
                        }:
                            continue
                        score = item.get("score", 0.0)
                        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
                            continue
                        safe_rankings.append({
                            "action": item["action"], "score": float(score),
                            "checkpoint_id": str(item.get("checkpoint_id", ""))[:160],
                        })
                    rankings = tuple(safe_rankings)
                except Exception as exc:
                    warnings.append(f"strategy_ranking_failed:{type(exc).__name__}")
            selected_ranking = next((item for item in rankings if item.get("action") in allowed_actions), None)
            strategy_action = selected_ranking["action"] if selected_ranking else default_action
            quota = min(base_quota, max(1, self.config.read_wave_size)) if strategy_action == "balance_documents" else base_quota
            wave_debug = {"wave": wave, "packets_read": 0, "documents": [], "action": strategy_action,
                          "policy_checkpoint": selected_ranking.get("checkpoint_id", "") if selected_ranking else "",
                          "policy_rankings": jsonable(rankings), "decision_context": decision_context,
                          "available_actions": sorted(allowed_actions), "packet_quota": quota}

            def finish_step():
                blocks, cards = evidence_progress(reader_artifacts)
                new_calls = [c for c in calls[calls_before:] if c.status != "budget_denied"]
                def output_tokens(call):
                    value = call.metrics.get("eval_count", 0)
                    return max(0, int(value)) if isinstance(value, (int, float)) and math.isfinite(value) else 0
                tokens = sum(output_tokens(c) for c in new_calls)
                elapsed = max(0.0, time.perf_counter() - wave_started)
                trajectory.append({
                    "step": wave, "action": strategy_action, "context": decision_context,
                    "next_context": context(wave + 1),
                    "cost": {"elapsed_fraction": min(1.0, elapsed / research_budget),
                             "token_fraction": 0.0, "call_fraction": 0.0},
                    "usage": {"elapsed_s": elapsed, "output_tokens": tokens, "model_calls": len(new_calls)},
                    "evidence_gain": {"new_blocks": max(0, blocks - before[0]),
                                      "new_source_bound_cards": max(0, cards - before[1]),
                                      "new_documents": max(0, len(state.documents_read) - documents_before)},
                })
            self._emit(progress, started, "adaptive_wave", f"Research wave {wave + 1} of {max_waves}",
                       current={"wave": wave + 1, "maximum_waves": max_waves})
            if wave:
                wave_debug["discovery"] = discover(wave)
            # Keep time for feedback and later decisions. A large first queue
            # must not spend the entire research allowance before review.
            available = synthesis_cutoff - time.perf_counter()
            wave_cutoff = min(synthesis_cutoff, time.perf_counter() + max(
                self.config.minimum_reader_window_s, available / (max_waves - wave)
            ))
            wave_debug["reader_window_s"] = max(0.0, wave_cutoff - time.perf_counter())
            attempts = 0
            while queue and attempts < quota:
                if synthesis_cutoff - time.perf_counter() < self.config.minimum_reader_window_s:
                    state.research_window_exhausted = True
                    break
                if (wave_debug["packets_read"] and
                        wave_cutoff - time.perf_counter() < self.config.minimum_reader_window_s):
                    break
                document_id = queue.pop_document()
                guard((document_id,))
                document = document_lookup[document_id]
                attempts += 1
                if document_id not in queue.packets:
                    try:
                        focus = question + ("\nResearch gaps: " + "; ".join(latest_queries) if latest_queries else "")
                        packets = self._pack_document(
                            run_id, focus, document, targets, role_targets, query_surfaces, exact_blocks,
                            allowed_document_ids=allowed_document_ids, exhaustive=selected_mode == "exhaustive",
                            deadline=wave_cutoff, started=started, progress=progress,
                            calls=calls, warnings=warnings,
                        )
                        packets, packet_priority = prioritize_reader_packets(focus, packets)
                        wave_debug.setdefault("packet_priority", []).append({
                            "document_revision_id": document_id, **packet_priority,
                        })
                        sections = self.store.sections_for_document(document_id, allowed_document_revision_ids=allowed_document_ids)
                        blocks = self.store.blocks_for_document(document_id, allowed_document_revision_ids=allowed_document_ids)
                        reference_index = ReferenceIndex(blocks, sections) if self.config.reference_closure else None
                        document_data[document_id] = (sections, blocks, reference_index)
                        state.sections_total += len(sections)
                        packet_totals[document_id] = queue.install(document_id, packets)
                        if not packet_totals[document_id]:
                            raise ValueError("no distinct reader packets")
                    except (PermissionError, RuntimeError):
                        raise
                    except Exception as exc:
                        errors.append(f"packet_build_failed:{document.title}:{type(exc).__name__}")
                        queue.discard(document_id)
                        self.store.finish_work(work_ids[document_id], state="failed", error_code=type(exc).__name__)
                        state.terminal_work_documents.add(document_id)
                        continue
                packet = queue.take(document_id)
                if packet is None:
                    continue
                remaining = reader_window(
                    time.perf_counter(), global_cutoff=synthesis_cutoff, soft_cutoff=wave_cutoff,
                    minimum_reader_s=self.config.minimum_reader_window_s,
                    packets_read=wave_debug["packets_read"],
                )
                if not remaining:
                    queue.pending.appendleft(document_id)
                    queue.packets[document_id].appendleft(packet)
                    # take() already requeued documents with another packet.
                    queue.pending = type(queue.pending)(dict.fromkeys(queue.pending))
                    break
                guard((document_id,))
                self._emit(progress, started, "reading", f"Reading {document.title}",
                           current={"wave": wave + 1, "packet": packet_counts[document_id] + 1,
                                    "document_packets": packet_totals[document_id]})
                try:
                    model_packet, alias_map, original_blocks = self._alias_reader_packet(packet)
                    artifact = self.reader.read(question, model_packet, entity_registry=role_targets,
                                                timeout_s=min(self.config.request_timeout_s, remaining))
                    artifact = self._restore_reader_artifact(artifact, alias_map, original_blocks)
                    reference_index = document_data[document_id][2]
                    if reference_index is not None:
                        artifact = self._with_reference_context(artifact, reference_index)
                    artifact = qualify_revisit_artifact(artifact, packet)
                    calls.append(artifact.call)
                    reader_artifacts.append(artifact)
                    warnings.extend(artifact.report.warnings)
                    packet_counts[document_id] += 1
                    read_blocks[document_id].update(block.block_id for block in artifact.blocks)
                    state.documents_read.add(document_id)
                    state.seen_section_ids.update(block.section_id for block in artifact.blocks if block.section_id)
                    wave_debug["packets_read"] += 1
                    wave_debug["documents"].append(document_id)
                    executed_actions.add("balance_documents")
                except Exception as exc:
                    if isinstance(exc, BrokerCallError):
                        calls.append(exc.record)
                    errors.append(f"reader_call_failed:{document.title}:{type(exc).__name__}")
                    queue.discard(document_id)
                    self.store.finish_work(work_ids[document_id], state="deferred", error_code="reader_error")
                    state.terminal_work_documents.add(document_id)
                    continue
                sections, blocks, _ = document_data[document_id]
                if {block.block_id for block in blocks} <= read_blocks[document_id]:
                    state.documents_fully_read.add(document_id)
                if packet_counts[document_id] == packet_totals[document_id]:
                    state.completed_work_documents.add(document_id)
                    self.store.finish_work(work_ids[document_id], state="completed", result={
                        "packets_read": packet_counts[document_id], "packets_total": packet_totals[document_id],
                        "complete_document": document_id in state.documents_fully_read,
                    })
                    state.terminal_work_documents.add(document_id)
            after = evidence_progress(reader_artifacts)
            wave_debug.update({"new_blocks": after[0] - before[0], "new_claims": after[1] - before[1]})
            state.debug["waves"].append(wave_debug)
            self._adaptive_event(run_id, "adaptive_wave_completed", wave_debug)
            if state.research_window_exhausted:
                state.debug["stop_reason"] = "research_deadline"
                finish_step()
                break
            if wave == max_waves - 1:
                state.debug["stop_reason"] = "wave_limit" if queue or state.catalogue_unqueued_ids else "queue_exhausted"
                finish_step()
                break
            if wave and after == before:
                state.debug["stop_reason"] = "no_progress"
                finish_step()
                break
            latest_queries = focused_queries(
                item for artifact in reader_artifacts for item in artifact.report.unresolved
                if not any(record.data.get("reason") == "search_silence_is_not_evidence" and record.text == item
                           for record in artifact.report.records)
            )
            if reader_artifacts and self.config.claim_review:
                review_budget = min(self.config.evidence_review_timeout_s,
                                    synthesis_cutoff - time.perf_counter() - self.config.minimum_reader_window_s)
                if review_budget >= 2.0:
                    guard(state.documents_read)
                    self._emit(progress, started, "evidence_review", "Checking which parts of the question still need evidence")
                    try:
                        assessment = self.evidence_reviewer.assess(
                            question, reader_artifacts, entity_targets=role_targets, timeout_s=review_budget,
                        )
                        state.last_assessment = assessment
                        state.evidence_at_last_assessment = evidence_progress(reader_artifacts)
                        if assessment.call is not None:
                            calls.append(assessment.call)
                        warnings.extend(assessment.warnings)
                        latest_queries = focused_queries((*assessment.focused_queries, *latest_queries))
                        missing_count = len(assessment.missing_obligations)
                        scope_count = len(assessment.claim_scope_issues)
                        review_debug = {"missing_obligations": assessment.missing_obligations,
                                        "focused_queries": latest_queries, "scope_issues": jsonable(assessment.claim_scope_issues),
                                        "sufficient_advisory": assessment.sufficient,
                                        "obligation_details": jsonable(getattr(assessment, "obligation_details", ())),
                                        "reviewed_card_ids": getattr(assessment, "reviewed_card_ids", ()),
                                        "evidence_fingerprint": getattr(assessment, "evidence_fingerprint", ""),
                                        "judgment_status": getattr(assessment, "judgment_status", "advisory")}
                        wave_debug["assessment"] = review_debug
                        self._adaptive_event(run_id, "adaptive_evidence_review", review_debug)
                        if (selected_mode != "exhaustive" and assessment.sufficient
                                and not assessment.missing_obligations and not assessment.claim_scope_issues
                                and _has_direct_evidence(reader_artifacts)):
                            state.semantic_queue_stop = bool(queue)
                            state.debug["stop_reason"] = "review_sufficient_advisory"
                            finish_step()
                            break
                    except (PermissionError, RuntimeError):
                        raise
                    except Exception as exc:
                        warnings.append(f"adaptive_review_failed:{type(exc).__name__}")
            if not queue and not state.catalogue_unqueued_ids and not latest_queries:
                state.debug["stop_reason"] = "queue_exhausted"
                finish_step()
                break
            finish_step()
        for work_id in page_work_ids - completed_page_work_ids:
            self.store.finish_work(work_id, state="deferred", result={"reason": state.debug["stop_reason"]})
        call_allowance = max(max_waves * (base_quota + 3), sum(s["usage"]["model_calls"] for s in trajectory), 1)
        token_allowance = max(self.config.context_tokens * call_allowance,
                              sum(s["usage"]["output_tokens"] for s in trajectory), 1)
        elapsed_allowance = max(research_budget, sum(s["usage"]["elapsed_s"] for s in trajectory), 1.0)
        for step in trajectory:
            step["cost"] = {"elapsed_fraction": step["usage"]["elapsed_s"] / elapsed_allowance,
                            "token_fraction": step["usage"]["output_tokens"] / token_allowance,
                            "call_fraction": step["usage"]["model_calls"] / call_allowance}
        state.debug.update({"distinct_packets": sum(packet_counts.values()),
                            "trajectory": trajectory,
                            "documents_read": len(state.documents_read),
                            "executed_actions": sorted(executed_actions),
                            "literal_query_calls": literal_queries_used,
                            "remaining_catalogue_documents": len(state.catalogue_unqueued_ids)})
        self._adaptive_event(run_id, "adaptive_research_stopped", state.debug)
        return state

    def _ask(
        self,
        question: str,
        *,
        session: Session | None,
        mode: str,
        progress: ProgressCallback | None,
        auth: PrincipalSnapshot,
    ) -> Answer:
        started = time.perf_counter()
        deadline = started + self.config.total_deadline_s
        # Research must stop early enough to leave the configured synthesis
        # window *and* the separate deterministic finalization window intact.
        synthesis_cutoff = (
            deadline
            - self.config.synthesis_reserve_s
            - self.config.finalize_reserve_s
        )
        selected_mode = _mode(question, mode)
        effective_question, prior_document_ids, prior_questions = _effective_question(question, session)
        inherited = _inherited_targets(prior_questions)
        resolution = self.registry.resolve(question, inherited=inherited)
        current_entity_surfaces = _explicit_entity_surfaces(question)
        targets = _question_targets(current_entity_surfaces, resolution)
        role_targets = self.registry.expand_prompt_partition(targets)
        query_surfaces = _unique((
            *exact_surfaces(question),
            *(item.surface for item in targets),
        ))
        forced_surfaces = _forced_surfaces(question, targets)
        snapshot = self.store.snapshot()
        searchable_ids = set(
            self.access.allowed_document_revision_ids(auth, "document.search")
        )
        citable_ids = set(
            self.access.allowed_document_revision_ids(auth, "document.cite")
        )
        allowed_document_ids = tuple(
            item
            for item in self.access.allowed_document_revision_ids(auth, "document.read")
            if item in searchable_ids and item in citable_ids
        )
        allowed_document_set = frozenset(allowed_document_ids)
        prior_document_ids = tuple(
            item for item in prior_document_ids if item in allowed_document_set
        )
        run_id = self.store.create_run(
            question,
            effective_question=effective_question,
            session_id=f"session_{id(session):x}" if session is not None else "",
            snapshot_id=snapshot.snapshot_id,
            authorization_scope_hash=self.access.authorization_scope_hash(auth),
            principal_id=auth.user_id,
            authorization_revision=auth.revision,
            completeness_mode=selected_mode,
            config={
                "model": self.config.model,
                "context_tokens": self.config.context_tokens,
                "deadline_s": self.config.total_deadline_s,
                "minimum_screen_window_s": self.config.minimum_screen_window_s,
                "minimum_reader_window_s": self.config.minimum_reader_window_s,
            },
        )
        calls: list[ModelCallRecord] = []
        warnings: list[str] = []
        errors: list[str] = []
        screening: list[ScreenDecision] = []
        reader_artifacts: list[ReaderArtifact] = []
        exact_hits_by_doc: dict[str, list[str]] = defaultdict(list)
        self._emit(progress, started, "authorizing", "Pinning the authorized corpus snapshot")
        self.store.set_run_state(run_id, "authorizing")
        self._event(run_id, "snapshot_authorized", {"snapshot_id": snapshot.snapshot_id})
        documents = self.store.documents(
            allowed_document_revision_ids=allowed_document_ids
        )
        document_lookup = {item.document_revision_id: item for item in documents}

        self._emit(progress, started, "exact_search", "Enumerating exact names and phrases")
        self.store.set_run_state(run_id, "exact_search")
        for surface in forced_surfaces:
            try:
                for document_id in self.store.exact_document_matches(
                    surface,
                    allowed_document_revision_ids=allowed_document_ids,
                ):
                    exact_hits_by_doc[document_id].append(surface)
            except Exception as exc:
                warnings.append(f"exact_enumeration_failed:{type(exc).__name__}")
        try:
            exact_blocks = self.store.exact_block_matches(
                forced_surfaces,
                allowed_document_revision_ids=allowed_document_ids,
            )
        except Exception as exc:
            exact_blocks = ()
            warnings.append(f"exact_block_enumeration_failed:{type(exc).__name__}")

        exact_occurrences_by_doc: dict[str, int] = defaultdict(int)
        occurrence_patterns = tuple(
            re.compile(rf"(?iu)(?<!\w){re.escape(surface)}(?!\w)")
            for surface in forced_surfaces
            if surface
        )
        for block in exact_blocks:
            exact_occurrences_by_doc[block.document_revision_id] += sum(
                len(pattern.findall(block.text)) for pattern in occurrence_patterns
            )

        corpus_order = {item.document_revision_id: index for index, item in enumerate(documents)}
        exact_specificities = _exact_match_specificities(exact_hits_by_doc, allowed_document_set)
        navigation_scores = {}
        navigation_stats = {"queries": 0, "terms_used": 0, "query_errors": 0}
        if self.config.adaptive_research:
            from .navigation_priority import navigation_priorities
            navigation_scores, navigation_stats = navigation_priorities(
                effective_question, self.store, allowed_document_ids,
            )
        candidate_priorities = dict(exact_specificities)
        for document_id, score in navigation_scores.items():
            candidate_priorities[document_id] = candidate_priorities.get(document_id, 0.0) + score
        self._event(run_id, "navigation_priority_computed", navigation_stats)

        discovery_state: dict[str, Any] = {
            "enabled": False,
            "authorized_documents": len(documents),
        }
        catalogue_unqueued_ids: tuple[str, ...] = ()
        if len(documents) <= self.config.manifest_full_scan_threshold:
            manifests = self.store.manifests(
                allowed_document_revision_ids=allowed_document_ids
            )
        else:
            discovery_query = " ".join(query_surfaces[:32]) or question
            try:
                page = self.store.discover_documents(
                    discovery_query,
                    page_size=self.config.discovery_page_size,
                    match_mode="all",
                    expected_snapshot_id=snapshot.snapshot_id,
                    allowed_document_revision_ids=allowed_document_ids,
                )
                if not page.document_revision_ids:
                    page = self.store.discover_documents(
                        discovery_query,
                        page_size=self.config.discovery_page_size,
                        match_mode="any",
                        expected_snapshot_id=snapshot.snapshot_id,
                        allowed_document_revision_ids=allowed_document_ids,
                    )
                candidate_capacity = self.config.discovery_page_size * 2
                candidate_ids = _admit_document_candidates(
                    prior_document_ids, page.document_revision_ids, candidate_priorities,
                    allowed_document_ids=allowed_document_set,
                    corpus_order=corpus_order, capacity=candidate_capacity,
                )
                if not candidate_ids:
                    candidate_ids = tuple(document_lookup)[: self.config.discovery_page_size]
                manifests = self.store.manifests(
                    candidate_ids,
                    allowed_document_revision_ids=allowed_document_ids,
                )
                selected_ids = {item.document_revision_id for item in manifests}
                catalogue_unqueued_ids = tuple(
                    item for item in document_lookup if item not in selected_ids
                )
                discovery_state = {
                    "enabled": True,
                    "query": discovery_query,
                    "match_mode": page.match_mode,
                    "matched_documents": page.total,
                    "page_offset": page.offset,
                    "page_returned": page.returned,
                    "next_offset": page.next_offset,
                    "exhausted": page.exhausted,
                    "screen_candidates": len(manifests),
                    "authorized_documents": len(documents),
                }
                if page.next_offset is not None:
                    discovery_state["work_id"] = self.store.enqueue_work(
                        run_id,
                        "discover_documents",
                        "catalogue_page",
                        str(page.next_offset),
                        source_lane="lexical_navigation",
                        wave=1,
                        queue_ordinal=0,
                        payload={
                            "query": discovery_query,
                            "match_mode": page.match_mode,
                            "snapshot_id": page.snapshot_id,
                            "offset": page.next_offset,
                        },
                    )
            except Exception as exc:
                warnings.append(f"catalogue_discovery_failed:{type(exc).__name__}")
                fallback_ids = tuple(document_lookup)[: self.config.discovery_page_size]
                manifests = self.store.manifests(
                    fallback_ids,
                    allowed_document_revision_ids=allowed_document_ids,
                )
                selected_ids = {item.document_revision_id for item in manifests}
                catalogue_unqueued_ids = tuple(
                    item for item in document_lookup if item not in selected_ids
                )
                discovery_state = {
                    "enabled": True,
                    "error": type(exc).__name__,
                    "screen_candidates": len(manifests),
                    "authorized_documents": len(documents),
                }
        if self.config.adaptive_research:
            manifests = _order_screen_manifests(
                manifests, prior_document_ids, candidate_priorities, corpus_order,
            )
        annotated_manifests = tuple(
            replace(
                manifest,
                exact_surfaces=_unique(exact_hits_by_doc.get(manifest.document_revision_id, ())),
            )
            for manifest in manifests
        )
        self._event(
            run_id,
            "exact_enumeration_completed",
            {
                "surfaces": list(forced_surfaces),
                "documents": list(exact_hits_by_doc),
                "blocks": len(exact_blocks),
            },
        )
        self._event(run_id, "catalogue_discovery_completed", discovery_state)

        self.store.set_run_state(run_id, "screening")
        screening_values, manifests_presented = self._screen_batches(
            effective_question,
            annotated_manifests,
            role_targets,
            deadline=synthesis_cutoff - self.config.minimum_reader_window_s,
            started=started,
            progress=progress,
            calls=calls,
            warnings=warnings,
            stage_label="Screening document manifests",
        )
        screening.extend(screening_values)
        self._event(run_id, "screening_completed", {"presented": manifests_presented})

        decision_by_doc = {item.document_revision_id: item for item in screening}

        def priority(document_id: str) -> tuple[float, float, int, int]:
            decision = decision_by_doc.get(document_id)
            semantic = {"read": 0.0, "maybe": 2.0, "unlikely": 4.0}.get(
                decision.decision if decision else "maybe", 2.0
            )
            if document_id in prior_document_ids:
                semantic -= 0.75
            hits = exact_hits_by_doc.get(document_id, ())
            specificity = candidate_priorities.get(document_id, 0.0)
            if hits:
                semantic -= 0.5
            return (
                semantic,
                -specificity,
                -exact_occurrences_by_doc.get(document_id, 0),
                corpus_order.get(document_id, 10**9),
            )

        candidate_document_ids = {
            manifest.document_revision_id for manifest in annotated_manifests
        }
        queued_ids = sorted(candidate_document_ids, key=priority)
        primary_ids = [
            document_id
            for document_id in queued_ids
            if (
                decision_by_doc.get(document_id, ScreenDecision("", document_id, "maybe")).decision == "read"
                or document_id in prior_document_ids
            )
        ]
        secondary_ids = [
            document_id
            for document_id in queued_ids
            if document_id not in primary_ids
            and decision_by_doc.get(document_id, ScreenDecision("", document_id, "maybe")).decision == "maybe"
        ]
        unlikely_ids = [item for item in queued_ids if item not in primary_ids and item not in secondary_ids]
        screened_out_all = _all_candidates_screened_unlikely(
            queued_ids,
            decision_by_doc,
            exact_ids=exact_hits_by_doc,
            prior_ids=prior_document_ids,
        )
        read_queue = [*primary_ids, *secondary_ids]
        if selected_mode == "exhaustive":
            read_queue.extend(unlikely_ids)
        if not read_queue and not screened_out_all:
            read_queue = list(queued_ids)

        strategy_action = ""
        strategy_debug: dict[str, Any] = {"enabled": self.config.strategy_learning, "recommended": [], "executed": ""}
        strategy_features = research_features(
            effective_question, mode=selected_mode, document_count=len(documents),
            full_scan_threshold=self.config.manifest_full_scan_threshold,
            has_exact_surfaces=bool(forced_surfaces),
        )
        strategy_scope = self.access.authorization_scope_hash(auth)

        work_ids: dict[str, str] = {}
        for index, document_id in enumerate(queued_ids):
            decision = decision_by_doc.get(document_id)
            lane = "exact" if document_id in exact_hits_by_doc else "semantic"
            work_ids[document_id] = self.store.enqueue_work(
                run_id,
                "read_document",
                "document_revision",
                document_id,
                source_lane=lane,
                wave=index // self.config.read_wave_size,
                queue_ordinal=index,
                payload={"decision": decision.decision if decision else "maybe"},
            )

        self.store.set_run_state(run_id, "reading")
        documents_read: set[str] = set()
        documents_fully_read: set[str] = set()
        completed_work_documents: set[str] = set()
        terminal_work_documents: set[str] = set()
        seen_section_ids: set[str] = set()
        sections_total = 0
        research_window_exhausted = False
        semantic_queue_stop = False
        secondary_probe_documents = 0
        adaptive_debug = {"enabled": False}
        synthesis_guidance = None
        if self.config.adaptive_research:
            adaptive = self._adaptive_read(
                run_id=run_id, question=effective_question, selected_mode=selected_mode,
                auth=auth, snapshot_id=snapshot.snapshot_id, document_lookup=document_lookup,
                allowed_document_ids=allowed_document_ids, targets=targets, role_targets=role_targets,
                query_surfaces=query_surfaces, exact_blocks=exact_blocks, read_queue=read_queue,
                queued_ids=queued_ids, work_ids=work_ids, annotated_manifests=annotated_manifests,
                discovery_state=discovery_state, catalogue_unqueued_ids=catalogue_unqueued_ids,
                screening=screening, reader_artifacts=reader_artifacts, calls=calls,
                warnings=warnings, errors=errors, synthesis_cutoff=synthesis_cutoff,
                started=started, progress=progress, strategy_action=strategy_action,
                strategy_features=strategy_features, strategy_scope=strategy_scope,
            )
            documents_read = adaptive.documents_read
            documents_fully_read = adaptive.documents_fully_read
            completed_work_documents = adaptive.completed_work_documents
            terminal_work_documents = adaptive.terminal_work_documents
            seen_section_ids = adaptive.seen_section_ids
            sections_total = adaptive.sections_total
            research_window_exhausted = adaptive.research_window_exhausted
            semantic_queue_stop = adaptive.semantic_queue_stop
            manifests_presented += adaptive.manifests_presented
            annotated_manifests = (*annotated_manifests, *adaptive.added_manifests)
            catalogue_unqueued_ids = adaptive.catalogue_unqueued_ids
            adaptive_debug = adaptive.debug
            if self.config.claim_review and adaptive.last_assessment is not None:
                try:
                    synthesis_guidance = build_synthesis_guidance(
                        effective_question, adaptive.last_assessment, reader_artifacts,
                        additional_evidence_since_review=(
                            evidence_progress(reader_artifacts) != adaptive.evidence_at_last_assessment
                        ),
                    )
                    if synthesis_guidance is not None:
                        adaptive_debug["synthesis_guidance"] = jsonable(synthesis_guidance)
                except Exception as exc:
                    warnings.append(f"synthesis_guidance_skipped:{type(exc).__name__}")
            screened_out_all = screened_out_all and not reader_artifacts
        else:
            for queue_index, document_id in enumerate(read_queue):
                if (
                    synthesis_cutoff - time.perf_counter()
                    < self.config.minimum_reader_window_s
                ):
                    research_window_exhausted = True
                    break
                is_secondary = document_id in secondary_ids
                direct_before_document = _has_direct_evidence(reader_artifacts)
                if _should_stop_secondary_queue(
                    selected_mode=selected_mode,
                    is_secondary=is_secondary,
                    artifacts=reader_artifacts,
                    probes_completed=secondary_probe_documents,
                    probe_limit=self.config.contradiction_probe_documents,
                ):
                    # Context, counter-evidence, and calculation operands never
                    # trigger early stopping. After direct evidence, retain a
                    # bounded secondary probe for contradictions before deferring
                    # the remaining lower-confidence documents.
                    semantic_queue_stop = True
                    break
                document = document_lookup[document_id]
                self._emit(
                    progress,
                    started,
                    "reading",
                    f"Reading {document.title}",
                    coverage={
                        "authorized_documents": len(documents),
                        "manifests_screened": manifests_presented,
                        "documents_read": len(documents_read),
                        "documents_queued": len(read_queue),
                    },
                    current={"title": document.title, "index": queue_index + 1, "total": len(read_queue)},
                )
                try:
                    packets = self._pack_document(
                        run_id,
                        effective_question,
                        document,
                        targets,
                        role_targets,
                        query_surfaces,
                        exact_blocks,
                        allowed_document_ids=allowed_document_ids,
                        exhaustive=selected_mode == "exhaustive",
                        deadline=synthesis_cutoff,
                        started=started,
                        progress=progress,
                        calls=calls,
                        warnings=warnings,
                    )
                except Exception as exc:
                    errors.append(f"packet_build_failed:{document.title}:{type(exc).__name__}")
                    self.store.finish_work(work_ids[document_id], state="failed", error_code=type(exc).__name__)
                    terminal_work_documents.add(document_id)
                    continue

                document_sections = self.store.sections_for_document(
                    document_id,
                    allowed_document_revision_ids=allowed_document_ids,
                )
                document_blocks = self.store.blocks_for_document(
                    document_id,
                    allowed_document_revision_ids=allowed_document_ids,
                )
                reference_index = (ReferenceIndex(document_blocks, document_sections)
                                   if self.config.reference_closure else None)
                sections_total += len(document_sections)
                if not packets:
                    errors.append(f"packet_build_empty:{document.title}")
                    self.store.finish_work(
                        work_ids[document_id],
                        state="failed",
                        error_code="no_reader_packets",
                    )
                    terminal_work_documents.add(document_id)
                    continue

                planned_block_ids = {
                    block.block_id for packet in packets for block in packet.blocks
                }
                source_block_ids = {block.block_id for block in document_blocks}
                planned_complete_document = bool(source_block_ids) and (
                    planned_block_ids == source_block_ids
                )
                document_completed = True
                document_packets_read = 0
                for packet in packets:
                    remaining = synthesis_cutoff - time.perf_counter()
                    if remaining < self.config.minimum_reader_window_s:
                        document_completed = False
                        research_window_exhausted = True
                        break
                    try:
                        model_packet, alias_map, original_blocks = self._alias_reader_packet(packet)
                        artifact = self.reader.read(
                            effective_question,
                            model_packet,
                            entity_registry=role_targets,
                            timeout_s=min(self.config.request_timeout_s, remaining),
                        )
                        artifact = self._restore_reader_artifact(
                            artifact, alias_map, original_blocks
                        )
                        if reference_index is not None:
                            artifact = self._with_reference_context(artifact, reference_index)
                        calls.append(artifact.call)
                        reader_artifacts.append(artifact)
                        warnings.extend(artifact.report.warnings)
                        document_packets_read += 1
                        seen_section_ids.update(
                            block.section_id
                            for block in artifact.blocks
                            if block.section_id
                        )
                    except BrokerCallError as exc:
                        calls.append(exc.record)
                        errors.append(f"reader_call_failed:{document.title}:{type(exc.__cause__).__name__ if exc.__cause__ else type(exc).__name__}")
                        document_completed = False
                        if time.perf_counter() >= synthesis_cutoff:
                            research_window_exhausted = True
                        break
                    except Exception as exc:
                        errors.append(f"reader_call_failed:{document.title}:{type(exc).__name__}")
                        document_completed = False
                        if time.perf_counter() >= synthesis_cutoff:
                            research_window_exhausted = True
                        break

                if document_packets_read:
                    documents_read.add(document_id)
                    if is_secondary and direct_before_document:
                        secondary_probe_documents += 1
                work_completed = document_completed and document_packets_read == len(packets)
                if work_completed:
                    completed_work_documents.add(document_id)
                if work_completed and planned_complete_document:
                    documents_fully_read.add(document_id)
                    seen_section_ids.update(section.section_id for section in document_sections)
                self.store.finish_work(
                    work_ids[document_id],
                    state="completed" if work_completed else "deferred",
                    result={
                        "packets_read": document_packets_read,
                        "packets_total": len(packets),
                        "complete_document": document_id in documents_fully_read,
                    },
                    error_code="" if work_completed else "deadline_or_reader_error",
                )
                terminal_work_documents.add(document_id)

        deferred_work_ids = tuple(
            item for item in queued_ids if item not in completed_work_documents
        )
        deferred_ids = tuple(
            item for item in read_queue if item not in completed_work_documents
        )
        for document_id in deferred_work_ids:
            if document_id in terminal_work_documents:
                continue
            try:
                self.store.finish_work(
                    work_ids[document_id],
                    state="deferred",
                    result={"reason": "not_processed_before_synthesis"},
                )
            except Exception:
                pass

        self.store.set_run_state(run_id, "synthesizing")
        self._emit(progress, started, "synthesizing", "Writing from the documents actually read")
        synthesis_answer = ""
        synthesis_cards: tuple[EvidenceCard, ...] = ()
        synthesis_raw = ""
        claim_review_debug: dict[str, Any] = {"enabled": self.config.claim_review, "attempted": False, "changed": False}
        no_source_evidence = bool(reader_artifacts) and not _has_source_backed_cards(reader_artifacts)
        no_packed_source_evidence = False
        synthesis_guard_debug = {
            "applied": no_source_evidence,
            "reason": "no_source_backed_reader_cards" if no_source_evidence else "",
            "reader_artifacts": len(reader_artifacts),
            "retained_cards": sum(len(item.report.cards) for item in reader_artifacts),
        }
        if no_source_evidence:
            # An answered report may survive even when every proposed card was
            # rejected. Neither its draft nor DOCUMENT_REPORT metadata is a
            # substitute for exact evidence. Preserve all research diagnostics.
            synthesis_answer = _no_source_evidence_text()
            warnings.append("synthesis_skipped_no_source_evidence")
        if (reader_artifacts and not no_source_evidence
                and deadline - time.perf_counter() > self.config.finalize_reserve_s + 1.0):
            remaining = deadline - self.config.finalize_reserve_s - time.perf_counter()
            if self.config.claim_review:
                # Repair has its own slice inside synthesis time; the existing
                # deterministic finalization reserve is never borrowed.
                remaining -= min(self.config.claim_review_reserve_s, remaining / 3.0)
            profile = load_profile(self.config)
            style = str(profile.get("answer_style", "Answer directly and naturally."))
            custom = profile.get("custom_instructions", ())
            if isinstance(custom, list) and custom:
                style += " " + " ".join(str(item) for item in custom[:12])
            try:
                artifact = self.synthesizer.synthesize(
                    effective_question,
                    reader_artifacts,
                    entity_registry=role_targets,
                    style_instruction=style,
                    timeout_s=min(self.config.request_timeout_s, remaining),
                    **({"evidence_guidance": synthesis_guidance} if synthesis_guidance is not None else {}),
                )
                if artifact.call is not None:
                    calls.append(artifact.call)
                warnings.extend(artifact.warnings)
                synthesis_answer = artifact.answer
                synthesis_cards = artifact.cards
                synthesis_raw = artifact.raw_output
                if not synthesis_cards and "synthesis_skipped_no_packed_source_evidence" in artifact.warnings:
                    # The role's atomic packer can reject all otherwise valid
                    # cards. Its deterministic result makes no model call and
                    # must not fall through to ungrounded report prose.
                    no_source_evidence = True
                    no_packed_source_evidence = True
                    synthesis_answer = _no_source_evidence_text(packing_limited=True)
                    synthesis_guard_debug.update({
                        "applied": True, "reason": "no_packed_source_evidence"})
                    warnings.append("synthesis_skipped_no_source_evidence")
            except BrokerCallError as exc:
                calls.append(exc.record)
                errors.append(f"synthesis_call_failed:{type(exc.__cause__).__name__ if exc.__cause__ else type(exc).__name__}")
            except Exception as exc:
                errors.append(f"synthesis_call_failed:{type(exc).__name__}")

        if not synthesis_answer:
            synthesis_answer, synthesis_cards = self._fallback_answer(reader_artifacts)
            if synthesis_answer:
                warnings.append("synthesis_fallback_used")

        repair_budget = min(self.config.claim_review_reserve_s,
                            deadline - self.config.finalize_reserve_s - time.perf_counter())
        if self.config.claim_review and synthesis_answer and synthesis_cards and repair_budget >= 2.0:
            self._reauthorize_answer_dependencies(auth, document_revision_ids=documents_read)
            if self.config.adaptive_research and self.store.snapshot().snapshot_id != snapshot.snapshot_id:
                raise RuntimeError("active corpus snapshot changed before claim review")
            self._emit(progress, started, "claim_review", "Checking the draft against its cited evidence")
            claim_review_debug["attempted"] = True
            try:
                repair = self.evidence_reviewer.repair(
                    effective_question, synthesis_answer, reader_artifacts,
                    allowed_cards=synthesis_cards, entity_targets=role_targets, timeout_s=repair_budget,
                )
                if repair.call is not None:
                    calls.append(repair.call)
                warnings.extend(repair.warnings)
                synthesis_answer = repair.answer
                claim_review_debug["changed"] = repair.changed
                claim_review_debug["warnings"] = repair.warnings
                claim_review_debug["repaired_units"] = getattr(repair, "repaired_units", ())
                claim_review_debug["evidence_fingerprint"] = getattr(repair, "evidence_fingerprint", "")
                claim_review_debug["judgment_status"] = getattr(repair, "judgment_status", "advisory")
            except Exception as exc:
                warnings.append(f"claim_review_failed:{type(exc).__name__}")
        elif self.config.claim_review:
            claim_review_debug["skip_reason"] = (
                "no_source_evidence" if no_source_evidence else "no_cited_draft_or_review_budget")
        self._adaptive_event(run_id, "claim_review_completed", claim_review_debug)

        self.store.set_run_state(run_id, "binding_citations")
        self._emit(progress, started, "binding_citations", "Binding exact source spans")
        block_lookup = {
            block.block_id: block
            for artifact in reader_artifacts
            for block in artifact.blocks
        }
        normalization = normalize_evidence_markers(synthesis_answer, tuple(card.card_id for card in synthesis_cards))
        synthesis_answer = normalization.text
        warnings.extend(normalization.warnings)
        synthesis_answer, qualification_warnings = qualify_reference_units(synthesis_answer, synthesis_cards, source_blocks=block_lookup)
        warnings.extend(qualification_warnings)
        if not no_source_evidence and not synthesis_cards and any(a.report.reference_issues for a in reader_artifacts):
            synthesis_answer = "A referenced source condition is unresolved; I cannot give a definite conclusion."
        rendered_text, citations, citation_warnings = self._bind_citations(
            synthesis_answer,
            synthesis_cards,
            block_lookup,
            document_lookup,
        )
        warnings.extend(citation_warnings)
        # Re-check after binding. A model can invent an E-card marker that is
        # correctly removed above; that must not make the now-uncited prose
        # look fully sourced merely because another paragraph cited a valid
        # card. Keep the useful prose, but label the result partial.
        bound_uncited_units = _uncited_answer_units(rendered_text)
        if bound_uncited_units:
            warnings.append(f"uncited_bound_answer_units:{bound_uncited_units}")

        # Do not infer a research timeout merely because synthesis naturally
        # runs after its cutoff. Only an observed research-window stop (or the
        # actual hard deadline) is a deadline event.
        hard_deadline_reached = time.perf_counter() >= deadline
        reached_deadline = research_window_exhausted or hard_deadline_reached
        evidence_count = sum(len(artifact.report.cards) for artifact in reader_artifacts)
        extraction_gaps = _read_extraction_gaps(documents, documents_read)
        incomplete: list[str] = []
        cited_cards = set(_CARD_MARKER.findall(synthesis_answer))
        reference_incomplete = (
            any(card.reference_issues and card.card_id in cited_cards for card in synthesis_cards)
            or any(item.startswith("reference_synthesis_budget:") for item in warnings)
            or (not synthesis_cards and any(a.report.reference_issues for a in reader_artifacts))
        )
        if reference_incomplete:
            incomplete.append("referenced source conditions were unresolved or omitted by a context budget")
        if deferred_ids:
            incomplete.append("documents remain in the research queue")
        if semantic_queue_stop:
            incomplete.append("semantic screening left lower-confidence documents queued")
        if catalogue_unqueued_ids:
            incomplete.append("additional authorized catalogue documents were not screened in this page")
        if research_window_exhausted:
            incomplete.append("interactive reading window ended to reserve synthesis")
        if hard_deadline_reached:
            incomplete.append("interactive deadline reached")
        if errors:
            incomplete.append("one or more stages reported a technical error")
        exhaustive_complete = (
            selected_mode == "exhaustive"
            and not deferred_ids
            and not catalogue_unqueued_ids
            and len(documents_fully_read) == len(documents)
            and not reference_incomplete
        )
        research_errors = tuple(
            item for item in errors if not item.startswith("synthesis_call_failed:")
        )
        research_complete = _research_complete(
            selected_mode=selected_mode,
            exhaustive_complete=exhaustive_complete,
            planned_document_ids=read_queue,
            completed_document_ids=completed_work_documents,
            catalogue_unqueued_ids=catalogue_unqueued_ids,
            semantic_queue_stop=semantic_queue_stop,
            research_window_exhausted=research_window_exhausted,
            errors=research_errors,
        )
        research_complete = research_complete and not reference_incomplete
        remaining_documents = len(deferred_ids) + len(catalogue_unqueued_ids)
        deferred_labels = [document_lookup[item].title for item in deferred_ids]
        if catalogue_unqueued_ids:
            deferred_labels.append(
                f"{len(catalogue_unqueued_ids)} additional authorized catalogue document(s)"
            )
        coverage = Coverage(
            snapshot_id=snapshot.snapshot_id,
            mode=selected_mode,
            authorized_documents=len(documents),
            manifests_screened=manifests_presented,
            documents_queued=len(read_queue),
            documents_read=len(documents_read),
            documents_fully_read=len(documents_fully_read),
            documents_remaining=remaining_documents,
            sections_seen=len(seen_section_ids),
            sections_total_for_opened_documents=sections_total,
            exact_match_documents=len(exact_hits_by_doc),
            evidence_cards=evidence_count,
            document_errors=tuple(errors),
            deferred_documents=tuple(deferred_labels),
            extraction_gaps=extraction_gaps,
            deadline_reached=reached_deadline,
            exhaustive=exhaustive_complete,
            complete=research_complete,
            provisional=not research_complete,
            incomplete_reasons=tuple(_unique(incomplete)),
            reference_checks_enabled=self.config.reference_closure,
            reference_checked_cards=sum(c.reference_checked for a in reader_artifacts for c in a.report.cards),
            reference_context_blocks=len({bid for card in synthesis_cards for bid in card.context_block_ids}),
            reference_incomplete=reference_incomplete,
        )

        explicit_not_found = bool(reader_artifacts) and all(
            artifact.report.answerability in {"not_found", "no", "none"}
            and not artifact.report.cards
            for artifact in reader_artifacts
        )
        if no_source_evidence:
            answer_status = "partial"
            rendered_text = _no_source_evidence_text(packing_limited=no_packed_source_evidence)
            citations = ()
            # This deterministic limitation makes no source claim requiring a
            # citation; do not mislabel it as ungrounded model-generated prose.
            warnings[:] = [item for item in warnings if not item.startswith(
                ("uncited_answer_units:", "uncited_bound_answer_units:"))]
        elif reference_incomplete and not synthesis_cards:
            answer_status = "partial"
            rendered_text = "A referenced source condition is unresolved; I cannot give a definite conclusion."
            citations = ()
        elif explicit_not_found and not errors:
            answer_status = "not_found"
            rendered_text = _bounded_not_found_text(content_was_read=True)
            citations = ()
            warnings[:] = [
                item
                for item in warnings
                if not item.startswith(("uncited_answer_units:", "uncited_bound_answer_units:"))
            ]
        elif (screened_out_all or not documents) and not reader_artifacts and not errors:
            answer_status = "not_found"
            rendered_text = _bounded_not_found_text(
                content_was_read=False,
                manifests_were_screened=manifests_presented > 0,
            )
            citations = ()
            warnings[:] = [
                item
                for item in warnings
                if not item.startswith(("uncited_answer_units:", "uncited_bound_answer_units:"))
            ]
        elif rendered_text:
            answer_status = (
                "partial"
                if (
                    (selected_mode == "exhaustive" and not exhaustive_complete)
                    or not citations
                    or bound_uncited_units
                    or reference_incomplete
                )
                else "answer"
            )
            if not citations:
                warnings.append("answer_has_no_bound_citations")
        else:
            answer_status = "error"
            rendered_text = (
                "The local research run did not produce a usable answer. "
                "The saved trace separates model, parsing, and document-read failures."
            )

        finalized_document_dependencies = {
            *documents_read,
            *(
                item.document_revision_id
                for item in (annotated_manifests if self.config.adaptive_research
                             else annotated_manifests[:manifests_presented])
            ),
        }
        try:
            if self.config.adaptive_research and self.store.snapshot().snapshot_id != snapshot.snapshot_id:
                raise RuntimeError("active corpus snapshot changed before answer finalization")
            self._reauthorize_answer_dependencies(
                auth,
                document_revision_ids=finalized_document_dependencies,
                sources=citations,
            )
        except PermissionError:
            try:
                self.store.set_run_state(run_id, "access_changed", finalized=True)
            except Exception:
                pass
            raise

        if self.config.strategy_learning and self.config.adaptive_research:
            try:
                from .strategy_learning import StrategyStore
                executed_actions = adaptive_debug.get("executed_actions", ())
                strategy_debug["executed_actions"] = executed_actions
                epoch = current_runtime_compatibility()
                strategy_debug["runtime_epoch_sha256"] = epoch["epoch_sha256"] if epoch is not None else None
                strategy_debug["runtime_compatibility_available"] = epoch is not None
                strategy_debug["executed"] = strategy_action if strategy_action in executed_actions else ""
                outcome = ("deadline" if research_window_exhausted else "error" if errors else
                           "new_evidence" if evidence_count else "no_progress")
                gaps = ("incomplete_coverage",) if remaining_documents else ()
                with StrategyStore(self.config.workspace_dir / "strategy_learning.sqlite3",
                                   runtime_compatibility=current_runtime_compatibility()) as memory:
                    trajectory = adaptive_debug.get("trajectory", ())
                    if trajectory:
                        episode = memory.record_episode(
                            principal=auth.user_id, authorization_scope=strategy_scope,
                            snapshot_id=snapshot.snapshot_id, trace_id=run_id,
                            question_sha256=hashlib.sha256(question.encode("utf-8")).hexdigest(),
                            steps=trajectory, outcome_code=outcome,
                        )
                        strategy_debug["episode_id"] = episode["id"]
                        strategy_debug["episode_verification"] = "unverified"
                    candidate_ids = []
                    for action in executed_actions:
                        candidate = memory.propose(
                            action, principal=auth.user_id, authorization_scope=strategy_scope,
                            features=strategy_features, gap_kinds=(),
                        )
                        memory.observe(candidate["id"], trace_id=run_id,
                                       question_sha256=hashlib.sha256(question.encode("utf-8")).hexdigest(),
                                       outcome_code=outcome, gaps=gaps)
                        candidate_ids.append(candidate["id"])
                strategy_debug.update({"observed_candidates": candidate_ids, "process_outcome": outcome,
                                       "automatic_promotion": False})
            except Exception as exc:
                warnings.append(f"strategy_observation_failed:{type(exc).__name__}")

        elapsed_before_trace = time.perf_counter() - started
        answer = Answer(
            status=answer_status,  # type: ignore[arg-type]
            text=rendered_text,
            sources=citations,
            warnings=_public_warnings(warnings, errors, coverage),
            timings={"answer_ready_s": elapsed_before_trace},
            coverage=coverage,
            debug={
                "run_id": run_id,
                "pipeline": "adaptive-evidence-research-v1" if self.config.adaptive_research else "authorize-route-screen-read-synthesize-v2",
                "adaptive_research": adaptive_debug,
                "claim_review": claim_review_debug,
                "synthesis_guard": synthesis_guard_debug,
                "strategy_learning": strategy_debug,
                "principal_id": auth.user_id,
                "authorization_revision": auth.revision,
                "model_calls": sum(c.status != "budget_denied" for c in calls),
                "answer_budget": self.broker.current_answer_budget.snapshot() if self.broker.current_answer_budget else None,
                "reference_closure_enabled": self.config.reference_closure,
                "reference_context_blocks": len({bid for card in synthesis_cards for bid in card.context_block_ids}),
                "reference_incomplete": reference_incomplete,
                "read_document_ids": tuple(documents_read),
                "screened_document_ids": tuple(
                    item.document_revision_id for item in annotated_manifests
                ),
                "cited_document_ids": _unique(item.document_revision_id for item in citations),
                "targets": tuple(jsonable(item) for item in targets),
                "prompt_identity_partition": tuple(
                    jsonable(item) for item in role_targets
                ),
                "ambiguities": resolution.ambiguities,
                "discovery": discovery_state,
                "errors": tuple(errors),
                "internal_warnings": tuple(_unique(warnings)),
            },
        )
        self._event(run_id, "answer_ready", {"status": answer_status, "sources": len(citations)})
        try:
            self.store.set_run_state(run_id, answer_status, finalized=True)
        except Exception:
            pass

        trace_payload = {
            "principal_id": auth.user_id,
            "authorization_revision": auth.revision,
            "request": {
                "question": question,
                "effective_question": effective_question,
                "mode": selected_mode,
                "deadline_s": self.config.total_deadline_s,
                "minimum_screen_window_s": self.config.minimum_screen_window_s,
                "minimum_reader_window_s": self.config.minimum_reader_window_s,
            },
            "corpus": jsonable(snapshot),
            "models": {
                "screen": self.config.effective_screen_model,
                "reader": self.config.effective_reader_model,
                "synthesis": self.config.effective_synthesis_model,
            },
            "exact_lane": {
                "surfaces": forced_surfaces,
                "documents": exact_hits_by_doc,
                "block_ids": [item.block_id for item in exact_blocks],
            },
            "discovery": discovery_state,
            "screening": [jsonable(item) for item in screening],
            "reader_reports": [jsonable(item.report) for item in reader_artifacts],
            "reader_packets": [
                {
                    "report_id": item.report.report_id,
                    "blocks": [jsonable(block) for block in item.blocks],
                }
                for item in reader_artifacts
            ],
            "evidence_cards": [jsonable(card) for card in synthesis_cards],
            "synthesis": {"raw_output": synthesis_raw, "rendered": rendered_text},
            "citation_checks": [jsonable(item) for item in citations],
            "coverage": jsonable(coverage),
            "timeline": list(self.store.events(run_id)),
            "model_calls": [jsonable(item) for item in calls],
            "answer": jsonable(answer),
            "warnings": list(_unique(warnings)),
            "errors": errors,
            "timings": {"answer_ready_s": elapsed_before_trace},
        }
        trace_path = self.traces.write(run_id, trace_payload)
        trace_s = time.perf_counter() - started - elapsed_before_trace
        answer.trace_path = str(trace_path) if trace_path is not None else ""
        if trace_path is not None:
            self.access.register_resource(
                "trace",
                run_id,
                stable_key=run_id,
                classification="restricted",
                owner_user_id=auth.user_id,
                metadata={
                    "resource_dependencies": [
                        f"document:{item.resource_id}" for item in citations
                    ]
                },
                actor_user_id=auth.user_id,
                affects_authorization=False,
            )
            if not (
                self.access.can(auth, "trace", run_id, "trace.read_own")
                or self.access.can(auth, "trace", run_id, "trace.read_any")
            ):
                answer.trace_path = ""
        answer.timings = {
            "answer_ready_s": elapsed_before_trace,
            "trace_s": max(0.0, trace_s),
            "total_s": time.perf_counter() - started,
        }
        if trace_path is None and self.config.trace_mode != "off":
            answer.warnings = tuple(_unique((*answer.warnings, f"trace_write_failed:{self.traces.last_error}")))
        if session is not None:
            session.record(question, answer)
        return answer


__all__ = ["SisuReader"]
