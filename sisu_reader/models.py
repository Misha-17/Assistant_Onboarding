from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


AnswerStatus = Literal["answer", "partial", "clarify", "not_found", "error"]
ScreenLabel = Literal["read", "maybe", "unlikely"]


@dataclass(frozen=True, slots=True)
class CorpusSnapshot:
    snapshot_id: str
    created_at: str
    manifest_sha256: str
    document_count: int
    section_count: int
    block_count: int


@dataclass(frozen=True, slots=True)
class DocumentRevision:
    document_revision_id: str
    logical_document_id: str
    title: str
    source_path: str
    source_sha256: str
    file_type: str
    extraction_coverage: str = "complete"
    warnings: tuple[str, ...] = ()
    token_estimate: int = 0
    body_sha256: str = ""


@dataclass(frozen=True, slots=True)
class Section:
    section_id: str
    document_revision_id: str
    parent_section_id: str | None
    ordinal: int
    depth: int
    heading: str
    section_path: str
    locator: str
    first_block_ordinal: int
    last_block_ordinal: int
    token_estimate: int


@dataclass(frozen=True, slots=True)
class SourceBlock:
    block_id: str
    document_revision_id: str
    section_id: str | None
    ordinal: int
    kind: str
    locator: str
    text: str
    text_sha256: str
    canonical_char_start: int
    canonical_char_end: int
    previous_block_id: str | None = None
    next_block_id: str | None = None
    table_id: str | None = None
    row_id: str | None = None
    headers: tuple[str, ...] = ()
    token_estimate: int = 0
    extraction_flags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class DocumentManifest:
    manifest_id: str
    document_revision_id: str
    title: str
    source_path: str
    file_type: str
    extraction_coverage: str
    outline: str
    lead_text: str
    exact_surfaces: tuple[str, ...]
    token_estimate: int
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class EntityTarget:
    entity_id: str
    surface: str
    canonical_name: str = ""
    basis: str = "question_surface"
    distinct_from: tuple[str, ...] = ()
    inherited: bool = False


@dataclass(frozen=True, slots=True)
class ResearchRequest:
    run_id: str
    snapshot_id: str
    question: str
    effective_question: str
    mode: str
    targets: tuple[EntityTarget, ...]
    prior_document_ids: tuple[str, ...]
    deadline_s: float


@dataclass(frozen=True, slots=True)
class ScreenDecision:
    manifest_id: str
    document_revision_id: str
    decision: ScreenLabel
    reason: str = ""
    relevant_entities: tuple[str, ...] = ()
    source_lane: str = "semantic"


@dataclass(frozen=True, slots=True)
class ReaderRecord:
    record_type: str
    text: str = ""
    subject: str = ""
    block_ids: tuple[str, ...] = ()
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EvidenceCard:
    card_id: str
    document_revision_id: str
    document_title: str
    subject: str
    claim: str
    block_ids: tuple[str, ...]
    role: str = "support"
    # Citation spans remain small. These additional immutable blocks carry
    # explicitly referenced conditions through the reader-to-synthesis boundary.
    context_block_ids: tuple[str, ...] = ()
    reference_issues: tuple[str, ...] = ()
    reference_checked: bool = False


@dataclass(frozen=True, slots=True)
class ReadReport:
    report_id: str
    document_revision_id: str
    document_title: str
    entity_scope: str
    answerability: str
    proposed_answer: str
    cards: tuple[EvidenceCard, ...]
    conflicts: tuple[str, ...] = ()
    unresolved: tuple[str, ...] = ()
    records: tuple[ReaderRecord, ...] = ()
    presented_block_ids: tuple[str, ...] = ()
    sections_seen: int = 0
    sections_total: int = 0
    complete_document_read: bool = False
    warnings: tuple[str, ...] = ()
    raw_output: str = ""
    reference_issues: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Citation:
    source_id: str
    card_id: str
    block_id: str
    document_revision_id: str
    title: str
    locator: str
    quote: str
    source_path: str
    quote_sha256: str
    resource_type: str = "document"
    resource_id: str = ""
    uri: str = ""
    timestamp_seconds: int | None = None
    provenance: str = "document"


@dataclass(frozen=True, slots=True)
class Coverage:
    snapshot_id: str
    mode: str
    authorized_documents: int
    manifests_screened: int
    documents_queued: int
    documents_read: int
    documents_fully_read: int
    documents_remaining: int
    sections_seen: int
    sections_total_for_opened_documents: int
    exact_match_documents: int
    evidence_cards: int
    document_errors: tuple[str, ...] = ()
    deferred_documents: tuple[str, ...] = ()
    extraction_gaps: tuple[str, ...] = ()
    deadline_reached: bool = False
    exhaustive: bool = False
    complete: bool = False
    provisional: bool = True
    incomplete_reasons: tuple[str, ...] = ()
    reference_checks_enabled: bool = False
    reference_checked_cards: int = 0
    reference_context_blocks: int = 0
    reference_incomplete: bool = False


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    stage: str
    label: str
    elapsed_s: float
    coverage: dict[str, Any] = field(default_factory=dict)
    current: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GenerationResult:
    content: str
    reasoning: str = ""
    tool_calls: tuple[dict[str, Any], ...] = ()
    metrics: dict[str, int | float | str] = field(default_factory=dict)
    raw_message: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ModelCallRecord:
    call_id: str
    role: str
    model: str
    prompt_version: str
    elapsed_s: float
    status: str
    metrics: dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""
    raw_output: str = ""
    prompt: str = ""
    error: str = ""


@dataclass
class Answer:
    status: AnswerStatus
    text: str
    sources: tuple[Citation, ...] = ()
    warnings: tuple[str, ...] = ()
    timings: dict[str, float] = field(default_factory=dict)
    trace_path: str = ""
    coverage: Coverage | None = None
    debug: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class BuildReport:
    documents: int
    sections: int
    blocks: int
    warnings: tuple[str, ...]
    elapsed_s: float
    snapshot_id: str


@dataclass(frozen=True, slots=True)
class PrincipalContext:
    """Trusted local identity attached by the host process.

    This is an authorization context, not a password or remote authentication
    token. The loopback application obtains it from trusted configuration.
    """

    user_id: str
    display_name: str
    system_roles: tuple[str, ...]
    group_ids: tuple[str, ...]
    authorization_revision: int
    active: bool = True


@dataclass(frozen=True, slots=True)
class PersonRecord:
    person_id: str
    display_name: str
    aliases: tuple[str, ...] = ()
    organization: str = ""
    active: bool = True
    created_at: str = ""
    updated_at: str = ""


@dataclass(frozen=True, slots=True)
class OrganizationalRole:
    role_id: str
    person_id: str
    person_name: str
    role_name: str
    responsibility: str = ""
    organization: str = ""
    valid_from: str = ""
    valid_to: str = ""
    status: str = "asserted"
    provenance_type: str = "manual"
    provenance_ref: str = ""
    access_classification: str = "restricted"
    active: bool = True
    created_at: str = ""
    updated_at: str = ""


@dataclass(frozen=True, slots=True)
class VideoChapter:
    chapter_id: str
    video_id: str
    title: str
    description: str
    start_seconds: int
    end_seconds: int | None = None
    keywords: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class VideoRecord:
    video_id: str
    title: str
    description: str
    uri: str
    speaker: str = ""
    recorded_date: str = ""
    duration_seconds: int | None = None
    tags: tuple[str, ...] = ()
    project: str = ""
    summary: str = ""
    access_classification: str = "restricted"
    chapters: tuple[VideoChapter, ...] = ()
    active: bool = True
    created_at: str = ""
    updated_at: str = ""


def jsonable(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return {key: jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    return value
