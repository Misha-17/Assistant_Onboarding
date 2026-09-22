from __future__ import annotations

import re
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Iterator, Mapping, Sequence


_DEICTIC = re.compile(
    r"\b(?:it|its|itself|they|them|their|theirs|he|him|his|she|her|hers|"
    r"this|that|these|those|former|latter|same|above|previous|"
    r"h\u00e4n|h\u00e4nen|heit\u00e4|heid\u00e4n|se|sen|ne|niiden|"
    r"t\u00e4m\u00e4|t\u00e4m\u00e4n|tuo|tuon|n\u00e4m\u00e4|"
    r"n\u00e4iden|nuo|noiden)\b",
    re.IGNORECASE,
)
_CONTINUATION = re.compile(
    r"^\s*(?:(?:and|also|then|plus)\b|(?:and\s+)?(?:what|how)\s+about\b|"
    r"(?:ent(?:a|\u00e4)|lis(?:a|\u00e4)ksi|sitten)\b)",
    re.IGNORECASE,
)
_NEW_TOPIC = re.compile(
    r"^\s*(?:(?:actually|instead)\s*[,;:]?\s+|"
    r"(?:new|different)\s+(?:question|topic)\s*[:;-]?\s+|"
    r"(?:itse asiassa|sen sijaan)\s*[,;:]?\s+|"
    r"(?:uusi|toinen)\s+(?:kysymys|aihe)\s*[:;-]?\s+)",
    re.IGNORECASE,
)
_FOLLOW_UP_WORD_LIMIT = 20


def _clean(value: Any, maximum: int = 4_000) -> str:
    return " ".join(str(value or "").split()).strip()[:maximum]


def _value(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(name, default)
    return getattr(item, name, default)


def _strings(value: Any, *, maximum: int = 64) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Sequence):
        return ()
    result: list[str] = []
    seen: set[str] = set()
    for raw in value[:maximum]:
        item = _clean(raw, 240)
        key = item.casefold()
        if item and key not in seen:
            result.append(item)
            seen.add(key)
    return tuple(result)


@dataclass(frozen=True)
class Turn:
    """Conversation memory safe to give back to the research engine.

    Deliberately absent: assistant answer prose. A previous generated answer is
    never promoted into evidence for a later question.
    """

    question: str
    status: str
    cited_document_ids: tuple[str, ...] = ()
    cited_source_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class SessionContext:
    is_follow_up: bool
    prior_user_questions: tuple[str, ...] = ()
    cited_document_ids: tuple[str, ...] = ()
    cited_source_ids: tuple[str, ...] = ()


@dataclass
class Session:
    """Small, bounded conversation state shared by the CLI and browser.

    The state retains user questions plus identifiers for documents that were
    actually cited. It never stores assistant prose, raw reasoning, or an
    uncited document as conversational evidence.
    """

    max_turns: int = 6
    turns: list[Turn] = field(default_factory=list)
    principal_id: str = ""
    authorization_revision: int = -1
    _lock: RLock = field(default_factory=RLock, init=False, repr=False)

    def __post_init__(self) -> None:
        self.max_turns = max(0, min(int(self.max_turns), 50))

    def bind_authorization(self, principal_id: str, revision: int) -> bool:
        """Bind conversation memory to one authorization snapshot.

        Any identity or policy-epoch change clears prior hints and questions.
        This is intentionally conservative: a follow-up must never carry a
        now-revoked document or a prior user's question into a new prompt.
        Returns ``True`` when existing state was cleared.
        """

        clean_id = _clean(principal_id, 160)
        clean_revision = int(revision)
        with self._lock:
            changed = bool(self.principal_id) and (
                self.principal_id != clean_id
                or self.authorization_revision != clean_revision
            )
            if changed:
                self.turns.clear()
            self.principal_id = clean_id
            self.authorization_revision = clean_revision
            return changed

    @contextmanager
    def locked(self) -> Iterator[None]:
        with self._lock:
            yield

    @property
    def turn_count(self) -> int:
        with self._lock:
            return len(self.turns)

    def is_follow_up(self, question: str) -> bool:
        clean = _clean(question)
        with self._lock:
            has_history = bool(self.turns)
        if not has_history or not clean or _NEW_TOPIC.search(clean):
            return False
        if _DEICTIC.search(clean):
            return True
        words = re.findall(r"[^\W_]+", clean, flags=re.UNICODE)
        return bool(
            len(words) <= _FOLLOW_UP_WORD_LIMIT and _CONTINUATION.search(clean)
        )

    def context(self, question: str) -> SessionContext:
        if not self.is_follow_up(question):
            return SessionContext(False)
        with self._lock:
            # The immediately preceding turn is enough to resolve ordinary
            # follow-ups and avoids silently blending two unrelated topics.
            recent = tuple(self.turns[-1:])
        return SessionContext(
            is_follow_up=True,
            prior_user_questions=tuple(turn.question for turn in recent),
            cited_document_ids=tuple(dict.fromkeys(
                document_id
                for turn in recent
                for document_id in turn.cited_document_ids
            )),
            cited_source_ids=tuple(dict.fromkeys(
                source_id
                for turn in recent
                for source_id in turn.cited_source_ids
            )),
        )

    def record(self, question: str, answer: Any) -> None:
        """Record a released turn without retaining generated answer text."""

        if self.max_turns == 0:
            return
        clean_question = _clean(question)
        status = _clean(_value(answer, "status", ""), 80).casefold()
        if not clean_question or status == "error":
            return

        document_ids: list[str] = []
        source_ids: list[str] = []
        sources = _value(answer, "sources", ())
        if not isinstance(sources, (str, bytes, Mapping)):
            try:
                source_values = tuple(sources or ())[:64]
            except TypeError:
                source_values = ()
            for source in source_values:
                document_id = _clean(
                    _value(source, "document_id", "")
                    or _value(source, "document_revision_id", "")
                    or _value(source, "document_revision", ""),
                    240,
                )
                source_id = _clean(
                    _value(source, "source_id", "")
                    or _value(source, "window_id", "")
                    or _value(source, "passage_id", ""),
                    240,
                )
                if document_id:
                    document_ids.append(document_id)
                if source_id:
                    source_ids.append(source_id)

        debug = _value(answer, "debug", {})
        if isinstance(debug, Mapping):
            document_ids.extend(_strings(
                debug.get("cited_document_ids") or (),
            ))

        turn = Turn(
            question=clean_question,
            status=status or "unknown",
            cited_document_ids=tuple(dict.fromkeys(document_ids)),
            cited_source_ids=tuple(dict.fromkeys(source_ids)),
        )
        with self._lock:
            # Engine implementations may record the turn themselves. Making
            # this operation idempotent lets the interaction shell safely call
            # it as well.
            if self.turns and self.turns[-1] == turn:
                return
            if self.turns and self.turns[-1].question == turn.question:
                self.turns[-1] = turn
            else:
                self.turns.append(turn)
            if len(self.turns) > self.max_turns:
                del self.turns[:-self.max_turns]

    def cited_document_hints(self, question: str) -> tuple[str, ...]:
        return self.context(question).cited_document_ids

    def prior_user_questions(self, question: str) -> tuple[str, ...]:
        return self.context(question).prior_user_questions

    def snapshot(self) -> tuple[Turn, ...]:
        with self._lock:
            return tuple(self.turns)

    def reset(self) -> None:
        with self._lock:
            self.turns.clear()


__all__ = ["Session", "SessionContext", "Turn"]
