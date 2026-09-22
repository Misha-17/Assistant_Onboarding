"""One bounded, request-local admission account shared by runtime and evaluation.

Only admitted requests spend calls. Their output ceilings are reserved before
dispatch, and reliable measured output refunds unused capacity after success.
Unknown output on failure spends the full admitted ceiling. Final synthesis
and optional claim repair retain capacity while research is being scheduled.
"""
from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Mapping

from .config import Config


RESERVATION_CONTRACT = 'synthesis_and_source_audit_v10'


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return value


def budget_descriptor(config: Config) -> dict:
    return {'kind': 'bounded_calls_tokens',
            'max_model_calls': _positive_integer(config.max_model_calls, 'max_model_calls'),
            'max_output_tokens': _positive_integer(config.max_output_tokens, 'max_output_tokens'),
            'reservation_contract': RESERVATION_CONTRACT}


class AnswerBudgetExceeded(TimeoutError):
    """A request was denied before dispatch and did not spend a model call."""


@dataclass(frozen=True, slots=True)
class Admission:
    number: int
    output_ceiling: int
    kind: str


class AnswerBudget:
    def __init__(self, config: Config, *, repair_prompt_version: str):
        self.descriptor = budget_descriptor(config)
        self.pending = {'synthesis': _positive_integer(config.synthesis_output_tokens, 'synthesis_output_tokens')}
        if config.claim_review:
            self.pending['repair'] = _positive_integer(config.review_output_tokens, 'review_output_tokens')
        if (self.descriptor['max_model_calls'] <= len(self.pending) or
                self.descriptor['max_output_tokens'] <= sum(self.pending.values())):
            raise ValueError('Answer budget must fit finalization plus at least one research call and token')
        self.repair_prompt_version = repair_prompt_version
        self._initial_pending = dict(self.pending)
        self._attempts = 0
        self._charged_tokens = 0
        self._inflight: dict[int, Admission] = {}
        self._denials: list[dict] = []
        self._denied_total = 0
        self._closed = False
        self._lock = threading.RLock()

    @property
    def attempts(self):
        with self._lock:
            return self._attempts

    @property
    def charged_tokens(self):
        with self._lock:
            return self._charged_tokens

    @property
    def denials(self):
        with self._lock:
            return [dict(d) for d in self._denials]

    def matches(self, config: Config, *, repair_prompt_version: str) -> bool:
        pending = {'synthesis':config.synthesis_output_tokens}
        if config.claim_review:
            pending['repair'] = config.review_output_tokens
        return (budget_descriptor(config) == self.descriptor and pending == self._initial_pending
                and repair_prompt_version == self.repair_prompt_version)

    def admit(self, *, role: str, prompt_version: str, requested_tokens: int) -> Admission:
        requested_tokens = _positive_integer(requested_tokens, 'requested_tokens')
        kind = ('synthesis' if role == 'synthesis' else 'repair'
                if role == 'review' and prompt_version == self.repair_prompt_version else 'research')
        with self._lock:
            protected = {k:v for k,v in self.pending.items() if k != kind}
            calls = self.descriptor['max_model_calls'] - self._attempts - len(protected)
            tokens = self.descriptor['max_output_tokens'] - self._charged_tokens - sum(protected.values())
            if self._closed or calls < 1 or tokens < 1:
                reason = 'answer_scope_closed' if self._closed else 'finalization_reserve_or_total_budget'
                # Bounded operational metadata only; no question, source or answer text.
                self._denied_total += 1
                if len(self._denials) < 128:
                    self._denials.append({'role':role, 'prompt_version':prompt_version[:120],
                                         'kind':kind, 'reason':reason})
                raise AnswerBudgetExceeded('Answer budget exhausted; remaining capacity is reserved for finalization')
            self.pending.pop(kind, None)
            self._attempts += 1
            admission = Admission(self._attempts, min(requested_tokens, tokens), kind)
            self._charged_tokens += admission.output_ceiling
            self._inflight[admission.number] = admission
            return admission

    def settle(self, admission: Admission, *, metrics: Mapping | None = None, failed: bool = False) -> int:
        """Return charged tokens, preserving reservations across concurrent calls."""
        with self._lock:
            if self._inflight.get(admission.number) is not admission:
                raise ValueError('Admission is foreign or already settled')
            measured = (metrics or {}).get('eval_count')
            valid = isinstance(measured, int) and not isinstance(measured, bool) and measured >= 0
            charged = measured if valid and not failed else admission.output_ceiling
            # Do not hide an invalid server overrun: charge reported actual output
            # even if it exceeds the requested ceiling and deny subsequent work.
            self._charged_tokens += charged - admission.output_ceiling
            del self._inflight[admission.number]
            return charged

    def close(self):
        with self._lock:
            self._closed = True

    def snapshot(self) -> dict:
        with self._lock:
            return {'descriptor':dict(self.descriptor), 'model_calls':self._attempts,
                    'charged_output_tokens':self._charged_tokens,
                    'denied_requests':self._denied_total, 'denials':[dict(d) for d in self._denials],
                    'denial_details_truncated':self._denied_total > len(self._denials),
                    'pending_finalization':dict(self.pending), 'inflight_calls':len(self._inflight),
                    'output_budget_exceeded':self._charged_tokens > self.descriptor['max_output_tokens']}
