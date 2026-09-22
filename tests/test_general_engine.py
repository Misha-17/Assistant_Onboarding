"""Actual corpus/authorization integration with a scripted, non-network broker."""
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import pytest

from sisu_reader.access import AccessManager
from sisu_reader.answer_budget import AnswerBudget
from sisu_reader.config import Config
from sisu_reader.engine import SisuReader, PIPELINE
from sisu_reader.grounded_answer import AUDIT_PROMPT_VERSION
from sisu_reader.models import GenerationResult, ModelCallRecord
from sisu_reader.session import Session
from sisu_reader.store import CorpusStore


QUESTION = "What are the hold duration and release condition?"
HOLD = "Hold samples for 13 days. Do not release while a review is active."
RELEASE = "Release requires approval from the duty engineer."


def planned(_):
    return {"facets": [{"id": "F1", "question": "Hold duration?", "queries": ["hold duration"]},
                       {"id": "F2", "question": "Release condition?", "queries": ["release condition"]}]}


def draft(payload, *, include_hold=True, include_release=True, incorrect_release=False, release_id="C2"):
    rows = []
    for marker, cid, facet, value in (
        ("Hold samples", "C1", "F1", HOLD),
        ("Release requires", release_id, "F2", RELEASE),
    ):
        if (facet == "F1" and not include_hold) or (facet == "F2" and not include_release):
            continue
        found = next(x for x in payload["passages"] if marker in x["text"])
        rows.append({"id": cid, "text": "Release requires nobody's approval." if incorrect_release and facet == "F2" else value,
                     "facet_ids": [facet], "source_ids": [found["id"]]})
    return {"claims": rows, "missing_facet_ids": [fid for fid, yes in (("F1", include_hold), ("F2", include_release)) if not yes]}


def clean_audit(_):
    return {"unsupported_claim_ids": [], "missing_facet_ids": [], "searches": [], "reason": "Both requested parts have source support."}


def gap_audit(_):
    return {"unsupported_claim_ids": [], "missing_facet_ids": ["F2"],
            "searches": [{"facet_id": "F2", "query": "release authorization"}], "reason": "Release is missing."}


class ScriptedBroker:
    def __init__(self, config, script):
        self.config, self.script = config, list(script)
        self.calls = []
        self.current_answer_budget = None

    @contextmanager
    def answer_budget(self, config):
        self.current_answer_budget = AnswerBudget(config, repair_prompt_version=AUDIT_PROMPT_VERSION)
        try:
            yield self.current_answer_budget
        finally:
            self.current_answer_budget.close()
            self.current_answer_budget = None

    def output_tokens_for(self, role):
        return getattr(self.config, role + "_output_tokens")

    def chat(self, **kwargs):
        admission = self.current_answer_budget.admit(role=kwargs["role"], prompt_version=kwargs["prompt_version"],
                                                    requested_tokens=kwargs["num_predict"])
        self.calls.append(kwargs)
        if not self.script:
            raise AssertionError("Unexpected extra model call")
        handler = self.script.pop(0)
        payload = json.loads(kwargs["messages"][1]["content"])
        result = handler(payload) if callable(handler) else handler
        if isinstance(result, Exception):
            self.current_answer_budget.settle(admission, failed=True)
            raise result
        raw = result if isinstance(result, str) else json.dumps(result)
        metrics = {"eval_count": 50}
        self.current_answer_budget.settle(admission, metrics=metrics)
        record = ModelCallRecord(str(len(self.calls)), kwargs["role"], "scripted", kwargs["prompt_version"], 0.01, "ok",
                                 metrics=metrics, raw_output=raw, prompt=json.dumps(kwargs["messages"]))
        return SimpleNamespace(generation=GenerationResult(raw, metrics=metrics), call=record)


class Retrieved:
    def __init__(self, store, choices=None, mutate=None):
        self.store, self.choices, self.mutate = store, choices, mutate
        self.calls = []

    def ensure(self):
        return {"test": True}

    def select(self, groups, allowed, token_budget):
        self.calls.append((groups, allowed, token_budget))
        all_docs = self.store.documents(allowed_document_revision_ids=allowed)
        selected = []
        for doc in all_docs:
            for block in self.store.blocks_for_document(doc.document_revision_id, allowed_document_revision_ids=allowed):
                if self.choices and not self.choices(len(self.calls), block):
                    continue
                selected.append({"id": "P" + str(len(selected) + 1), "text": block.text,
                                 "block_id": block.block_id, "document_revision_id": doc.document_revision_id,
                                 "document_title": doc.title, "locator": block.locator,
                                 "char_start": 0, "char_end": len(block.text), "estimated_tokens": 50})
        if self.mutate:
            self.mutate(selected)
        return {"evidence": selected, "stats": {"selected_passages": len(selected)}}


@pytest.fixture
def make(tmp_path):
    engines = []
    def build(script=None, *, trace="full", choices=None, mutate=None, access_clock=None, **overrides):
        case = tmp_path / str(len(engines))
        docs = case / "documents"
        docs.mkdir(parents=True)
        (docs / "hold.txt").write_text(HOLD, encoding="utf-8")
        (docs / "release.txt").write_text(RELEASE, encoding="utf-8")
        config = Config(case, case / "workspace", model="scripted", trace_mode=trace,
                        strategy_learning=False, **overrides)
        store = CorpusStore(config)
        store.rebuild(docs)
        access = AccessManager(config, clock=access_clock)
        broker = ScriptedBroker(config, script if script is not None else [planned, draft, clean_audit])
        retrieval = Retrieved(store, choices, mutate)
        engine = SisuReader(config, store=store, broker=broker, access=access, retrieval=retrieval)
        engine._loaded = True  # Readiness/network is outside these orchestration tests.
        engines.append(engine)
        return engine, broker, retrieval, docs
    yield build
    for engine in engines:
        engine.store.close()


def test_normal_path_actual_binding_coverage_session_and_trace(make):
    engine, broker, _, _ = make()
    session = Session()
    result = engine.ask(QUESTION, session=session)
    assert result.status == "answer" and len(result.sources) == 2
    assert [x["role"] for x in broker.calls] == ["screen", "synthesis", "review"]
    assert result.coverage.documents_read == result.coverage.documents_fully_read == 2
    assert result.coverage.complete
    for citation in result.sources:
        block = engine.store.blocks_by_ids((citation.block_id,))[0]
        assert citation.quote in block.text
        assert citation.quote_sha256 == hashlib.sha256(citation.quote.encode()).hexdigest()
    trace = json.loads(Path(result.trace_path).read_text(encoding="utf-8"))
    assert trace["pipeline"] == PIPELINE
    assert trace["answer"]["text"] == result.text
    assert session.turns[0].question == QUESTION and not hasattr(session.turns[0], "answer")
    assert result.debug["answer_budget"]["pending_finalization"] == {}


def test_v9_strategy_policy_cannot_be_activated(tmp_path):
    with pytest.raises(ValueError, match="V9 learned controller is incompatible"):
        SisuReader(Config(tmp_path, tmp_path / "workspace", strategy_learning=True))


def test_bad_audit_does_not_clear_answer(make):
    engine, _, _, _ = make([planned, draft, "not valid JSON"])
    result = engine.ask(QUESTION)
    assert result.status == "partial"
    assert "review_not_cleared" in result.coverage.incomplete_reasons


def test_revision_cannot_silently_drop_prior_unflagged_fact(make):
    engine, broker, _, _ = make([planned, lambda p: draft(p, include_release=False), gap_audit,
                                lambda p: draft(p, include_hold=False), clean_audit],
                               choices=lambda turn, block: turn == 1 or "Release requires" in block.text)
    result = engine.ask(QUESTION)
    assert "13 days" in result.text
    assert any("retain" in warning.lower() for warning in result.warnings)
    revision_payload = json.loads(broker.calls[3]["messages"][1]["content"])
    assert any("Hold samples" in row["text"] for row in revision_payload["passages"])
    assert any("Release requires" in row["text"] for row in revision_payload["passages"])


def test_successful_revision_retains_old_context_and_recovers_missing_facet(make):
    engine, broker, _, _ = make([planned, lambda p: draft(p, include_release=False), gap_audit, draft, clean_audit],
                               choices=lambda turn, block: "Hold samples" in block.text if turn == 1 else "Release requires" in block.text)
    result = engine.ask(QUESTION)
    assert result.status == "answer" and "13 days" in result.text and "duty engineer" in result.text
    assert len(broker.calls) == 5 and result.coverage.documents_read == 2
    revised = json.loads(broker.calls[3]["messages"][1]["content"])
    assert revised["prior_claims"][0]["source_ids"]
    assert "quotes" not in revised["prior_claims"][0]


def test_old_audit_cannot_clear_a_revised_draft_with_reused_or_new_ids(make, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("sisu_reader.engine.time.perf_counter", lambda: clock[0])
    def audit(_):
        clock[0] = 70
        return {"unsupported_claim_ids": ["C2"], "missing_facet_ids": [], "searches": [], "reason": "C2 contradicts approval."}
    def revise(payload):
        clock[0] = 86
        return draft(payload, release_id="C3")
    engine, broker, _, _ = make([planned, lambda p: draft(p, incorrect_release=True), audit, revise])
    result = engine.ask(QUESTION)
    assert len(broker.calls) == 4
    assert result.status == "partial" and "review_not_cleared" in result.coverage.incomplete_reasons
    assert "nobody" not in result.text and "duty engineer" in result.text


def test_invalid_claim_sibling_retains_valid_content_and_reports_issue(make):
    def malformed(payload):
        value = draft(payload)
        value["claims"][1]["source_ids"] = ["P999"]
        return value
    engine, _, _, _ = make([planned, malformed], claim_review=False)
    result = engine.ask(QUESTION)
    assert "13 days" in result.text and len(result.sources) == 1 and result.status == "partial"
    assert result.warnings


def test_failed_draft_dispatch_does_not_report_selected_sources_as_read(make):
    engine, _, _, _ = make([planned, TimeoutError("scripted synthesis timeout")])
    result = engine.ask(QUESTION)
    assert result.status == "error" and not result.sources
    assert result.coverage.documents_read == 0


def test_acl_revocation_stops_next_source_bearing_call(make):
    engine, broker, _, _ = make()
    document = engine.store.documents()[0]
    def revoke(_):
        engine.access.grant("local-admin", "document", document.logical_document_id, "document.read", effect="deny")
        return planned(None)
    broker.script = [revoke, draft, clean_audit]
    with pytest.raises(PermissionError):
        engine.ask(QUESTION)
    assert len(broker.calls) == 1


def test_time_based_access_change_without_revision_bump_blocks_publication(make):
    now = [datetime(2026, 9, 22, tzinfo=timezone.utc)]
    engine, broker, _, _ = make(access_clock=lambda: now[0], claim_review=False)
    document = engine.store.documents()[0]
    engine.access.grant("local-admin", "document", document.logical_document_id, "document.read", effect="deny",
                        valid_from=now[0] + timedelta(seconds=1))
    revision = engine.access.authorization_revision()
    def advance(payload):
        result = draft(payload)
        now[0] += timedelta(seconds=2)
        return result
    broker.script = [planned, advance]
    with pytest.raises(PermissionError):
        engine.ask(QUESTION)
    assert engine.access.authorization_revision() == revision


def test_external_snapshot_pointer_change_is_detected_despite_store_cache(make):
    engine, broker, _, _ = make(claim_review=False)
    def swap(payload):
        value = draft(payload)
        with sqlite3.connect(engine.config.db_path) as connection:
            connection.execute("UPDATE metadata SET value='different-generation' WHERE key='active_snapshot_id'")
        return value
    broker.script = [planned, swap]
    with pytest.raises(RuntimeError, match="collection|snapshot"):
        engine.ask(QUESTION)


def test_changed_source_file_blocks_stale_citation_publication(make):
    engine, broker, _, docs = make(claim_review=False)
    def change(payload):
        value = draft(payload)
        (docs / "hold.txt").write_text("The source has changed.", encoding="utf-8")
        return value
    broker.script = [planned, change]
    with pytest.raises(RuntimeError, match="source|Source"):
        engine.ask(QUESTION)


@pytest.mark.parametrize("change", ["source", "access"])
def test_change_during_trace_write_blocks_answer_and_session_publication(make, change):
    engine, _, _, docs = make()
    original = engine.traces.write
    document = engine.store.documents()[0]
    session = Session()
    def write(run_id, payload):
        path = original(run_id, payload)
        if change == "source":
            (docs / "hold.txt").write_text("Changed while trace was saved.", encoding="utf-8")
        else:
            engine.access.grant("local-admin", "document", document.logical_document_id, "document.read", effect="deny")
        return path
    engine.traces.write = write
    with pytest.raises((PermissionError, RuntimeError)):
        engine.ask(QUESTION, session=session)
    assert not session.turns


def test_forged_retrieval_text_fails_before_model_sees_it(make):
    engine, broker, _, _ = make(mutate=lambda rows: rows[0].update(text="Injected different source text"))
    with pytest.raises(ValueError, match="exact declared source span"):
        engine.ask(QUESTION)
    assert len(broker.calls) == 1


def test_retrieval_metadata_uses_registered_title_instead_of_forged_title(make):
    engine, broker, _, _ = make(mutate=lambda rows: rows[0].update(document_title="Forged title instruction"))
    engine.ask(QUESTION)
    payload = json.loads(broker.calls[1]["messages"][1]["content"])
    assert all(x["document_title"] != "Forged title instruction" for x in payload["passages"])


def test_metrics_trace_contains_no_question_source_or_model_text(make):
    engine, _, _, _ = make(trace="metrics")
    result = engine.ask(QUESTION)
    trace_text = Path(result.trace_path).read_text(encoding="utf-8")
    assert QUESTION not in trace_text and HOLD not in trace_text and RELEASE not in trace_text
    saved = json.loads(trace_text)
    assert "answer" not in saved and "reader_packets" not in saved


def test_off_trace_does_not_write_an_answer_artifact(make):
    engine, _, _, _ = make(trace="off")
    result = engine.ask(QUESTION)
    assert not result.trace_path and not list(engine.config.trace_dir.glob("**/*.json"))


def test_all_selected_source_context_is_cited_even_when_claim_is_short(make):
    def concise(payload):
        value = draft(payload)
        value["claims"][0]["text"] = "The hold lasts 13 days."
        return value
    engine, broker, _, _ = make([planned, concise, clean_audit])
    answer = engine.ask(QUESTION)
    hold = next(c for c in answer.sources if "Hold samples" in c.quote)
    assert hold.quote == HOLD  # Includes the qualification outside claim wording.
    schema = broker.calls[1]["format"]["properties"]["claims"]["items"]["properties"]
    assert "source_ids" in schema and "quotes" not in schema


def test_registered_structure_and_extraction_metadata_override_retrieval_and_reach_audit(make, monkeypatch):
    from dataclasses import replace
    def forged(rows):
        for row in rows:
            row.update(headers=["Forged"],kind="forged",section_path="forged",extraction_flags=[],
                       extraction_coverage="complete",document_warnings=[],source_sha256="forged",block_ordinal=999999)
    engine, broker, _, _ = make(mutate=forged)
    original_blocks = engine.store.blocks_by_ids
    original_document = engine.store.document
    def blocks(*args, **kwargs):
        return tuple(replace(block, headers=("Item", "Days"), extraction_flags=("ocr_uncertain",))
                     for block in original_blocks(*args, **kwargs))
    def document(*args, **kwargs):
        return replace(original_document(*args, **kwargs), extraction_coverage="partial",
                       warnings=("Some images have no text layer.",))
    monkeypatch.setattr(engine.store, "blocks_by_ids", blocks)
    monkeypatch.setattr(engine.store, "document", document)
    answer = engine.ask(QUESTION)
    for call in broker.calls[1:]:
        payload = json.loads(call["messages"][1]["content"])
        for row in payload["passages"]:
            assert row["headers"] == ["Item", "Days"] and row["extraction_flags"] == ["ocr_uncertain"]
            assert row["extraction_coverage"] == "partial"
            assert row["document_warnings"] == ["Some images have no text layer."]
            assert row["kind"] != "forged" and row["section_path"] != "forged"
            assert "source_sha256" not in row and "block_ordinal" not in row and "block_id" not in row
    for row in json.loads(Path(answer.trace_path).read_text(encoding='utf-8'))['evidence_cards']:
        original = original_blocks([row["block_id"]])[0]
        assert row["block_ordinal"] == original.ordinal != 999999
        assert row['source_sha256'] != 'forged' and row['extraction_flags'] == ['ocr_uncertain']


def test_audit_identity_changes_when_actual_source_metadata_changes(make):
    from sisu_reader.engine import _draft_identity
    from sisu_reader.grounded_answer import parse_draft, fallback_question_plan
    engine, _, retrieval, _ = make()
    allowed = [d.document_revision_id for d in engine.store.documents()]
    evidence, _ = engine._validated_evidence(retrieval.select({}, allowed, 1000)["evidence"], allowed)
    plan = fallback_question_plan(QUESTION)
    raw = json.dumps({"claims":[{"id":"C1","text":"A source-bound fact.","facet_ids":["F1"],
                                "source_ids":[evidence[0]["id"]]}], "missing_facet_ids":[]})
    parsed = parse_draft(raw, plan, evidence)
    old = _draft_identity(parsed, evidence)
    changed = [dict(row) for row in evidence]
    changed[0]["headers"] = ["Different unit"]
    assert old != _draft_identity(parsed, changed)


def test_initial_fit_accounts_for_full_prompts_schema_and_future_audit_without_splicing(tmp_path):
    from sisu_reader.engine import _fit_initial_evidence
    from sisu_reader.grounded_answer import fallback_question_plan
    sources=[{'id':f'P{i+1}','text':'Detailed original passage. '*200,'block_id':f'block{i}',
              'document_revision_id':f'doc{i}','document_title':'Manual','locator':f'paragraph {i}'} for i in range(12)]
    config=Config(tmp_path,tmp_path/'workspace',context_tokens=12000)
    selected,stats=_fit_initial_evidence(QUESTION,fallback_question_plan(QUESTION),sources,config,3000,1600)
    assert 0 < len(selected) < len(sources)
    assert selected == sources[:len(selected)]
    assert stats['omitted_passage_ids'] == [r['id'] for r in sources[len(selected):]]
    assert stats['initial_request_budget']['draft']['fits']
    assert stats['initial_request_budget']['prospective_audit']['fits']
    assert stats['initial_request_budget']['prospective_audit']['additional_headroom_tokens'] >= 3000
    assert not stats['untrimmed_request_budget']['fits']
    original,unchanged=_fit_initial_evidence(QUESTION,fallback_question_plan(QUESTION),sources[:1],config,3000,1600)
    assert original==sources[:1] and not unchanged['omitted_passage_ids']


def test_too_small_request_context_is_rejected_before_any_model_call(make):
    engine,broker,_,_=make(context_tokens=1024)
    answer=engine.ask(QUESTION)
    assert not broker.calls and not answer.sources and answer.coverage.documents_read==0
    assert any('PromptBudgetExceeded' in err for err in answer.debug['errors'])
    assert answer.debug['request_budgets'] and not answer.debug['request_budgets'][0]['fits']


@pytest.mark.parametrize('typed',[True,False])
def test_provider_context_overflow_is_identified_and_never_retried_unchanged(make,typed):
    from sisu_reader.broker import BrokerCallError
    from sisu_reader.ollama import OllamaHTTPError
    body={'error':{'type':'exceed_context_size_error','n_prompt_tokens':33000,'n_ctx':32768}} if typed else {'error':'invalid sampler grammar'}
    cause=OllamaHTTPError(400,json.dumps(body),path='/api/chat')
    record=ModelCallRecord('failed','synthesis','scripted','test',0.01,'error')
    failure=BrokerCallError('provider rejected',record=record);failure.__cause__=cause
    engine,broker,_,_=make([planned,failure,draft])
    answer=engine.ask(QUESTION)
    assert len(broker.calls)==2 and len(broker.script)==1 and not answer.sources
    assert ('answer:ProviderContextOverflow' in answer.debug['errors']) is typed
    assert any('not repeated unchanged' in warning for warning in answer.warnings) is typed


def test_oversized_actual_audit_keeps_draft_and_declines_call(make):
    def big_draft(payload):
        source=payload['passages'][0]['id']
        return {'claims':[{'id':f'C{i+1}','text':str(i+1)*750,'facet_ids':['F1'],
                            'source_ids':[source]} for i in range(32)],'missing_facet_ids':['F2']}
    engine,broker,_,_=make([planned,big_draft,clean_audit])
    answer=engine.ask(QUESTION)
    assert len(broker.calls)==2 and answer.sources and answer.status=='partial'
    assert 'review:PromptBudgetExceeded' in answer.debug['errors']
    assert not answer.debug['request_budgets'][-1]['fits']
