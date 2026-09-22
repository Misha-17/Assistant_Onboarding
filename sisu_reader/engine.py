"""Question-wide evidence retrieval and directly grounded answering."""
from __future__ import annotations
from dataclasses import asdict
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3
import time
from .legacy_engine import SisuReader as LegacyReader, _effective_question, _mode, _unique
from .models import Answer, Citation, Coverage, jsonable
from .broker import BrokerCallError

PIPELINE = 'question-evidence-grounding-v10'
REVISION_PROMPT_VERSION = 'grounded-answer-v10.4-retained-source-revision'
REQUEST_BUDGET_POLICY = 'rendered-messages-schema-output-v1'


class PromptBudgetExceeded(RuntimeError):
    """A complete rendered request exceeds the conservative context budget."""


class ProviderContextOverflow(PromptBudgetExceeded):
    """The provider rejected the context despite the local estimate."""


def _request_budget(messages, schema, context_tokens, output_tokens, *, headroom_tokens=0):
    from .screen_budget import conservative_tokens, screen_prompt_cost
    prompt = screen_prompt_cost(messages)
    schema_tokens = conservative_tokens(json.dumps(schema, ensure_ascii=False, separators=(',',':')))
    required = prompt + schema_tokens + output_tokens + headroom_tokens + 512
    return {'policy':REQUEST_BUDGET_POLICY,'prompt_tokens_estimate':prompt,'schema_tokens_estimate':schema_tokens,
            'output_tokens_reserved':output_tokens,'additional_headroom_tokens':headroom_tokens,
            'template_margin_tokens':512,'required_tokens_estimate':required,'context_tokens':context_tokens,
            'fits':required <= context_tokens,'estimate_is_exact':False}


def _fit_initial_evidence(question, plan, evidence, config, draft_tokens, audit_tokens):
    """Keep the highest-priority selected prefix that fits full request overhead.

    The prospective audit reserves room for the upcoming draft as well as its
    own response. Actual audits are checked again with the generated claims.
    No source text is shortened or joined and no aliases are renumbered.
    """
    from . import grounded_answer as grounded
    def measure(rows):
        draft = _request_budget(grounded.build_draft_messages(question,plan,rows),
            grounded.draft_schema(plan,rows),config.context_tokens,draft_tokens)
        audit = None
        if config.claim_review:
            empty=grounded.GroundedDraft((),tuple(f.id for f in plan.facets))
            audit = _request_budget(grounded.build_audit_messages(question,plan,rows,empty),
                grounded.audit_schema(plan),config.context_tokens,audit_tokens,headroom_tokens=draft_tokens+512)
        return {'draft':draft,'prospective_audit':audit,'fits':draft['fits'] and (audit is None or audit['fits'])}
    original=measure(evidence)
    selected=list(evidence)
    checked=original
    if not original['fits']:
        low,high=0,len(evidence)
        while low < high:
            middle=(low+high+1)//2
            if measure(evidence[:middle])['fits']:low=middle
            else:high=middle-1
        selected=list(evidence[:low]);checked=measure(selected)
    return selected,{'initial_request_budget':checked,'untrimmed_request_budget':original,
        'selected_passages_before':len(evidence),'selected_passages_after':len(selected),
        'omitted_passage_ids':[row['id'] for row in evidence[len(selected):]],
        'omission_reason':'request_context_budget' if len(selected)<len(evidence) else None}


def _draft_identity(draft, evidence):
    from .grounded_answer import _evidence
    value = {'claims':[asdict(c) for c in draft.claims], 'missing_facet_ids':draft.missing_facet_ids,
             'binding_policy':draft.binding_policy, 'evidence':list(_evidence(evidence).values())}
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode('utf-8')).hexdigest()


def _merge_evidence(previous, additions, token_budget):
    """Keep prior source context/aliases; admit new immutable spans within budget."""
    from .screen_budget import conservative_tokens
    def key(row):
        return row['block_id'],row['char_start'],row['char_end'],row['text_sha256']
    def cost(row):
        return 32+conservative_tokens(row['text'])+conservative_tokens(
            json.dumps([row['document_title'],row['locator'],row.get('headers',[])],ensure_ascii=False))
    merged=[dict(row) for row in previous]
    seen={key(row) for row in merged}
    aliases={row['id'] for row in merged}
    used=sum(cost(row) for row in merged)
    omitted=0
    for row in additions:
        if key(row) in seen:continue
        if len(merged)>=128 or used+cost(row)>token_budget:
            omitted+=1
            continue
        number=1
        while f'P{number}' in aliases:number+=1
        alias=f'P{number}'
        merged.append({**row,'id':alias})
        aliases.add(alias);seen.add(key(row));used+=cost(row)
    return merged,{'retained_passages':len(previous),'added_passages':len(merged)-len(previous),
                   'new_passages_omitted':omitted,'estimated_tokens_used':used,'token_budget':token_budget}


def _retention_omissions(original, revised, evidence, flagged_ids):
    by_id={row['id']:row for row in evidence}
    def signature(claim):
        spans=[]
        for quote in claim.quotes:
            source=by_id[quote.source_id]
            spans.append((source['block_id'],source['char_start']+quote.start,
                          source['char_start']+quote.end,quote.quote))
        return claim.text,tuple(sorted(claim.facet_ids)),tuple(sorted(spans))
    new={signature(claim) for claim in revised.claims}
    return tuple(claim.id for claim in original.claims if claim.id not in flagged_ids and signature(claim) not in new)


class SisuReader(LegacyReader):
    """Preserve authorization/resource routing; replace document orchestration."""
    def __init__(self, *args, retrieval=None, **kwargs):
        super().__init__(*args, **kwargs)
        if self.config.strategy_learning:
            raise ValueError('The V9 learned controller is incompatible: this pipeline does not support strategy_learning.')
        if retrieval is None:
            from .hybrid_retrieval import HybridIndex
            retrieval = HybridIndex(self.config, self.store)
        self.retrieval = retrieval
        self._index_stats = {}

    def load(self):
        super().load()
        self._index_stats = self.retrieval.ensure()
        self._status.update(pipeline=PIPELINE, strategy_learning=False, retrieval_index=self._index_stats)
        return self

    def rebuild(self, paths):
        report = super().rebuild(paths)
        self._index_stats = self.retrieval.ensure()
        return report

    def _validated_evidence(self, rows, allowed):
        """Only immutable corpus bytes, within the allowed scope, are admitted."""
        rows = list(rows)
        blocks = {x.block_id:x for x in self.store.blocks_by_ids(
            tuple(dict.fromkeys(str(x['block_id']) for x in rows)),
            allowed_document_revision_ids=tuple(allowed))}
        seen, output = set(), []
        documents,sections={},{}
        for item in rows:
            row = dict(item)
            if row['id'] in seen:raise ValueError('Duplicate evidence identity')
            seen.add(row['id'])
            block = blocks.get(row['block_id'])
            if block is None or block.document_revision_id not in allowed:
                raise PermissionError('Evidence is outside the authorized snapshot')
            if row['document_revision_id'] != block.document_revision_id:
                raise ValueError('Evidence document identity mismatch')
            if block.document_revision_id not in documents:
                documents[block.document_revision_id]=self.store.document(
                    block.document_revision_id,allowed_document_revision_ids=tuple(allowed))
                sections[block.document_revision_id]={section.section_id:section.section_path
                    for section in self.store.sections_for_document(block.document_revision_id,
                        allowed_document_revision_ids=tuple(allowed))}
            document=documents[block.document_revision_id]
            if document is None:raise PermissionError('Evidence document is no longer available')
            if hashlib.sha256(block.text.encode('utf-8')).hexdigest() != block.text_sha256:
                raise ValueError('Source block integrity check failed')
            start = row.get('char_start',row.get('start',0))
            end = row.get('char_end',row.get('end',len(block.text)))
            if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(block.text):
                raise ValueError('Invalid evidence offsets')
            if row['text'] != block.text[start:end]:
                raise ValueError('Evidence is not the exact declared source span')
            row.update(char_start=start,char_end=end,block_sha256=block.text_sha256,
                locator=block.locator,document_title=document.title,
                section_id=block.section_id,section_path=sections[block.document_revision_id].get(block.section_id,''),
                headers=list(block.headers),kind=block.kind,table_id=block.table_id,row_id=block.row_id,
                block_ordinal=block.ordinal,
                extraction_flags=list(block.extraction_flags),extraction_coverage=document.extraction_coverage,
                document_warnings=list(document.warnings),file_type=document.file_type,
                source_sha256=document.source_sha256,block_char_length=len(block.text),
                is_excerpt=start!=0 or end!=len(block.text),
                text_sha256=hashlib.sha256(row['text'].encode('utf-8')).hexdigest())
            output.append(row)
        return output, blocks

    def _ask(self, question, *, session, mode, progress, auth):
        from . import grounded_answer as grounded
        started = time.perf_counter()
        deadline = started + self.config.total_deadline_s
        snapshot = self.store.snapshot()
        effective_question, _, _ = _effective_question(question, session)
        allowed = set(self.access.allowed_document_revision_ids(auth,'document.search'))
        allowed.intersection_update(self.access.allowed_document_revision_ids(auth,'document.read'))
        allowed.intersection_update(self.access.allowed_document_revision_ids(auth,'document.cite'))
        documents = self.store.documents(allowed_document_revision_ids=tuple(allowed))
        lookup = {x.document_revision_id:x for x in documents}
        allowed = set(lookup)
        selected_mode = _mode(question,mode)
        run_id = self.store.create_run(question,effective_question=effective_question,
            session_id=f'session_{id(session):x}' if session is not None else '',snapshot_id=snapshot.snapshot_id,
            authorization_scope_hash=self.access.authorization_scope_hash(auth),principal_id=auth.user_id,
            authorization_revision=auth.revision,completeness_mode=selected_mode,
            config={'pipeline':PIPELINE,'model':self.config.model,'deadline_s':self.config.total_deadline_s,
                    'context_tokens':self.config.context_tokens,'direct_source_evidence':True})
        calls,warnings,errors,operations,audits,selections = [],[],[],[],[],[]
        audit_identities,revision_history,presented = [],[],{}
        request_budgets=[]
        plan = grounded.fallback_question_plan(effective_question)
        draft,evidence,blocks = None,[],{}

        def emit(stage,label,**current):
            self._emit(progress,started,stage,label,current=current)
            self._event(run_id,stage,current)

        def remaining():
            return max(0.0,deadline-time.perf_counter()-self.config.finalize_reserve_s)

        def guard(rows=(),sources=()):
            # CorpusStore caches its generation; inspect the active on-disk
            # pointer too, including externally rebuilt/replaced databases.
            try:
                with closing(sqlite3.connect(self.config.db_path.resolve().as_uri()+'?mode=ro',uri=True,timeout=1.0)) as connection:
                    active=connection.execute("SELECT value FROM metadata WHERE key='active_snapshot_id'").fetchone()
                if not active or active[0]!=snapshot.snapshot_id or self.store.snapshot().snapshot_id!=snapshot.snapshot_id:
                    raise RuntimeError('The document collection snapshot changed; retry.')
                self._reauthorize_answer_dependencies(auth,
                    document_revision_ids={row['document_revision_id'] for row in rows},sources=sources)
            except (PermissionError,RuntimeError,sqlite3.Error):
                try:self.store.set_run_state(run_id,'error',finalized=True)
                except Exception:pass
                raise

        def call(role,messages,schema,version,tokens,reserve=0.0,cap=None,source_rows=()):
            guard(source_rows)
            output_tokens=min(tokens,self.broker.output_tokens_for(role))
            budget=_request_budget(messages,schema,self.config.context_tokens,output_tokens)
            request_budgets.append({'role':role,'prompt_version':version,**budget})
            if not budget['fits']:
                raise PromptBudgetExceeded('Rendered messages, schema and output reserve exceed the model context budget')
            available = remaining()-reserve
            if available < 2.0:raise TimeoutError('No request time remains for this stage')
            try:
                result = self.broker.chat(role=role,messages=messages,format=schema,prompt_version=version,
                    num_predict=output_tokens,
                    timeout_s=min(available,cap or self.config.request_timeout_s),
                    reasoning=self.config.reasoning,temperature=0.0,seed=7)
            except BrokerCallError as exc:
                calls.append(exc.record)
                from .screen_budget import is_context_overflow
                if is_context_overflow(exc):
                    warnings.append('The model rejected an oversized context; that request was not repeated unchanged.')
                    raise ProviderContextOverflow('Provider rejected the request context') from exc
                raise
            calls.append(result.call)
            for row in source_rows:
                presented[(row['block_id'],row['char_start'],row['char_end'])]=dict(row)
            guard(source_rows)
            return result.generation.content

        try:
            emit('authorizing','Checking access to the current document collection')
            if documents:
                emit('screening','Separating the parts of your question')
                try:
                    raw=call('screen',grounded.build_plan_messages(effective_question),grounded.PLAN_SCHEMA,
                             grounded.PLAN_PROMPT_VERSION,640,reserve=35.0,cap=16.0)
                    plan=grounded.parse_plan(raw,effective_question)
                except (BrokerCallError,ValueError,TimeoutError,PromptBudgetExceeded) as exc:
                    warnings.append('Question planning was unavailable; the original question was used.')
                    errors.append('planning:'+type(exc).__name__)
                groups={f.id:list(dict.fromkeys([effective_question,f.question,*f.queries])) for f in plan.facets}
                emit('exact_search','Finding evidence by meaning and exact wording',facets=len(plan.facets))
                guard()
                selection=self.retrieval.select(groups,tuple(allowed),token_budget=min(10000,self.config.context_tokens//2))
                selections.append(selection['stats'])
                evidence,blocks=self._validated_evidence(selection['evidence'],allowed)
                evidence,fit_stats=_fit_initial_evidence(effective_question,plan,evidence,self.config,
                    min(3000,self.broker.output_tokens_for('synthesis')),min(1600,self.broker.output_tokens_for('review')))
                selections.append(fit_stats)
                if fit_stats['omitted_passage_ids']:
                    warnings.append('Some selected passages did not fit the model request budget and were not inspected.')
                if not fit_stats['initial_request_budget']['fits']:
                    raise PromptBudgetExceeded('The question and required request overhead exceed the model context budget')
                if evidence:
                    emit('reading','Reading original passages together',passages=len(evidence),
                         documents=len({x['document_revision_id'] for x in evidence}))
                    raw=call('synthesis',grounded.build_draft_messages(effective_question,plan,evidence),
                        grounded.draft_schema(plan,evidence),grounded.DRAFT_PROMPT_VERSION,3000,reserve=14.0,source_rows=evidence)
                    draft=grounded.parse_draft(raw,plan,evidence)
                    operations.append({'stage':'draft','draft':asdict(draft)})
                    if self.config.claim_review and remaining() >= 8.0:
                        emit('claim_review','Checking source support and question coverage')
                        try:
                            raw=call('review',grounded.build_audit_messages(effective_question,plan,evidence,draft),
                                grounded.audit_schema(plan),grounded.AUDIT_PROMPT_VERSION,1600,cap=22.0,source_rows=evidence)
                            audit=grounded.parse_audit(raw,plan)
                            audits.append(audit)
                            audit_identities.append(_draft_identity(draft,evidence))
                            actionable=bool({c.id for c in draft.claims}.intersection(audit.unsupported_claim_ids)
                                            or audit.missing_facet_ids or draft.rejections)
                            if actionable and remaining() >= 15.0:
                                expanded={key:list(value) for key,value in groups.items()}
                                for request in audit.searches:
                                    if request.facet_id in expanded:expanded[request.facet_id].insert(0,request.query)
                                if audit.searches:
                                    emit('adaptive_wave','Looking for evidence for the remaining parts')
                                    guard(evidence)
                                    extra=self.retrieval.select(expanded,tuple(allowed),
                                        token_budget=min(12000,self.config.context_tokens//2))
                                    selections.append(extra['stats'])
                                    new_evidence,_=self._validated_evidence(extra['evidence'],allowed)
                                    revised_evidence,merge_stats=_merge_evidence(evidence,new_evidence,
                                        token_budget=min(12000,self.config.context_tokens//2))
                                    revised_evidence,revised_blocks=self._validated_evidence(revised_evidence,allowed)
                                    selections.append({'revision_context_merge':merge_stats})
                                else:revised_evidence,revised_blocks=evidence,blocks
                                emit('synthesizing','Revising the answer using the checked source passages')
                                messages=list(grounded.build_draft_messages(effective_question,plan,revised_evidence,audit=audit))
                                flagged=set(audit.unsupported_claim_ids) if not audit.rejections else set()
                                messages[0]={**messages[0],'content':messages[0]['content']+
                                    ' Preserve every prior claim not flagged by the audit with exactly the same text, facet IDs and source IDs; '
                                    'add separate claims for newly established facts. The prior draft is not independent evidence. '
                                    'Reconsider flagged claims against actual source passages, not audit confidence.'}
                                data=json.loads(messages[1]['content'])
                                data['prior_claims']=[{'id':c.id,'text':c.text,'facet_ids':list(c.facet_ids),
                                    'source_ids':list(dict.fromkeys(q.source_id for q in c.quotes))} for c in draft.claims]
                                data['preserve_claim_ids']=[c.id for c in draft.claims if c.id not in flagged]
                                messages[1]={**messages[1],'content':json.dumps(data,ensure_ascii=False)}
                                raw=call('synthesis',messages,grounded.draft_schema(plan,revised_evidence),
                                    REVISION_PROMPT_VERSION,3000,reserve=6.0,source_rows=revised_evidence)
                                revised=grounded.parse_draft(raw,plan,revised_evidence)
                                omitted=_retention_omissions(draft,revised,revised_evidence,flagged)
                                accepted=not omitted and bool(revised.claims or not draft.claims)
                                revision_history.append({'accepted':accepted,'omitted_unflagged_claim_ids':list(omitted),
                                    'candidate_draft_identity':_draft_identity(revised,revised_evidence)})
                                operations.append({'stage':'revision_candidate','draft':asdict(revised),'accepted':accepted,
                                                   'omitted_unflagged_claim_ids':list(omitted)})
                                if omitted:
                                    warnings.append('The revision omitted previously source-bound content; the earlier answer was retained.')
                                if accepted:
                                    draft,evidence,blocks=revised,revised_evidence,revised_blocks
                                    operations.append({'stage':'revision','draft':asdict(draft)})
                                    if remaining() >= 8.0:
                                        raw=call('review',grounded.build_audit_messages(effective_question,plan,evidence,draft),
                                            grounded.audit_schema(plan),grounded.AUDIT_PROMPT_VERSION,1400,cap=18.0,source_rows=evidence)
                                        audits.append(grounded.parse_audit(raw,plan))
                                        audit_identities.append(_draft_identity(draft,evidence))
                        except (BrokerCallError,ValueError,TimeoutError,PromptBudgetExceeded) as exc:
                            warnings.append('The evidence review could not finish; available source-bound content is retained.')
                            errors.append('review:'+type(exc).__name__)
                else:warnings.append('No source passages were available to answer this question.')
        except (PermissionError,ValueError):
            self.store.set_run_state(run_id,'error',finalized=True)
            raise
        except Exception as exc:
            errors.append('answer:'+type(exc).__name__)
            warnings.append('The local model could not complete every stage of the answer.')

        guard(evidence)
        evidence,blocks=self._validated_evidence(evidence,allowed)
        citations=[]
        facet_gaps=tuple(f.id for f in plan.facets)
        rendered=None
        if draft is not None:
            rendered=grounded.format_answer(draft,plan)
            facet_gaps=rendered.facet_gaps
            by_alias={x['id']:x for x in evidence}
            seen=set()
            for binding in rendered.citation_bindings:
                if binding.source_id in seen:continue
                seen.add(binding.source_id)
                source=by_alias[binding.evidence_id]
                block=blocks[source['block_id']]
                document=lookup[block.document_revision_id]
                if source['text'][binding.start:binding.end] != binding.quote:
                    raise ValueError('Rendered citation span integrity failure')
                citations.append(Citation(source_id=binding.source_id,card_id=binding.claim_id,
                    block_id=block.block_id,document_revision_id=block.document_revision_id,
                    title=document.title,locator=block.locator,quote=binding.quote,source_path=document.source_path,
                    quote_sha256=hashlib.sha256(binding.quote.encode('utf-8')).hexdigest(),
                    resource_type='document',resource_id=document.logical_document_id,
                    provenance='Full immutable selected passage bound by source ID; semantic support remains a model judgment'))
        current_identity=_draft_identity(draft,evidence) if draft is not None else None
        bound_audit=next((audit for audit,identity in reversed(list(zip(audits,audit_identities)))
                          if identity==current_identity),None)
        if bound_audit is not None:
            outstanding=set(bound_audit.unsupported_claim_ids).intersection(c.id for c in draft.claims)
            if outstanding:warnings.append('The source review flagged possible unsupported content; inspect the cited passages.')
            unresolved_audit=bool(outstanding or bound_audit.missing_facet_ids or bound_audit.rejections)
            if bound_audit.rejections:warnings.append('The source review returned incomplete or invalid records; it did not clear the answer.')
        else:unresolved_audit=bool(self.config.claim_review and draft)
        if unresolved_audit and audits and bound_audit is None:
            warnings.append('The revised answer has no completed review bound to its current text and sources.')
        if draft is not None and draft.rejections:
            warnings.append('Some generated answer records were invalid; valid source-bound content was retained.')
        read_evidence=list(presented.values())
        _,read_blocks=self._validated_evidence(read_evidence,allowed)
        read_ids={x['document_revision_id'] for x in read_evidence}
        guard(read_evidence,citations)
        def verify_source_files():
            try:
                for document_id in read_ids:
                    document=lookup[document_id]
                    digest=hashlib.sha256()
                    with Path(document.source_path).open('rb') as stream:
                        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
                    if digest.hexdigest()!=document.source_sha256:
                        raise RuntimeError('A source file changed after indexing; rebuild the index before answering.')
            except (OSError,RuntimeError) as exc:
                self.store.set_run_state(run_id,'error',finalized=True)
                raise RuntimeError('A source file changed or is unavailable; rebuild the index before answering.') from exc
        complete_ids=[]
        full_spans={x['block_id'] for x in read_evidence if x['char_start']==0 and x['char_end']==len(read_blocks[x['block_id']].text)}
        for document_id in read_ids:
            full=self.store.blocks_for_document(document_id,allowed_document_revision_ids=tuple(allowed))
            if full and all(x.block_id in full_spans for x in full):complete_ids.append(document_id)
        extraction_gaps=tuple(x.document_revision_id for x in documents if x.document_revision_id in read_ids
                              and x.extraction_coverage!='complete')
        reasons=[]
        if len(complete_ids)!=len(documents):reasons.append('selected_passages_only')
        if facet_gaps:reasons.append('unresolved_question_parts')
        if extraction_gaps:reasons.append('incomplete_extraction')
        if unresolved_audit:reasons.append('review_not_cleared')
        coverage=Coverage(snapshot_id=snapshot.snapshot_id,mode=selected_mode,authorized_documents=len(documents),
            manifests_screened=0,documents_queued=len(read_ids),documents_read=len(read_ids),
            documents_fully_read=len(complete_ids),documents_remaining=len(documents)-len(complete_ids),
            sections_seen=len({x.section_id for x in read_blocks.values() if x.section_id}),
            sections_total_for_opened_documents=sum(len(self.store.sections_for_document(x,
                allowed_document_revision_ids=tuple(allowed))) for x in read_ids),
            exact_match_documents=0,evidence_cards=len(citations),extraction_gaps=extraction_gaps,
            deadline_reached=remaining()<=0,exhaustive=not reasons,complete=not reasons,
            provisional=bool(reasons),incomplete_reasons=tuple(reasons))
        text=rendered.text if rendered is not None and draft.claims else 'I could not establish an answer from the inspected source passages.'
        status=('answer' if citations and not facet_gaps and not unresolved_audit else 'partial' if citations else
                'error' if errors else 'not_found')
        if selected_mode=='exhaustive' and not coverage.complete and status=='answer':status='partial'
        emit('binding_citations','Attaching exact source passages to the answer',sources=len(citations))
        answer=Answer(status=status,text=text,sources=tuple(citations),warnings=_unique(warnings),coverage=coverage,
            timings={'answer_ready_s':time.perf_counter()-started},debug={'run_id':run_id,'pipeline':PIPELINE,
            'principal_id':auth.user_id,'authorization_revision':auth.revision,'model_calls':len(calls),
            'answer_budget':self.broker.current_answer_budget.snapshot() if self.broker.current_answer_budget else None,
            'question_plan':asdict(plan),'facet_gaps':list(facet_gaps),
            'request_budgets':request_budgets,
            'claim_review':[{**asdict(x),'draft_identity':identity,'applies_to_final':identity==current_identity}
                            for x,identity in zip(audits,audit_identities)],'revision_history':revision_history,
            'selection':selections,'read_document_ids':sorted(read_ids),'cited_document_ids':sorted({c.document_revision_id for c in citations}),
            'errors':errors,'strategy_learning':{'enabled':False}})
        self.store.set_run_state(run_id,status,finalized=True)
        guard(read_evidence,citations)
        verify_source_files()
        payload={'pipeline':PIPELINE,'principal_id':auth.user_id,'authorization_revision':auth.revision,
            'request':{'question':question,'effective_question':effective_question,'mode':selected_mode},
            'corpus':jsonable(snapshot),'models':{'model':self.config.model},
            'model_calls':[jsonable(x) for x in calls],'reader_packets':[{'blocks':[jsonable(x) for x in read_blocks.values()]}],
            'evidence_cards':evidence,'synthesis':{'operations':operations,'rendered':text},
            'citation_checks':[jsonable(x) for x in citations],'coverage':jsonable(coverage),
            'timeline':list(self.store.events(run_id)),'answer':jsonable(answer),'warnings':warnings,'errors':errors}
        trace=self.traces.write(run_id,payload)
        if trace is not None:
            self.access.register_resource('trace',run_id,stable_key=run_id,classification='restricted',owner_user_id=auth.user_id,
                metadata={'resource_dependencies':[f'document:{lookup[x].logical_document_id}' for x in read_ids]},
                actor_user_id=auth.user_id,affects_authorization=False)
            if self.access.can(auth,'trace',run_id,'trace.read_own') or self.access.can(auth,'trace',run_id,'trace.read_any'):
                answer.trace_path=str(trace)
        elif self.config.trace_mode!='off':answer.warnings=(*answer.warnings,'The local trace could not be saved.')
        guard(read_evidence,citations)
        verify_source_files()
        answer.timings['total_s']=time.perf_counter()-started
        if session is not None:session.record(question,answer)
        return answer


__all__=['SisuReader']
