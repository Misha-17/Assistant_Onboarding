"""Optional post-answer citation audit; no answer, trace or policy mutations."""
from __future__ import annotations
from dataclasses import asdict, is_dataclass
from contextlib import closing
import builtins
import copy
import importlib.util
import math
import os
import secrets
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import sys
import threading
import time

SUPPORT = Path(__file__).resolve().parent / 'support'


def _load_support(name, contract=None):
    """Load relocated, byte-identical helpers without global bare-module aliases.

    The frozen helpers use one bare support_contract import. Redirect only that
    import inside their module builtins; other processes/tests cannot cause a
    cached research support_client to select its old filesystem root.
    """
    qualified = __package__ + '._' + name
    spec = importlib.util.spec_from_file_location(qualified, SUPPORT / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    if contract is not None:
        def local_import(module_name, globals=None, locals=None, fromlist=(), level=0):
            if module_name == 'support_contract' and level == 0:
                return contract
            return builtins.__import__(module_name, globals, locals, fromlist, level)
        module.__dict__['__builtins__'] = {**vars(builtins), '__import__': local_import}
    sys.modules[qualified] = module
    spec.loader.exec_module(module)
    return module


_contract = _load_support('support_contract')
_planner = _load_support('support_planner', _contract)
_client_module = _load_support('support_client', _contract)
SupportClient = _client_module.SupportClient
canonical_sha256 = _contract.canonical_sha256
text_sha256 = _contract.text_sha256
AuditScope = _planner.AuditScope
plan_audit = _planner.plan_audit
validate_unit_response = _planner.validate_unit_response

VERSION = 'optional-post-answer-citation-audit-v3'
MAX_TRACE_BYTES = 8 * 1024 * 1024
MAX_SOURCE_BYTES = 64 * 1024 * 1024


class AuditError(RuntimeError):
    def __init__(self, code, message, status=409):
        super().__init__(message); self.code=code; self.message=message; self.status=status


def row(value): return asdict(value) if is_dataclass(value) else dict(value)


def _file_hash(path, deadline=None, maximum=MAX_SOURCE_BYTES):
    path=Path(path)
    if path.stat().st_size > maximum: raise AuditError('audit_input_limit','A source exceeds the optional audit size limit.')
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        while block:=stream.read(1024*1024):
            if deadline is not None and time.monotonic() >= deadline: raise AuditError('audit_budget_exhausted','The separate audit time budget expired.')
            digest.update(block)
    return digest.hexdigest()


def _sources_hash(core_dir):
    return canonical_sha256({p.name:_file_hash(p) for p in sorted(Path(core_dir).iterdir())
                             if p.is_file() and p.suffix in {'.py', '.ps1'}})


class AuditService:
    def __init__(self, application, *, client_factory=None, runtime_identity_provider=None, budget_s=20.0, startup_budget_s=60.0, maximum_units=6):
        if isinstance(budget_s, bool) or not math.isfinite(budget_s) or not 0 < budget_s <= 20 or type(maximum_units) is not int or not 0 <= maximum_units <= 6: raise ValueError('Audit budget exceeds facade contract')
        if isinstance(startup_budget_s, bool) or not math.isfinite(startup_budget_s) or not 0 < startup_budget_s <= 60: raise ValueError('Audit startup budget exceeds facade contract')
        self.application=application; self.budget_s=float(budget_s); self.maximum_units=maximum_units
        self.startup_budget_s=float(startup_budget_s)
        self._client_factory=client_factory or self._default_client
        self._runtime_identity_provider=runtime_identity_provider or self._current_runtime_identity
        self._client=None; self._lock=threading.RLock(); self._items={}; self._active=None; self._closed=False
        import sisu_reader.web as core_web
        self.core_dir=Path(core_web.__file__).resolve().parent
        self.core_build_sha256=_sources_hash(self.core_dir)
        own=Path(__file__).resolve().parent
        self.build_sha256=canonical_sha256({'version':VERSION,'core_source_sha256':self.core_build_sha256,
            'facade_sources':{str(p.relative_to(own)):_file_hash(p) for p in
                sorted([own/'service.py',own/'audit_app.py',own/'minicheck_cpu_run.py',
                        own/'minicheck_flan_cpu_run.py',*SUPPORT.glob('*.py')]) if p.exists()}})

    def _default_client(self):
        root=Path(__file__).resolve().parent.parent
        interpreter=root/'.audit-venv'/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
        interpreter=Path(os.environ.get('SISU_READER_AUDIT_PYTHON') or interpreter)
        manifest=Path(os.environ.get('SISU_READER_AUDIT_MANIFEST') or root/'models/minicheck/manifest.json')
        if not interpreter.is_file() or not manifest.is_file():
            raise AuditError('audit_unavailable',
                'The optional citation checker is not installed. Follow docs/OPTIONAL_CITATION_AUDIT.md to enable it.',503)
        return SupportClient(interpreter,manifest,
                             max_pending=0,startup_timeout_s=self.startup_budget_s)

    def _active_snapshot_id(self):
        # CorpusStore caches its snapshot. An external index rebuild must also
        # invalidate audit delivery while that engine still holds the old one.
        uri=self.application.config.db_path.resolve().as_uri()+'?mode=ro'
        with closing(sqlite3.connect(uri,uri=True,timeout=0.5)) as connection:
            value=connection.execute("SELECT value FROM metadata WHERE key='active_snapshot_id'").fetchone()
        return value[0] if value else None

    def _current_runtime_identity(self, deadline=None):
        # Metadata only. This neither loads an inference model nor changes the
        # core's strategy-learning flag, active epoch, or checkpoint selection.
        from sisu_reader.runtime_compatibility import build_runtime_compatibility, resolve_role_digests
        from sisu_reader.ollama import OllamaClient
        timeout = 3.0 if deadline is None else min(3.0, deadline - time.monotonic())
        if timeout <= 0:
            raise AuditError('audit_budget_exhausted', 'The separate audit time budget expired.')
        config = self.application.config
        roles = resolve_role_digests(config, refresh=True, fetch=lambda:
            OllamaClient(config)._request_json('GET', '/api/tags', timeout_s=timeout))
        return build_runtime_compatibility(config, model_roles=roles)

    def _job(self, job_id, browser):
        current_browser=self.application.sessions.get(browser.token)
        if current_browser is None: raise AuditError('session_expired','Start a new browser conversation.',401)
        job=self.application.jobs.get(job_id,browser.token)
        if job is None: raise AuditError('job_not_found','No such job in this browser conversation.',404)
        if not self.application.job_authorized(job,current_browser):
            raise AuditError('access_changed','The answer is unavailable under the current access policy.',403)
        if job.status!='complete' or job.answer is None: raise AuditError('answer_not_complete','Wait for the original answer to finish.')
        return job,current_browser

    def _capture(self, job_id, browser, *, deadline=None):
        try:
            return self._capture_checked(job_id, browser, deadline=deadline)
        except AuditError:
            raise
        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError):
            raise AuditError('access_changed', 'The answer, source, or runtime binding is no longer verifiable.', 403) from None

    def _capture_checked(self, job_id, browser, *, deadline=None):
        from sisu_reader.web import _sources
        job,browser=self._job(job_id,browser)
        original_job=copy.deepcopy(job.answer)
        visible,error=self.application.read_trace(job,browser)
        if error or visible is None: raise AuditError('audit_trace_unavailable','The authorized full answer trace is unavailable for this audit.')
        path=Path(job.trace_path).resolve(strict=True)
        trace_root=self.application.config.trace_dir.resolve()
        if not path.is_relative_to(trace_root) or path.suffix.lower()!='.json': raise AuditError('access_changed','The answer trace identity changed.',403)
        if path.stat().st_size > MAX_TRACE_BYTES: raise AuditError('audit_trace_unavailable','The answer trace exceeds the optional audit limit.')
        raw=path.read_bytes(); trace=json.loads(raw.decode('utf-8'))
        answer=trace.get('answer',{}); citations=answer.get('sources')
        # Answer-mode traces intentionally omit the duplicate citation_checks.
        # The raw original answer still retains full unsanitized Citation rows.
        if 'citation_checks' in trace and trace['citation_checks'] != citations:
            raise AuditError('access_changed','The duplicate citation bindings disagree.',403)
        run_id=original_job.get('debug',{}).get('run_id')
        if (not run_id or answer.get('text')!=original_job.get('text') or
            _sources(citations)!=original_job.get('sources') or
            answer.get('debug',{}).get('run_id')!=run_id or trace.get('principal_id')!=browser.principal_id):
            raise AuditError('access_changed','The saved answer or citation bindings changed.',403)
        if not isinstance(citations,list) or len(citations)>96: raise AuditError('audit_input_limit','The citation list exceeds this optional audit limit.')
        if any(c.get('resource_type','document')!='document' for c in citations):
            raise AuditError('audit_representation_unavailable','This optional audit currently checks document citations only.')
        snapshot_id=trace.get('corpus',{}).get('snapshot_id')
        if snapshot_id!=original_job.get('coverage',{}).get('snapshot_id'):
            raise AuditError('access_changed','The answer snapshot binding changed.',403)
        engine=self.application._ready_engine(); store=engine.store
        if self._active_snapshot_id()!=snapshot_id or store.snapshot().snapshot_id!=snapshot_id:
            raise AuditError('access_changed','The document index changed after this answer.',403)
        recorded_epoch=answer.get('debug',{}).get('strategy_learning',{}).get('runtime_epoch_sha256')
        if original_job.get('debug',{}).get('strategy_learning',{}).get('runtime_epoch_sha256')!=recorded_epoch:
            raise AuditError('access_changed','The core answer runtime identity changed.',403)
        if _sources_hash(self.core_dir)!=self.core_build_sha256:
            raise AuditError('access_changed','Core runtime sources changed; restart before auditing.',403)
        try:
            current_identity=self._runtime_identity_provider(deadline)
            from sisu_reader.runtime_compatibility import validate_runtime_compatibility
            current_identity=validate_runtime_compatibility(current_identity)
        except AuditError:
            raise
        except Exception:
            raise AuditError('audit_runtime_identity_unavailable',
                             'Current local core/model metadata is unavailable; no claim was scored.') from None
        epoch=current_identity['epoch_sha256']
        if recorded_epoch is not None and recorded_epoch!=epoch:
            raise AuditError('access_changed','The current core/model configuration differs from the answer.',403)
        roles=current_identity['descriptor']['model_roles']
        for name, configured in trace.get('models',{}).items():
            if name in roles and str(configured).removesuffix(':latest')!=roles[name]['name']:
                raise AuditError('access_changed','The answer used a different model configuration.',403)
        identity_basis='recorded_epoch_verified_current' if recorded_epoch is not None else 'current_metadata_only'
        principal=self.application.access.principal_snapshot(browser.principal_id)
        allowed=set(self.application.access.allowed_document_revision_ids(principal,'document.read'))
        for action in ('document.search','document.cite'):
            allowed.intersection_update(self.application.access.allowed_document_revision_ids(principal,action))
        cited_revisions={c['document_revision_id'] for c in citations}
        if not cited_revisions<=allowed: raise AuditError('access_changed','Citation access changed.',403)
        documents=[]; total_size=0
        for revision in sorted(cited_revisions):
            document=store.document(revision,allowed_document_revision_ids=tuple(allowed))
            if document is None: raise AuditError('access_changed','A cited document revision changed.',403)
            value=row(document); file=Path(value['source_path'])
            total_size+=file.stat().st_size
            if total_size>MAX_SOURCE_BYTES: raise AuditError('audit_input_limit','The source bytes exceed this optional audit limit.')
            if _file_hash(file,deadline)!=value['source_sha256']: raise AuditError('access_changed','A cited source file changed after indexing.',403)
            documents.append(value)
        blocks=[row(b) for b in store.blocks_by_ids(tuple(dict.fromkeys(c['block_id'] for c in citations)),
                                                   allowed_document_revision_ids=tuple(allowed))]
        scope=AuditScope(run_id,principal.user_id,self.application.access.authorization_scope_hash(principal),
                         snapshot_id,epoch,frozenset(allowed))
        # Recheck policy and immutable bindings after file I/O, not merely before it.
        if not self.application.job_authorized(job,browser) or self._active_snapshot_id()!=snapshot_id or store.snapshot().snapshot_id!=snapshot_id or job.answer!=original_job:
            raise AuditError('access_changed','Answer access or content changed during audit preparation.',403)
        reference_ids=frozenset(c['source_id'] for c in citations if any(
            card.get('card_id')==c.get('card_id') and card.get('reference_issues') for card in trace.get('evidence_cards',[])))
        payload={'answer':answer['text'],'citations':citations,'documents':documents,'blocks':blocks,
                 'scope':scope,'unresolved_citation_ids':reference_ids,
                 'runtime_identity_basis':identity_basis,
                 'recorded_runtime_epoch_sha256':recorded_epoch,
                 'reference_metadata_available':'evidence_cards' in trace,
                 'trace_sha256':hashlib.sha256(raw).hexdigest(),
                 'job_answer_sha256':canonical_sha256(original_job)}
        payload['fingerprint']=canonical_sha256({k:asdict(v) if k=='scope' else v for k,v in payload.items()
            if k not in {'scope','unresolved_citation_ids'}} | {
                'scope':{**asdict(scope),'authorized_document_revision_ids':sorted(scope.authorized_document_revision_ids)},
                'unresolved_citation_ids':sorted(reference_ids)})
        return payload

    def start(self, job_id, browser):
        self._job(job_id,browser)
        with self._lock:
            if self._closed: raise AuditError('audit_unavailable','The audit service is closed.',503)
            if job_id in self._items: return self.get(job_id,browser)
            if self._active is not None: raise AuditError('audit_busy','Another optional citation audit is running.',429)
            if len(self._items)>=128:
                oldest=next(iter(self._items)); self._items.pop(oldest)
            record={'audit_id':'audit_'+secrets.token_hex(12),'audit_saved':False,'status':'queued','units':[],'counts':{'total':0,'scored':0,'literal_quotes':0,'unchecked':0},
                'elapsed_s':0.0,'elapsed_scope':'in_progress','timing_receipt_saved':False,'budget_s':self.budget_s,'maximum_scored_units':self.maximum_units,
                'work_budget_s':self.budget_s,'startup_budget_s':self.startup_budget_s,
                'total_work_ceiling_s':self.budget_s+self.startup_budget_s,
                'effective_work_cutoff_s':self.budget_s,'phase':'queued',
                'advisory':True,'learning_truth':False,'original_answer_changed':False,
                'version':VERSION,'build_sha256':self.build_sha256,'core_source_sha256':self.core_build_sha256,
                'costs':{'responses':[],'client_events':[],'cutoff_s':None,
                         'startup_elapsed_s':0.0,'startup_budget_used_s':0.0,
                         'startup_cutoff_to_return_s':0.0,
                         'startup_outcome':'not_needed','preflight_elapsed_s':None,
                         'scoring_work_elapsed_s':0.0,
                         'teardown_s':None,'post_budget_elapsed_s':0.0,'parent_validation_s':0.0,
                         'prepublication_validation_s':0.0,'publication_s':None,
                         'timing_receipt_publication_s':None,'client_events_may_be_incomplete':False,
                         'teardown_note':'The frozen client includes cancellation in call elapsed time; internal teardown is not separately measured.'},
                'limitations':['A high score predicts source support, not truth. A low score does not prove falsehood.',
                               'Unchecked units and verified literal quotations are separate from scored generated claims.']}
            self._items[job_id]={'public':record,'capture':None}; self._active=job_id
            threading.Thread(target=self._run,args=(job_id,browser),daemon=True,name='sisu-optional-citation-audit').start()
            return copy.deepcopy(record)

    def _record_client_events(self, record, client, prior_events):
        # A bounded ring's length is not a cursor. Keep prior objects alive so
        # IDs cannot be reused and only this operation's appended events enter
        # the current principal's audit, including during worker startup.
        current_events=tuple(client.events)
        new_events=[event for event in current_events
                    if not any(event is prior for prior in prior_events)]
        with self._lock:
            record['costs']['client_events'].extend(copy.deepcopy(new_events))
            if prior_events and len(current_events)>=128 and not any(
                event is prior for event in current_events for prior in prior_events):
                record['costs']['client_events_may_be_incomplete']=True

    def _retire_client(self, client):
        # A failed or revoked startup must not leave a live private worker, and
        # a permanently closed client must not poison the next explicit audit.
        try:
            client.close()
        finally:
            with self._lock:
                if self._client is client:self._client=None

    @staticmethod
    def _plan_units(plan):
        return [{'unit_id':unit.unit_id,'text':unit.exact_text,'claim':unit.claim_text,
                 'start':unit.start,'end':unit.end,'state':unit.status if unit.status!='planned' else 'unchecked',
                 'support_score':None,'reason':unit.unchecked_reason if unit.status!='planned' else 'pending',
                 'citation_ids':[e.source_id for e in unit.evidence]} for unit in plan.units]

    def _run(self, job_id, browser):
        started=time.monotonic(); deadline=started+self.budget_s
        whole_deadline=started+self.budget_s+self.startup_budget_s
        item=self._items[job_id]; item['started_monotonic']=started
        record=item['public']; plan=None
        try:
            with self._lock: record.update(status='running',phase='preflight')
            capture=self._capture(job_id,browser,deadline=deadline)
            with self._lock:
                item['capture']=capture
                record['runtime_identity_basis']=capture['runtime_identity_basis']
                record['runtime_epoch_sha256']=capture['scope'].runtime_epoch_sha256
                record['recorded_runtime_epoch_sha256']=capture['recorded_runtime_epoch_sha256']
                record['reference_metadata_available']=capture['reference_metadata_available']
                if not capture['reference_metadata_available']:
                    record['limitations'].append('This trace omits reader-card reference-debt metadata. Scores concern only the quoted citation text, not completeness of required source context.')
                if capture['runtime_identity_basis']=='current_metadata_only':
                    record['limitations'].append('This answer has no recorded historical epoch. The audit binds current core/model metadata; it does not prove the earlier model weights.')
            def make_plan(contract_sha):
                return plan_audit(capture['answer'],capture['citations'],documents=capture['documents'],blocks=capture['blocks'],
                    scope=capture['scope'],deadline_monotonic=deadline,now=time.monotonic(),maximum_claims=self.maximum_units,
                    contract_sha256=contract_sha,unresolved_citation_ids=capture['unresolved_citation_ids'])
            plan=make_plan(_contract.CONTRACT_SHA256)
            with self._lock:
                record['units']=self._plan_units(plan)
                self._counts(record)
            if any(unit.status=='planned' for unit in plan.units):
                with self._lock:
                    if self._closed: raise AuditError('audit_unavailable','The audit service closed.',503)
                    if self._client is None: self._client=self._client_factory()
                    client=self._client
                # Validation and cheap client construction consume the ordinary
                # work budget, before a separate bounded worker-load phase.
                try:
                    current=self._capture(job_id,browser,deadline=deadline)
                    if current['fingerprint']!=capture['fingerprint']:
                        raise AuditError('access_changed','Answer or evidence changed before worker startup.',403)
                except Exception:
                    self._retire_client(client)
                    raise
                if time.monotonic()>=deadline:
                    raise AuditError('audit_budget_exhausted','The separate audit work budget expired before startup.')
                startup_started=time.monotonic()
                startup_deadline=min(startup_started+self.startup_budget_s,whole_deadline)
                prior_events=tuple(client.events)
                with self._lock:
                    record['phase']='loading_model'
                    record['costs']['preflight_elapsed_s']=startup_started-started
                    record['costs']['startup_outcome']='running'
                try:
                    startup_remaining=startup_deadline-time.monotonic()
                    if startup_remaining<=0:raise TimeoutError('Startup allowance expired before dispatch')
                    client.start(timeout_s=startup_remaining)
                    if time.monotonic()>=startup_deadline:
                        raise TimeoutError('Worker readiness arrived after startup deadline')
                    with self._lock:
                        if self._closed:raise AuditError('audit_unavailable','The audit service closed during startup.',503)
                        record['costs']['startup_outcome']='ready'
                except TimeoutError:
                    with self._lock:record['costs']['startup_outcome']='timed_out'
                    self._retire_client(client)
                    raise AuditError('audit_startup_timeout','The local citation checker did not load within its separate startup allowance.') from None
                except Exception:
                    with self._lock:record['costs']['startup_outcome']='failed'
                    self._retire_client(client)
                    raise
                finally:
                    loaded=time.monotonic()
                    startup_elapsed=loaded-startup_started
                    # Credit only actual loading time, capped at its allowance.
                    # Unused loading allowance never becomes extra scoring time.
                    startup_credit=min(startup_elapsed,self.startup_budget_s)
                    deadline=min(started+self.budget_s+startup_credit,whole_deadline)
                    with self._lock:
                        record['costs']['startup_elapsed_s']=startup_elapsed
                        record['costs']['startup_budget_used_s']=startup_credit
                        record['costs']['startup_cutoff_to_return_s']=max(0.0,loaded-startup_deadline)
                        record['effective_work_cutoff_s']=deadline-started
                    self._record_client_events(record,client,prior_events)
                try:
                    current=self._capture(job_id,browser,deadline=deadline)
                    if current['fingerprint']!=capture['fingerprint']:
                        raise AuditError('access_changed','Answer or evidence changed during worker startup.',403)
                except Exception:
                    self._retire_client(client)
                    raise
                # Recreate every bound request after readiness, with the final
                # shared scoring deadline and actual scorer contract identity.
                plan=make_plan(self._client.identity['contract_sha256'])
                record['scorer_contract_sha256']=self._client.identity['contract_sha256']
                record['scorer_model_identity_sha256']=self._client.identity['model_identity_sha256']
            with self._lock:
                record['phase']='scoring'
                record['units']=self._plan_units(plan)
                if record['costs']['preflight_elapsed_s'] is None:
                    record['costs']['preflight_elapsed_s']=time.monotonic()-started
                self._counts(record)
            for index,unit in enumerate(plan.units):
                if unit.status!='planned': continue
                if time.monotonic()>=deadline:
                    with self._lock: record['units'][index]['reason']='budget_exhausted'
                    continue
                with self._lock:
                    if self._closed: raise AuditError('audit_unavailable','The audit service closed.',503)
                checked_started=time.monotonic()
                current=self._capture(job_id,browser,deadline=deadline)
                record['costs']['parent_validation_s']+=time.monotonic()-checked_started
                if current['fingerprint']!=capture['fingerprint']:
                    raise AuditError('access_changed','Answer or evidence changed before scoring.',403)
                call_started=time.monotonic()
                prior_events=tuple(self._client.events)
                try:
                    response=self._client.score(unit.request())
                finally:
                    self._record_client_events(record,self._client,prior_events)
                returned=time.monotonic()
                # Validate the wire result at its actual return time, before
                # mandatory post-call authorization/source checks consume time.
                generation=(self._client.readiness or {}).get('worker_generation')
                checked=validate_unit_response(plan,unit.unit_id,response,current_answer=current['answer'],current_scope=current['scope'],
                    citations=current['citations'],documents=current['documents'],blocks=current['blocks'],now=None,
                    expected_worker_generation=generation if response['status']=='scored' else response['worker_generation'],
                    expected_model_identity_sha256=self._client.identity['model_identity_sha256'])
                with self._lock:
                    record['costs']['responses'].append({'unit_id':unit.unit_id,'usage':checked['usage'],
                        'elapsed_s':checked['elapsed_s'],'parent_call_elapsed_s':returned-call_started,
                        'cutoff_to_return_s':max(0.0,returned-deadline),
                        'input_tokens':checked['input_tokens'],'status':checked['status']})
                # Identity validation above deliberately has no clock argument:
                # reject a late positive below without discarding measured cost.
                delivered=checked if returned<deadline or checked['status']!='scored' else {
                    **checked,'status':'unchecked','support_score':None,'unchecked_reason':'deadline_exceeded'}
                checked_started=time.monotonic()
                current=self._capture(job_id,browser)
                record['costs']['parent_validation_s']+=time.monotonic()-checked_started
                if current['fingerprint']!=capture['fingerprint']:
                    raise AuditError('access_changed','Answer or evidence changed during scoring.',403)
                with self._lock:
                    record['units'][index].update(state=delivered['status'],support_score=delivered['support_score'],reason=delivered['unchecked_reason'])
                    self._counts(record)
            current=self._capture(job_id,browser)
            if current['fingerprint']!=capture['fingerprint']:
                raise AuditError('access_changed','Answer or evidence changed before audit delivery.',403)
            with self._lock: record['status']='complete'
        except AuditError as exc:
            with self._lock:
                if exc.code=='audit_budget_exhausted' and plan is not None:
                    record['status']='complete'
                else:
                    record.update(status='failed',error={'code':exc.code,'message':exc.message})
                if exc.code=='access_changed': record['units']=[]
        except Exception:
            with self._lock:
                record.update(status='failed',error={'code':'audit_failed','message':'The optional audit could not complete safely.'})
        finally:
            elapsed=time.monotonic()-started
            with self._lock:
                for unit in record['units']:
                    if unit['reason']=='pending':
                        unit['reason']='budget_exhausted' if time.monotonic()>=deadline else 'audit_incomplete'
                self._counts(record)
                record['costs']['scoring_work_elapsed_s']=max(0.0,elapsed-record['costs']['startup_elapsed_s'])
                self._elapsed(record,elapsed,'through_score_processing')
                record['phase']='publishing'
            publication_started=time.monotonic()
            try:
                published=self._persist(job_id,browser,item)
                main_published=time.monotonic()
                with self._lock:
                    record['costs']['publication_s']=main_published-publication_started
                    self._elapsed(record,main_published-started,'through_main_artifact_publication')
                receipt_started=time.monotonic()
                self._persist_completion(job_id,browser,item,published)
                finished=time.monotonic()
                with self._lock:
                    record['costs']['timing_receipt_publication_s']=finished-receipt_started
                    self._elapsed(record,finished-started,'through_timing_receipt_publication')
            finally:
                with self._lock:
                    record['phase']=record['status']
                    if self._active==job_id: self._active=None

    def _elapsed(self, record, elapsed, scope):
        record['elapsed_s']=elapsed
        record['elapsed_scope']=scope
        cutoff=record['effective_work_cutoff_s']
        record['costs']['post_budget_elapsed_s']=max(0.0,elapsed-cutoff)
        if elapsed>=cutoff:record['costs']['cutoff_s']=cutoff

    @staticmethod
    def _publish_json(target,payload):
        """Atomically publish without replacing an earlier artifact or temp."""
        temporary=target.with_name(target.name+'.'+secrets.token_hex(8)+'.tmp')
        created=False
        payload={**payload,'artifact_content_sha256':canonical_sha256(payload)}
        try:
            with temporary.open('x',encoding='utf-8',newline='\n') as stream:
                created=True
                json.dump(payload,stream,ensure_ascii=False,allow_nan=False,indent=2)
                stream.write('\n');stream.flush();os.fsync(stream.fileno())
            actual_sha256=_file_hash(temporary,maximum=MAX_TRACE_BYTES)
            # Same-volume hard-link publication is atomic and never overwrites.
            os.link(temporary,target)
            return actual_sha256
        finally:
            if created:
                try:temporary.unlink(missing_ok=True)
                except OSError:pass

    def _persist(self, job_id, browser, item):
        """One separate private artifact; source-check again at publication."""
        record=item['public'];capture=item['capture']
        validation_started=time.monotonic()
        try:
            if capture is None:
                self._job(job_id,browser)
            else:
                current=self._capture(job_id,browser)
                if current['fingerprint']!=capture['fingerprint']:
                    raise AuditError('access_changed','Answer or source identity changed before publication.',403)
        except AuditError:
            with self._lock:
                record.update(status='failed',error={'code':'access_changed',
                    'message':'Answer access or evidence changed before the audit artifact was published.'})
                record['units']=[];self._counts(record)
        validated=time.monotonic()
        with self._lock:
            record['costs']['prepublication_validation_s']=validated-validation_started
            self._elapsed(record,validated-item['started_monotonic'],'through_prepublication_validation')
        run_id=capture['scope'].run_id if capture is not None else job_id
        safe_run=run_id if re.fullmatch(r'[A-Za-z0-9_-]{1,160}',run_id) else text_sha256(run_id)[:32]
        root=self.application.config.trace_dir.resolve()
        directory=root/'citation_audits'
        try:
            directory.mkdir(parents=True,exist_ok=True)
            if not directory.resolve().is_relative_to(root):
                raise ValueError('Audit output escaped private trace directory')
            target=directory/(safe_run+'_'+self.build_sha256[:12]+'_'+record['audit_id']+'.json')
            provenance=None
            if capture is not None:
                docs={d['document_revision_id']:d for d in capture['documents']}
                blocks={b['block_id']:b for b in capture['blocks']}
                provenance={'answer_sha256':text_sha256(capture['answer']),
                    'job_answer_sha256':capture['job_answer_sha256'],'trace_sha256':capture['trace_sha256'],
                    'capture_fingerprint':capture['fingerprint'],'authorization_scope':capture['scope'].authorization_scope,
                    'snapshot_id':capture['scope'].snapshot_id,
                    'source_bindings':[{'source_id':c['source_id'],'card_id':c['card_id'],
                        'document_revision_id':c['document_revision_id'],'block_id':c['block_id'],
                        'source_sha256':docs[c['document_revision_id']]['source_sha256'],
                        'block_text_sha256':blocks[c['block_id']]['text_sha256'],'quote_sha256':c['quote_sha256']}
                        for c in capture['citations']]}
            with self._lock:
                saved_record={**copy.deepcopy(record),'audit_saved':True}
                payload={'schema_version':2,'principal_id':browser.principal_id,'job_id':job_id,
                    'run_id':run_id,'provenance':provenance,'audit':saved_record,
                    'timing_note':'Main artifact timings stop before its own serialization/fsync/link. A separately hash-bound completion receipt measures publication.'}
            artifact_sha=self._publish_json(target,payload)
            with self._lock:record['audit_saved']=True
            return {'path':target,'sha256':artifact_sha}
        except (OSError,ValueError,KeyError,TypeError):
            with self._lock:
                record['audit_saved']=False
                record['audit_save_error']='The separate private audit artifact could not be saved.'
            return None

    def _persist_completion(self,job_id,browser,item,published):
        """Record observed main-file publication, never its own future fsync."""
        record=item['public']
        if published is None:return
        try:
            root=self.application.config.trace_dir.resolve()
            directory=published['path'].parent/'completion_receipts'
            directory.mkdir(parents=True,exist_ok=True)
            if not directory.resolve().is_relative_to(root):raise ValueError('Timing output escaped private directory')
            payload={'schema_version':1,'audit_id':record['audit_id'],'job_id':job_id,
                'principal_id':browser.principal_id,'build_sha256':self.build_sha256,
                'audit_artifact':{'name':published['path'].name,'sha256':published['sha256']},
                'elapsed_s':record['elapsed_s'],'elapsed_scope':record['elapsed_scope'],
                'publication_s':record['costs']['publication_s'],
                'prepublication_validation_s':record['costs']['prepublication_validation_s'],
                'post_budget_elapsed_s':record['costs']['post_budget_elapsed_s'],
                'timing_note':'Measured through completed main-artifact publication. This receipt cannot measure its own future serialization/fsync/link; the live API reports that separate observed duration.'}
            self._publish_json(directory/published['path'].name,payload)
            with self._lock:record['timing_receipt_saved']=True
        except (OSError,ValueError,KeyError,TypeError):
            with self._lock:record['timing_receipt_saved']=False

    @staticmethod
    def _counts(record):
        record['counts']={'total':len(record['units']),'scored':sum(u['state']=='scored' for u in record['units']),
            'literal_quotes':sum(u['state']=='literal_quote_verified' for u in record['units']),
            'unchecked':sum(u['state']=='unchecked' for u in record['units'])}

    def get(self, job_id, browser):
        self._job(job_id,browser)
        with self._lock:
            item=self._items.get(job_id)
            if item is None: raise AuditError('audit_not_found','No optional audit has been requested for this answer.',404)
            capture=item['capture']
        if capture is not None:
            current=self._capture(job_id,browser)
            if current['fingerprint']!=capture['fingerprint']: raise AuditError('access_changed','Answer or evidence changed after the audit.',403)
        with self._lock:
            value=copy.deepcopy(item['public'])
            if self._active==job_id and value['status'] in {'complete','failed'}:
                value['status']='running'  # separate artifact publication is still finishing
            return value

    def close(self):
        with self._lock:self._closed=True;client=self._client
        if client is not None:client.close()
