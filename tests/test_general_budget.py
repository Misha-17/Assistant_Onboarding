from dataclasses import replace
import pytest
from sisu_reader.answer_budget import AnswerBudget, AnswerBudgetExceeded
from sisu_reader.broker import InferenceBroker, BrokerCallError
from sisu_reader.config import Config
from sisu_reader.grounded_answer import AUDIT_PROMPT_VERSION, PLAN_PROMPT_VERSION, DRAFT_PROMPT_VERSION
from sisu_reader.models import GenerationResult

def test_planning_cannot_spend_reserved_draft_or_actual_source_audit(tmp_path):
    cfg=Config(project_dir=tmp_path,workspace_dir=tmp_path,max_model_calls=3,
        max_output_tokens=6000,synthesis_output_tokens=3000,review_output_tokens=2000)
    class Client:
        def __init__(self,*args,**kwargs):pass
        def chat(self,**kwargs):return GenerationResult('{}',metrics={'eval_count':10})
    broker=InferenceBroker(cfg,client_factory=Client)
    def call(role,version):return broker.chat(role=role,prompt_version=version,messages=({'role':'user','content':'test'},))
    with broker.answer_budget() as account:
        call('screen',PLAN_PROMPT_VERSION)
        with pytest.raises(BrokerCallError):call('screen',PLAN_PROMPT_VERSION)
        call('synthesis',DRAFT_PROMPT_VERSION)
        call('review',AUDIT_PROMPT_VERSION)
        assert account.snapshot()['pending_finalization']=={}
        assert account.snapshot()['model_calls']==3
        assert account.snapshot()['charged_output_tokens']==30
        assert account.snapshot()['descriptor']['reservation_contract']=='synthesis_and_source_audit_v10'

def test_failed_audit_spends_full_ceiling_and_cannot_restart_budget(tmp_path):
    cfg=Config(project_dir=tmp_path,workspace_dir=tmp_path,max_model_calls=3,max_output_tokens=6000)
    account=AnswerBudget(cfg,repair_prompt_version=AUDIT_PROMPT_VERSION)
    for role,version,ceiling in [('screen',PLAN_PROMPT_VERSION,640),('synthesis',DRAFT_PROMPT_VERSION,3000),('review',AUDIT_PROMPT_VERSION,1600)]:
        admission=account.admit(role=role,prompt_version=version,requested_tokens=ceiling)
        account.settle(admission,metrics={'eval_count':1},failed=True)
    assert account.charged_tokens==5240
    with pytest.raises(AnswerBudgetExceeded):account.admit(role='review',prompt_version=AUDIT_PROMPT_VERSION,requested_tokens=1)
