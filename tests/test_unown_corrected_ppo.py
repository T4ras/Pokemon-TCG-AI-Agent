"""Focused revised protocol checks; zero new native matches."""
import copy
from types import SimpleNamespace
import numpy as np,pytest
import unown_corrected_ppo_core as core

def test_released_loader_cannot_override_registered_thread_backend(monkeypatch):
 def loader():core.torch.set_num_threads(2);return object()
 monkeypatch.setattr(core,'_load_model',loader)
 core.load_model();assert core.torch.get_num_threads()==1
def test_three_zero_fresh_cohorts_and_reset_after_progress():
 streak=core.ZeroCohorts()
 assert [streak.record(n) for n in [0,0,1,0,0,0]]==[False,False,False,False,False,True]
def test_balanced_fresh_cohort_schedule_deterministic():
 a=core.cohort_jobs(0);b=core.cohort_jobs(1)
 assert a==core.cohort_jobs(0) and len(a)==16
 assert [sum(j['learner_seat']==s for j in a) for s in [0,1]]==[8,8]
 assert {j['id'] for j in a}.isdisjoint(j['id'] for j in b)
@pytest.mark.parametrize('phase_pre,phase_post,accepted',[(0.,0.,True),(0.,.031,False),(.021,0.,False)])
def test_phase_only_guards_and_exact_rollback(monkeypatch,phase_pre,phase_post,accepted,tmp_path):
 batch=core.batch;state={'value':0};execution={'step':7,'phase':1};calls=[]
 monkeypatch.setattr(batch,'PRIVATE',tmp_path);monkeypatch.setattr(batch,'OUT',tmp_path)
 def snap(m,o,e):return dict(model=dict(state),execution=copy.deepcopy(e))
 def restore(m,o,s,e):state.clear();state.update(s['model']);e.clear();e.update(copy.deepcopy(s['execution']))
 def metrics(m,r):calls.append(1);return dict(phase_KL=phase_pre if len(calls)==1 else phase_post,initialization_KL=999.)
 def step(*a,**k):state['value']=1;return {'policy_loss':0.}
 monkeypatch.setattr(batch,'snapshot',snap);monkeypatch.setattr(batch,'restore',restore);monkeypatch.setattr(batch,'diagnose',metrics);monkeypatch.setattr(batch,'step',step);monkeypatch.setattr(batch.scalar,'state_fingerprints',lambda x:x)
 result=batch.transaction(None,None,[1],[1],execution,'test')
 assert result['accepted']==accepted
 assert state['value']==int(accepted) and execution['step']==(8 if accepted else 7)
 if phase_post>.03:assert result['rollback_exact'] and (tmp_path/'test-rejected.pt').exists()
def test_corrected_collection_stores_exact_policy_packet_without_rebasing():
 # Use the real wrapper, with a forced-free toy actor base to expose the
 # original packet capture pitfall independently of model/numerical machinery.
 from unown_start_order_actor import StartingOrder
 original_class=core.actor_class;original_runtime=core.runtime
 class Parent:
  def __init__(self,model,*a,**kw):self.records=[];self.candidate=True;self.order=StartingOrder()
  def act(self,raw):
   self.order.observe(raw);p=SimpleNamespace(row=np.zeros(80,np.float32),layout=SimpleNamespace(offsets={'global_f':(10,74)}));p.row[41]=raw['current']['yourIndex'];self.records.append(dict(packet=p,old_probs=np.array([.2,.8],np.float32),log_probability=-.22314355,seat=raw['current']['yourIndex']));return [1]
 try:
  core.actor_class=lambda rt,ctx:Parent
  ctx=dict(candidate_model=None,actors=[]);cls=core.corrected_actor_class(ctx);actor=cls(object(),collect=True)
  actor.act(dict(current=dict(round=2,yourIndex=1,firstPlayer=1)))
  r=actor.records[0];assert r['packet'].row[41]==0 and r['seat']==r['physical_seat']==1
  assert r['log_probability']==-.22314355 and r['old_probs'].tolist()==pytest.approx([.2,.8])
 finally:core.actor_class=original_class
