import itertools,collections,copy,json
import numpy as np
import pytest
from unown_belief_learning_source_repair.common import *
from unown_belief_learning_source_repair.distributions import allocation,moments
from unown_belief_bayes_repair.baseline import Facts
from unown_belief_bayes_repair.allocation import allocation_one

def facts(h,d,p,basic={1}):
 f=Facts(metadata()['cards']);f.basic=basic;f.sizes=dict(H=h,D=d,P=p,L=0);return f

@pytest.mark.parametrize('condition',[None,True,False])
@pytest.mark.parametrize('cards',[[1,1,2,2],[1,2,2,3]])
def test_exact_count_distributions(condition,cards):
 f=facts(2,1,1);f.setup_basic=condition;q=sorted(set(cards));a=allocation(f,cards,q,5);expected=np.zeros_like(a);n=0
 for order in itertools.permutations(range(4)):
  H=order[:2];D=order[2:3];P=order[3:]
  if condition is not None and any(cards[i] in f.basic for i in H)!=condition:continue
  n+=1
  for i,c in enumerate(q):
   for z,zone in enumerate((H,D,P)):expected[i,z,sum(cards[j]==c for j in zone)]+=1
 np.testing.assert_allclose(a,expected/n,atol=1e-10)
 ref=allocation_one(f,cards,q);np.testing.assert_allclose(moments(a),[ref[c] for c in q],atol=1e-10)

def test_basic_energy_above_four_and_known_hand():
 f=facts(5,10,2,basic={1});a=allocation(f,[1]*17,[1],18);assert a[0,0,5]==pytest.approx(1) and a[0,1,10]==pytest.approx(1)
 f=facts(2,1,1);f.copies={9:(1,frozenset({'H'}))};a=allocation(f,[1,1,2,2],[1,2],5);ref=allocation_one(f,[1,1,2,2],[1,2]);np.testing.assert_allclose(moments(a),[ref[c] for c in (1,2)],atol=1e-10)

def test_ambiguous_domain_and_hidden_basic():
 for hidden in (0,1):
  f=facts(1,1,1,basic={1,2});cards=[1,2,2,3] if hidden else [1,2,3];f.unknown_basic=hidden
  if not hidden:f.copies={10:(1,frozenset({'H','D'}))}
  a=allocation(f,cards,[1,2,3],5);ref=allocation_one(f,cards,[1,2,3]);np.testing.assert_allclose(moments(a),[ref[c] for c in (1,2,3)],atol=1e-10)

def synthetic():
 from unown_belief_learning_source_repair.features import GLOBAL_DIM,EVENT_DIM
 S=18;N=2;Q=3;A=np.zeros((N,Q,3,S),np.float32);A[...,0]=.6;A[...,1]=.4
 return dict(global_X=np.zeros((N,153),np.float32),local_X=np.zeros((N,Q,77),np.float32),sequence=np.zeros((6,140),np.float32),reset=np.array([1,0,0,1,0,0],bool),selected=np.array([2,5]),A=A,lower=np.zeros((N,Q,3),np.uint8),upper=np.ones((N,Q,3),np.uint8),deck_A=np.full((N,37),1/37,np.float32),games=np.array([1,2]),known=np.zeros((N,Q),np.uint8),zone_sizes=np.ones((N,3)),ledger=np.zeros((N,143)),conditionals=np.empty(0),conditional_indices=np.empty(0),conditional_offsets=np.array([0,0,0]))

@pytest.mark.parametrize('family',['MLP','GRU'])
def test_neural_shapes_loss_and_legal_support(family):
 from unown_belief_learning_source_repair.models import make_neural,neural_predict,neural_loss
 model=make_neural(family,11);f=synthetic();d,p=neural_predict(model,f);assert d.shape==(2,37) and p.shape==(2,3,3,18);assert sum(x.numel() for x in model.parameters())<1000000
 np.testing.assert_allclose(p.sum(-1),1,atol=1e-6);assert p[...,2:].sum()==0
 t=dict(deck=np.array([0,1]),counts=np.zeros((2,3,3),int));loss=neural_loss(model,f,t);loss.backward();assert np.isfinite(float(loss.detach()))

def test_gru_resets_and_keeps_early_prefix():
 import torch
 from unown_belief_learning_source_repair.models import make_neural,neural_predict
 model=make_neural('GRU',11)
 with torch.no_grad():model.deck.weight.normal_(0,.2);model.query[-1].weight.normal_(0,.2)
 f=synthetic();a=neural_predict(model,f);g=copy.deepcopy(f);g['sequence'][0]=10;b=neural_predict(model,g)
 np.testing.assert_allclose(a[0][1],b[0][1],atol=0);np.testing.assert_allclose(a[1][1],b[1][1],atol=0);assert not np.allclose(a[0][0],b[0][0],atol=1e-9)

def test_tree_shared_inputs_and_one_proper_gradient_stage():
 from unown_belief_learning_source_repair.models import BoostedTrees,tree_inputs
 f=synthetic();positions,X=tree_inputs(f);i,q,z=positions.T;np.testing.assert_array_equal(X[:,:153],f['global_X'][i]);np.testing.assert_array_equal(X[:,153:230],f['local_X'][i,q])
 m=BoostedTrees(11);t=dict(deck=np.array([0,1]),counts=np.zeros((2,3,3),int));items=[dict(id='a',opponent_list=next(iter(prior())))];m.stage(items,0,lambda i:(f,t),lambda:None);d,p=m.predict(f);np.testing.assert_allclose(d.sum(-1),1,atol=1e-6);np.testing.assert_allclose(p.sum(-1),1,atol=1e-6);assert p[...,2:].sum()==0

def test_match_weighting_not_length():
 from unown_belief_learning_source_repair.models import point_weights
 np.testing.assert_allclose(point_weights(np.array([1,1,2])),[.25,.25,.5])

def test_shard_roundtrip_and_locked_loader(monkeypatch,tmp_path):
 import unown_belief_learning_source_repair.storage as s
 monkeypatch.setattr(s,'R',tmp_path);monkeypatch.setattr(s,'OUT',tmp_path/'out');monkeypatch.setattr(s,'PRIVATE',tmp_path/'private');sh=s.Shards()
 sh.append(dict(legal=dict(id='m',entries=[dict(observation={})]),targets=dict(id='m',points=[dict(key=[1,0,1,0])]),provenance=dict(id='m',partition='training')));sh.flush_all();r=sh.committed[0];assert s.load(r,'legal')['id']=='m';assert s.load(r,'targets')['points'][0]['key']==[1,0,1,0]
 locked=dict(r,partition='locked-test')
 with pytest.raises(PermissionError):s.load(locked,'targets')
 with pytest.raises(PermissionError):s.load(locked,'legal')

def test_splits_nested_balanced_and_reserved():
 entries=read(OUT/'split-manifest.json')['entries'];byid={x['id']:x for x in entries};assert len(byid)==5920;assert collections.Counter(x['partition'] for x in entries)==dict(training=4736,validation=592,**{'locked-test':592})
 assert all(x['partition']=='training' for x in entries if x['reused']);reserved=read(OUT/'reserved-ID-reconciliation.json')['ids'];assert all(byid[i]['partition']=='locked-test' for i in reserved)
 subsets=read(OUT/'training-subsets.json');prev=set()
 for n in SIZES:
  ids=set(subsets[str(n)]);assert len(ids)==37*n and prev<=ids;assert collections.Counter((byid[i]['opponent_list'],byid[i]['chooser']) for i in ids)=={(d,s):n//2 for d in prior() for s in (0,1)};prev=ids

def test_scoring_schema_and_uncertainty_subset():
 from unown_belief_learning_source_repair.scoring import decision_scores,METRICS
 f=synthetic();d=f['deck_A'];p=f['A'];t=dict(deck=np.array([0,1]),counts=np.zeros((2,3,3),int));a,b=decision_scores(f,t,d,p);assert a.shape==(2,len(METRICS));assert a[:,0]==pytest.approx([.16,.16]);assert a[:,1]==pytest.approx([.16,.16])

def test_legal_encoder_rejects_hidden_information_and_has_fixed_dimensions(monkeypatch):
 import unown_belief_learning_source_repair.features as e
 monkeypatch.setattr(e,'check_time',lambda:None)
 o=dict(current=dict(round=1,yourIndex=0,turn=0,turnActionCount=1,firstPlayer=-1,players=[dict(deckCount=60,handCount=0,hand=[],prize=[],active=[],bench=[],discard=[]),dict(deckCount=60,handCount=0,hand=None,prize=[],active=[],bench=[],discard=[])],stadium=[],looking=None),select=dict(context=41,minCount=1,maxCount=1,option=[{},{}]),logs=[])
 hist=[dict(observation=o,action=[0])];a,info=e.encode(hist,{'0':{}},True);x=copy.deepcopy(hist);x[0]['observation']['current']['players'][1]['hand']=[dict(id=999,serial=999)];x[0]['truth']='forbidden';b,_=e.encode(x,{'0':{}},True)
 for k in a:np.testing.assert_array_equal(a[k],b[k])
 assert a['global_X'].shape==(1,153) and a['local_X'].shape==(1,143,77) and a['sequence'].shape==(1,140)


def test_streaming_and_full_prefix_neural_equivalence():
 import torch
 from unown_belief_learning_source_repair.models import make_neural,streaming_predict
 from unown_belief_learning_source_repair.fit import predict
 f=synthetic();model=make_neural('GRU',11)
 with torch.no_grad():model.deck.weight.normal_(0,.1);model.query[-1].weight.normal_(0,.1)
 full=predict(model,f,'GRU');d,p,updates,readouts=streaming_predict(model,f,'GRU',predict)
 np.testing.assert_allclose(d,full[0],rtol=1e-5,atol=1e-7);np.testing.assert_allclose(p,full[1],rtol=1e-5,atol=1e-7);assert len(updates)==6 and len(readouts)==2

def test_frozen_uncertainty_mask_wins_over_storage_roundoff():
 from unown_belief_learning_source_repair.scoring import decision_scores
 f=synthetic();f['uncertain']=np.zeros((2,3),bool);t=dict(deck=np.array([0,1]),counts=np.zeros((2,3,3),int));a,_=decision_scores(f,t,f['deck_A'],f['A']);assert np.isnan(a[:,0]).all()

def test_paired_bootstrap_seedmean_keeps_whole_matches():
 from unown_belief_learning_source_repair.analysis import summarize
 from unown_belief_learning_source_repair.scoring import METRICS
 entries=[x for x in read(OUT/'split-manifest.json')['entries'] if x['partition']=='validation'];results=[]
 for method,delta in (('A',0),('MLP-n8-s11',-.01),('MLP-n8-s29',-.01)):
  rec=[]
  for i,x in enumerate(entries):
   m={k:.1 for k in METRICS};m['uncertain_brier']=.2+(i%8)/100+delta;m.update({f'bin{j}_{v}':0 for j in range(10) for v in ('count','pred','label')});rec.append(dict(id=x['id'],opponent_list=x['opponent_list'],chooser=x['chooser'],metrics=m,by_game={'1':m},by_stage={'early':m}))
  results.append(dict(key=method,state='COMPLETE',completed_BO3=592,records=rec))
 a=summarize(results,200)
 for pop in ('eligible','full_field'):
  v=a['minus_A'][pop]['MLP-n8-seedmean']['uncertain_brier'];assert v['difference']==pytest.approx(-.01);assert v['CI95']==pytest.approx([-.01,-.01])
 assert a['validation_BO3']==592 and a['locked_test_scored'] is False


def test_complete_fit_caps_fit_aggregate_and_controller_imports():
 import unown_belief_learning_source_repair.run as controller
 jobs=read(OUT/'fit-schedule.json');assert len(jobs)==14;assert sum(fit_budget(j) for j in jobs)+1000<FIT_CAP
 for n in SIZES:assert fit_budget(dict(family='MLP',size=n))==fit_budget(dict(family='GRU',size=n))
 assert callable(controller.feasibility) and callable(controller.neural_workload_probe)


def test_transient_atomic_write_retries_without_restarting_work(monkeypatch):
 import unown_belief_learning_source_repair.common as c
 calls=[]
 def transient(path,value):
  calls.append((path,value))
  if len(calls)<3:raise PermissionError('transient sharing violation')
 monkeypatch.setattr(c,'_atomic_write',transient);monkeypatch.setattr(c.time,'sleep',lambda _:None);c.write('telemetry',{'CPU':1});assert len(calls)==3


def test_repaired_allocator_exact_old_measure_above_old_cap(monkeypatch):
 from unown_belief_learning_source_repair import baseline as repaired
 from unown_belief_bayes_repair import baseline as old
 monkeypatch.setattr(repaired,'check_time',lambda:None)
 groups=[((('D','H','P'),()),collections.Counter({j+1:4})) for j in range(4)]
 capacity=dict(D=22,H=6,P=6,L=0)
 with pytest.raises(UnsupportedEvent):old.assignments(groups,capacity)
 expected=old.assignments(groups,capacity,cap=1000000)
 actual=repaired.assignments(groups,capacity)
 assert actual==expected
 assert sum(x[0] for x in actual)==pytest.approx(1)

def test_repaired_allocator_constraints_and_resource_guard(monkeypatch):
 from unown_belief_learning_source_repair import baseline as repaired
 from unown_belief_bayes_repair import baseline as old
 monkeypatch.setattr(repaired,'check_time',lambda:None)
 groups=[((('D','H'),(0,)),collections.Counter({1:2})),((('D','H','P'),(0,)),collections.Counter({2:2}))]
 constraints=[(frozenset(range(4)),'H',2,3)]
 capacity=dict(D=3,H=3,P=1,L=0)
 assert repaired.assignments(groups,capacity,constraints=constraints)==old.assignments(groups,capacity,1000000,constraints)
 with pytest.raises(UnsupportedEvent):repaired.assignments(groups,capacity,cap=1,constraints=constraints)

def test_recovery_never_reissues_old_attempts_and_same_subsets():
 reg=read(OUT/'recovery-registration.json');old=R/'experiments'/reg['parent'];started={p.stem for p in (old/'native-starts').glob('*.json')};jobs=[j for b in read(OUT/'collection-schedule.json') for j in b['jobs']]
 assert len(jobs)==67 and len({j['id'] for j in jobs})==67
 assert not {j['id'] for j in jobs}&started
 assert read(OUT/'training-subsets.json')==read(old/'training-subsets.json')
 assert all(not set(read(OUT/'training-subsets.json')[str(n)])&set(reg['lost_ids']) for n in SIZES)
 assert set(reg['lost_ids'])<=set(read(OUT/'training-subsets.json')['128'])
 assert reg['no_native_retries'] and not reg['actor_changes']


def test_durable_result_survives_until_shard_commit(monkeypatch,tmp_path):
 import unown_belief_learning_source_repair.storage as s
 monkeypatch.setattr(s,'R',tmp_path);monkeypatch.setattr(s,'OUT',tmp_path/'out');monkeypatch.setattr(s,'PRIVATE',tmp_path/'private')
 journal=tmp_path/'private/pending-results/m.json.gz';journal.parent.mkdir(parents=True);journal.write_bytes(b'recoverable-result')
 sh=s.Shards();sh.append(dict(legal=dict(id='m',entries=[{}]),targets=dict(id='m',points=[{}]),provenance=dict(id='m',partition='training')))
 assert journal.exists() and not sh.committed
 sh.flush_all();assert not journal.exists() and sh.committed[0]['id']=='m'

def test_eligible_recovery_union_is_exact_and_original_plan_incomplete():
 committed={r['id'] for r in read(OUT/'inherited-dataset-index.json')};remaining={j['id'] for b in read(OUT/'collection-schedule.json') for j in b['jobs']};final=committed|remaining
 assert len(final)==3518 and not committed&remaining
 entries=read(OUT/'split-manifest.json')['entries']
 assert {e['id'] for e in entries if e['partition'] in ('validation','locked-test')}<=final
 subsets=read(OUT/'training-subsets.json')
 assert all(set(subsets[str(n)])<=final for n in SIZES)
 assert not set(subsets['128'])<=final


def attachment_obs(hand,deck,logs):
 return dict(current=dict(round=1,yourIndex=1,turn=14,turnActionCount=1,firstPlayer=0,players=[dict(deckCount=deck,handCount=hand,prize=[],active=[],bench=[],discard=[]),dict(deckCount=50,handCount=7,hand=[],prize=[],active=[],bench=[],discard=[])],stadium=[],looking=None),select=dict(context=1,minCount=1,maxCount=1,option=[{}]),logs=logs)

def test_crispin_exhausted_deck_then_manual_attachment():
 from unown_belief_learning_source_repair.baseline import Facts
 f=Facts(metadata()['cards']);f.game=1;f.begun=True;f.sizes=dict(D=1,H=2,P=0,L=0);f.crispin_pending=True
 f.transfer('D','H',5,101);assert f.sizes['D']==0 and not f.crispin_pending
 # No Shuffle event is emitted by the official empty-deck path. Serial12 was
 # not known previously; the next ordinary attachment must consume the hand.
 o=attachment_obs(2,0,[dict(type=11,playerIndex=0,cardId=7,serial=12,cardIdTarget=1071,serialTarget=29)])
 f.observe(o);assert f.sizes['H']==2 and f.sizes['D']==0 and f.copies[12][1]=={'B'}

def test_crispin_direct_last_deck_attachment_preserved():
 from unown_belief_learning_source_repair.baseline import Facts
 f=Facts(metadata()['cards']);f.game=1;f.begun=True;f.sizes=dict(D=1,H=2,P=0,L=0);f.crispin_pending=True
 f.observe(attachment_obs(2,0,[dict(type=11,playerIndex=0,cardId=7,serial=12,cardIdTarget=1071,serialTarget=29)]))
 assert f.sizes['D']==0 and f.sizes['H']==2 and not f.crispin_pending

def test_negative_source_guard_remains_strict():
 from unown_belief_learning_source_repair.baseline import Facts
 f=Facts(metadata()['cards']);f.sizes=dict(D=0,H=1,P=0,L=0)
 with pytest.raises(InvariantFailure,match='negative source count'):f.transfer('D','H',5,101)

def test_correctness_drain_preserves_started_results_without_dispatch(monkeypatch,tmp_path):
 import concurrent.futures as cf
 import unown_belief_learning_source_repair.run as controller
 monkeypatch.setattr(controller,'OUT',tmp_path);seen=[]
 def record(result,shards):
  seen.append(result['id'])
  if result['status']=='FAILED':raise InvariantFailure('saved independent failure')
 monkeypatch.setattr(controller,'record_result',record);monkeypatch.setattr(controller,'check',lambda _:None)
 class Shards:
  flushed=False
  def flush_all(self):self.flushed=True
 a=cf.Future();a.set_result(dict(id='a',status='FAILED'));b=cf.Future();b.set_result(dict(id='b',status='DONE'));pending={a:dict(id='a'),b:dict(id='b')};sh=Shards()
 records=controller.drain_started(pending,sh,None)
 assert seen==['a','b'] and not pending and sh.flushed
 assert records[0]['status']=='FAILED' and records[1]['status']=='COMMITTED'
 assert not(tmp_path/'STOP.json').exists() and not(tmp_path/'issued').exists()
