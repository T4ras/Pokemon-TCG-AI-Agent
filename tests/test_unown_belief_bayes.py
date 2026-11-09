import itertools,collections,copy,json
import numpy as np
import pytest
from unown_belief_bayes.common import *
from unown_belief_bayes.baseline import Facts,baseline_one
from unown_belief_bayes.allocation import allocation_one
from unown_belief_bayes.deck import fit_theta,deck_weights,mix_allocations,catalogue

def facts(H,D,P,basic={1}):
 f=Facts(metadata()['cards']);f.basic=set(basic);f.sizes=dict(H=H,D=D,P=P,L=0);return f

def reference(cards,H,D,P,basic,condition=None,known_token=None):
 values=[]
 for perm in itertools.permutations(range(len(cards))):
  hh=perm[:H];dd=perm[H:H+D];pp=perm[H+D:]
  if condition is not None and any(cards[t] in basic for t in hh)!=condition:continue
  if known_token is not None and known_token not in hh+dd:continue
  values.append([collections.Counter(cards[t] for t in zone) for zone in (hh,dd,pp)])
 return {c:[sum(x[0][c]>0 for x in values)/len(values)]+[float(np.mean([x[z][c] for x in values])) for z in range(3)]+[float(np.var([x[z][c] for x in values])) for z in range(3)] for c in set(cards)}

@pytest.mark.parametrize('cards',[[1,1,2,2],[1,2,2,3]])
@pytest.mark.parametrize('condition',[None,True,False])
def test_moments_against_full_physical_enumeration(cards,condition):
 f=facts(2,1,1);f.setup_basic=condition;actual=allocation_one(f,cards,sorted(set(cards)));expected=reference(cards,2,1,1,{1},condition)
 old=baseline_one(f,cards,sorted(set(cards)))
 for c,v in actual.items():
  assert v==pytest.approx(expected[c],abs=1e-12);assert v[:4]==pytest.approx([old[c][k] for k in ('p_unknown_hand','hand_unknown_mean','deck_mean','prize_mean')],abs=1e-12)

def test_ambiguous_known_copy_constraints_and_variance():
 cards=[1,2,2,3];f=facts(1,2,1);f.copies={77:(1,frozenset({'H','D'}))};a=allocation_one(f,cards,[1,2,3]);ref=reference(cards,1,2,1,{1},known_token=0)
 for c in a:assert a[c]==pytest.approx(ref[c],abs=1e-12)

def test_hidden_basic_slot_mixture_preserves_mean_and_between_variance():
 f=facts(1,1,1,basic={1,2});f.unknown_basic=1;cards=[1,2,2,3];a=allocation_one(f,cards,[1,2,3]);old=baseline_one(f,cards,[1,2,3]);samples=[]
 for p in itertools.permutations(range(4)):
  if cards[p[0]] in f.basic:samples.append(cards[p[1]])
 for c in a:
  assert a[c][:4]==pytest.approx([old[c][k] for k in ('p_unknown_hand','hand_unknown_mean','deck_mean','prize_mean')]);assert a[c][4]==pytest.approx(np.var([int(x==c) for x in samples]))

def test_known_hand_copy_is_excluded_from_additional_presence():
 f=facts(1,2,1);f.copies={77:(1,frozenset({'H'}))};a=allocation_one(f,[1,1,2,2],[1]);assert a[1][0]==0 and a[1][1]==0 and a[1][4]==0

def test_theta_matches_registered_formula_and_no_labelled_impossible_card():
 decks=np.array([[30,30],[60,0]],dtype=float);x=np.array([[3,1],[0,2],[4,0]],dtype=float);y=[0,0,1];theta=fit_theta(x,y,decks,30);assert theta[0]==pytest.approx([(3+15)/36,(3+15)/36]);assert theta[1]==pytest.approx([1,0])
 with pytest.raises(InvariantFailure):fit_theta([[0,1]],[1],decks,30)

def test_weight_beta_zero_and_stable_large_ledger():
 theta=np.array([[.9,.1],[.1,.9]]);prior=np.array([.3,.7]);assert deck_weights([0,0],[True,True],prior,theta,1)==pytest.approx(prior)
 assert deck_weights([50,0],[True,True],prior,theta,0)==pytest.approx(prior)
 w=deck_weights([50,0],[True,True],prior,theta,.25);assert np.isfinite(w).all() and sum(w)==pytest.approx(1) and w[0]>.999
 assert deck_weights([1,0],[False,True],prior,theta,1)==pytest.approx([0,1])

def test_total_variance_includes_both_terms():
 a=np.zeros((2,1,7));a[0,0,1]=0;a[1,0,1]=2;a[:,:,4]=.5;v=mix_allocations([.5,.5],a);assert v[0,1]==1 and v[0,4]==pytest.approx(1.5)

def test_four_folds_and_three_inner_folds_are_grouped_and_balanced():
 f=read(OUT/'folds.json');inputs=read(OUT/'inputs.json');byid={x['id']:x for x in inputs};allids=set(byid);assert len(allids)==296
 for i,ids in enumerate(f['outer_folds']):
  assert len(ids)==74;assert collections.Counter((byid[k]['opponent_list'],byid[k]['chooser']) for k in ids)=={(d,s):1 for d in prior() for s in (0,1)}
  training=allids-set(ids);assert len(training)==222
  for j,inner in enumerate(f['outer_folds']):
   if i==j:continue
   assert set(inner)<=training and not set(inner)&set(ids);assert len(training-set(inner))==148

def test_reveal_ledger_max_across_games_and_duplicate_scope():
 from unown_belief_bayes.adapter import legal
 f=facts(0,60,0);o=dict(current=dict(round=1,yourIndex=0,turn=0,turnActionCount=1,players=[dict(deckCount=60,handCount=0,hand=[],prize=[],active=[],bench=[],discard=[]),dict(deckCount=60,handCount=0,hand=None,prize=[],active=[],bench=[],discard=[])],stadium=[],looking=None),select=None,logs=[]);f.observe(o);f.note(1,10,'B');f.note(1,11,'B');assert f.lower[1]==2
 o['current']['round']=2;f.observe(o);f.note(1,10,'B');f.note(1,11,'B');assert f.lower[1]==2
 old=copy.deepcopy(f.lower);assert not f.observe(o);assert f.lower==old
 x=copy.deepcopy(o);x['truth']='forbidden';x['scheduled_opponent']='forbidden';x['current']['players'][1]['hand']=[dict(id=99999,serial=1)];assert legal(x)==legal(o)

def test_metang_and_dwebble_saved_source_regressions():
 import gzip
 from unown_belief_bayes.adapter import Compiler
 byid={x['id']:x for x in read(OUT/'inputs.json')}
 for ident,target,source in (('belief-b6802d5c931cf73960c0',345,'D'),('belief-4028a82a3b2d7716fc7f',8,'L')):
  c=Compiler(metadata()['cards']);seen=False
  for e in json.load(gzip.open(R/byid[ident]['corpus']['legal_path'],'rt')):
   result=c.compile(e['observation'])
   seen |= any(x.get('op')=='move' and x.get('cid')==target and x.get('src')==source for x in result['operations'])
  assert seen

def test_parameter_selection_never_accesses_outer_labels(monkeypatch,tmp_path):
 import unown_belief_bayes.cv as cv
 decks,cards,counts,pop=catalogue();folds=read(OUT/'folds.json')['outer_folds'];inputs=read(OUT/'inputs.json');outer=set(folds[0]);features={x['id']:dict(final_ledger=[0]*len(cards),points=[dict(game=1,lower=[0]*len(cards),compatible=list(range(37)))]) for x in inputs}
 class Labels(dict):
  def __getitem__(self,key):
   if key in outer:raise AssertionError('Outer label leaked into selection')
   return super().__getitem__(key)
 labels=Labels({x['id']:decks.index(x['opponent_list']) for x in inputs});monkeypatch.setattr(cv,'R',tmp_path);monkeypatch.setattr(cv,'OUT',tmp_path);monkeypatch.setattr(cv,'PRIVATE',tmp_path/'private');theta,selection=cv.select_parameters(0,features,labels,folds);assert selection['alpha']==120 and selection['beta']==.25;assert not set(selection['training_ids'])&outer

def test_legal_predictions_and_end_to_end_synthetic_scoring(monkeypatch,tmp_path):
 import gzip
 import unown_belief_bayes.cache as cache
 import unown_belief_bayes.evaluate as evaluate
 cats=lambda:(['a','b'],[1,2],np.array([[60,0],[0,60]],dtype=float),np.array([.5,.5]));p={'a':dict(cards=[1]*60,prior=.5),'b':dict(cards=[2]*60,prior=.5)}
 for m in (cache,evaluate):
  monkeypatch.setattr(m,'OUT',tmp_path);monkeypatch.setattr(m,'PRIVATE',tmp_path/'private');monkeypatch.setattr(m,'R',tmp_path);monkeypatch.setattr(m,'catalogue',cats);monkeypatch.setattr(m,'guard',lambda:None)
 monkeypatch.setattr(cache,'prior',lambda:p)
 o=dict(current=dict(round=1,yourIndex=0,turn=0,turnActionCount=1,players=[dict(deckCount=60,handCount=0,hand=[],prize=[],active=[],bench=[],discard=[]),dict(deckCount=60,handCount=0,hand=None,prize=[],active=[],bench=[],discard=[])],stadium=[],looking=None),select=dict(context=41,minCount=1,maxCount=1,option=[{},{}]),logs=[])
 legal=tmp_path/'legal.gz';truth=tmp_path/'truth.gz'
 with gzip.open(legal,'wt') as f:json.dump([dict(observation=o,action=[0])],f)
 with gzip.open(truth,'wt') as f:json.dump([dict(key=[1,0,1,0],hand={},hand_serials=[],deck={'1':60},prize={})],f)
 write(tmp_path/'query-set.json',dict(card_ids=[1,2],categories={}));e=dict(legal_path=legal.name,legal_sha256=sha(legal),selection={'0':{'game':1}},selected_decisions=1);record=cache.cache_one(dict(id='synthetic',corpus=e));assert record['status']=='COMPLETE',record['error'];features=json.load(gzip.open(tmp_path/record['feature_path'],'rt'));item=dict(id='synthetic',opponent_list='a',chooser=0,outer_fold=0,corpus=dict(e,truth_path=truth.name));out=evaluate.evaluate_one(item,record,np.array([[1.,0.],[0.,1.]]),.5,features);assert out['status']=='COMPLETE' and out['scored']==1;assert out['metrics']['A']['all_in/all/prediction/brier']==out['metrics']['B']['all_in/all/prediction/brier']==0

def test_paired_bootstrap_difference_keeps_shared_indices():
 from unown_belief_bayes.scoring import summarize
 rows=[]
 for d in prior():
  for seat in (0,1):
   for j in range(4):rows.append(dict(partition='development',status='DONE',opponent_list=d,anchor_seat=seat,belief_metrics={'A/all_in/uncertain/prediction/brier':j/10,'B/all_in/uncertain/prediction/brier':j/10-.01,'A/deck/log_loss':j/10+1,'B/deck/log_loss':j/10+.8}))
 s=summarize(rows,replicates=200)
 for field in ('eligible','full_field'):
  assert s['point'][field]['B_minus_A/all_in/uncertain/prediction/brier']==pytest.approx(-.01);assert s['CI95'][field]['B_minus_A/all_in/uncertain/prediction/brier']==pytest.approx([-.01,-.01])
