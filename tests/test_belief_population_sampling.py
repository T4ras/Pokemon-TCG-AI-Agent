import copy
import numpy as np
import pytest
from unown_belief_population_sampling.common import *
from unown_belief_population_sampling.metrics import score
from unown_belief_full_training.models import make_neural,neural_loss
from unown_belief_deck_audit.diagnostics import components,initial_counts,digest
from test_belief_deck_audit import fixture

def test_expected_population_coefficients_identical():
 train=entries('training');n=collections.Counter((r['opponent_list'],r['chooser']) for r in train)
 for r in train:assert r['neural_loss_weight']/2334==pytest.approx(prior()[r['opponent_list']]['prior']/(2*n[r['opponent_list'],r['chooser']]),abs=1e-15)
 assert sum(r['neural_loss_weight'] for r in train)/2334==pytest.approx(1)

def test_frozen_schedules_exact_budget_and_eligible_pool():
 schedule=read(OUT/'schedules.json');ids={r['id'] for r in entries('training')}
 for j in read(OUT/'jobs.json'):
  s=schedule[j['key']];assert len(s)==14004 and set(s)<=ids and j['sampling_seed']!=j['seed']
  if j['arm']=='control':
   for start in range(0,14004,2334):assert len(set(s[start:start+2334]))==2334
 assert read(OUT/'sampler-check.json')['passed']

@pytest.mark.parametrize('seed',[11,29])
def test_initial_models_and_fresh_optimizer_states_identical(seed):
 import torch
 a,b=make_neural('MLP',seed),make_neural('MLP',seed);assert digest(a.state_dict())==digest(b.state_dict())
 oa=torch.optim.AdamW(a.parameters(),lr=.001,weight_decay=.0001);ob=torch.optim.AdamW(b.parameters(),lr=.001,weight_decay=.0001)
 assert oa.state_dict()==ob.state_dict() and oa.state_dict()['state']=={}
 f,t=fixture();deck,count,_,_=components(a,f,t);torch.testing.assert_close(neural_loss(a,f,t),deck+count)
 # No optimizer step is performed by this test.

def test_secondary_selection_includes_initial_and_ties_earlier():
 row=lambda u,v:dict(updates=u,summary=dict(populations=dict(full_field=dict(metrics=dict(objective_NLL=v)))))
 assert choose_secondary([row(2334,2),row(0,1),row(14004,1)])['updates']==0
 assert choose_secondary([row(0,2),row(2334,1),row(14004,.5)])['updates']==14004

def test_test_partition_and_unknown_ids_rejected():
 with pytest.raises(PermissionError):entries('locked-test')
 with pytest.raises(PermissionError):load('not-in-frozen-non-test-pool')
 assert len(entries('training'))==2334 and len(entries('validation'))==592

def test_all_four_objectives_and_zone_metrics_unchanged():
 from unown_belief_full_training.fit import predict
 f,t=fixture();d,p=predict(make_neural('MLP',11),f,'MLP');m=score(f,t,d,p)
 assert m['count_NLL']==pytest.approx(sum(m['count_NLL_'+z] for z in ('hand','draw_pile','prizes'))/3,abs=5e-6)
 assert m['all_brier']==pytest.approx(m['presence_hand'])
 assert m['objective_NLL']==pytest.approx(m['deck_log_loss']+m['count_NLL'])


def test_end_to_end_analysis_keeps_final_primary_and_selection_secondary(monkeypatch,tmp_path):
 import unown_belief_population_sampling.analysis as analysis
 valid=entries('validation');(tmp_path/'validation').mkdir();jobs=read(OUT/'jobs.json');write(tmp_path/'jobs.json',jobs)
 def records(value):
  return [dict(id=r['id'],opponent_list=r['opponent_list'],chooser=r['chooser'],metrics={k:(value if k=='deck_log_loss' else .2) for k in analysis.ALL_METRICS}) for r in valid]
 write(tmp_path/'baseline-validation.json',dict(records=records(1.5)))
 for j in jobs:
  for u in CHECKPOINTS:
   value=.5 if u==0 else (2. if j['arm']=='control' else 1.)
   write(tmp_path/'validation'/f'{j["key"]}-u{u:05d}.json',dict(updates=u,BO3=592,records=records(value),summary=dict(populations=dict(full_field=dict(metrics=dict(objective_NLL=value))))))
 monkeypatch.setattr(analysis,'OUT',tmp_path);monkeypatch.setattr(analysis,'entries',lambda p:valid);monkeypatch.setattr(analysis,'plot',lambda a:None)
 result=analysis.run(replicates=32)
 assert result['primary_updates']==14004
 for pop in ('full_field','eligible'):
  c=result['populations'][pop]['contrasts']['population_minus_control_seedmean']['deck_log_loss']
  assert c['difference']==pytest.approx(-1) and c['CI95']==pytest.approx([-1,-1])
 assert all(r['updates']==0 for r in result['selection'].values())
 assert not result['conclusion']['treatment_vs_initial']['improvement_established']
 assert not result['test_access']
