import collections,copy
import numpy as np
import pytest
from unown_belief_full_training.common import *

def synthetic():
 N,Q,S=2,3,18;A=np.zeros((N,Q,3,S),np.float32);A[...,0]=.6;A[...,1]=.4
 return dict(global_X=np.zeros((N,153),np.float32),local_X=np.zeros((N,Q,77),np.float32),sequence=np.zeros((6,140),np.float32),reset=np.array([1,0,0,1,0,0],bool),selected=np.array([2,5]),A=A,lower=np.zeros((N,Q,3),np.uint8),upper=np.ones((N,Q,3),np.uint8),deck_A=np.full((N,37),1/37,np.float32),games=np.array([1,2]),known=np.zeros((N,Q),np.uint8),zone_sizes=np.ones((N,3)),ledger=np.zeros((N,143)),conditionals=np.empty(0),conditional_indices=np.empty(0),conditional_offsets=np.array([0,0,0]))

def test_full_manifest_retains_every_committed_train_record_and_weights():
 full=read(OUT/'full-training-manifest.json')['entries'];index=read(OUT/'dataset-index.json');assert len(full)==2334 and {x['id'] for x in full}=={x['id'] for x in index if x['partition']=='training'}
 totals=collections.Counter()
 for x in full:totals[x['opponent_list'],x['chooser']]+=x['neural_loss_weight']/2334
 for (d,s),w in totals.items():assert w==pytest.approx(prior()[d]['prior']/2)
 assert sum(x['neural_loss_weight'] for x in full)==pytest.approx(2334)
 assert all(x['neural_loss_weight']==pytest.approx(37*prior()[x['opponent_list']]['prior']) for x in full if x['stratum_count']==2334/74)

def test_tree_original_schedule_full_exposure_and_balanced_batches():
 entries={x['id']:x for x in read(OUT/'full-training-manifest.json')['entries']}
 for seed in (11,29):
  s=read(OUT/f'tree-schedule-s{seed}.json');exposure=collections.Counter(i for stage in s['stages'] for i in stage)
  assert len(s['stages'])==32 and set(exposure)==set(entries) and min(exposure.values())>=2 and exposure==s['exposure_counts']
  for stage in s['stages']:assert collections.Counter((entries[i]['opponent_list'],entries[i]['chooser']) for i in stage)=={(d,c):4 for d in prior() for c in (0,1)}

def test_bayesian_equal_chooser_weights_and_balanced_parity():
 from unown_belief_full_training.fit import fit_theta
 from unown_belief_bayes_repair.deck import fit_theta as old
 counts=np.array([[30.,20.,10.],[20.,20.,20.]]);x=np.array([[1.,0.,0.],[1.,0.,0.],[0.,2.,0.],[0.,1.,1.]]);y=[0,0,0,1]
 actual=fit_theta(x,y,counts,30,[.75,.75,1.5,1]);expected=(np.array([[1.5,3,0],[0,1,1]])+30*counts/60)/np.array([[34.5],[32.]])
 np.testing.assert_allclose(actual,expected)
 np.testing.assert_array_equal(fit_theta(x,y,counts,30),old(x,y,counts,30))

@pytest.mark.parametrize('family',['MLP','GRU'])
def test_architecture_forward_and_loss_unchanged(family):
 import torch
 from unown_belief_full_training.models import make_neural,neural_loss
 from unown_belief_learning_source_repair.models import make_neural as old_make,neural_loss as old_loss
 a=make_neural(family,11);b=old_make(family,11);assert sum(p.numel() for p in a.parameters())==dict(MLP=57691,GRU=177755)[family]
 for k,v in a.state_dict().items():torch.testing.assert_close(v,b.state_dict()[k],rtol=0,atol=0)
 f=synthetic();t=dict(deck=np.array([0,1]),counts=np.zeros((2,3,3),int));torch.testing.assert_close(neural_loss(a,f,t),old_loss(b,f,t),rtol=0,atol=0)

def test_no_locked_test_in_training_validation_or_timing():
 from unown_belief_full_training.storage import load
 entries=read(OUT/'split-manifest.json')['entries'];train={x['id'] for x in entries if x['partition']=='training'};val={x['id'] for x in entries if x['partition']=='validation'};test={x['id'] for x in entries if x['partition']=='locked-test'}
 assert len(train)==2334 and len(val)==len(test)==592 and not train&val and not train&test and not val&test
 assert set(read(OUT/'online-timing-manifest.json')['ids'])<=val
 for row in read(OUT/'dataset-index.json'):
  if row['partition']=='locked-test':
   with pytest.raises(PermissionError):load(row,'legal')

def test_paired_full_minus592_uses_same_whole_matches():
 from unown_belief_full_training.analysis import summarize
 from unown_belief_full_training.scoring import METRICS
 entries=[x for x in read(OUT/'split-manifest.json')['entries'] if x['partition']=='validation'];results=[]
 for key,shift in [('A',0),('MLP-n16-s11',-.01),('MLP-n16-s29',-.01),('MLP-nfull-s11',-.02),('MLP-nfull-s29',-.02)]:
  records=[]
  for i,x in enumerate(entries):
   metrics={m:.2+(i%7)/100+shift for m in METRICS};metrics.update({f'bin{j}_{v}':0 for j in range(10) for v in ('count','pred','label')});records.append(dict(id=x['id'],opponent_list=x['opponent_list'],chooser=x['chooser'],metrics=metrics,by_game={'1':metrics},by_stage={'early':metrics}))
  results.append(dict(key=key,state='COMPLETE',completed_BO3=592,records=records))
 a=summarize(results,100)
 for pop in ('eligible','full_field'):
  c=a['minus_592'][pop]['MLP-nfull-seedmean']['uncertain_brier'];assert c['difference']==pytest.approx(-.01) and c['CI95']==pytest.approx([-.01,-.01])

def test_caps_and_no_native_entrypoint():
 from unown_belief_full_training import run
 jobs=read(OUT/'fit-schedule.json');assert len(jobs)==7 and sum(fit_budget(j) for j in jobs)==31200 and FIT_CAP==36000 and CPU_CAP==72000 and ELAPSED_CAP==86400
 assert not hasattr(run,'collect_one') and not(R/'scripts/unown_belief_full_training/worker.py').exists()


def test_finished_report_uses_all_data_and_discloses_regressions(monkeypatch,tmp_path):
 import shutil
 import unown_belief_full_training.report_full as report
 fake=tmp_path/'experiments'/'report-fixture';fake.mkdir(parents=True);(tmp_path/'docs').mkdir();(tmp_path/'docs/STATUS.md').write_text('historical status')
 for name in ['manifest-audit.json','registration.json','stratum-counts.json','tree-schedule-s11.json','tree-schedule-s29.json']:shutil.copy2(OUT/name,fake/name)
 # Use frozen historical metrics as report fixture only, never as full-data evidence.
 a=read(PARENT/'analysis.json');a['minus_592']={'eligible':{},'full_field':{}}
 write(fake/'analysis.json',a);monkeypatch.setattr(report,'R',tmp_path);monkeypatch.setattr(report,'OUT',fake);monkeypatch.setattr(report,'FULL_REPORT',tmp_path/'report.md')
 report.render('COMPLETE','Synthetic report fixture',mirror=tmp_path/'mirror')
 text=(tmp_path/'report.md').read_text();assert 'all2,334 valid training' in text and 'locked-test' in text
 assert 'historical status' in (tmp_path/'docs/STATUS.md').read_text()
