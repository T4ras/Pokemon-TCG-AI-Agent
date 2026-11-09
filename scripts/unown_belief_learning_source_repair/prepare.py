"""Outcome-independent deficit schedules, nested splits and frozen model rules."""
from .common import *
from unown_belief_bayes_repair.deck import catalogue
import random

def main():
 reuse=read(OUT/'reuse-primary.json');p=prior();allids={x['id'] for x in reuse};entries=[];new=[];reserved=read(R/'experiments/astra-unown-belief-native-validation-002/held-out-schedule.json')
 for x in reuse:entries.append(dict(id=x['id'],partition='training',opponent_list=x['opponent_list'],chooser=x['chooser'],origin=x['origin'],reused=True))
 for part,total in (('training',128),('validation',16),('locked-test',16)):
  for d in sorted(p):
   for seat in (0,1):
    inherited=[x for x in entries if x['partition']==part and x['opponent_list']==d and x['chooser']==seat]
    reserved_group=sorted([x for x in reserved if part=='locked-test' and x['opponent_list']==d and x['anchor_seat']==seat],key=lambda x:x['id'])
    for k in range(len(inherited),total//2):
     old=reserved_group[k] if k<len(reserved_group) else None;ident=old['id'] if old else 'belief-v1-'+stable(SEED,part,d,seat,k)[:20]
     assert ident not in allids;allids.add(ident)
     row=dict(id=ident,partition=part,opponent_list=d,chooser=seat,origin='reserved-held-out-ID' if old else 'new',reused=False)
     entries.append(row);new.append(dict(row,a=anchor(),b=d,seat=seat,seed=old['policy_seed'] if old else job_seed(ident),tracker_seed=0,startup_check=False))
 subsets={}
 for n in SIZES:
  ids=[]
  for d in sorted(p):
   for seat in (0,1):
    group=[x for x in entries if x['partition']=='training' and x['opponent_list']==d and x['chooser']==seat]
    # Original8/list anchor the nested curve; then all eligible retained matches.
    group.sort(key=lambda x:(0 if x['origin']=='legacy-belief' else 1 if x['reused'] else 2,stable(SEED,'subset',x['id'])))
    ids.extend(x['id'] for x in group[:n//2])
  subsets[str(n)]=sorted(ids,key=lambda i:stable(SEED,'subset-order',n,i))
 batches=[]
 groups={(part,d,seat):sorted([x for x in new if x['partition']==part and x['b']==d and x['seat']==seat],key=lambda x:stable(SEED,'collection-order',x['id'])) for part in ('training','validation','locked-test') for d in sorted(p) for seat in (0,1)}
 # First allocated batch is fresh validation, exactly2/list, chooser-balanced.
 for level in range(64):
  for part in ('validation','locked-test','training'):
   batch=[]
   for d in sorted(p):
    for seat in (0,1):
     g=groups[(part,d,seat)]
     if level<len(g):batch.append(g[level])
   if batch:
    random.Random(SEED+len(batches)).shuffle(batch);batches.append(dict(id=len(batches),partition=part,jobs=batch))
 flat=[x for b in batches for x in b['jobs']];assert len(flat)==5920-len(reuse) and len({x['id'] for x in flat})==len(flat)
 for x in flat[:2]:x['startup_check']=True
 detail_ids=set(sorted([x['id'] for x in flat],key=lambda i:stable(SEED,'detail',i))[:100])
 for x in flat:x['retain_detail']=x['id'] in detail_ids
 write(OUT/'split-manifest.json',dict(schema_version='belief-dataset-v1',entries=entries,partition_counts=dict(training=4736,validation=592,locked_test=592),all_viewpoints_and_rounds_grouped=True,previously_inspected_training_only=True))
 write(OUT/'training-subsets.json',subsets);write(OUT/'collection-schedule.json',batches)
 d,c,n,w=catalogue();S=int(n.max())+1
 write(OUT/'schema.json',dict(version='belief-dataset-v1',query_ids=c,count_support=list(range(S)),library_max_counts=n.max(0).astype(int).tolist(),prediction_points='up to24 hash-selected genuine decisions/game; full forced/genuine legal callback sequence retained',raw_components=['legal JSONL.gz','diagnostic targets JSONL.gz','provenance JSONL.gz'],BO3_per_raw_shard=16,legal_keys=['current','select','logs'],excluded_model_features=['game index','match score','previous outcomes','opponent identity','hidden targets','native opaque state'],global_features='16-bin signed sketches of match lower bounds; exact known H/D/P/B counts; within-game revealed log identities;20 public scalars;37 legal baseline deck weights',local_features='8-bit unique query index;7 normalized A moments; known-hand count;3 lower/upper legal count bounds; maximum library copies; full A count marginals',event_features='same six legal sketches,20 public scalars,24 legal log-type frequencies; full ordered within-game callback sequence',dimensions=dict(global_X=153,event_X=140,local_X=23+3*S),sketch='SHA256 card-sketch-v1;16 signed bins; fixed lossy representation, no text vectors',neural_reset='zero GRU state at every game; only legal match deck lower bounds cross games; no history truncation',distribution='marginal unknown-H/full-D/full-P counts; original count support up to'+str(S-1),mask='legal known lower bounds, physical-copy domains, public sizes and maximum compatible-list copies; no hidden identity',reconciliation='same legal support masking and normalization for all methods; no joint zone-total projection; report remaining consistency violations',limitations='marginals do not define coherent joint hidden worlds; event encoding is lossy; raw records support further legal-history encodings, not arbitrary counterfactual simulation'))
 models=dict(sizes=list(SIZES),seeds=[11,29],training_population='n/100, equally weighted chooser, then BO3/game/decision/query/zone',baseline='exact A marginal extension verified against frozen moments; no fit',bayes=dict(alpha=[30,120],beta=[.25,.5,1.],objective='validation full-field BO3 deck log loss',tie='lower beta then higher alpha within1e-12',one_final_ledger_per_training_BO3=True),trees=dict(implementation='sklearn DecisionTreeRegressor multi-output stochastic gradient boosting',stages=32,learning_rate=.1,max_depth=3,max_leaf_nodes=8,min_samples_leaf=128,max_features=32,stage_BO3_per_list=8,stage_sampling='cyclic balanced4/4 chooser windows; every training BO3 covered at least twice at128/list; same registered queries/points, deterministic counts have zero gradient',loss='masked multinomial log loss for deck and per-zone count; negative-gradient boosting, sample weights preserve match hierarchy',base='masked A deck/count probabilities; count floor1e-5 within legal support'),feedforward=dict(hidden_layers=[128,128],count_head_hidden=64,shared_query_head=True),GRU=dict(width=128,layers=1,event_dimension=140,full_current_game_prefix=True,reset_every_game=True,readout='GRU state plus same current global/local feature matrices'),neural=dict(parameter_cap=1000000,epochs=6,AdamW=dict(lr=.001,weight_decay=.0001),gradient_clip_norm=1,loss='deck CE + mean masked count CE; equal registered match hierarchy',batch='one complete BO3 at a time; MLP point chunks up to128 accumulated before one match update; GRU complete games with full BPTT and no state across games',initialization='zero output residuals around A logits; same legal support',early_stopping='fixed validation weighted deck+count NLL; patience3; minimum improvement1e-5; retain best validation epoch, both seeds reported',inference='no stochastic sampling of predictions'),per_fit_CPU_seconds=dict(Bayes=600,Trees=900,MLP_and_GRU='500+46*n, n=BO3 per list; identical caps across neural families/seeds'),total_fit_CPU_seconds=72000,bootstrap=dict(replicates=20000,seed=SEED,unit='paired wholeBO3 within deck/chooser strata; same indices across methods and sizes; seed outputs reported individually and averaged inside BO3',conditional_on_fitted_models=True),plots='validation uncertain-hand Brier and deck accuracy versus independent training BO3 count',no_test_scoring=True)
 write(OUT/'model-config.json',models)
 order=[]
 for n in SIZES:
  order.append(dict(size=n,family='Bayes',seed=0))
  for j,seed in enumerate((11,29)):
   families=['Trees','MLP','GRU'];rotation=(SIZES.index(n)+j)%3;families=families[rotation:]+families[:rotation]
   order.extend(dict(size=n,family=f,seed=seed) for f in families)
 write(OUT/'fit-schedule.json',order)
 timing=[]
 for family in sorted({v['archetype'] for v in p.values()}):
  deck=min([d for d,v in p.items() if v['archetype']==family],key=lambda d:stable(SEED,'timing-deck',d))
  for seat in (0,1):timing.append(min([x['id'] for x in entries if x['partition']=='validation' and x['opponent_list']==deck and x['chooser']==seat],key=lambda i:stable(SEED,'timing-ID',i)))
 assert len(timing)==16
 write(OUT/'online-timing-manifest.json',dict(ids=timing,selection='one outcome-independent list/archetype, one validation BO3/chooser;16 total',workload='every legal callback, GRU update and genuine decision readout; no new native matches',labels_opened=False))
 print(json.dumps(dict(reused=len(reuse),new_attempts=len(flat),new_by_partition=dict(collections.Counter(x['partition'] for x in flat)),subset_sizes={k:len(v) for k,v in subsets.items()},first_batch=len(batches[0]['jobs']),reserved_ids_retained=len(reserved),count_support=S)))
if __name__=='__main__':main()
