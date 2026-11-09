"""Freeze all committed records; no native execution or locked-test parsing."""
from .common import *
from .storage import load
import shutil

def main():
 start=time.process_time();index=read(PARENT/'dataset-index.json');old_entries={x['id']:x for x in read(PARENT/'split-manifest.json')['entries']};ids=[x['id'] for x in index]
 if len(ids)!=len(set(ids)):raise InvariantFailure('Duplicate committed match IDs')
 counts=collections.Counter(r['partition'] for r in index)
 if counts!=dict(training=2334,validation=592,**{'locked-test':592}):raise InvariantFailure('Unexpected committed partition allocation')
 allpaths={v['path']:v['sha256'] for r in index for v in r['paths'].values()}
 for path,digest in allpaths.items():
  if sha(R/path)!=digest:raise InvariantFailure('Committed shard changed: '+path)
 entries=[];fingerprints={};verified=0
 for row in index:
  item=old_entries[row['id']]
  if item['partition']!=row['partition']:raise InvariantFailure('Partition identity mismatch')
  entries.append(item)
  if row['partition']=='locked-test':continue
  hist=load(row,'legal');fingerprint=stable(json.dumps(hist['entries'],sort_keys=True,separators=(',',':')))
  if fingerprint in fingerprints:raise InvariantFailure('Duplicate whole legal BO3 history')
  fingerprints[fingerprint]=row['id'];cache=read(OUT/'feature-results'/(row['id']+'.json'))
  if cache['id']!=row['id'] or cache['partition']!=row['partition']:raise InvariantFailure('Cache ID/partition mismatch')
  for k in ('feature','target','detail'):
   if sha(R/cache[k+'_path'])!=cache[k+'_sha256']:raise InvariantFailure('Cache checksum mismatch: '+row['id'])
  verified+=1
 write(OUT/'dataset-index.json',index);write(OUT/'split-manifest.json',dict(entries=entries,source='committed3518 only; no planned/missing IDs',locked_test_parsed=False))
 train=sorted([x for x in entries if x['partition']=='training'],key=lambda x:stable(SEED,'full-train-order',x['id']));strata=collections.Counter((x['opponent_list'],x['chooser']) for x in train);deck=collections.Counter(x['opponent_list'] for x in train)
 if len(strata)!=74 or min(strata.values())<=0:raise InvariantFailure('Training strata missing')
 manifest=[]
 for x in train:
  d=x['opponent_list'];seat=x['chooser'];n=strata[d,seat]
  manifest.append(dict(x,stratum_count=n,neural_loss_weight=2334*prior()[d]['prior']/(2*n),bayes_conditional_weight=deck[d]/(2*n)))
 write(OUT/'full-training-manifest.json',dict(BO3=2334,entries=manifest,sampling='all records once per neural epoch; uniform permutation; no downsampling',stratum_weights='p(deck)/2 divided by available stratum count'))
 write(OUT/'training-subsets.json',dict(full=[x['id'] for x in train]));write(OUT/'stratum-counts.json',[dict(deck=d,archetype=prior()[d]['archetype'],chooser=s,BO3=strata[d,s],population_weight=prior()[d]['prior']/2) for d in sorted(prior()) for s in (0,1)])
 for seed in (11,29):
  groups={(d,s):sorted([x for x in train if x['opponent_list']==d and x['chooser']==s],key=lambda x:stable(seed,'tree-batch',x['id'])) for d in sorted(prior()) for s in (0,1)};stages=[];exposure=collections.Counter()
  for stage in range(32):
   batch=[g[(stage*4+j)%len(g)]['id'] for g in groups.values() for j in range(min(4,len(g)))];stages.append(batch);exposure.update(batch)
  if set(exposure)!=set(x['id'] for x in train) or min(exposure.values())<2:raise InvariantFailure('Original tree coverage rule fails; amendment required before fitting')
  write(OUT/f'tree-schedule-s{seed}.json',dict(stages=stages,exposure_counts=dict(exposure),minimum=min(exposure.values()),maximum=max(exposure.values()),unique_BO3=len(exposure),draws=sum(exposure.values()),amendment_needed=False))
 schedule=[dict(size='full',family=f,seed=s,training_BO3=2334) for f,s in [('Bayes',0),('Trees',11),('MLP',11),('GRU',11),('GRU',29),('MLP',29),('Trees',29)]];write(OUT/'fit-schedule.json',schedule)
 model_config=read(OUT/'model-config.json');model_config.update(training_population='all2334 committed matches; n/100 deck and equal chooser objective; family-specific weighting in registration',sizes=['full'],per_fit_CPU_seconds=dict(Bayes=600,Trees=2700,MLP=5400,GRU=7200),total_fit_CPU_seconds=36000);write(OUT/'model-config.json',model_config)
 # Historical validation metrics are immutable references, not rerun candidates.
 (OUT/'validation-results').mkdir()
 for p in (PARENT/'validation-results').glob('*.json'):shutil.copy2(p,OUT/'validation-results'/p.name)
 write(OUT/'historical-results-manifest.json',dict(files={str(p.relative_to(R)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in (PARENT/'validation-results').glob('*.json')},old_report_sha256=sha(R/'reports/astra-unown-belief-classifier-learning-curves.md')))
 write(OUT/'manifest-audit.json',dict(at_utc=iso(),passed=True,CPU_seconds=time.process_time()-start,partition_counts=dict(counts),unique_match_IDs=len(ids),unique_non_test_legal_histories=len(fingerprints),verified_non_test_caches=verified,missing_features=0,origin_counts=dict(collections.Counter(x['origin'] for x in train)),deck_count_min=min(deck.values()),deck_count_max=max(deck.values()),stratum_count_min=min(strata.values()),stratum_count_max=max(strata.values()),locked_test_payload_parsed=False,locked_test_model_access=False,raw_shards_hashed_without_test_parsing=len(allpaths),native_matches=0))
 print(read(OUT/'manifest-audit.json'),flush=True)
if __name__=='__main__':main()
