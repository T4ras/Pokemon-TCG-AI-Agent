"""Offline recovery of retained failure; no native retries or test scoring."""
from .common import *
from .storage import Shards
from .audit import integrity
from .truth import select_points,align
from unown_belief_bayes_repair.adapter import legal
import shutil

def main():
 start=time.process_time();parent=R/'experiments/astra-unown-belief-learning-dataset-v1-resume-004';ident='belief-v1-aaefc2f725d6ac542d7c'
 exclusive(OUT/'reconcile-intent.json',dict(at_utc=iso(),id=ident,new_native_matches=0))
 diagnostic=R/'data/private'/parent.name/'detail-temp'/(ident+'-integrity.json.gz');x=gzip_read(diagnostic);job=x['job'];hist=[dict(observation=legal(e['observation']),action=e['action']) for e in x['legal_history']];selection=select_points(ident,hist);targets=align([hist[i] for i in sorted(selection)],x['native_frames'],job['seat']);checks=integrity(hist,targets,job['b'])
 old_records=rows(parent/'matches.jsonl');receipt=next(r for r in old_records if r['id']==ident);assert receipt['policy_unchanged'] and receipt['policy_tensor']==TENSOR
 snapshots=[read(p) for p in (parent/'inflight').glob('*.json')];slot=next(r for r in snapshots if r['job']['id']==ident);assert slot['completed']==ident
 games=slot['games'];outcomes=[f['current']['result'] for f in x['native_frames'] if (f.get('current') or {}).get('result',-1)>=0];assert outcomes==[g['result'] for g in games] and len(games) in (2,3)
 wins=[sum(g['result']==i for g in games) for i in (0,1,2)];winner=0 if wins[0]>wins[1] else 1 if wins[1]>wins[0] else 2
 provenance=dict(id=ident,partition=job['partition'],observer_deck=job['a'],opponent_list=job['b'],observer_seat=job['seat'],chooser=job['seat'],policy_tensor=TENSOR,games=games,match_result=winner,native_receipt=receipt,status='RECOVERED_OFFLINE',checks=checks,CPU_core_seconds=receipt['CPU_core_seconds'],UTC_seconds=receipt['UTC_seconds'],policy_unchanged=True,origin='new',original_integrity_failure=receipt['error'],recovery_source_sha256=sha(diagnostic),recovery_source_manifest_sha256=sha(parent/'source-manifest.json'),recovery_native_replay=False)
 payload=dict(legal=dict(id=ident,entries=hist,selection={str(i):v for i,v in selection.items()}),targets=dict(id=ident,opponent_list=job['b'],points=targets),provenance=provenance)
 shards=Shards();assert ident not in {r['id'] for r in shards.committed};shards.append(payload);shards.flush_all()
 shutil.copy2(OUT/'dataset-index.json',OUT/'inherited-dataset-index.json');shutil.copy2(OUT/'dataset-shards.json',OUT/'inherited-dataset-shards.json')
 records=[]
 for r in old_records:
  if r['id']==ident:r=dict(r,status='DONE',recovered_offline=True,original_integrity_status='FAILED',original_integrity_error=r['error'],error=None,recovery_CPU_seconds=time.process_time()-start)
  records.append(r)
 reg=read(OUT/'recovery-registration.json')
 for missing in reg['lost_ids']:
  if any(r['id']==missing for r in records):continue
  job=read(parent/'issued'/(missing+'.json'))['job'];records.append(dict(job,status='FAILED',error='Interrupted before durable full-history receipt; user elected no retry',failure_kind='MISSING_HISTORY',CPU_core_seconds=0,CPU_cost_note='Included in prior controller group meter; no per-match value recoverable',native_execution_status='UNRESOLVED',payload=None,policy_tensor=TENSOR))
 (OUT/'matches.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
 for missing in reg['lost_ids']:append(OUT/'attempt-ledger.jsonl',dict(event='RESULT',id=missing,status='FAILED',reason='INTERRUPTED_MISSING_HISTORY_NO_RETRY',at_utc=iso(),partition='training'))
 append(OUT/'attempt-ledger.jsonl',dict(event='OFFLINE_RECOVERY',id=ident,status='DONE',at_utc=iso(),new_native_matches=0))
 reg['recovered_ids']=read(parent/'recovery-registration.json')['recovered_ids']+[ident];reg['state']='PREPARED';reg['remaining_jobs_sha256']=sha(OUT/'collection-schedule.json');write(OUT/'recovery-registration.json',reg)
 write(OUT/'recovery-reconciliation.json',dict(at_utc=iso(),CPU_seconds=time.process_time()-start,adopted_BO3=len(shards.committed)-1,recovered_ids=[ident],unrecoverable_ids=reg['lost_ids'],unrecoverable_ids_reissued=False,parent_native_starts=len(list((parent/'native-starts').glob('*.json'))),remaining_previously_unissued=sum(len(b['jobs']) for b in read(OUT/'collection-schedule.json')),expected_final_archive_BO3=len(shards.committed)+67,expected_cumulative_native_starts=3158,expected_cumulative_native_complete=3154,expected_missing_history_failures=4,required_learning_curve_sizes=list(SIZES),unavailable_sizes=[32,64,128],locked_test_scores=0,original_failure_preserved=str((parent/'failure.json').relative_to(R))))
 print(read(OUT/'recovery-reconciliation.json'))
if __name__=='__main__':main()
