"""Audit every reusable match; inspected histories are training-only."""
from .common import *
from .truth import select_points,align,key
from unown_belief_bayes_repair.adapter import legal
from .baseline import Facts
from unown_belief_bayes_repair.deck import catalogue

def integrity(hist,targets,opponent):
 selected=select_points('unused',[]) # selection is supplied by each record below
 if not hist or not targets:raise InvariantFailure('Empty legal history/targets')
 labels={tuple(t['key']):t for t in targets}
 if len(labels)!=len(targets):raise InvariantFailure('Duplicate target alignment key')
 f=Facts(metadata()['cards']);games=set();last={};bounds=collections.Counter(prior()[opponent]['cards'])
 for entry in hist:
  o=entry['observation']
  if legal(o)!=o:raise InvariantFailure('Legal input contains forbidden fields')
  f.observe(o);cur=o['current'];games.add(cur['round'])
  if any(f.lower.get(c,0)<n for c,n in last.items()):raise InvariantFailure('Match facts lost on reset')
  last=dict(f.lower)
  if collections.Counter(f.lower)-bounds:raise InvariantFailure('Certified lower bound exceeds frozen list')
  t=labels.get(key(cur))
  if t:
   known=f.known_hand();hand={int(c):n for c,n in t['hand'].items()}
   if any(n>hand.get(c,0) for c,n in known.items()):raise InvariantFailure('Known hand exceeds target')
   if not {s for s,(c,d) in f.copies.items() if d=={'H'}}<=set(t['hand_serials']):raise InvariantFailure('Known hand serial missing in target')
   if any(sum(map(int,t[z].values()))!=n for z,n in (('hand',cur['players'][1-cur['yourIndex']]['handCount']),('deck',cur['players'][1-cur['yourIndex']]['deckCount']),('prize',len(cur['players'][1-cur['yourIndex']]['prize'] or [])))):raise InvariantFailure('Target/public count mismatch')
 if games!={1,2} and games!={1,2,3}:raise InvariantFailure('Incomplete BO3 legal history')
 if set(labels)-{key(e['observation']['current']) for e in hist}:raise InvariantFailure('Unaligned diagnostic keys')
 return dict(legal_callbacks=len(hist),selected_points=len(targets),games=len(games),final_ledger_sha256=stable([f.lower.get(c,0) for c in catalogue()[1]]))

def from_replay(detail,seat):
 env=detail['environment'];frames=env['steps'][0][0].get('visualize');hist=[]
 if not frames:raise InvariantFailure('Retained replay lacks native visualization targets')
 for i,step in enumerate(env['steps'][:-1]):
  row=step[seat];o=row.get('observation') or {}
  if row['status']=='ACTIVE' and o.get('current') and o.get('select'):
   action=env['steps'][i+1][seat].get('action')
   if action is None:raise InvariantFailure('Missing aligned response action')
   hist.append(dict(observation=legal(o),action=action))
 selected=select_points(detail['job']['id'],hist);targets=align([hist[i] for i in sorted(selected)],frames,seat)
 return hist,targets,selected

def main():
 start=time.process_time();begun=now();inventory=[];primary=[];seen=set();digests={};old=read(BASE/'inputs.json');manifest=read(SCREEN/'manifest.json');assert manifest['model_tensor_sha256']==TENSOR
 records=rows(R/'experiments/astra-unown-belief-development-repair-003/native-matches.jsonl')+read(R/'experiments/astra-unown-belief-development-repair-003/inherited-native-records.json');native={x['id']:x for x in records}
 for item in old:
  ident=item['id'];e=item['corpus'];hist=gzip_read(R/e['legal_path']);targets=gzip_read(R/e['truth_path']);assert sha(R/e['legal_path'])==e['legal_sha256'] and sha(R/e['truth_path'])==e['truth_sha256']
  checks=integrity(hist,targets,item['opponent_list']);record=native[ident];assert record['policy_tensor']==TENSOR and record['policy_unchanged']
  fingerprint=stable(e['legal_sha256'],e['truth_sha256']);assert ident not in seen and fingerprint not in digests;seen.add(ident);digests[fingerprint]=ident
  view=dict(id=ident,partition='training',opponent_list=item['opponent_list'],chooser=item['chooser'],observer_deck=anchor(),observer_seat=item['chooser'],previous_use='belief development, PF/RB diagnostics and Bayesian nested CV',origin='legacy-belief',corpus=e,provenance=dict(source_record_run='astra-unown-belief-development-repair-003',games=record['games'],policy_tensor=TENSOR,policy_unchanged=True,SDK_source_manifest=str((BASE/'source-manifest.json').relative_to(R))),checks=checks)
  primary.append(view);inventory.append(dict(id=ident,duplicate=False,primary_eligible=True,partition='training',origin=view['origin'],deck_pair=[anchor(),item['opponent_list']],observer_seat=item['chooser'],chooser=item['chooser'],complete_legal=True,aligned_targets=True,previous_use=view['previous_use'],policy_tensor=TENSOR))
 retained=read(SCREEN/'retention.json')['successes'];aux=[]
 for retained_item in retained:
  detail=gzip_read(R/retained_item['path']);job=detail['job'];ident=job['id'];assert ident not in seen;seen.add(ident)
  is_primary=anchor() in (job['a'],job['b']);seat=job['seat'] if job['a']==anchor() else 1-job['seat'] if is_primary else int(stable(SEED,ident,'aux-view')[-1],16)%2
  # Mirrors use a fixed hash-selected viewpoint, independent of outcomes.
  if job['a']==job['b']==anchor():seat=int(stable(SEED,ident,'mirror-view')[-1],16)%2
  opp=job['b'] if seat==job['seat'] else job['a'];observer=job['a'] if seat==job['seat'] else job['b']
  views=[]
  for v in ([seat] if is_primary else [0,1]):
   hist,targets,selected=from_replay(detail,v);opponent=job['b'] if v==job['seat'] else job['a'];checks=integrity(hist,targets,opponent)
   root=PRIVATE/('imports' if is_primary else 'auxiliary')/ident;lp=root/('legal-'+str(v)+'.json.gz');tp=root/('targets-'+str(v)+'.json.gz');gzip_write(lp,hist);gzip_write(tp,targets)
   corpus=dict(legal_path=str(lp.relative_to(R)),legal_sha256=sha(lp),truth_path=str(tp.relative_to(R)),truth_sha256=sha(tp),selection={str(k):z for k,z in selected.items()},selected_decisions=len(selected),legal_observations=len(hist))
   views.append(dict(viewpoint=v,corpus=corpus,checks=checks))
  entry=dict(id=ident,duplicate=False,primary_eligible=is_primary,partition='training' if is_primary else 'auxiliary-training-only',origin='retained-screen-'+job['stage'],deck_pair=[job['a'],job['b']],observer_seat=seat,chooser=seat,complete_legal=True,aligned_targets=True,previous_use='screen outcome/deck selection and retained SDK alignment development',policy_tensor=TENSOR,source_replay=retained_item['path'],source_sha256=sha(R/retained_item['path']),SDK_source_manifest=str((SCREEN/'manifest.json').relative_to(R)))
  inventory.append(entry)
  if is_primary:
   view=views[0];primary.append(dict(entry,opponent_list=opp,observer_deck=observer,corpus=view['corpus'],checks=view['checks'],provenance=dict(games=detail['games'],source_replay=entry['source_replay'],source_sha256=entry['source_sha256'],policy_tensor=TENSOR,policy_unchanged=True,SDK_source_manifest=entry['SDK_source_manifest'])))
  else:aux.append(dict(entry,views=views))
 # Native reserved identities are reconciled by ID only; no target/outcome access.
 reserved=read(R/'experiments/astra-unown-belief-native-validation-002/held-out-schedule.json');assert len(reserved)==296 and not set(x['id'] for x in reserved)&seen
 write(OUT/'reuse-primary.json',primary);write(OUT/'inventory.json',inventory);write(OUT/'auxiliary-manifest.json',aux);write(OUT/'reserved-ID-reconciliation.json',dict(ids=[x['id'] for x in reserved],count=len(reserved),previous_native_attempts=0,new_partition='locked-test',training_or_validation_overlap=False,targets_or_scores_inspected=False,source_manifest=str((R/'experiments/astra-unown-belief-native-validation-002/held-out-schedule.json').relative_to(R))))
 summary=dict(at_utc=iso(),unique_audited_BO3=len(inventory),duplicates=0,legacy_primary=296,screen_primary=sum(x['origin'].startswith('retained-screen') for x in primary),primary_reused=len(primary),auxiliary_BO3=len(aux),auxiliary_viewpoints=2*len(aux),compact_screen_outcomes_not_histories=3456-len(retained),primary_training_only=True,CPU_seconds=time.process_time()-start,elapsed_seconds=(now()-begun).total_seconds(),new_native_matches=0,held_out_targets_accessed=False)
 write(OUT/'reuse-audit.json',summary);print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
