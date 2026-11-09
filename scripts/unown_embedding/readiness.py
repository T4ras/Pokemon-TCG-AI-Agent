"""Bounded saved-observation checks. Never starts native games."""
import copy,json,resource,subprocess,sys,types
from .common import *
from .runtime import AdaptationRuntime
from .model import *
from unown_runner.runtime import Runtime

def saved_new_probes():
 rt=Runtime(context());model=rt.load_model(PRIVATE/'initialization.pt','cpu');rt.runtime.freeze_inference(model)
 sample=read(ROOT/'experiments/top50-models-decks-novelty-904104-001/analysis.json');decks=read(OUT/'decks.json');rows=[];sources=[]
 try:
  for deck in ('D10','D04'):
   source=next(o for m in sample['models'] for o in m['observations'] if o['deck_id']==deck);eid=source['episode_id'];seat=source['seat']
   path=ROOT/'data/private/top50-models-decks-novelty-904104-001/prefixes'/f'{eid}.json-prefix'
   text=path.read_text();match=text.index('"steps"');offset=text.index('[',match)+1;decoder=json.JSONDecoder();count=0;frames=0
   actors=[];actor_context=dict(candidate_model=model,actors=actors,actual_starter=None,retain_history=False)
   actor=rt.corrected.corrected_actor_class(actor_context)(model,decks[deck]['cards'],sample=False,collect=True,seed=SEED)
   while count<64:
    while offset<len(text) and text[offset] in ' \r\n\t,':offset+=1
    try:frame,offset=decoder.raw_decode(text,offset)
    except (ValueError,IndexError):break
    frames+=1;entry=frame[seat]
    raw=entry.get('observation') or {}
    if frames==1 or entry.get('status')=='ACTIVE':
     before=len(actor.records);actor.act(raw)
     for row in actor.records[before:]:
      if count>=64:break
      row=copy.deepcopy(row);row.update(advantage=1. if count%2 else -1.,return_target=row['value']+(.25 if count%2 else -.25),software_probe=True,replayed_off_policy=True,deck=deck,source_episode=eid,source_frame=frames)
      rows.append(row);count+=1
   if count<16:raise RuntimeError(f'Insufficient retained conditional prefixes for {deck}: {count}')
   sources.append(dict(deck=deck,episode=eid,seat=seat,rows=count,frames=frames,prefix_SHA256=sha(path),native_matches=0))
  torch.save(rows,PRIVATE/'new-probes.pt');write(OUT/'probe-provenance.json',dict(sources=sources,off_policy=True,not_training_experience=True,targets='Synthetic fixed+/- .25 value offsets and+/-1 advantages for software tests only',rows=len(rows)))
 finally:rt.close()

def mechanisms():
 rows=read(OUT/'trainable-rows.json');source=PRIVATE/'assets/released/submission/ptcg/rl/encoder.py';lines=source.read_text().splitlines()
 cite=lambda word:[dict(line=i+1,text=line.strip()) for i,line in enumerate(lines) if word in line]
 requirements=[
  dict(cards=[1348,1349,1374,1375,1380],mechanism='Hide n Sneak identities, discard thresholds4/6, and permanent attack/ability protection',observables='Public discard identities/counts, field identities, current active/bench; conditional effects are engine-resolved',missing_information=None,limitation='No bespoke counter-of-Hide-n-Sneak feature is added; frozen projections/network must use existing identity/count tokens'),
  dict(cards=[1340],mechanism='Patrat blocks moving damage counters on either side',observables='Patrat visible identity, board HP/status and legal target options',missing_information=None,limitation='No new prevent-movement numerical flag; permanent property must be learned from identity and board context'),
  dict(cards=[1331,1387,1406],mechanism='Evolution, extra-energy damage, optional energy discard, forced switching, deck discard/search',observables='Evolution IDs, physical energy-card references plus provided-energy counts, selection context/source/target pointers and revealed deck/looking zones',missing_information=None,limitation='Existing threat helper uses base/known-modifier damage and omits these conditional bonuses; attached energy counts and discard decisions remain in actor input'),
  dict(cards=[1413],mechanism='Opponent-prize threshold, opponent hand sent to deck bottom, draw3',observables='Public prize/hand/deck counts, legally revealed move logs and legal item option',missing_information=None,limitation='Hidden hand identities remain hidden; tracker exposes legal known-card lower bounds, not an invented exact hand or deck order'),
  dict(cards=[1428,1430],mechanism='Choose/discard cards and draw; once-per-turn stadium use',observables='Own hand identities, legal choices/count limits, stadium identity, current turn and offered usable action',missing_information=None,limitation='No generic ability-used flag is introduced; current action availability encodes immediate use legality')]
 write(OUT/'mechanic-readiness.json',dict(at_utc=iso(),status='PASS_SCOPED_SOURCE_REVIEW',requirements=requirements,material_missing_observable_found=False,scope='Selected new cards only; not a broad all-card audit or proof of learned understanding',sources=dict(encoder_SHA256=sha(source),discard=cite('discard'),energy=cite('energies'),stadium=cite('stadium'),counts=cite('COUNT'),option_pointers=cite('opt_i'),native_API=sha(PRIVATE/'assets/environment/cg/sim.py')),native_matches=0))

def main():
 started=time.monotonic();cpu=time.process_time();usage=resource.getrusage(resource.RUSAGE_CHILDREN)
 number=len(list(OUT.glob('readiness-*.json')))+1;folder=PRIVATE/'readiness'/f'{number:03d}';folder.mkdir(exist_ok=True,parents=True)
 try:
  mechanisms()
  if not (PRIVATE/'new-probes.pt').exists():saved_new_probes()
  env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=''
  with (folder/'pytest.log').open('w') as stream:result=subprocess.run([sys.executable,'-m','pytest','tests/test_unown_embedding.py','tests/test_unown_embedding_collection.py','-q','--basetemp='+str(folder/'cases')],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=600)
  receipt=dict(at_utc=iso(),status='PASS' if result.returncode==0 else 'FAILED',returncode=result.returncode,log=str((folder/'pytest.log').relative_to(ROOT)),native_attempts=0,GPU_seconds=0,elapsed_seconds=time.monotonic()-started)
  write(OUT/f'readiness-{number:03d}.json',receipt);write(OUT/'readiness.json',receipt);print(json.dumps(receipt));return result.returncode
 finally:
  after=resource.getrusage(resource.RUSAGE_CHILDREN);append(OUT/'local-costs.jsonl',dict(stage='readiness',at_utc=iso(),elapsed_seconds=time.monotonic()-started,CPU_core_hours=(time.process_time()-cpu+after.ru_utime+after.ru_stime-usage.ru_utime-usage.ru_stime)/3600,native_attempts=0,GPU_seconds=0))

if __name__=='__main__':raise SystemExit(main())
