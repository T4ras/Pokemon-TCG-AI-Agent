"""Build isolated deployment repair from the exact previous archive directory."""
from pathlib import Path
import sys,shutil,json,hashlib,tarfile,ast,datetime as dt
R=Path(__file__).resolve().parents[1];RUN='astra-unown-belief-hosted-repair-003';O=R/'experiments'/RUN;P=R/'data/private'/RUN;PKG=P/'package';OLD=R/'data/private/astra-unown-belief-hydrapple-submission-repair-002/package'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(v,indent=2)+'\n')
def main():
 assert not PKG.exists();shutil.copytree(OLD,PKG,ignore=shutil.ignore_patterns('__pycache__','*.pyc','runtime-cache'))
 old=read(R/'experiments/astra-unown-belief-hydrapple-submission-repair-002/package-manifest.json')
 for n,v in old['files'].items():assert sha(PKG/n)==v['sha256']
 shutil.copy2(R/'scripts/unown_belief_deployment_guard.py',PKG/'frozen_belief/deployment_guard.py')
 p=PKG/'frozen_unown/runtime.py';s=p.read_text()
 s=s.replace('from frozen_belief.integration import packet_belief,with_belief','from frozen_belief.deployment_guard import SupportedBelief,with_supported_belief as with_belief,supported_outputs')
 before="belief=packet_belief(packet,next(net.parameters()).device,state['r'].dtype)\n        logits,v,delta=corrected_outputs(net,state,belief)"
 assert s.count(before)==1;s=s.replace(before,'logits,v,delta=supported_outputs(net,state,packet)')
 assert s.count('self.belief_observer=OnlineBelief()')==1;s=s.replace('self.belief_observer=OnlineBelief()','self.belief_observer=SupportedBelief(OnlineBelief())');compile(s,str(p),'exec');p.write_text(s)
 files={p.relative_to(PKG).as_posix():dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(PKG.rglob('*')) if p.is_file() and p.name!='frozen-candidate.json'}
 modified=[n for n,v in files.items() if n not in old['files'] or v['sha256']!=old['files'][n]['sha256']]
 assert set(modified)=={'frozen_belief/deployment_guard.py','frozen_unown/runtime.py'}
 assert not(set(old['files'])-set(files));manifest=dict(old,files=files,modified_files=modified,unsupported_behavior='Expected belief support/resource failures latch base-policy action logits for rest of BO3; new match resets; numeric/invariant failures remain errors',parent_submission=58069431,repair='runtime guard for out-of-catalogue legal evidence; weights/deck/decoder unchanged')
 write(PKG/'frozen-candidate.json',manifest);write(O/'package-manifest.json',manifest)
 # Fixed packaging allocation: three existing decks, two initial chooser seats each.
 import sys;sys.path.insert(0,str(R/'src'))
 from ptcg_agent.engine import load_sdk,validate_deck
 api,_=load_sdk();cards={c.cardId:c for c in api.all_card_data()}
 stage=read(R/'experiments/astra-stage-g-deck-policy-screen-001/manifest.json')
 choices=[('OceanMix',stage['packages']['G']['deck']),('Abomasnow',stage['panel'][-1]['deck']),('Hydrapple', [int(x) for x in (PKG/'deck.csv').read_text().split()])]
 import random
 jobs=[]
 for i,(name,deck) in enumerate(choices):
  validate_deck(deck,cards)
  for seat in (0,1):jobs.append(dict(id=f'guard-package-{i}-{seat}',learner_seat=seat,opponent_deck=name,opponent_cards=deck,seed=904100+i*2+seat))
 random.Random(904100).shuffle(jobs);write(O/'schedule.json',jobs)
 write(O/'repair-plan.json',dict(registered_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),diagnosis='Latest58069431invalidated after initial validation; four inspected public failures are UnknownProvenance outside37-listcardcatalogue',scope='Inference-only runtime repair; historical training, observer and all old packages unchanged',expected_caught=['UnsupportedEvent including UnknownProvenance','OutOfLibrary','IncompatibleAllocation','OperationLimit'],uncaught=['InvariantFailure','FloatingPointError','unexpected software exceptions'],fallback='Exact base logits; disable correction through rest of BO3 including forced observations/game resets; normal belief resumes only for next match',tests='Focused tests; replay10retainedsupportedhistories for exact old parity; reproduce earliest retrieved hosted crash with original and repaired package;6officialnativeBO3 threefixeddecks x2chooser; replay4OODnativehistoriesagainstfrozenbase',new_native_BO3=6,extra_native_diagnostics=0,GPU=0,training=0,paid_model_API_requests=0,submission_calls=1,opponent='Previously submitted frozen embedding-base argmax on existing decks; runtime checks only, not strength estimates'))
 print(json.dumps(dict(package=str(PKG),modified=modified,jobs=len(jobs))))
if __name__=='__main__':main()
