"""Freeze the one retained final checkpoint for the original CPU benchmark."""
import shutil,sys
from .common import *

def prepare():
 path=OUT/'checkpoint-benchmark-registration.json'
 if path.exists():raise RuntimeError('Benchmark already registered; do not replace its allocation')
 assert not records(OUT/'evaluation-final-records.jsonl')
 assert not (OUT/'unown_embedding_continuation-evaluate-final-process.json').exists()
 training=read(OUT/'training-completion.json');assert training['actor_changed']
 assert sha(PRIVATE/'final.pt')==training['final_file_sha256']
 initial=read(OUT/'evaluation-initial-completion.json');assert initial['completed']==512 and initial['failures']==0
 schedule=read(OUT/'evaluation-final-schedule.json');assert len(schedule)==512
 assert set(collections.Counter((j['learner_deck'],j['opponent_deck'],j['learner_seat']) for j in schedule).values())=={32}
 from unown_runner.runtime import Runtime
 rt=Runtime(context())
 try:
  candidate=rt.load_model(PRIVATE/'final.pt','cpu')
  assert rt.state_hash(candidate)==training['final_tensor_hash']
  original=rt.load_model(PRIVATE/'initialization.pt','cpu');assert rt.state_hash(original)==INITIAL_TENSOR
 finally:rt.close()
 frozen=list((ROOT/'scripts/unown_runner').rglob('*.py'))
 frozen += list((ROOT/'scripts/unown_embedding_continuation').glob('*.py'))
 frozen += [ROOT/'scripts/unown_embedding_continuation_benchmark_owner.py']
 frozen += [PRIVATE/'final.pt',PRIVATE/'initialization.pt',PRIVATE/'text-tables.npz']
 frozen += [OUT/name for name in ('training-completion.json','completion.json','gpu-launch.json','GPU-accounting.json','decks.json','evaluation-final-schedule.json','evaluation-initial-schedule.json','evaluation-initial-records.jsonl','evaluation-initial-completion.json')]
 frozen += [p for p in (PRIVATE/'assets').rglob('*') if p.is_file() and p.suffix!='.pyc' and '__pycache__' not in p.parts]
 registration=dict(at_utc=iso(),scope='explicit_frozen_checkpoint_CPU_evaluation',authority='User: launch the benchmark',
  repeated_looks='Additional final-checkpoint evaluation after a preceding checkpoint was evaluated; pointwise intervals only, no sequential/family-wise error claim or pooling.',
  attempts=512,earlier_strength_attempts=1024,project_strength_attempts_after=1536,new_initial_attempts=0,initial_attempts_reused=512,training_allowed=False,GPU_allowed=False,uploads_allowed=False,paid_model_API_requests=0,
  final_file_sha256=training['final_file_sha256'],final_tensor_hash=training['final_tensor_hash'],
  checkpoint_rule='Exact final continuation export after reaching502384 collected decisions and15996 updates; fixed before any final match. No checkpoint selection or further learning.',
  schedule='Original final schedule: D10/D04 x Hydro/D02/D01/D03 x64,32/32 initial chooser; both conditional argmax; original initialization opponents.',
  statistics='Original independent native deals,20000 whole-match bootstrap draws within checkpoint/deck/opponent/chooser strata, equal weights; primary final-minus-initial across8 cells; fixed seed2026103001.',
  failures='Keep every attempted match, no replacements or sample extension; first2 scheduled attempts supply startup evidence, no extra native diagnostics.',
  historical_status='Training COMPLETE at the fresh-decision target. User explicitly authorizes512 additional CPU attempts. Prior1024 attempts and earlier stopped-checkpoint comparison remain separate historical evidence; reused native schedule/seeds do not pair deals.',
  frozen_files={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(frozen))})
 exclusive(path,registration)
 print(json.dumps(dict(status='REGISTERED',attempts=512,new_native_attempts=0,GPU_seconds=0,checkpoint_tensor=training['final_tensor_hash'])))

def verify():
 from .benchmark_authorization import final_evaluation_allowed
 assert final_evaluation_allowed()
 print(json.dumps(dict(status='PASS',frozen_inputs_exact=True,training_evidence_unchanged=True)))

if __name__=='__main__':prepare() if sys.argv[1:]==['prepare'] else verify()
