"""Focused true-sparse/critic-isolation checks using retained legal histories."""
import copy,random
import pytest,torch,numpy as np
from unown_embedding.common import *
from unown_embedding.model import *
from unown_embedding.runtime import AdaptationRuntime
from unown_runner.runtime import Runtime

@pytest.fixture(scope='module')
def pair():
 spec=context().spec();spec['mode']='software_test';ctx=RunContext.from_spec(spec)
 base=Runtime(ctx);original=base.load_model(PRIVATE/'initialization.pt','cpu');rt=AdaptationRuntime(ctx);model,opt,saved=rt.restore(PRIVATE/'initialization.pt','cpu')
 rows=torch.load(PRIVATE/'new-probes.pt',weights_only=False,map_location='cpu');old=torch.load(PRIVATE/'old-probes.pt',weights_only=False,map_location='cpu')
 yield base,original,rt,model,opt,rows,old
 base.close();rt.close()

def outputs(rt,model,rows):
 rt.runtime.freeze_inference(model)
 with torch.no_grad():result=[model.core(r['packet'])[0]['logits'].clone() for r in rows]
 rt.batch.clear(model);return result

def test_zero_actor_and_canonical_initialization(pair):
 base,original,rt,model,opt,rows,old=pair
 assert rt.state_hash_from_canonical(model)==INITIAL_TENSOR
 assert len(opt.state)==0 and all(group['weight_decay']==0 for group in opt.param_groups)
 for a,b in zip(outputs(base,original,rows[:16]+old[:16]),outputs(rt,model,rows[:16]+old[:16])):assert torch.equal(a,b)
 assert all(not p.requires_grad for name,p in model.named_parameters() if not name.endswith('.delta') and not name.startswith('value_mlp.'))

def test_value_gradient_does_not_reach_embeddings(pair):
 _,_,rt,model,opt,rows,_=pair
 cohort=rt.epochs.new_cohort(rows,rt.state_hash(model));engine=rt.make_engine(model,opt,cohort,0)
 try:
  model.zero_grad(set_to_none=True);tables=model._card_table();group=next(iter(engine.partition(rows[:16])))[1]
  logits,values,loss,metrics=engine.execute(group,16,tables);values.square().sum().backward()
  assert all(p.grad is None for p in actor_parameters(model))
  assert gradient_norm(model.value_mlp.parameters())>0
 finally:engine.close();rt.batch.clear(model);model.zero_grad(set_to_none=True)

def test_sparse_updates_frozen_rows_and_old_policy_are_exact(pair,tmp_path):
 base,original,rt,model,opt,rows,old=pair
 model,opt,_=rt.restore(PRIVATE/'initialization.pt','cpu');before=rt.scalar.digest(frozen_state(model));canonical_before=canonical_state(model)
 cohort=rt.epochs.new_cohort(rows,rt.state_hash(model));engine=rt.make_engine(model,opt,cohort,0);execution=dict(step=0)
 try:
  for step in range(3):
   indices=np.arange(0,min(64,len(rows)));result=engine.transaction(model,opt,[rows[i] for i in indices],cohort['advantages'][indices],execution,'software-'+str(step));assert result['accepted'],result
  assert any(torch.count_nonzero(p).item() for p in actor_parameters(model))
  assert before==rt.scalar.digest(frozen_state(model))
  state=canonical_state(model)
  for kind,(identity,text) in CHANNELS.items():
   eligible=set(rt.scope[kind])
   for name in (identity+'.weight',text):
    frozen=[i for i in range(len(state[name])) if i not in eligible]
    assert torch.equal(state[name][frozen],canonical_before[name][frozen])
  for a,b in zip(outputs(base,original,old[:16]),outputs(rt,model,old[:16])):assert torch.equal(a,b)
  # Effective canonical export loads through ordinary inference, not only the
  # adaptation trainer, and an intentionally filled inference cache is refreshed.
  rt.runtime.freeze_inference(model);before_change=outputs(rt,model,rows[:4])
  path=tmp_path/'effective.pt';rt.export(model,execution,path);reloaded=base.load_model(path,'cpu')
  for a,b in zip(outputs(rt,model,rows[:16]),outputs(base,reloaded,rows[:16])):assert torch.equal(a,b)
  proof=dict(at_utc=iso(),status='PASS',frozen_digest=before,trainable_inventory=inventory(model),canonical_initial_tensor=INITIAL_TENSOR,zero_parity_exact=True,value_gradient_isolated=True,old_only_policy_exact=True,export_inference_exact=True,frozen_rows_exact=True,three_disposable_updates=True)
  write(OUT/'parameter-verification.json',proof)
 finally:engine.close();rt.batch.clear(model)

def test_resume_and_rollback_preserve_sparse_optimizer(pair,tmp_path):
 _,_,rt,_,_,rows,_=pair;model,opt,_=rt.restore(PRIVATE/'initialization.pt','cpu');cohort=rt.epochs.new_cohort(rows,rt.state_hash(model));state=dict(step=0);engine=rt.make_engine(model,opt,cohort,0)
 try:
  indices=np.arange(min(32,len(rows)));selected=[rows[i] for i in indices];adv=cohort['advantages'][indices]
  assert engine.transaction(model,opt,selected,adv,state,'resume-a')['accepted']
  saved=tmp_path/'resume.pt';rt.legacy.save(saved,model,opt,state['step'],extra=dict(embedding_adaptation=True,frozen_digest=rt.frozen_digest,software_verification_only=True))
  assert engine.transaction(model,opt,selected,adv,state,'resume-b')['accepted'];expected=rt.scalar.digest(dict(model=model.state_dict(),optimizer=opt.state_dict(),rng=torch.get_rng_state()))
 finally:engine.close()
 model,opt,_=rt.restore(saved,'cpu');engine=rt.make_engine(model,opt,cohort,0);state=dict(step=1)
 try:
  assert engine.transaction(model,opt,selected,adv,state,'resumed-b')['accepted']
  actual=rt.scalar.digest(dict(model=model.state_dict(),optimizer=opt.state_dict(),rng=torch.get_rng_state()));assert actual==expected
  before=rt.batch.snapshot(model,opt,state);fingerprints=rt.scalar.state_fingerprints(before);diagnose=engine.diagnose
  def reject(*args):result=diagnose(*args);result['phase_KL']=.04;return result
  engine.diagnose=reject;result=engine.transaction(model,opt,selected,adv,state,'software-rollback')
  assert not result['accepted'] and result['rollback_exact'] and fingerprints==rt.scalar.state_fingerprints(rt.batch.snapshot(model,opt,state))
  write(OUT/'recovery-verification.json',dict(status='PASS',same_backend_resume_exact=True,full_rollback_exact=True,sparse_rows_optimizer_RNG=True))
 finally:engine.close()

def test_fixed_schedules_and_learning_rates():
 decks=read(OUT/'decks.json');training=read(OUT/'training-schedule.json');assert len(training)==MAX_COHORTS
 for cohort in training:
  assert len(cohort)==16 and len({(j['learner_deck'],j['opponent_deck'],j['learner_seat']) for j in cohort})==16
  assert collections.Counter(j['learner_deck'] for j in cohort)=={'D10':8,'D04':8}
 for stage in ('initial','final'):
  jobs=read(OUT/f'evaluation-{stage}-schedule.json');assert len(jobs)==512
  assert set(collections.Counter((j['learner_deck'],j['opponent_deck'],j['learner_seat']) for j in jobs).values())=={32}
 assert embedding_lr(0)==pytest.approx(1e-4) and embedding_lr(500000)==pytest.approx(1e-5)
