"""Frozen scalar/batch comparisons, disposable updates and honest timing."""
from unown_batched_core import *

def pad_row(r):
 from unown_v2_runtime import Layout,Packet
 p=r['packet'];old=p.layout;new=Layout(tokens=old.tokens*2,options=old.options+17,tools=old.tools,oracle=old.oracle)
 a=np.zeros(new.size,np.float32)
 for field in old.offsets:
  src=p.row[slice(*old.offsets[field])];dest=a[slice(*new.offsets[field])]
  if field in ['tok_int','tok_float','opt_int','opt_float','opt_mask']:
   width={'tok_int':3+old.tools,'tok_float':56,'opt_int':6,'opt_float':8,'opt_mask':1}[field]
   if field.startswith('opt'):
    dest[:(old.options-1)*width]=src[:(old.options-1)*width]
    dest[(new.options-1)*width:new.options*width]=src[(old.options-1)*width:old.options*width]
   else:dest[:len(src)]=src
  else:dest[:]=src
 out=dict(r,packet=Packet(a,new),scalar_reference_packet=p,action=new.stop if r['action']==old.stop else r['action'])
 for field in ['old_probs','initial_probs']:
  if field in r:
   v=np.zeros(new.options,np.float32);v[:old.options-1]=r[field][:old.options-1];v[-1]=r[field][-1];out[field]=v
 return out

def assert_close(a,b,atol,rtol=0):torch.testing.assert_close(a,b,atol=atol,rtol=rtol)

@torch.no_grad()
def compare(model,rows,exact_length=True,physical=16,padded=False):
 runtime.freeze_inference(model);maxima=dict(logit=0.,value=0.,logprob=0.,TV=0.,KL=0.,mask_errors=0)
 refs=[]
 for r in rows:
  l,v=legacy.forward(model,[r]);refs.append((l[0],v[0]))
 for indices,group in groups(rows,physical,exact_length):
  feed=[pad_row(r) for r in group] if padded else group
  l,v=forward(model,feed)
  for j,i in enumerate(indices):
   a,b=refs[i];x=l[j]
   if padded:x=torch.cat([x[:len(a)-1],x[-1:]])
   mask=rows[i]['packet'].row[slice(*rows[i]['packet'].layout.offsets['opt_mask'])]>0
   assert float(x.softmax(-1)[~mask].sum())==0
   assert_close(x,a,1e-4,1e-5);assert_close(v[j],b,1e-5,1e-5)
   lp=x.log_softmax(-1);ref=a.log_softmax(-1);p=ref.exp();tv=float((lp.exp()-p).abs().sum()/2)
   difference=abs(float(lp[rows[i]['action']]-ref[rows[i]['action']]));kl=float((p*(ref-lp)).sum())
   maxima['logit']=max(maxima['logit'],float((x[mask]-a[mask]).abs().max()));maxima['value']=max(maxima['value'],abs(float(v[j]-b)))
   maxima['logprob']=max(maxima['logprob'],difference);maxima['TV']=max(maxima['TV'],tv);maxima['KL']=max(maxima['KL'],kl)
   if difference>1e-5 or tv>1e-5 or kl>1e-6:
    write(OUT/'batch-probability-failure.json',dict(device=str(x.device),row=i,exact_length=exact_length,physical=physical,padded=padded,metrics=maxima,
     token_count=token_count(rows[i]),group_token_counts=[token_count(r) for r in group],legal_options=int(mask.sum()),chosen_action=rows[i]['action']))
    raise AssertionError('Unchanged likelihood/distribution tolerance exceeded')
 clear(model);return maxima

def grad_compare(rows,device,padded=False):
 a,ao,_=restore_checkpoint(PRIVATE/'start.pt',device,rng=False);b,bo,_=restore_checkpoint(PRIVATE/'start.pt',device,rng=False)
 adv=advantages(rows);ma=gradients(a,rows,adv,False);mb=gradients(b,[pad_row(r) for r in rows] if padded else rows,adv,True)
 ga=dict(a.named_parameters());gb=dict(b.named_parameters());delta=0.;norm=0.;maximum=0.
 for name,p in ga.items():
  q=gb[name];assert (p.grad is None)==(q.grad is None),name
  if p.grad is None:continue
  d=(p.grad-q.grad).double();delta+=float(d.square().sum());norm+=float(p.grad.double().square().sum());maximum=max(maximum,float(d.abs().max()))
 relative=(delta/max(norm,1e-30))**.5
 suffix='-padded' if padded else ''
 result=dict(decisions=len(rows),synthetic_extra_padding=padded,gradient_relative_L2=relative,gradient_max_abs=maximum,scalar_losses=ma,batch_losses=mb)
 write(OUT/(device+f'-gradient-{len(rows)}{suffix}.json'),result)
 assert relative<=1e-4 and maximum<=1e-5,'Averaged gradient disagreement'
 for k in ma:assert np.isclose(ma[k],mb[k],atol=1e-6,rtol=1e-5),k
 torch.nn.utils.clip_grad_norm_(a.parameters(),1.5,error_if_nonfinite=True);torch.nn.utils.clip_grad_norm_(b.parameters(),1.5,error_if_nonfinite=True);ao.step();bo.step()
 change=0.
 for name,p in a.state_dict().items():
  assert_close(p,b.state_dict()[name],1e-6,1e-5)
  if p.is_floating_point():change=max(change,float((p-b.state_dict()[name]).abs().max()))
 sa,sb=ao.state_dict(),bo.state_dict();assert sa['param_groups']==sb['param_groups']
 moment_delta=0.
 for key,values in sa['state'].items():
  for name,value in values.items():
   other=sb['state'][key][name]
   if torch.is_tensor(value):
    assert_close(value,other,1e-6,1e-5);moment_delta=max(moment_delta,float((value-other).abs().max()))
   else:assert value==other
 result['optimizer_parameter_max_difference']=change
 result['optimizer_state_max_difference']=moment_delta
 # The resumed and uninterrupted optimized paths use the same state, packets,
 # optimizer moments and RNG; no parameter is borrowed from these probes.
 path=PRIVATE/(device+'-disposable-resume.pt');legacy.save(path,b,bo,1)
 step(b,bo,rows,adv);uninterrupted_state=dict(model=copy.deepcopy(b.state_dict()),optimizer=copy.deepcopy(bo.state_dict()),torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),CUDA_rng=torch.cuda.get_rng_state_all() if device=='cuda' else None);uninterrupted=digest(uninterrupted_state)
 c,co,_=restore_checkpoint(path,device,rng=True);step(c,co,rows,adv)
 resumed_state=dict(model=c.state_dict(),optimizer=co.state_dict(),torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),CUDA_rng=torch.cuda.get_rng_state_all() if device=='cuda' else None);resumed=digest(resumed_state)
 if uninterrupted!=resumed:
  differences={name:dict(uninterrupted=digest(v),resumed=digest(resumed_state[name]),equal=digest(v)==digest(resumed_state[name])) for name,v in uninterrupted_state.items()}
  differences['parameter_max_difference']=max(float((p-c.state_dict()[name]).abs().max()) for name,p in b.state_dict().items())
  write(OUT/(device+f'-resume-{len(rows)}-failure.json'),differences)
 assert uninterrupted==resumed,'Save/resume model+optimizer+RNG mismatch'
 result['resume_exact']=True;write(OUT/(device+f'-gradient-{len(rows)}{suffix}.json'),result);return result

def benchmark(rows,device):
 # Retained rollout files predate attaching the initialization reference.
 # Compute that diagnostic channel separately; stored old_probs stay intact.
 anchor=load_model(PRIVATE/'initialization.pt',device);attach_initial(rows,anchor)
 cohort=torch.load(PRIVATE/'last-rollout.pt',map_location='cpu',weights_only=False)
 fixed=torch.load(PRIVATE/'fixed-diagnostic.pt',map_location='cpu',weights_only=False)
 attach_initial(cohort,anchor);attach_initial(fixed,anchor)
 times={};adv=advantages(rows)
 for batched,label in [(False,'scalar'),(True,'batched')]:
  model,opt,_=restore_checkpoint(PRIVATE/'start.pt',device,rng=False)
  if device=='cuda':torch.cuda.synchronize()
  begin=time.monotonic();pre=diagnose(model,rows,batched);step(model,opt,rows,adv,batched);post=diagnose(model,rows,batched)
  legacy.save(PRIVATE/(device+'-'+label+'-benchmark.pt'),model,opt,1)
  if device=='cuda':torch.cuda.synchronize()
  elapsed=time.monotonic()-begin
  begin=time.monotonic();diagnose(model,cohort,batched);diagnose(model,fixed,batched)
  if device=='cuda':torch.cuda.synchronize()
  boundary=time.monotonic()-begin
  phase=687.119350383/7+int(np.ceil(len(cohort)/128))*elapsed+boundary
  times[label]=dict(seconds=elapsed,logical_decisions=len(rows),includes='Pre/post full-distribution diagnostics, one logical update, atomic model+optimizer+RNG checkpoint',pre=pre,post=post,
   full_saved_cohort_decisions=len(cohort),fixed_diagnostic_decisions=len(fixed),cohort_boundary_diagnostic_seconds=boundary,
   projected_full_collection_update_diagnostic_phase_seconds=phase,projected_steady_BO3_per_hour=16/phase*3600,
   projection_limitation='Previous actual collection plus saved-cohort actual decision count. Initial cohort likelihood/reference computation and provision/export are separately measured by the real job, so steady-rate projection is optimistic rather than an all-in guarantee.')
 ratio=times['scalar']['seconds']/times['batched']['seconds']
 result=dict(device=device,workload=times,speedup=ratio,not_promoted=True,
  historical_collection_BO3_per_second=112/687.119350383,
  limitation='Projection uses previous measured native collection and observed decisions/update mix; the new job reports actual total throughput including startup/checks/fixed-KL and export. CPU and GPU comparisons are never pooled.')
 write(OUT/(device+'-benchmark.json'),result);return result

def verify(device='cpu',limit=None):
 rows=torch.load(PRIVATE/'verification.pt',map_location='cpu',weights_only=False)
 model,opt,_=restore_checkpoint(PRIVATE/'start.pt',device,rng=False)
 result=dict(device=device,registered_tolerances=read(OUT/'registration.json')['numerical_tolerances'],comparisons={})
 cases=[('exact_length',rows,True,16,False),('reverse_composition',list(reversed(rows[:96])),True,7,False),('mixed_padding',rows[:64],False,16,False),('enlarged_layout',rows[:32],True,16,True)]
 for name,sample,exact,physical,padded in cases:
  if limit and time.monotonic()>=limit:raise TimeoutError('Thirty-minute GPU pretraining check cap')
  result['comparisons'][name]=compare(model,sample,exact,physical,padded);progress('VERIFICATION_CASE',device=device,case=name)
 result['gradients']=[]
 adoption=read(OUT/'completed-gradient-stages.json') if device=='cpu' and (OUT/'completed-gradient-stages.json').exists() else {}
 reusable=adoption.get('core_sha256')==sha(REPO/'scripts/unown_batched_core.py') and adoption.get('verification_sha256')==sha(PRIVATE/'verification.pt')
 for n in [128,51,1]:
  path=OUT/(device+f'-gradient-{n}.json')
  if reusable and path.exists() and adoption.get('files',{}).get(path.name)==sha(path):
   done=read(path);assert done['resume_exact'];result['gradients'].append(done)
  else:result['gradients'].append(grad_compare(rows[:n],device))
 result['gradients'].append(grad_compare(rows[:51],device,padded=True))
 write(OUT/(device+'-verification-stages.json'),result)
 result['benchmark']=benchmark(rows[:128],device)
 result.update(status='PASS',deterministic_algorithms=True,CPU_threads=1,autograd_parallel_branches=False,source='Immutable retained packets; no rebasing, truncation or promoted probe weights',
  mismatch_investigation='Raw batched float32 GEMM failed the unchanged1e-5 chosen-logprob gate; immutable first failure retained. Production preserves current scalar forward values using detached numerical residuals and differentiates the same operations in batches; stored rollout probabilities are never changed. Independent losses/gradients/update tests validate this hybrid. Exact-valid-length bucketing limits padding; synthetic extra padding retains the original scalar packet as its canonical reference.')
 write(OUT/(device+'-verification.json'),result);return result

if __name__=='__main__':
 torch.set_num_threads(2);verify('cpu')
