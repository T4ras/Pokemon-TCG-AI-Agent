"""Full saved-workload numerical checks and timings, never candidate training."""
import resource,shutil,traceback
from unown_round_core import *
import unown_round_lean as lean

def prepare():
 cases=[];inventory={}
 for label,phase in [('first',0),('last',47)]:
  p=SAVED/(label+'-rollout.pt');rows=torch.load(p,map_location='cpu',weights_only=False);order=np.random.default_rng(900005+phase).permutation(len(rows));adv=core.advantages(rows)
  selected=order[:512];remainder=order[len(rows)//128*128:]
  if len(remainder)==0:remainder=order[-17:]
  groups=[('128',selected[:128]),('remainder',remainder),('17',selected[:17])]
  for kind,indices in groups:cases.append(dict(name=label+'-'+kind,phase=phase,rows=[rows[i] for i in indices],adv=adv[indices],indices=indices.tolist()))
  inventory[label]=dict(rows=len(rows),source_sha256=sha(p),permutation_seed=900005+phase,window_indices=selected.tolist(),actual_remainder_indices=remainder.tolist())
  cases.append(dict(name=label+'-window',phase=phase,rows=[rows[i] for i in selected],adv=adv[selected],indices=selected.tolist()))
 torch.save(cases,PRIVATE/'workloads.pt');shutil.copy2(SAVED/'phase-input.pt',PRIVATE/'phase-input.pt')
 historical=ROOT/'data/private/astra-unown-batched-continuation-execution-repair-001';oldrows=torch.load(historical/'verification.pt',map_location='cpu',weights_only=False)[:128];cases.append(dict(name='historical-raw128',phase=-1,rows=oldrows,adv=core.advantages(oldrows),indices=list(range(len(oldrows)))))
 torch.save(cases,PRIVATE/'workloads.pt');shutil.copy2(historical/'start.pt',PRIVATE/'historical-input.pt');inventory['historical']=dict(workload_sha256=sha(historical/'verification.pt'),checkpoint_sha256=sha(historical/'start.pt'),purpose='Directraw reproduction on the retained old verification workload; arithmetic only,never trainingexperience')
 # Initial optimizer state is genuinely empty; mature state is immutable phase47.
 m=load_model(PRIVATE/'initialization.pt','cpu');o=batch.optimizer(m);legacy.save(PRIVATE/'first-input.pt',m,o,0)
 write(OUT/'workload-manifest.json',dict(at_utc=iso(),selection=inventory,workload_sha256=sha(PRIVATE/'workloads.pt'),first_checkpoint_sha256=sha(PRIVATE/'first-input.pt'),late_checkpoint_sha256=sha(PRIVATE/'phase-input.pt'),disposable_only=True))

def sync(device):
 if device=='cuda':torch.cuda.synchronize()
def loaded(case,device):return batch.restore_checkpoint(PRIVATE/('historical-input.pt' if case['phase']==-1 else 'first-input.pt' if case['phase']==0 else 'phase-input.pt'),device,rng=True)[:2]
def outputs(model,rows,kind):
 runtime.freeze_inference(model);answer={}
 with torch.no_grad():
  for ii,g in batch.groups(rows):
   l,v=batch.forward(model,g,raw=kind=='raw')
   for i,ll,vv in zip(ii,l,v):answer[i]=(ll.detach(),vv.detach())
 batch.clear(model);return answer
def distribution(a,b,rows):
 metrics=dict(max_logit=0.,probability_max_abs=0.,TV_max=0.,KL_max=0.,value_max_abs=0.,value_scaled_error=0.,greedy_disagreements=0,near_tie_violations=0,chosen_LP_max=0.);kls=[];tvs=[]
 for i,row in enumerate(rows):
  l,v=a[i];r,w=b[i];mask=torch.as_tensor(row['packet'].row[slice(*row['packet'].layout.offsets['opt_mask'])]>0,device=l.device);assert torch.equal(l.softmax(-1)[~mask],r.softmax(-1)[~mask])
  p=l.softmax(-1);q=r.softmax(-1);d=float((l[mask]-r[mask]).abs().max());tv=float((p-q).abs().sum()/2);kl=float((p*(l.log_softmax(-1)-r.log_softmax(-1))).sum());kls.append(kl);tvs.append(tv)
  metrics['max_logit']=max(metrics['max_logit'],d);metrics['probability_max_abs']=max(metrics['probability_max_abs'],float((p-q).abs().max()));metrics['TV_max']=max(metrics['TV_max'],tv);metrics['KL_max']=max(metrics['KL_max'],kl);metrics['value_max_abs']=max(metrics['value_max_abs'],float((v-w).abs()));metrics['value_scaled_error']=max(metrics['value_scaled_error'],float((v-w).abs()/(1e-4+1e-4*v.abs())))
  metrics['chosen_LP_max']=max(metrics['chosen_LP_max'],float((l.log_softmax(-1)[row['action']]-r.log_softmax(-1)[row['action']]).abs()))
  if int(l.argmax())!=int(r.argmax()):
   metrics['greedy_disagreements']+=1;top=l[mask].topk(min(2,int(mask.sum()))).values;margin=float(top[0]-top[-1]);metrics['near_tie_violations']+=int(margin>2*d+1e-7)
 metrics.update(KL_mean=float(np.mean(kls)),TV_mean=float(np.mean(tvs)));return metrics

def verify(case,device):
 from unown_batched_verify import pad_row
 rows=case['rows'];adv=case['adv'];padded=case['name'].endswith('padded')
 a,ao=loaded(case,device);b,bo=loaded(case,device);scalar=outputs(a,rows,'scalar');raw=outputs(b,rows,'raw');dist=distribution(scalar,raw,rows);criteria=read(OUT/'registration.json')['numerical']['criteria']
 ma=batch.gradients(a,rows,adv);mb,_=lean.gradients(b,rows,adv);aa=dict(a.named_parameters());bb=dict(b.named_parameters());delta=norm=maximum=0.
 for k,p in aa.items():
  q=bb[k];assert (p.grad is None)==(q.grad is None)
  if p.grad is not None:d=(p.grad-q.grad).double();delta+=float(d.square().sum());norm+=float(p.grad.double().square().sum());maximum=max(maximum,float(d.abs().max()))
 relative=(delta/max(norm,1e-30))**.5
 losses={k:dict(old=ma[k],revised=mb[k],accepted=bool(np.isclose(ma[k],mb[k],atol=1e-5,rtol=1e-4))) for k in ma}
 for model,opt in [(a,ao),(b,bo)]:torch.nn.utils.clip_grad_norm_(model.parameters(),1.5,error_if_nonfinite=True);opt.step();model.eval()
 param=max(float((v-b.state_dict()[k]).abs().max()) for k,v in a.state_dict().items());post=distribution(outputs(a,rows,'scalar'),outputs(b,rows,'scalar'),rows)
 # One backend and identical constructor/configuration; exact state/RNG resume.
 legacy.save(PRIVATE/'benchmark-resume.pt',b,bo,1);lean.transaction(b,bo,rows,adv,dict(step=1),'resume-probe');expected=batch.digest(dict(model=b.state_dict(),optimizer=bo.state_dict(),torch=torch.get_rng_state(),numpy=np.random.get_state(),python=random.getstate(),cuda=torch.cuda.get_rng_state_all() if device=='cuda' else None))
 c,co,_=batch.restore_checkpoint(PRIVATE/'benchmark-resume.pt',device,rng=True);lean.transaction(c,co,rows,adv,dict(step=1),'resume-probe');actual=batch.digest(dict(model=c.state_dict(),optimizer=co.state_dict(),torch=torch.get_rng_state(),numpy=np.random.get_state(),python=random.getstate(),cuda=torch.cuda.get_rng_state_all() if device=='cuda' else None));resume=expected==actual
 gates=dict(probability=dist['probability_max_abs']<=5e-5,TV=dist['TV_max']<=1e-4,KL=dist['KL_max']<=1e-4 and dist['KL_mean']<=1e-6,values=dist['value_scaled_error']<=1,losses=all(v['accepted'] for v in losses.values()),gradients=relative<=1e-3 and maximum<=1e-4,optimizer=param<=2.1e-5,postupdate=post['TV_max']<=1e-3 and post['TV_mean']<=1e-4,ties=dist['near_tie_violations']==0,resume=resume)
 result=dict(case=case['name'],n=len(rows),device=device,forward=dist,losses=losses,gradient_relative_L2=relative,gradient_max_abs=maximum,optimizer_parameter_max_abs=param,postupdate=post,resume_exact=resume,gates=gates,status='PASS' if all(gates.values()) else 'REJECTED');append(OUT/(device+'-verification.jsonl'),result);return result

def timing(case,device):
 records=[]
 for repeat in range(3):
  for variant in ['old','revised']:
   model,opt=loaded(case,device);execution=dict(step=0);wall=time.monotonic();timers=collections.Counter();accepted=0;checkpoint_count=0;oldmeters=copy.deepcopy(core.METERS);newmeters=copy.deepcopy(lean.METERS)
   for offset in range(0,len(case['rows']),128):
    rows=case['rows'][offset:offset+128];adv=case['adv'][offset:offset+128];sync(device);t=time.monotonic()
    result=(core.transaction if variant=='old' else lean.transaction)(model,opt,rows,adv,execution,'benchmark-'+variant);sync(device);timers['transaction_seconds']+=time.monotonic()-t
    if not result['accepted']:break
    accepted+=1
    if variant=='old' or execution['step']%8==0 or offset+128>=len(case['rows']):
     t=time.monotonic();legacy.save(PRIVATE/('benchmark-'+variant+'.pt'),model,opt,execution['step']);timers['checkpoint_seconds']+=time.monotonic()-t;checkpoint_count+=1
   if variant=='old':timers.update(diagnostic_seconds=core.METERS['phase_diagnostic_seconds']-oldmeters['phase_diagnostic_seconds'],gradient_optimizer_seconds=core.METERS['optimizer_seconds']-oldmeters['optimizer_seconds'])
   else:
    for k in lean.METERS:timers[k]+=lean.METERS[k]-newmeters[k]
   records.append(dict(repeat=repeat,variant=variant,elapsed_seconds=time.monotonic()-wall,accepted=accepted,checkpoint_count=checkpoint_count,**timers))
 summary={}
 for name in ['old','revised']:
  rs=[r for r in records if r['variant']==name];summary[name]={k:float(np.median([r.get(k,0) for r in rs])) for k in ['elapsed_seconds','transaction_seconds','checkpoint_seconds','accepted','checkpoint_count','diagnostic_seconds','gradient_optimizer_seconds','pre_metrics_seconds','post_diagnostic_seconds','gradient_seconds','optimizer_seconds','snapshot_seconds']}
 result=dict(case=case['name'],decisions=len(case['rows']),device=device,repeats=records,medians=summary,complete_saved_window_speedup=summary['old']['elapsed_seconds']/summary['revised']['elapsed_seconds'],same_accepted_update_counts=summary['old']['accepted']==summary['revised']['accepted']);append(OUT/(device+'-timings.jsonl'),result);return result

def run(device):
 t=time.monotonic();cpu=time.process_time();cases=torch.load(PRIVATE/'workloads.pt',map_location='cpu',weights_only=False);results=[]
 if device=='cuda':
  from unown_pilot_execution import cuda_startup
  cuda_startup(torch,lambda x:write(OUT/'gpu-runtime.json',x));torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
 for case in cases:
  if case['name'].endswith('window'):continue
  if device=='cpu':check_time()
  results.append(verify(case,device))
  if case['name'].endswith('128'):
   from unown_batched_verify import pad_row
   p=dict(case,name=case['name']+'-padded',rows=[pad_row(r) for r in case['rows'][:16]],adv=case['adv'][:16]);results.append(verify(p,device))
 timings=[timing(c,device) for c in cases if c['name'].endswith(('window','remainder'))]
 write(OUT/(device+'-benchmark.json'),dict(status='PASS' if all(r['status']=='PASS' for r in results) and all(r['same_accepted_update_counts'] for r in timings) else 'REJECTED',device=device,verification=results,timings=timings,elapsed_seconds=time.monotonic()-t,CPU_core_hours=(time.process_time()-cpu)/3600,disposable_weights_not_candidates=True))
if __name__=='__main__':
 import sys
 try:prepare() if sys.argv[1]=='prepare' else run(sys.argv[1])
 except BaseException as e:write(OUT/'benchmark-failure.json',dict(at_utc=iso(),error=str(e),traceback=traceback.format_exc()));raise
