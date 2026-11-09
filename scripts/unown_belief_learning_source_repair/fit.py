"""One capped supervised fit, with immutable match splits and recovery snapshots."""
from .common import *
from .preprocess import load_features
from .models import *
from unown_belief_bayes_repair.deck import catalogue,fit_theta,deck_weights
import pickle,signal,traceback,resource,copy

class FitLimit(RuntimeError):pass

def allowed_items(partition):
 if partition not in ('training','validation'):raise PermissionError('Locked test is excluded from model workflow')
 return [x for x in read(OUT/'split-manifest.json')['entries'] if x['partition']==partition]

def fit_key(job):return f"{job['family']}-n{job['size']}-s{job['seed']}"

def bayes_predict(theta,beta,f):
 pop=catalogue()[3];ds=[];cp=[]
 for j,x in enumerate(f['ledger']):
  w=deck_weights(x,f['deck_A'][j]>0,pop,theta,beta);lo,hi=f['conditional_offsets'][j:j+2];inds=f['conditional_indices'][lo:hi];ds.append(w);cp.append(np.einsum('d,dqzs->qzs',w[inds],f['conditionals'][lo:hi]))
 return np.array(ds),np.array(cp)

def predict(model,f,family):
 from .distributions import mask_counts
 if family=='A':d,p=f['deck_A'],f['A']
 elif family=='Bayes':d,p=bayes_predict(model['theta'],model['beta'],f)
 elif family=='Trees':d,p=model.predict(f)
 else:d,p=neural_predict(model,f)
 p=mask_counts(p,f['lower'],f['upper']);d=np.maximum(d,0)*(f['deck_A']>0);d/=d.sum(-1,keepdims=True)
 return d,p

def validation_loss(model,family,items,check):
 loss=0.
 for item in items:
  check();f,t=load_features(item['id']);d,p=predict(model,f,family);w=point_weights(f['games']);deck=-np.log(np.maximum(d[np.arange(len(d)),t['deck']],1e-30));prob=np.take_along_axis(p,t['counts'][...,None],axis=-1)[...,0];count=-np.log(np.maximum(prob,1e-30)).mean((1,2));loss+=prior()[item['opponent_list']]['prior']/16*float(w@(deck+count))
 return loss

def main(job):
 import torch
 torch.set_num_threads(3);torch.set_num_interop_threads(1)
 start=time.process_time();limit=fit_budget(job);key=fit_key(job);root=PRIVATE/'models'/key;root.mkdir(parents=True,exist_ok=True);last_model=None;state='INCOMPLETE';epochs=0;best_loss=float('inf');best=None
 exclusive(OUT/'fit-intents'/(key+'.json'),dict(job=job,at_utc=iso(),pid=os.getpid()));write(OUT/'active-fit.json',dict(key=key,pid=os.getpid(),at_utc=iso(),job=job))
 def check():
  check_time()
  if time.process_time()-start>=limit-30:raise FitLimit('Registered size/family CPU cap with30s export reserve')
 def abort(*_):raise FitLimit('Fit CPU/session cap or controller stop')
 signal.signal(signal.SIGPROF,abort);signal.signal(signal.SIGTERM,abort);signal.setitimer(signal.ITIMER_PROF,limit-30)
 try:
  ids=read(OUT/'training-subsets.json')[str(job['size'])];byid={x['id']:x for x in allowed_items('training')};train=[byid[i] for i in ids];valid=allowed_items('validation');assert len(train)==37*job['size'] and len(valid)==592
  if job['family']=='Bayes':
   decks,cards,counts,pop=catalogue();features=[load_features(i)[0]['final_ledger'] for i in ids];labels=[decks.index(byid[i]['opponent_list']) for i in ids];grid=[]
   for beta in (.25,.5,1.):
    for alpha in (120,30):
     check();theta=fit_theta(features,labels,counts,alpha);loss=0.
     for item in valid:
      check();f,t=load_features(item['id']);ds=np.array([deck_weights(x,a>0,pop,theta,beta) for x,a in zip(f['ledger'],f['deck_A'])]);loss+=prior()[item['opponent_list']]['prior']/16*float(point_weights(f['games'])@(-np.log(np.maximum(ds[np.arange(len(ds)),t['deck']],1e-30))))
     grid.append(dict(alpha=alpha,beta=beta,loss=loss))
   choice=grid[0]
   for x in grid[1:]:
    if x['loss']<choice['loss']-1e-12:choice=x
   last_model=dict(theta=fit_theta(features,labels,counts,choice['alpha']),beta=choice['beta'],alpha=choice['alpha']);npwrite(root/'model.npz',**last_model);write(root/'selection.json',dict(grid=grid,choice=choice,validation_only=True));epochs=1
  elif job['family']=='Trees':
   last_model=BoostedTrees(job['seed']);groups={}
   for d in sorted(prior()):
    for seat in (0,1):groups[(d,seat)]=sorted([x for x in train if x['opponent_list']==d and x['chooser']==seat],key=lambda x:stable(job['seed'],'tree-batch',x['id']))
   for stage in range(32):
    check();batch=[]
    for g in groups.values():batch.extend(g[(stage*4+j)%len(g)] for j in range(min(4,len(g))))
    last_model.stage(batch,stage,load_features,check);epochs=stage+1
    with (root/'recovery.pkl').open('wb') as f:pickle.dump(last_model,f)
    write(root/'progress.json',dict(stages=epochs,CPU_seconds=time.process_time()-start))
   with (root/'model.pkl').open('wb') as f:pickle.dump(last_model,f)
  else:
   last_model=make_neural(job['family'],job['seed']);optimizer=torch.optim.AdamW(last_model.parameters(),lr=.001,weight_decay=.0001);rng=np.random.default_rng(job['seed']);bad=0;curve=[]
   for epoch in range(6):
    last_model.train();total=0.
    for i in rng.permutation(len(train)):
     check();item=train[int(i)];f,t=load_features(item['id']);optimizer.zero_grad(set_to_none=True);loss=neural_loss(last_model,f,t)*37*prior()[item['opponent_list']]['prior']
     if not torch.isfinite(loss):raise InvariantFailure('Nonfinite supervised loss')
     loss.backward();norm=torch.nn.utils.clip_grad_norm_(last_model.parameters(),1.)
     if not torch.isfinite(norm):raise InvariantFailure('Nonfinite supervised gradient')
     optimizer.step();total+=float(loss.detach())
    epochs=epoch+1;score=validation_loss(last_model,job['family'],valid,check);curve.append(dict(epoch=epochs,validation_NLL=score,training_loss=total/len(train),CPU_seconds=time.process_time()-start))
    if score<best_loss-1e-5:best_loss=score;best=copy.deepcopy(last_model.state_dict());bad=0
    else:bad+=1
    torch.save(dict(model=last_model.state_dict(),optimizer=optimizer.state_dict(),torch_rng=torch.get_rng_state(),numpy_rng=rng.bit_generator.state,epochs=epochs,best=best,best_loss=best_loss),root/'recovery.pt');write(root/'curve.json',curve)
    if bad>=3:break
   last_model.load_state_dict(best);torch.save(dict(state_dict=last_model.state_dict(),family=job['family'],seed=job['seed'],epochs=epochs,validation_NLL=best_loss,parameters=sum(p.numel() for p in last_model.parameters())),root/'model.pt')
  check();state='COMPLETE'
 except BaseException as exc:
  write(root/'failure.json',dict(at_utc=iso(),type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc(),epochs_or_stages=epochs));raise
 finally:
  signal.setitimer(signal.ITIMER_PROF,0);files={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in root.iterdir() if p.is_file()};write(OUT/'fit-results'/(key+'.json'),dict(at_utc=iso(),job=job,key=key,state=state,epochs_or_stages=epochs,CPU_seconds=time.process_time()-start,model_directory=str(root.relative_to(R)),files=files,peak_RSS_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,locked_test_accessed=False))

def load_model(job):
 key=fit_key(job);record=read(OUT/'fit-results'/(key+'.json'))
 if record['state']!='COMPLETE':raise InvariantFailure('Incomplete fit cannot be evaluated as a candidate')
 root=R/record['model_directory']
 if job['family']=='Bayes':
  with np.load(root/'model.npz') as x:return {k:x[k] for k in x.files}
 if job['family']=='Trees':
  with (root/'model.pkl').open('rb') as f:return pickle.load(f)
 import torch
 model=make_neural(job['family'],job['seed']);data=torch.load(root/'model.pt',map_location='cpu',weights_only=False);model.load_state_dict(data['state_dict']);return model

if __name__=='__main__':
 import sys
 main(read(Path(sys.argv[1])))
