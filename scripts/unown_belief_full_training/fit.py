"""One capped supervised fit, with immutable match splits and recovery snapshots."""
from .common import *
from .preprocess import load_features
from .models import *
from unown_belief_bayes_repair.deck import catalogue,deck_weights
from .preprocess import LOAD_STATS
import pickle,signal,traceback,resource,copy

def fit_theta(revealed,labels,deck_counts,alpha,weights=None):
 """Same conditional estimator/smoothing with equal chooser influence per deck."""
 revealed=np.asarray(revealed,dtype=float);labels=np.asarray(labels,dtype=int);weights=np.ones(len(labels)) if weights is None else np.asarray(weights,dtype=float);counts=np.zeros_like(deck_counts,dtype=float)
 if len(revealed)!=len(labels) or len(weights)!=len(labels) or np.any(weights<=0):raise InvariantFailure('Invalid Bayesian training weights')
 for x,y,w in zip(revealed,labels,weights):
  if np.any(x<0) or np.any(x>deck_counts[y]):raise InvariantFailure('Training reveal ledger exceeds labelled deck')
  counts[y]+=w*x
 theta=(counts+float(alpha)*deck_counts/60.)/(counts.sum(axis=1,keepdims=True)+float(alpha));assert np.allclose(theta.sum(axis=1),1);return theta

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
 loss=deck_total=count_total=0.
 for item in items:
  check();f,t=load_features(item['id']);d,p=predict(model,f,family);w=point_weights(f['games']);deck=-np.log(np.maximum(d[np.arange(len(d)),t['deck']],1e-30));prob=np.take_along_axis(p,t['counts'][...,None],axis=-1)[...,0];count=-np.log(np.maximum(prob,1e-30)).mean((1,2));weight=prior()[item['opponent_list']]['prior']/16;loss+=weight*float(w@(deck+count));deck_total+=weight*float(w@deck);count_total+=weight*float(w@count)
 return dict(combined=loss,deck=deck_total,count=count_total)

def main(job):
 import torch
 torch.set_num_threads(3);torch.set_num_interop_threads(1)
 wall=time.monotonic();start=time.process_time();seen=set();exposure=collections.Counter();updates=0;best_epoch=None;best_components=None;limit=fit_budget(job);key=fit_key(job);root=PRIVATE/'models'/key;root.mkdir(parents=True,exist_ok=True);last_model=None;state='INCOMPLETE';epochs=0;best_loss=float('inf');best=None
 exclusive(OUT/'fit-intents'/(key+'.json'),dict(job=job,at_utc=iso(),pid=os.getpid()));write(OUT/'active-fit.json',dict(key=key,pid=os.getpid(),at_utc=iso(),job=job))
 def check():
  check_time()
  if time.process_time()-start>=limit-30:raise FitLimit('Registered size/family CPU cap with30s export reserve')
 def abort(*_):raise FitLimit('Fit CPU/session cap or controller stop')
 signal.signal(signal.SIGPROF,abort);signal.signal(signal.SIGTERM,abort);signal.setitimer(signal.ITIMER_PROF,limit-30)
 try:
  ids=read(OUT/'training-subsets.json')[str(job['size'])];byid={x['id']:x for x in allowed_items('training')};train=[byid[i] for i in ids];valid=allowed_items('validation');assert len(train)==2334 and len(valid)==592;training_weights={x['id']:x for x in read(OUT/'full-training-manifest.json')['entries']}
  if job['family']=='Bayes':
   decks,cards,counts,pop=catalogue();features=[load_features(i)[0]['final_ledger'] for i in ids];labels=[decks.index(byid[i]['opponent_list']) for i in ids];weights=[training_weights[i]['bayes_conditional_weight'] for i in ids];seen.update(ids);exposure.update(ids);grid=[]
   for beta in (.25,.5,1.):
    for alpha in (120,30):
     check();theta=fit_theta(features,labels,counts,alpha,weights);loss=0.
     for item in valid:
      check();f,t=load_features(item['id']);ds=np.array([deck_weights(x,a>0,pop,theta,beta) for x,a in zip(f['ledger'],f['deck_A'])]);loss+=prior()[item['opponent_list']]['prior']/16*float(point_weights(f['games'])@(-np.log(np.maximum(ds[np.arange(len(ds)),t['deck']],1e-30))))
     grid.append(dict(alpha=alpha,beta=beta,loss=loss))
   choice=grid[0]
   for x in grid[1:]:
    if x['loss']<choice['loss']-1e-12:choice=x
   last_model=dict(theta=fit_theta(features,labels,counts,choice['alpha'],weights),beta=choice['beta'],alpha=choice['alpha']);npwrite(root/'model.npz',**last_model);write(root/'selection.json',dict(grid=grid,choice=choice,validation_only=True));epochs=1
  elif job['family']=='Trees':
   last_model=BoostedTrees(job['seed']);schedule=read(OUT/f'tree-schedule-s{job["seed"]}.json')['stages']
   for stage in range(32):
    check();batch=[byid[i] for i in schedule[stage]]
    last_model.stage(batch,stage,load_features,check);epochs=stage+1;seen.update(x['id'] for x in batch);exposure.update(x['id'] for x in batch)
    with (root/'recovery.pkl').open('wb') as f:pickle.dump(last_model,f)
    write(root/'progress.json',dict(stages=epochs,CPU_seconds=time.process_time()-start))
   with (root/'model.pkl').open('wb') as f:pickle.dump(last_model,f)
  else:
   last_model=make_neural(job['family'],job['seed']);optimizer=torch.optim.AdamW(last_model.parameters(),lr=.001,weight_decay=.0001);rng=np.random.default_rng(job['seed']);bad=0;curve=[]
   for epoch in range(6):
    last_model.train();total=0.
    for i in rng.permutation(len(train)):
     check();item=train[int(i)];f,t=load_features(item['id']);optimizer.zero_grad(set_to_none=True);loss=neural_loss(last_model,f,t)*training_weights[item['id']]['neural_loss_weight']
     if not torch.isfinite(loss):raise InvariantFailure('Nonfinite supervised loss')
     loss.backward();norm=torch.nn.utils.clip_grad_norm_(last_model.parameters(),1.)
     if not torch.isfinite(norm):raise InvariantFailure('Nonfinite supervised gradient')
     optimizer.step();updates+=1;seen.add(item['id']);exposure.update([item['id']]);total+=float(loss.detach())
    epochs=epoch+1;components=validation_loss(last_model,job['family'],valid,check);score=components['combined'];curve.append(dict(epoch=epochs,validation_NLL=score,validation_components=components,training_loss=total/len(train),optimizer_updates=updates,unique_training_BO3_seen=len(seen),CPU_seconds=time.process_time()-start))
    if score<best_loss-1e-5:best_epoch=epochs;best_components=components;best_loss=score;best=copy.deepcopy(last_model.state_dict());bad=0
    else:bad+=1
    torch.save(dict(model=last_model.state_dict(),optimizer=optimizer.state_dict(),torch_rng=torch.get_rng_state(),numpy_rng=rng.bit_generator.state,epochs=epochs,best=best,best_loss=best_loss,optimizer_updates=updates,exposure_counts=dict(exposure),best_epoch=best_epoch,best_components=best_components),root/'recovery.pt');write(root/'curve.json',curve)
    if bad>=3:break
   last_model.load_state_dict(best);torch.save(dict(state_dict=last_model.state_dict(),family=job['family'],seed=job['seed'],epochs=epochs,validation_NLL=best_loss,best_epoch=best_epoch,selected_losses=best_components,parameters=sum(p.numel() for p in last_model.parameters())),root/'model.pt')
  check()
  if len(seen)!=2334:raise InvariantFailure('Complete fit did not use every valid training BO3')
  if job['family']=='Trees' and min(exposure.values())<2:raise InvariantFailure('Tree exposure coverage failed')
  state='COMPLETE'
 except BaseException as exc:
  write(root/'failure.json',dict(at_utc=iso(),type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc(),epochs_or_stages=epochs));raise
 finally:
  signal.setitimer(signal.ITIMER_PROF,0);files={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in root.iterdir() if p.is_file()};write(OUT/'fit-results'/(key+'.json'),dict(at_utc=iso(),job=job,key=key,state=state,epochs_or_stages=epochs,CPU_seconds=time.process_time()-start,model_directory=str(root.relative_to(R)),files=files,peak_RSS_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,locked_test_accessed=False,elapsed_seconds=time.monotonic()-wall,unique_training_BO3_seen=len(seen),optimizer_updates=updates,exposure_counts=dict(exposure),best_epoch=best_epoch,selected_validation_losses=best_components,cache_loading=dict(LOAD_STATS),parameter_details=parameter_details(last_model,job['family'])))

def parameter_details(model,family):
 if model is None:return {}
 if family in ('MLP','GRU'):return dict(trainable_parameters=sum(p.numel() for p in model.parameters()))
 if family=='Bayes':return dict(theta_values=int(model['theta'].size),normalized_theta_free_parameters=int(model['theta'].shape[0]*(model['theta'].shape[1]-1)),alpha=float(model['alpha']),beta=float(model['beta']))
 trees=model.deck+model.count;internal=sum(int((t.tree_.children_left>=0).sum()) for t in trees);leaf_values=sum(int(t.tree_.value[t.tree_.children_left<0].size) for t in trees)
 return dict(trees=len(trees),split_thresholds=internal,leaf_prediction_values=leaf_values,fitted_numeric_parameters=internal+leaf_values,discrete_split_features=internal)

def load_model(job):
 key=fit_key(job);record=read(OUT/'fit-results'/(key+'.json'))
 if record['state']!='COMPLETE':raise InvariantFailure('Incomplete fit cannot be evaluated as a candidate')
 root=R/record['model_directory']
 for name,v in record['files'].items():
  if sha(root/name)!=v['sha256']:raise InvariantFailure('Fitted artifact changed: '+name)
 if job['family']=='Bayes':
  with np.load(root/'model.npz') as x:return {k:x[k] for k in x.files}
 if job['family']=='Trees':
  with (root/'model.pkl').open('rb') as f:return pickle.load(f)
 import torch
 model=make_neural(job['family'],job['seed']);data=torch.load(root/'model.pt',map_location='cpu',weights_only=False);model.load_state_dict(data['state_dict']);return model

if __name__=='__main__':
 import sys
 main(read(Path(sys.argv[1])))
