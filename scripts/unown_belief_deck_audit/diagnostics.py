"""Read-only checkpoint, loss, input and gradient diagnostics."""
from .common import *
from unown_belief_full_training.models import make_neural,point_weights
from unown_belief_full_training.fit import predict,bayes_predict
from unown_belief_full_training.preprocess import load_features
from unown_belief_full_training.scoring import decision_scores,aggregate,METRICS
from unown_belief_bayes_repair.deck import catalogue
import numpy as np,torch,copy

TORCH_F=torch.nn.functional

def digest(state):
 h=hashlib.sha256()
 for k,v in sorted(state.items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()

def components(model,f,t):
 d,c=model(f);N,Q,Z,S=c.shape;yd=torch.as_tensor(t['deck'],dtype=torch.long);yc=torch.as_tensor(t['counts'],dtype=torch.long);w=torch.as_tensor(point_weights(f['games']))
 deck=(TORCH_F.cross_entropy(d,yd,reduction='none')*w).sum();count=(TORCH_F.cross_entropy(c.reshape(-1,S),yc.reshape(-1),reduction='none').reshape(N,Q,Z).mean((1,2))*w).sum()
 return deck,count,d,c

def initial_counts(f):
 p=np.maximum(f['A'],1e-5);k=np.arange(p.shape[-1]);p*=((k>=f['lower'][...,None])&(k<=f['upper'][...,None]));return p/p.sum(-1,keepdims=True)

class Accumulator:
 def __init__(self,entries):
  self.entries={x['id']:x for x in entries};self.counts=collections.Counter((x['opponent_list'],x['chooser']) for x in entries);self.num={p:collections.Counter() for p in ('eligible','full_field','collection')};self.den={p:collections.Counter() for p in self.num};self.mass={p:np.zeros(37) for p in self.num};self.perdeck={};self.matches=0
 def add(self,item,metrics,mass):
  self.matches+=1;d=item['opponent_list'];s=item['chooser'];n=self.counts[d,s]
  for pop in self.num:
   if pop=='eligible' and prior()[d]['group']!='passing':continue
   weight=1/len(self.entries) if pop=='collection' else prior()[d]['entries']/(82 if pop=='eligible' else 100)/(2*n);self.mass[pop]+=weight*mass
   for k,v in metrics.items():
    if v is not None:self.num[pop][k]+=weight*v;self.den[pop][k]+=weight
  pd=self.perdeck.setdefault(d,dict(chooser={}))['chooser'].setdefault(str(s),dict(count=0,num=collections.Counter(),den=collections.Counter()));pd['count']+=1
  for k,v in metrics.items():
   if v is not None:pd['num'][k]+=v;pd['den'][k]+=1
 def result(self):
  decks={}
  for d,r in self.perdeck.items():
   decks[d]={k:sum(v['num'][k]/v['den'][k]/2 for v in r['chooser'].values() if v['den'][k]) for k in set().union(*(set(v['num']) for v in r['chooser'].values()))}
  return dict(matches=self.matches,populations={p:dict(metrics={k:self.num[p][k]/self.den[p][k] for k in self.num[p] if self.den[p][k]},coverage=dict(self.den[p]),predicted_deck_mass=self.mass[p].tolist()) for p in self.num},per_deck=decks)

def build_bank(check):
 bank={};aliases={};info={};recoveries={};curves=[]
 for rec in read(OUT/'models.json'):
  check();fit=read(R/rec['fit_record']);root=R/fit['model_directory']
  for name,v in fit['files'].items():
   if sha(root/name)!=v['sha256']:raise InvariantFailure('Historical checkpoint/curve changed')
  chosen=torch.load(R/rec['model'],map_location='cpu',weights_only=False);last=torch.load(R/rec['recovery'],map_location='cpu',weights_only=False)
  states={'selected':chosen['state_dict'],'last':last['model']}
  if digest(chosen['state_dict'])!=digest(last['best']):raise InvariantFailure('Selected tensor is not saved best')
  curve=read(R/rec['curve']);best_loss=float('inf');best_epoch=None;bad=0
  for row in curve:
   if row['validation_NLL']<best_loss-1e-5:best_loss=row['validation_NLL'];best_epoch=row['epoch'];bad=0
   else:bad+=1
  if abs(float(chosen['validation_NLL'])-best_loss)>5e-6:raise InvariantFailure('Selected loss does not match frozen patience rule')
  curves.append(dict(model=rec['name'],rows=curve,selected_epoch=best_epoch,last_epoch=last['epochs'],selected_NLL=float(chosen['validation_NLL']),initial_eligible=False,unsaved_epoch_training_components='unavailable; only online combined training_loss was retained',checkpoint_selected_matches_saved_best=True))
  for kind,state in states.items():
   name=rec['name']+'-'+kind;key=digest(state);aliases[name]=key
   if key not in bank:
    model=make_neural(rec['family'],rec['seed']);model.load_state_dict(state);model.eval();bank[key]=model
   info[name]=dict(rec,kind=kind,tensor_sha256=key,epoch=best_epoch if kind=='selected' else last['epochs'])
  if rec['origin']=='full':recoveries[rec['name']]=last
 for family in ('MLP','GRU'):
  for seed in (11,29):
   name=f'zero-{family}-s{seed}';model=make_neural(family,seed);model.eval();key=digest(model.state_dict());aliases[name]=key;bank[key]=model;info[name]=dict(family=family,seed=seed,kind='zero',origin='zero',tensor_sha256=key,epoch=0)
 write(OUT/'checkpoint-selection.json',curves);write(OUT/'checkpoint-map.json',dict(aliases=aliases,info=info,unique_states=len(bank)))
 return bank,aliases,info,recoveries

def score(f,t,d,p):
 values,_=decision_scores(f,t,d,p);metrics=dict(zip(METRICS,aggregate(values,f['games'])));w=point_weights(f['games']);mass=w@d
 compatible=f['deck_A']>0;uniform=compatible/compatible.sum(1,keepdims=True);safe=np.maximum(d,1e-30);metrics['KL_to_uniform_compatible']=float(w@np.sum(d*(np.log(safe)-np.log(np.maximum(uniform,1e-30))),axis=1));metrics['KL_to_A']=float(w@np.sum(d*(np.log(safe)-np.log(np.maximum(f['deck_A'],1e-30))),axis=1))
 metrics['deck_entropy']=float(w@(-np.sum(d*np.log(safe),axis=1)))
 return metrics,mass

def validate_input(item,f,t):
 decks,q,counts,pop=catalogue();label=decks.index(item['opponent_list'])
 if not np.all(t['deck']==label):raise InvariantFailure('Deck label/index/viewpoint chain mismatch')
 if not np.all(f['deck_A'][:,label]>0):raise InvariantFailure('True deck removed by legal mask')
 if list(q)!=spec()['query_ids'] or f['deck_A'].shape[1]!=37:raise InvariantFailure('Class/query order changed')
 np.testing.assert_array_equal(f['global_X'][:,-37:],f['deck_A'])
 if np.any(t['counts']<f['lower']) or np.any(t['counts']>f['upper']):raise InvariantFailure('Targets outside legal support')
 return dict(decisions=len(t['deck']),singleton=int((np.sum(f['deck_A']>0,axis=1)==1).sum()),multiple=int((np.sum(f['deck_A']>0,axis=1)>1).sum()))

def parity_checks(model,f,t):
 model.eval()
 with torch.no_grad():
  a,b,d,c=components(model,f,t);p,q=predict(model,f,model.family);manual=-np.log(np.maximum(p[np.arange(len(p)),t['deck']],1e-30));count=-np.log(np.maximum(np.take_along_axis(q,t['counts'][...,None],axis=-1)[...,0],1e-30)).mean((1,2));w=point_weights(f['games']);err=max(abs(float(a)-float(w@manual)),abs(float(b)-float(w@count)))
  if err>5e-6:raise InvariantFailure('Training/report component loss mismatch')
  model.train();td,tc=model(f);model.eval();mode=float(torch.max(torch.abs(td-d)))
  if mode>1e-6 or not torch.allclose(tc,c,atol=1e-6,rtol=1e-6):raise InvariantFailure('Train/eval forward differs')
  g=dict(f);g['local_X']=f['local_X'][:,::-1].copy();pd,_=model(g);query=float(torch.max(torch.abs(pd-d)))
  if query>1e-7:raise InvariantFailure('Deck output depends on card-query order')
  if np.any(p[f['deck_A']<=0]!=0) or np.any(np.abs(p.sum(1)-1)>1e-6):raise InvariantFailure('Deck mask/normalization broken')
 return dict(loss_error=err,mode_error=mode,query_permutation_error=query)

def gradient_row(model,f,t,weight):
 model.eval();deck,count,_,_=components(model,f,t);params=list(model.named_parameters());dv=torch.autograd.grad(deck,[p for _,p in params],retain_graph=True,allow_unused=True);cv=torch.autograd.grad(count,[p for _,p in params],allow_unused=True)
 groups={'shared':('summary.','gru.'),'deck':('deck.',),'count':('query.',)};result={};total2=0.
 for group,prefixes in groups.items():
  ds=[];cs=[]
  for (name,p),d,c in zip(params,dv,cv):
   if not name.startswith(prefixes):continue
   ds.append((torch.zeros_like(p) if d is None else d).detach().reshape(-1).double());cs.append((torch.zeros_like(p) if c is None else c).detach().reshape(-1).double())
  d=torch.cat(ds);c=torch.cat(cs);dn=float(torch.linalg.vector_norm(d));cn=float(torch.linalg.vector_norm(c));joint=d+c;jn=float(torch.linalg.vector_norm(joint));total2+=jn*jn;result[group]=dict(deck_gradient_norm=dn,count_gradient_norm=cn,combined_norm=jn,cosine=float(torch.dot(d,c)/(torch.linalg.vector_norm(d)*torch.linalg.vector_norm(c))) if dn and cn else None)
 norm=math.sqrt(total2);clip=min(1.,1/(weight*norm+1e-6));unit_clip=min(1.,1/(norm+1e-6));result.update(deck_loss=float(deck.detach()),count_loss=float(count.detach()),record_weight=weight,unweighted_global_norm=norm,weighted_unclipped_norm=weight*norm,clip_factor=clip,unit_clip_factor=unit_clip,effective_weight_after_clip=weight*clip/unit_clip,weighted_clipped_norm=weight*norm*clip)
 if result['deck']['count_gradient_norm']>1e-10 or result['count']['deck_gradient_norm']>1e-10:raise InvariantFailure('Unintended head-to-head gradient path')
 return result
