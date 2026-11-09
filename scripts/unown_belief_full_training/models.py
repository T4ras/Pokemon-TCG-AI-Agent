"""Small neural models and proper-loss stochastic gradient-boosted trees."""
from .common import *
import numpy as np

def softmax(logits,mask):
 a=np.where(mask,logits,-1e30);a-=a.max(axis=-1,keepdims=True);p=np.exp(a)*mask;return p/p.sum(axis=-1,keepdims=True)

def point_weights(games):
 gs=np.unique(games);return np.array([1/len(gs)/np.sum(games==g) for g in games],np.float32)

def make_neural(family,seed):
 import torch
 from torch import nn
 torch.manual_seed(seed);S=len(spec()['count_support']);L=spec()['dimensions']['local_X']
 class BeliefNetwork(nn.Module):
  def __init__(self):
   super().__init__();self.family=family;self.gru=nn.GRU(140,128,batch_first=True) if family=='GRU' else None
   self.summary=nn.Sequential(nn.Linear(153+(128 if self.gru else 0),128),nn.ReLU(),nn.Linear(128,128),nn.ReLU())
   self.deck=nn.Linear(128,37);self.query=nn.Sequential(nn.Linear(128+L,64),nn.ReLU(),nn.Linear(64,3*S))
   nn.init.zeros_(self.deck.weight);nn.init.zeros_(self.deck.bias);nn.init.zeros_(self.query[-1].weight);nn.init.zeros_(self.query[-1].bias)
  def forward(self,f):
   x=torch.as_tensor(f['global_X'],dtype=torch.float32)
   if self.gru:
    seq=torch.as_tensor(f['sequence'],dtype=torch.float32);starts=np.flatnonzero(f['reset']);ends=list(starts[1:])+[len(seq)];states=[]
    for start,end in zip(starts,ends):
     # Complete current-game history, including all forced callbacks.
     out,_=self.gru(seq[start:end][None,:,:],torch.zeros(1,1,128))
     ids=f['selected'][(f['selected']>=start)&(f['selected']<end)]-start
     states.append(out[0,torch.as_tensor(ids),:])
    x=torch.cat((x,torch.cat(states)),dim=-1)
   return self.readout(f,x)
  def readout(self,f,x):
   context=self.summary(x);local=torch.as_tensor(f['local_X'],dtype=torch.float32);Q=local.shape[1];residual=self.query(torch.cat((context[:,None,:].expand(-1,Q,-1),local),dim=-1)).reshape((-1,Q,3,S))
   A=torch.as_tensor(f['A']);low=torch.as_tensor(f['lower']);high=torch.as_tensor(f['upper']);k=torch.arange(S);mask=(k>=low[...,None])&(k<=high[...,None])
   count_logits=(torch.log(torch.clamp(A,min=1e-5))+residual).masked_fill(~mask,-1e30)
   deck_A=torch.as_tensor(f['deck_A']);deck_logits=(torch.log(torch.clamp(deck_A,min=1e-30))+self.deck(context)).masked_fill(deck_A<=0,-1e30)
   return deck_logits,count_logits
 model=BeliefNetwork();assert sum(p.numel() for p in model.parameters())<1000000;return model

def neural_loss(model,f,t):
 import torch
 import torch.nn.functional as F
 d,c=model(f);N,Q,Z,S=c.shape;y=torch.as_tensor(t['counts'],dtype=torch.long);yd=torch.as_tensor(t['deck'],dtype=torch.long)
 deck=F.cross_entropy(d,yd,reduction='none');counts=F.cross_entropy(c.reshape((-1,S)),y.reshape(-1),reduction='none').reshape((N,Q,Z)).mean((1,2));w=torch.as_tensor(point_weights(f['games']))
 return ((deck+counts)*w).sum()

def neural_predict(model,f):
 import torch
 model.eval()
 with torch.inference_mode():d,c=model(f);return d.softmax(-1).numpy(),c.softmax(-1).numpy()

def tree_inputs(f):
 N,Q,L=f['local_X'].shape;positions=np.argwhere(f['upper']>f['lower']);X=np.empty((len(positions),153+L+3),np.float32)
 if len(positions):
  i,q,z=positions.T;X[:,:153]=f['global_X'][i];X[:,153:153+L]=f['local_X'][i,q];X[:,153+L:]=np.eye(3,dtype=np.float32)[z]
 return positions,X

class BoostedTrees:
 def __init__(self,seed):self.seed=seed;self.deck=[];self.count=[];self.rate=.1
 def predict(self,f):
  ds=np.log(np.maximum(f['deck_A'],1e-30)).astype(float)
  for tree in self.deck:ds+=self.rate*tree.predict(f['global_X'])
  dp=softmax(ds,f['deck_A']>0);S=f['A'].shape[-1];k=np.arange(S);mask=(k>=f['lower'][...,None])&(k<=f['upper'][...,None]);cp=np.zeros_like(f['A']);positions,X=tree_inputs(f)
  # Deterministic legal counts have zero loss and need no learned readout.
  i,q,z=np.indices(f['lower'].shape);fixed=f['lower']==f['upper'];cp[i[fixed],q[fixed],z[fixed],f['lower'][fixed]]=1
  if len(positions):
   i,q,z=positions.T;logits=np.log(np.maximum(f['A'][i,q,z],1e-5)).astype(float)
   for tree in self.count:logits+=self.rate*tree.predict(X)
   cp[i,q,z]=softmax(logits,mask[i,q,z])
  return dp,cp
 def stage(self,items,stage,load,check):
  from sklearn.tree import DecisionTreeRegressor
  dx=[];dg=[];dw=[];cx=[];cg=[];cw=[]
  for item in items:
   check();f,t=load(item['id']);d,p=self.predict(f);pw=point_weights(f['games'])*37*prior()[item['opponent_list']]['prior'];target=np.eye(37)[t['deck']]
   dx.append(f['global_X']);dg.append(target-d);dw.append(pw)
   positions,X=tree_inputs(f)
   if len(positions):
    i,q,z=positions.T;g=np.eye(p.shape[-1])[t['counts'][i,q,z]]-p[i,q,z];cx.append(X);cg.append(g);cw.append(pw[i]/f['A'].shape[1]/3)
  config=dict(max_depth=3,max_leaf_nodes=8,min_samples_leaf=128,max_features=32,random_state=self.seed+stage)
  dt=DecisionTreeRegressor(**config);dt.fit(np.concatenate(dx),np.concatenate(dg),sample_weight=np.concatenate(dw));self.deck.append(dt)
  if cx:
   ct=DecisionTreeRegressor(**config);ct.fit(np.concatenate(cx),np.concatenate(cg),sample_weight=np.concatenate(cw));self.count.append(ct)


def point_slice(f,j):
 fields=('global_X','local_X','A','lower','upper','deck_A','ledger','known','games','zone_sizes')
 out={k:f[k][j:j+1] for k in fields if k in f}
 if 'conditionals' in f:
  lo,hi=f['conditional_offsets'][j:j+2];out.update(conditionals=f['conditionals'][lo:hi],conditional_indices=f['conditional_indices'][lo:hi],conditional_offsets=np.array([0,hi-lo]))
 return out

def streaming_predict(model,f,family,predict):
 """Actual one-callback/one-readout execution, including forced GRU updates."""
 from contextlib import nullcontext
 from .distributions import mask_counts
 neural=family in ('MLP','GRU')
 if neural:import torch
 outputs=[];probs=[];updates=[];readouts=[];selected={int(i):j for j,i in enumerate(f['selected'])};hidden=torch.zeros(1,1,128) if family=='GRU' else None
 if family in ('MLP','GRU'):model.eval()
 with (torch.inference_mode() if neural else nullcontext()):
  for i,event in enumerate(f['sequence']):
   start=time.process_time()
   if family=='GRU':
    if f['reset'][i]:hidden=torch.zeros(1,1,128)
    state,hidden=model.gru(torch.as_tensor(event)[None,None,:],hidden)
   updates.append(time.process_time()-start)
   if i not in selected:continue
   j=selected[i];single=point_slice(f,j);start=time.process_time()
   if family=='GRU':
    x=torch.cat((torch.as_tensor(single['global_X']),state[:,0,:]),dim=-1);d,p=model.readout(single,x);d=d.softmax(-1).numpy();p=p.softmax(-1).numpy();p=mask_counts(p,single['lower'],single['upper']);d*=single['deck_A']>0;d/=d.sum(-1,keepdims=True)
   else:d,p=predict(model,single,family)
   readouts.append(time.process_time()-start);outputs.append(d[0]);probs.append(p[0])
 return np.array(outputs),np.array(probs),updates,readouts
