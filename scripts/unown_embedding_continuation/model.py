"""True sparse parameter objects; frozen rows are never optimizer members."""
import copy
import torch
from torch.nn.utils import parametrize
CHANNELS=dict(cards=('card_id_emb','card_text'),attacks=('attack_id_emb','att_text'),skills=('skill_id_emb','skill_text'))

class RowCorrection(torch.nn.Module):
 def __init__(self,indices,width,device,dtype):
  super().__init__();self.register_buffer('rows',torch.tensor(indices,dtype=torch.long,device=device))
  self.delta=torch.nn.Parameter(torch.zeros((len(indices),width),dtype=dtype,device=device))
 def forward(self,base):return torch.index_add(base,0,self.rows,self.delta)

def install(model,scope):
 if getattr(model,'embedding_adaptation',False):raise ValueError('Double adaptation installation')
 keys=list(model.state_dict());model._canonical_keys=keys
 for p in model.parameters():p.requires_grad_(False)
 for kind,(identity,text) in CHANNELS.items():
  ids=scope[kind];module=getattr(model,identity);base=module.weight
  parametrize.register_parametrization(module,'weight',RowCorrection(ids,base.shape[1],base.device,base.dtype))
  base=getattr(model,text);parametrize.register_parametrization(model,text,RowCorrection(ids,base.shape[1],base.device,base.dtype))
 for p in model.value_mlp.parameters():p.requires_grad_(True)
 model.embedding_adaptation=True
 return model

def actor_parameters(model):return [p for name,p in model.named_parameters() if name.endswith('.delta')]
def optimizer(model,lr):
 groups=[dict(params=actor_parameters(model),lr=lr,name='eligible_identity_and_text_corrections'),dict(params=list(model.value_mlp.parameters()),lr=1e-5,name='detached_blind_value')]
 assert len({id(p) for g in groups for p in g['params']})==sum(len(g['params']) for g in groups)
 assert {id(p) for p in model.parameters() if p.requires_grad}=={id(p) for g in groups for p in g['params']}
 return torch.optim.AdamW(groups,betas=(.95,.999),eps=1e-8,weight_decay=0.,foreach=False)

def canonical_state(model):
 result={}
 for key in model._canonical_keys:
  parent,_,leaf=key.rpartition('.');module=model.get_submodule(parent) if parent else model
  result[key]=getattr(module,leaf).detach().cpu().clone()
 return result

def actor_hash(model,digest):return digest({name:p.detach() for name,p in model.named_parameters() if name.endswith('.delta')})
def frozen_state(model):return {k:v for k,v in model.state_dict().items() if '.delta' not in k and not k.startswith('value_mlp.')}
def gradient_norm(parameters):return float(sum(float(p.grad.detach().double().square().sum()) for p in parameters if p.grad is not None)**.5)

def inventory(model):
 names={id(p):name for name,p in model.named_parameters()}
 return [dict(name=name,shape=list(p.shape),elements=p.numel(),role='embedding/text' if name.endswith('.delta') else 'detached value' if name.startswith('value_mlp.') else 'frozen',trainable=p.requires_grad) for name,p in model.named_parameters()]

def bind_detached_value(runtime):
 def region(model,layout,length):
  core_fn=runtime.tensor.core_with_tables(layout,length)
  def run(observation,card_table,attack_table,old,old_lp,actions,mask,returns,advantages,real_mask,denominator):
   state=core_fn(model,observation,(card_table,attack_table))
   values=model.value_mlp(torch.cat([state['r'],state['g']],-1).detach()).squeeze(-1)
   loss,metrics=runtime.tensor.objective(state['logits'],values,old,old_lp,actions,mask,returns,advantages,real_mask,denominator)
   return state['logits'],values,loss,metrics
  return run
 runtime.engine.region=region
