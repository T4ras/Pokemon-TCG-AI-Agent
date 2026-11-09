"""Detached residual actor and independent, actor-visible residual critic."""
from .common import *
import torch,numpy as np,math
from torch import nn

class Correction(nn.Module):
 def __init__(self):
  super().__init__();self.actor_belief=nn.Sequential(nn.Linear(BELIEF_DIM,128),nn.GELU());self.actor=nn.Sequential(nn.Linear(384+256+128,256),nn.GELU(),nn.Linear(256,1));self.critic_belief=nn.Sequential(nn.Linear(BELIEF_DIM,128),nn.GELU());self.critic=nn.Sequential(nn.Linear(384+128,256),nn.GELU(),nn.Linear(256,1))
  for head in (self.actor[-1],self.critic[-1]):nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)
 def action_delta(self,state,candidate,belief):
  state=state.detach();candidate=candidate.detach();belief=belief.detach();b=self.actor_belief(belief);x=torch.cat([state[:,None,:].expand(-1,candidate.shape[1],-1),candidate,b[:,None,:].expand(-1,candidate.shape[1],-1)],-1);return self.actor(x).squeeze(-1)
 def value_delta(self,state,belief):return self.critic(torch.cat([state.detach(),self.critic_belief(belief.detach())],-1)).squeeze(-1)

def configure(model):
 for name,p in model.named_parameters():p.requires_grad_(name.startswith(('value_mlp.','correction.')))
 groups=[dict(params=list(model.correction.actor_belief.parameters())+list(model.correction.actor.parameters()),lr=1e-4,name='actor_correction'),dict(params=list(model.correction.critic_belief.parameters())+list(model.correction.critic.parameters()),lr=1e-4,name='new_critic'),dict(params=list(model.value_mlp.parameters()),lr=1e-5,name='original_value_head')]
 return torch.optim.AdamW(groups,betas=(.95,.999),eps=1e-8,weight_decay=0.,foreach=False)
def learning_rate(opt,fresh):
 progress=min(max(fresh/250000.,0.),1.);rate=1e-5+.5*(1e-4-1e-5)*(1+math.cos(math.pi*progress))
 for group in opt.param_groups:group['lr']=1e-5 if group['name']=='original_value_head' else rate
 return rate

def attach(model,seed=SEED):
 devices=[next(model.parameters()).device.index or 0] if next(model.parameters()).is_cuda else []
 with torch.random.fork_rng(devices=devices):
  torch.manual_seed(seed);model.correction=Correction().to(next(model.parameters()).device)
 for name,p in model.named_parameters():p.requires_grad_(name.startswith(('value_mlp.','correction.')))
 model._pilot_q=None
 def capture(module,args,output):model._pilot_q=output.detach()
 model.q_mlp.register_forward_hook(capture)
 return model

def corrected_outputs(model,state,belief):
 features=torch.cat([state['r'],state['g']],-1).detach();q=model._pilot_q
 if q is None or q.shape[0]!=features.shape[0]:raise InvariantFailure('Missingdetachedcandidatefeatures')
 if belief.shape!=(features.shape[0],BELIEF_DIM):raise InvariantFailure('Belief/statebatchmismatch')
 delta=model.correction.action_delta(features,q,belief);raw=state['logits'].detach();mask=state['pilot_mask'];logits=(raw+delta).masked_fill(~mask,-1e9);value=model.value_mlp(features).squeeze(-1)+model.correction.value_delta(features,belief);return logits,value,delta

def frozen_digest(model):return tensor_digest({k:v for k,v in model.state_dict().items() if not k.startswith(('value_mlp.','correction.'))})
