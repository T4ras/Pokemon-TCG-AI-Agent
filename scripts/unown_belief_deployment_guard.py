"""Deployment boundary: unsupported belief evidence disables only its correction.

This does not alter the historical observer or training path. It never invents
probabilities for a deck outside the frozen catalogue or weakens invariants.
"""
from frozen_belief.common import *
from unown_belief_bayes_repair.cache import OperationLimit
from unown_belief_correction.correction import corrected_outputs
import torch

ZERO=np.zeros(BELIEF_DIM,np.float32);ZERO.flags.writeable=False
UNAVAILABLE=(UnsupportedEvent,OutOfLibrary,IncompatibleAllocation,OperationLimit)

class SupportedBelief:
 def __init__(self,inner):
  self.inner=inner;self.disabled=False;self.disabled_reason=None;self.fallback_count=0
 def __getattr__(self,name):return getattr(self.inner,name)
 def reset_match(self):
  self.inner.reset_match();self.disabled=False;self.disabled_reason=None;self.fallback_count=0
 def disable(self,error):
  if not self.disabled:
   self.disabled=True;self.disabled_reason=(type(error).__name__+': '+str(error))[:320];self.fallback_count+=1
   print('BELIEF_UNAVAILABLE_BASE_POLICY '+self.disabled_reason,file=sys.stderr)
 def observe(self,raw):
  if raw.get('select') is None:self.reset_match();return
  if self.disabled:return
  try:self.inner.observe(raw)
  except UNAVAILABLE as error:self.disable(error)
 def readout(self):
  if self.disabled:return ZERO
  try:return self.inner.readout()
  except UNAVAILABLE as error:self.disable(error);return ZERO

def with_supported_belief(packet,actor):
 observer=actor.belief_observer
 vector=observer.readout()
 if vector.shape!=(BELIEF_DIM,) or not np.isfinite(vector).all():raise InvariantFailure('Badactorbeliefinput')
 out=copy.copy(packet);out.belief=vector;out.belief_disabled=observer.disabled;return out

def supported_outputs(model,state,packet):
 if getattr(packet,'belief_disabled',False):
  # Zeroing the channel alone would still invoke the learned state residual.
  # Bypass the entire correction to retain the exact frozen base action logits.
  raw=state['logits'];features=torch.cat([state['r'],state['g']],-1)
  return raw,model.value_mlp(features).squeeze(-1),torch.zeros_like(raw)
 from frozen_belief.integration import packet_belief
 belief=packet_belief(packet,next(model.parameters()).device,state['r'].dtype)
 return corrected_outputs(model,state,belief)
