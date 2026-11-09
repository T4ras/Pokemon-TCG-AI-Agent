"""Memoize immutable library descriptors; preserve all observation-dependent math."""
import sys,hashlib,json,collections
import numpy as np
import unown_belief_correction.belief as belief
import unown_belief_bayes_repair.deck as decks

class FixedDataCache:
 def __init__(self):
  self.originals=dict(prior=belief.prior,metadata=belief.metadata,catalogue=decks.catalogue);self.data={k:f() for k,f in self.originals.items()};self.calls=collections.Counter();self.patches=[]
  for x in self.data['catalogue']:
   if isinstance(x,np.ndarray):x.flags.writeable=False
  self.initial=self.digest()
 def digest(self):
  h=hashlib.sha256(json.dumps({k:self.data[k] for k in ('prior','metadata')},sort_keys=True).encode())
  for x in self.data['catalogue']:
   h.update(x.tobytes() if isinstance(x,np.ndarray) else json.dumps(x).encode())
  return h.hexdigest()
 def install(self):
  for module in list(sys.modules.values()):
   name=getattr(module,'__name__','')
   if not name.startswith(('unown_belief_','unown_runner')):continue
   for key,original in self.originals.items():
    if getattr(module,key,None) is not original:continue
    def cached(_key=key):self.calls[_key]+=1;return self.data[_key]
    self.patches.append((module,key,original));setattr(module,key,cached)
  return self
 def close(self):
  assert self.digest()==self.initial,'Immutable library mutated'
  for module,key,original in reversed(self.patches):setattr(module,key,original)
  self.patches=[]
 def __enter__(self):return self.install()
 def __exit__(self,*args):self.close()

_INSTALLED=None
def install():
 global _INSTALLED
 if _INSTALLED is None:_INSTALLED=FixedDataCache().install()
 return _INSTALLED
