from dataclasses import dataclass
import numpy as np
import torch
"""Reusable declarations extracted from unown_v2_runtime.py; dependencies injected per run."""

@dataclass(frozen=True)
class Layout:
 tokens:int=160
 options:int=64
 tools:int=2
 oracle:tuple=(24,24,8)
 @property
 def offsets(self):
  sizes=dict(tok_int=self.tokens*(3+self.tools),tok_float=self.tokens*56,
   opt_int=self.options*6,opt_float=self.options*8,opt_mask=self.options,global_f=64,
   dec_int=8,dec_float=16,oracle=self.oracle[0]*2+self.oracle[1]*3+self.oracle[2]*2)
  off=0;out={}
  for k,n in sizes.items():out[k]=(off,off+n);off+=n
  return out
 @property
 def size(self):return self.offsets['oracle'][1]
 @property
 def stop(self):return self.options-1


@dataclass
class Packet:
 row:np.ndarray
 layout:Layout
 def tensor(self,device='cpu'):return torch.as_tensor(self.row,device=device).reshape(1,-1)

