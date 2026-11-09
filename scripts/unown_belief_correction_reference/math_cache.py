"""Bounded pure-function memoization and exact one-to-one PMF array copying."""
from .cache import FixedDataCache
from pathlib import Path
import functools,inspect,textwrap,sys,ast
import unown_belief_full_training.distributions as distributions
import unown_belief_full_training.baseline as baseline
ORIGINAL_HYPERGEOM=distributions.hypergeom
ORIGINAL_ORDINARY=distributions.ordinary
_SOURCE=Path(distributions.__file__).read_text()
_SOURCE_ORDINARY=ast.get_source_segment(_SOURCE,next(n for n in ast.parse(_SOURCE).body if isinstance(n,ast.FunctionDef) and n.name=='ordinary'))
ORIGINAL_CHOOSE=baseline.choose

class ExactBeliefCache(FixedDataCache):
 def install(self):
  super().install();self.math={}
  original=distributions.hypergeom
  @functools.lru_cache(maxsize=8192)
  def hypergeom(K,C,n,S):
   out=ORIGINAL_HYPERGEOM(K,C,n,S);out.flags.writeable=False;return out
  self.patches.append((distributions,'hypergeom',original));distributions.hypergeom=hypergeom;self.math['hypergeom']=hypergeom
  choose=baseline.choose;cached_choose=functools.lru_cache(maxsize=8192)(ORIGINAL_CHOOSE);self.math['choose']=cached_choose
  for module in list(sys.modules.values()):
   if getattr(module,'__name__','').startswith('unown_belief_') and getattr(module,'choose',None) is choose:
    self.patches.append((module,'choose',choose));module.choose=cached_choose
  old=distributions.ordinary;source=textwrap.dedent(_SOURCE_ORDINARY);start=source.index(' for i,c in enumerate(queries):',source.index(' raw=np.fft.irfft'));end=source.index(' if out.min()',start)
  replacement=""" shifts=np.array([[fixed['H'][c]-known[c],fixed['D'][c],fixed['P'][c]] for c in queries],dtype=np.int64)
 positions=np.arange(raw.shape[-1])[None,None,:]+shifts[:,:,None]
 if np.any((np.abs(raw)>1e-8)&((positions<0)|(positions>=S))):raise InvariantFailure('Count support overflow')
 sources=np.arange(S)[None,None,:]-shifts[:,:,None]
 valid=(sources>=0)&(sources<raw.shape[-1])
 out=np.where(valid,np.take_along_axis(raw,np.clip(sources,0,raw.shape[-1]-1),axis=-1),0.)
"""
  namespace=distributions.__dict__;compiled={};exec(compile(source[:start]+replacement+source[end:],'<exact-belief-PMF-copy>','exec'),namespace,compiled);self.patches.append((distributions,'ordinary',old));distributions.ordinary=compiled['ordinary'];return self

_INSTALLED=None
def install():
 global _INSTALLED
 if _INSTALLED is None:_INSTALLED=ExactBeliefCache().install()
 return _INSTALLED
