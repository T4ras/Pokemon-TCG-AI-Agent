from pathlib import Path
from types import SimpleNamespace
import sys,copy,json
import numpy as np,torch,pytest
R=Path(__file__).resolve().parents[1];P=R/'data/private/astra-unown-belief-hosted-repair-003/package'
sys.path.insert(0,str(P))
from frozen_belief import deployment_guard as guard
from frozen_belief.common import UnsupportedEvent,OutOfLibrary,IncompatibleAllocation,InvariantFailure
from unown_belief_bayes_repair.provenance import UnknownProvenance,validate_identity

class Stub:
 def __init__(self,error=None,read_error=False):self.error=error;self.read_error=read_error;self.calls=0;self.resets=0
 def observe(self,raw):
  self.calls+=1
  if self.error and not self.read_error:raise self.error
 def readout(self):
  if self.error and self.read_error:raise self.error
  return np.ones(1479,np.float32)
 def reset_match(self):self.resets+=1

@pytest.mark.parametrize('kind',[UnknownProvenance,UnsupportedEvent,OutOfLibrary,IncompatibleAllocation,guard.OperationLimit])
@pytest.mark.parametrize('read_error',[False,True])
def test_support_failure_latches_without_loosening_observer(kind,read_error):
 inner=Stub(kind('unsupported legal evidence'),read_error);g=guard.SupportedBelief(inner)
 g.observe({'select':{'context':1},'current':{'round':1}});v=g.readout()
 assert g.disabled and np.array_equal(v,guard.ZERO) and g.fallback_count==1
 before=inner.calls;g.observe({'select':{'context':1},'current':{'round':2}})
 assert g.disabled and inner.calls==before
 g.observe({'select':None});assert not g.disabled and inner.resets==1
 with pytest.raises(UnknownProvenance):validate_identity(4,1,{},set())

@pytest.mark.parametrize('kind',[InvariantFailure,FloatingPointError,TypeError])
def test_unexpected_or_numeric_defect_stays_visible(kind):
 g=guard.SupportedBelief(Stub(kind('do not swallow')))
 with pytest.raises(kind):g.observe({'select':{'context':1}})
 assert not g.disabled

def test_supported_path_exact_packet_and_no_mutation():
 g=guard.SupportedBelief(Stub());p=SimpleNamespace(row=np.arange(3));q=guard.with_supported_belief(p,SimpleNamespace(belief_observer=g))
 assert q is not p and not hasattr(p,'belief') and not q.belief_disabled
 assert np.array_equal(q.belief,np.ones(1479,np.float32))

def test_fallback_bypasses_entire_residual_and_preserves_mask(monkeypatch):
 def forbidden(*args):raise AssertionError('correction must not run')
 monkeypatch.setattr(guard,'corrected_outputs',forbidden)
 state=dict(logits=torch.tensor([[.2,-1e9,.8]]),r=torch.ones(1,2),g=torch.ones(1,2))
 m=SimpleNamespace(value_mlp=torch.nn.Linear(4,1));p=SimpleNamespace(belief_disabled=True)
 logits,value,delta=guard.supported_outputs(m,state,p)
 assert logits is state['logits'] and torch.equal(delta,torch.zeros_like(logits)) and logits.argmax().item()==2

def test_exact_private_assets_and_limited_source_change():
 from unown_belief_hosted_repair_prepare import sha
 old=R/'data/private/astra-unown-belief-hydrapple-submission-repair-002/package'
 manifest=json.loads((R/'experiments/astra-unown-belief-hosted-repair-003/package-manifest.json').read_text())
 assert set(manifest['modified_files'])=={'frozen_belief/deployment_guard.py','frozen_unown/runtime.py'}
 for name in ['weights.pt','belief-classifier.pt','deck.csv','belief-prior.json','belief-schema.json']:
  assert sha(P/name)==sha(old/name)
 assert sha(P/'frozen_belief/deployment_guard.py')==sha(R/'scripts/unown_belief_deployment_guard.py')
