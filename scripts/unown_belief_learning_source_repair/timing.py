"""Full online workload timing on fixed validation histories; no extra games."""
from .common import *
from .storage import load
from .features import encode
from .truth import meaningful
import numpy as np

def prepare(index,check):
 byid={r['id']:r for r in index};results=[]
 for ident in read(OUT/'online-timing-manifest.json')['ids']:
  check();row=byid[ident]
  if row['partition']!='validation':raise PermissionError('Timing must use validation, never locked test')
  source=load(row,'legal');selection={i:{} for i,e in enumerate(source['entries']) if meaningful(e['observation'])};f,info=encode(source['entries'],selection,True);path=PRIVATE/'online-timing'/(ident+'.npz');npwrite(path,**f);info.pop('points',None);info.update(id=ident,path=str(path.relative_to(R)),sha256=sha(path),all_genuine_readouts=len(selection),all_callbacks=len(source['entries']),target_labels_opened=False);results.append(info)
 write(OUT/'online-timing-inputs.json',results);return results

def evaluate(model,family,predict,check):
 records=[]
 for item in read(OUT/'online-timing-inputs.json'):
  check()
  if sha(R/item['path'])!=item['sha256']:raise InvariantFailure('Full-workload timing input changed')
  with np.load(R/item['path']) as raw:f={k:raw[k] for k in raw.files}
  from .models import streaming_predict
  start=time.process_time();d,p,updates,readouts=streaming_predict(model,f,family,predict);increment=time.process_time()-start
  if len(d)!=item['all_genuine_readouts']:raise InvariantFailure('Full workload readout count mismatch')
  records.append(dict(id=item['id'],all_callbacks=item['all_callbacks'],all_genuine_readouts=item['all_genuine_readouts'],common_CPU_seconds=item['CPU_seconds'],model_and_reconciliation_CPU_seconds=increment,total_CPU_seconds=item['CPU_seconds']+increment,model_update_CPU_p50_p95=np.quantile(updates,[.5,.95]).tolist(),model_readout_CPU_p50_p95=np.quantile(readouts,[.5,.95]).tolist()))
 return dict(expected=16,completed=len(records),records=records,complete_recurring_CPU_seconds_per_BO3=sum(x['total_CPU_seconds'] for x in records)/len(records),incremental_model_CPU_seconds_per_BO3=sum(x['model_and_reconciliation_CPU_seconds'] for x in records)/len(records),streaming_one_callback_at_a_time=True,all_callbacks_and_genuine_readouts=True,sampled_readout_cost=False,new_native_matches=0)
