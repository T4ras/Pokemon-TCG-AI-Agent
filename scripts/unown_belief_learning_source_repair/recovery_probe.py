"""Bounded saved-failure replay; no native collection or learned scoring."""
from .common import *
from .audit import integrity
from .features import encode,target_arrays
from .truth import select_points,align
from unown_belief_bayes_repair.adapter import legal
import gzip,traceback

def main():
 ident='belief-v1-aaefc2f725d6ac542d7c';p=OUT/'failed-history-replay-gate.json'
 x=gzip_read(R/'data/private/astra-unown-belief-learning-dataset-v1-resume-004/detail-temp'/ (ident+'-integrity.json.gz'));hist=[dict(observation=legal(e['observation']),action=e['action']) for e in x['legal_history']];job=x['job'];sel=select_points(ident,hist);start=time.process_time()
 try:
  targets=align([hist[i] for i in sorted(sel)],x['native_frames'],job['seat']);checks=integrity(hist,targets,job['b']);print('INTEGRITY',checks,flush=True)
  f,info=encode(hist,{str(k):v for k,v in sel.items()},False);target_arrays(targets,f,info['points'],job['b'])
  write(p,dict(passed=True,callbacks=len(hist),selected_points=len(targets),CPU_seconds=time.process_time()-start,checks=checks,legal_features_only=True,target_alignment_passed=True,new_native_matches=0,feature_shape=f['A'].shape,max_readout_CPU_seconds=max(info['latency_CPU_seconds'])))
  print(read(p),flush=True)
 except BaseException as e:
  write(p,dict(passed=False,error=type(e).__name__+':'+str(e),CPU_seconds=time.process_time()-start,traceback=traceback.format_exc()));raise
if __name__=='__main__':main()
