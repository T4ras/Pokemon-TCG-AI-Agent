"""Feature caches are train/validation-only; sealed test gets no model pass."""
from .common import *
from .storage import load
from .features import encode,target_arrays
import numpy as np,resource

def one(row):
 start=time.process_time()
 if row['partition']=='locked-test':raise PermissionError('No test feature extraction/model inference')
 legal=load(row,'legal');f,info=encode(legal['entries'],legal['selection'],row['partition']=='validation')
 labels=load(row,'targets');t=target_arrays(labels['points'],f,info['points'],labels['opponent_list'])
 fp=PRIVATE/'features'/(row['id']+'.npz');tp=PRIVATE/'supervision'/(row['id']+'.npz');npwrite(fp,**f);npwrite(tp,**t)
 info.update(id=row['id'],partition=row['partition'],feature_path=str(fp.relative_to(R)),feature_sha256=sha(fp),target_path=str(tp.relative_to(R)),target_sha256=sha(tp),total_CPU_seconds=time.process_time()-start,peak_RSS_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
 detail=PRIVATE/'feature-info'/(row['id']+'.json');write(detail,info)
 public={k:v for k,v in info.items() if k!='points'};public.update(selected_points=len(info['points']),detail_path=str(detail.relative_to(R)),detail_sha256=sha(detail));write(OUT/'feature-results'/(row['id']+'.json'),public);return public

def load_features(ident):
 r=read(OUT/'feature-results'/(ident+'.json'))
 if r['partition'] not in ('training','validation'):raise PermissionError('Test features are forbidden')
 for k in ('feature','target'):
  if sha(R/r[k+'_path'])!=r[k+'_sha256']:raise InvariantFailure('Encoded cache checksum mismatch')
 with np.load(R/r['feature_path']) as a:f={k:a[k] for k in a.files}
 with np.load(R/r['target_path']) as a:t={k:a[k] for k in a.files}
 return f,t
