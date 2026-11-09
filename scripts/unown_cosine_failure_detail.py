"""Bound new optional detail before writing; never delete saved evidence."""
import io,json
from unown_cosine_common import *
class FailureSink:
 def __init__(self,path,original_open,job):
  self.path=Path(path);assert self.path.resolve().parent==PRIVATE.resolve();self.original_open=original_open;self.job=job;self.stream=io.StringIO();self.bytes=0;self.overflow=False;self.closed=False
 def write(self,value):
  self.bytes+=len(value.encode())
  if self.bytes<=2*1024**2:self.stream.write(value)
  else:self.overflow=True
  return len(value)
 def flush(self):pass
 def __enter__(self):return self
 def __exit__(self,*error):self.close();return False
 def close(self):
  if self.closed:return
  self.closed=True;path=OUT/'failure-detail-retention.json';ledger=read(path) if path.exists() else dict(signatures={},raw_bytes=0,omitted=0)
  raw=self.stream.getvalue();payload=None
  if not self.overflow:
   try:payload=json.loads(raw)
   except json.JSONDecodeError:pass
  record=payload.get('record') if isinstance(payload,dict) else None
  if record is None:
   stage=OUT/self.job['stage']/'matches.jsonl'
   try:record=json.loads(stage.read_text().splitlines()[-1])
   except Exception:record=dict(self.job,error='Failuredetail unavailable; intent/compactrecord preserved')
  signature=str(record.get('error')).split(':')[0];count=ledger['signatures'].get(signature,0);ledger['signatures'][signature]=count+1
  keep=payload is not None and not self.overflow and count<10 and ledger['raw_bytes']+self.bytes<=64*1024**2
  assert not self.path.exists(),'Neveroverwriteexisting failureevidence'
  with self.original_open(self.path,'wt') as file:
   if keep:file.write(raw);ledger['raw_bytes']+=self.bytes
   else:json.dump(dict(record=record,optional_raw_detail_omitted=True,reason='Registeredfirst10persignature/2MiBperitem/64MiBtotalbudget; everycompactfailurepreserved'),file);ledger['omitted']+=1
  write(path,ledger);self.stream.close()
def opener(original_open,job):
 def open_file(path,*args,**kw):
  if str(path).endswith('-failure.json.gz'):return FailureSink(path,original_open,job)
  return original_open(path,*args,**kw)
 return open_file
