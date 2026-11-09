"""Shared software gate only; no classifier fitting or comparative scores."""
from .common import *
from .baseline import Facts,hard_layout
from .adapter import Compiler
import gzip,copy,time,traceback,collections,concurrent.futures as cf,multiprocessing as mp

def one(item):
 started=time.process_time();hist=json.load(gzip.open(R/item['corpus']['legal_path'],'rt'));f=Facts(metadata()['cards']);c=Compiler(metadata()['cards']);compiler_error=None;error=None;last_lower={};snapshots=[];duplicates=0;selected={int(k) for k in item['corpus']['selection']};processed=0
 for i,e in enumerate(hist):
  try:
   raw=e['observation'];f.observe(raw);processed+=1
   assert all(f.lower.get(k,0)>=v for k,v in last_lower.items());last_lower=dict(f.lower)
   if compiler_error is None:
    try:c.compile(raw)
    except UnsupportedEvent as exc:compiler_error=dict(index=i,error=type(exc).__name__+': '+str(exc))
   if i in selected:
    known=f.known_hand();snapshots.append(dict(index=i,key=[raw['current'].get('round',1),raw['current']['turn'],raw['current']['turnActionCount'],raw['current']['yourIndex']],known=dict(known),serials=sorted(s for s,(cid,where) in f.copies.items() if where=={'H'}),lower=dict(f.lower)))
   # One deterministic duplicate per game verifies no count/evidence advance.
   if raw['current']['turnActionCount']==1:
    before=(copy.deepcopy(f.copies),dict(f.lower),dict(f.sizes));assert not f.observe(raw);assert before==(f.copies,f.lower,f.sizes);duplicates+=1
  except BaseException as exc:error=dict(index=i,type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc());break
 # True list and truth labels enter this separate gate/scoring section only.
 if not error:
  try:
   truth={tuple(t['key']):t for t in json.load(gzip.open(R/item['corpus']['truth_path'],'rt'))};deck=collections.Counter(prior()[item['opponent_list']]['cards'])
   for snap in snapshots:
    t=truth[tuple(snap['key'])];hand={int(k):v for k,v in t['hand'].items()};assert set(snap['serials'])<=set(t['hand_serials']);assert all(n<=hand.get(k,0) for k,n in snap['known'].items());assert not(collections.Counter(snap['lower'])-deck)
  except BaseException as exc:error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
 return dict(id=item['id'],status='FAILED' if error else 'COMPLETE',callbacks=len(hist),processed=processed,selected=len(snapshots),expected_selected=item['corpus']['selected_decisions'],duplicate_checks=duplicates,compiler_unsupported=compiler_error,error=error,CPU_seconds=time.process_time()-started,final_ledger=dict(f.lower))

def main():
 start=now();records=[]
 with cf.ProcessPoolExecutor(4,mp_context=mp.get_context('spawn')) as pool:
  for result in pool.map(one,read(OUT/'inputs.json'),chunksize=1):
   records.append(result);append(OUT/'shared-validation.jsonl',result)
 failures=collections.Counter((x['error']['type']+': '+x['error']['message']) for x in records if x['error']);summary=dict(at_utc=iso(),expected=296,attempted=len(records),completed=sum(x['status']=='COMPLETE' for x in records),failed=sum(x['status']!='COMPLETE' for x in records),compiler_unsupported=sum(x['compiler_unsupported'] is not None for x in records),failure_signatures=dict(failures),CPU_seconds=sum(x['CPU_seconds'] for x in records),elapsed_seconds=(now()-start).total_seconds(),new_native_matches=0,held_out_accessed=False,comparative_scores_examined=False)
 write(OUT/'shared-validation-summary.json',summary);print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
