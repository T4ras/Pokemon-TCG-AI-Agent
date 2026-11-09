"""Registered saved-record pooling; no models, native games or remote compute."""
from .common import *
from .statistics import analyze
import collections,json,time,math

def main():
 start=iso();wall=time.perf_counter();cpu=time.process_time()
 reg=read(OUT/'registration.json');decks=read(ROOT/reg['decks_file']['path'])
 assert sha(ROOT/reg['decks_file']['path'])==reg['decks_file']['sha256']
 rows=[];sources={};seen=set();input_hashes={}
 for label,info in reg['input_files'].items():
  path=ROOT/info['path'].replace('\\','/');assert sha(path)==info['sha256']
  data=read(path);ids=[x['id'] for x in data]
  assert len(ids)==len(set(ids)) and not seen.intersection(ids)
  seen.update(ids);input_hashes[label]=sha(path)
  sources[label]={}
  for agent in ('original','belief','oracle'):
   subset=[x for x in data if x['agent']==agent]
   sources[label][agent]=dict(attempts=len(subset),completed=sum(x['status']=='DONE' for x in subset),failures=sum(x['status']!='DONE' for x in subset))
  for row in data:
   assert row['learner_seat'] in (0,1) and row['opponent_deck'] in decks['passing']
   if row['status']=='DONE':
    assert row['result'] in (0,1,2) and row['games']
    assert all(g['completion_status']=='DONE' and g['result'] in (0,1,2) for g in row['games'])
   rows.append(dict(row,source_run=label))
 assert len(rows)==reg['expected_attempts']
 assert sum(x['status']=='DONE' for x in rows)==reg['expected_completed_BO3']
 failed=[x for x in rows if x['status']!='DONE']
 assert len(failed)==reg['expected_failed_or_interrupted']
 audit=dict(source_counts=sources,attempts=len(rows),completed=sum(x['status']=='DONE' for x in rows),failures=len(failed),unique_IDs=len(seen),input_hashes=input_hashes,failed_records=[dict(id=x['id'],agent=x['agent'],source_run=x['source_run'],error=x.get('error'),retained_games=len(x.get('games',[]))) for x in failed],usable_first_games_from_failed=sum(bool(x.get('games')) and x['games'][0].get('completion_status')=='DONE' for x in failed))
 write(OUT/'count-audit.json',audit)
 with (OUT/'evaluation-all-attempts.jsonl').open('w',encoding='utf-8') as f:
  for row in rows:f.write(json.dumps(row,sort_keys=True,separators=(',',':'))+'\n')
 result=analyze(rows,decks)
 # Independent direct count-weighting check of every point estimate.
 for agent,summary in result['agents'].items():
  for metric in ('BO3','first_game'):
   expected=sum(decks['training_probabilities'][cell['deck']]/2*cell[metric]['W']/cell[metric]['completed'] for cell in summary['cells'])
   assert math.isclose(summary[metric]['rate'],expected,abs_tol=1e-14)
  assert sum(cell['BO3']['completed'] for cell in summary['cells'])==summary['completed']
  summary['raw_BO3_counts']={k:sum(cell['BO3'][k] for cell in summary['cells']) for k in ('W','D','L')}
  summary['raw_first_game_counts']={k:sum(cell['first_game'][k] for cell in summary['cells']) for k in ('W','D','L')}
  summary['completed_per_chooser']={str(seat):sum(x['status']=='DONE' for x in rows if x['agent']==agent and x['learner_seat']==seat) for seat in (0,1)}
 for label,info in reg['input_files'].items():assert sha(ROOT/info['path'].replace('\\','/'))==input_hashes[label]
 write(OUT/'analysis.json',result)
 write(OUT/'completion.json',dict(state='COMPLETE',started_at_utc=start,ended_at_utc=iso(),elapsed_seconds=time.perf_counter()-wall,process_CPU_seconds=time.process_time()-cpu,GPU_seconds=0,new_native_BO3=0,training=False,uploads=0,paid_model_API_requests=0,checks=['Registered immutable input hashes verified before/after analysis','Unique within-source and cross-source match IDs','1639 attempts / 1636 completed / 3 failed or interrupted','No outcome imputation; valid first-game retention','Independent direct count-weighted point estimates agree'],output_hashes={name:sha(OUT/name) for name in ('count-audit.json','analysis.json','evaluation-all-attempts.jsonl')}))
 print(json.dumps(dict(counts=audit,agents={k:{m:v[m] for m in ('attempts','completed','failures','BO3','first_game','raw_BO3_counts','raw_first_game_counts')} for k,v in result['agents'].items()},contrasts=result['contrasts'])))

if __name__=='__main__':main()
