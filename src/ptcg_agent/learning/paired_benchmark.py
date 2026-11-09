"""Resumable unchanged-deck CPU evaluations using the existing benchmark policy."""
import argparse,concurrent.futures,datetime,hashlib,json,itertools,multiprocessing,random,time
from pathlib import Path
from .benchmark import worker,summarize
from .live import balanced_schedule
from .round_robin import write_json

def run(results,output,count=400,workers=4,seed=904116,teachers=(0,1,3)):
    root=Path(results).resolve();out=Path(output).resolve();out.mkdir(parents=True,exist_ok=True)
    sha=hashlib.sha256((root/'selected.pt').read_bytes()).hexdigest();entries=json.loads((root/'agents.json').read_text())['agents']
    if len(entries)!=4 or any(e['sha256']!=sha for e in entries):raise ValueError('Checkpoint provenance mismatch')
    jobs=[]
    for t in teachers:
        for o in ['random','heuristic-v1','heuristic-v2']:
            jobs.extend((t,o,i,s) for i,s in enumerate(balanced_schedule(count,seed+t*7+['random','heuristic-v1','heuristic-v2'].index(o))))
    rng=random.Random(seed);rng.shuffle(jobs);ids=[f'bench-{t}-{o}-{i}' for t,o,i,s in rng.sample(jobs,min(100,len(jobs)))]
    expected={f'bench-{t}-{o}-{i}':(t,o,i,s) for t,o,i,s in jobs}
    registration=dict(checkpoint_sha256=sha,count=count,workers=workers,seed=seed,teachers=list(teachers),planned=len(jobs),chooser_balance='Half per pairing, shuffled',intervals='Pointwise equal-chooser Wilson-MOVER95%',native_rng_seeded=False,diagnostic_cap_bytes=2*1024**3,sampled_visible_traces=len(ids))
    p=out/'registration.json'
    if p.exists() and json.loads(p.read_text())!=registration:raise ValueError('Resume configuration changed')
    if not p.exists():write_json(p,registration)
    rows=[];seen=set()
    for p in out.rglob('matches.jsonl'):
        for line in p.read_text().splitlines():
            r=json.loads(line);mid=r['match_id']
            if mid in seen or mid not in expected:raise ValueError('Duplicate or unknown planned match')
            t,o,i,seat=expected[mid]
            required=dict(teacher=t,opponent=o,learner_seat=seat,checkpoint_sha256=sha,scheduler_seed=seed,inference_temperature=1.,deck_sha256=hashlib.sha256(json.dumps(entries[t]['deck']).encode()).hexdigest())
            if any(r.get(k)!=v for k,v in required.items()):raise ValueError('Recorded provenance changed')
            seen.add(mid);rows.append(r)
    if (out/'report.json').exists() and len(rows)==len(jobs):return json.loads((out/'report.json').read_text())
    old=list(rows);remaining=[j for j in jobs if f'bench-{j[0]}-{j[1]}-{j[2]}' not in seen]
    interrupted=[json.loads(p.read_text()) for p in out.rglob('active.json') if json.loads(p.read_text())['match_id'] not in seen]
    segments=sorted(out.glob('segment-*'));segment=out/f'segment-{len(segments)+1:03d}';segment.mkdir()
    write_json(segment/'registration.json',dict(retained=len(old),remaining=len(remaining),interrupted_attempts=interrupted,note='Unsaved in-flight outcomes unknown; linked retry counts once per planned match',started_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    retained_bytes=sum(p.stat().st_size for p in out.rglob('sampled-visible-traces/*.json'))
    work=[dict(results=str(root),output=str(segment),worker=i,jobs=remaining[i::workers],threads=1,seed=seed,diagnostic_ids=ids,interrupted_ids=[v['match_id'] for v in interrupted],diagnostic_cap=max(0,2*1024**3-retained_bytes)//workers) for i in range(workers)]
    start=time.monotonic()
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(worker,j) for j in work];pending=set(futures)
        while pending:
            _,pending=concurrent.futures.wait(pending,timeout=30,return_when=concurrent.futures.FIRST_COMPLETED)
            current=list(old)
            for p in segment.rglob('matches.jsonl'):
                for line in p.read_text().splitlines():
                    try:current.append(json.loads(line))
                    except json.JSONDecodeError:pass  # Concurrent final line only; final collection is strict.
            write_json(out/'status.json',dict(stage='RUNNING',completed=len(current),planned=len(jobs),failures=sum(r['status']!='DONE' for r in current),updated_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
        rows=old+[r for f in futures for r in f.result()]
    metrics=[dict(teacher=t,source_team=entries[t]['source_team'],opponent=o,**summarize([r for r in rows if r['teacher']==t and r['opponent']==o])) for t in teachers for o in ['random','heuristic-v1','heuristic-v2']]
    failed=sum(r['status']!='DONE' for r in rows);violations=sum(m['chooser_protocol_violations'] for m in metrics)
    report=dict(status='COMPLETE' if not failed and not violations else 'COMPLETED_WITH_FAILURES',matches=len(rows),failures=failed,chooser_protocol_violations=violations,wall_seconds=time.monotonic()-start,matchups=metrics,checkpoint_sha256=sha,segments=len(segments)+1,interrupted_attempts=interrupted)
    write_json(out/'report.json',report);write_json(out/'status.json',dict(stage=report['status'],completed=len(rows),planned=len(jobs),failures=failed));return report
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',required=True);p.add_argument('--output',required=True);p.add_argument('--count',type=int,default=400);p.add_argument('--workers',type=int,default=4);p.add_argument('--teachers',type=int,nargs='+',default=[0,1,3]);p.add_argument('--seed',type=int,default=904116);a=p.parse_args()
    print(json.dumps(run(a.results,a.output,a.count,a.workers,seed=a.seed,teachers=a.teachers)),flush=True)
