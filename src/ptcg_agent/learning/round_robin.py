"""Cross-deck round robin for the four fixed shared-BC deck instances."""
import argparse, concurrent.futures, datetime, hashlib, itertools, json, multiprocessing, random, time
from pathlib import Path
import torch
from .benchmark import summarize
from .live import NeuralAgent, play_bo3, balanced_schedule
from .model import PolicyNetwork, metadata_tables
from .training import load_checkpoint
from ..engine import load_sdk

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_json(path,value):
    path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)

def completed_rows(out, jobs, sha, seed, entries):
    """Retain failed attempts too; a retry is never a new planned trial."""
    planned={f'cross-{a}-{b}-{i}':(a,b,i,seat) for a,b,i,seat in jobs}
    rows=[];seen=set()
    for path in sorted(Path(out).rglob('matches.jsonl')):
        for line in path.read_text().splitlines():
            row=json.loads(line)  # Never silently discard a damaged record.
            mid=row['match_id']
            if mid in seen or mid not in planned:raise ValueError(f'Duplicate or unknown match: {mid}')
            a,b,i,seat=planned[mid]
            expected=dict(teacher=a,opponent=b,learner_seat=seat,checkpoint_sha256=sha,scheduler_seed=seed,policy_seed=seed+a*100000+b*10000+i,inference_temperature=1.0,deck_sha256=hashlib.sha256(json.dumps(entries[a]['deck']).encode()).hexdigest(),opponent_deck_sha256=hashlib.sha256(json.dumps(entries[b]['deck']).encode()).hexdigest())
            if any(row.get(k)!=v for k,v in expected.items()):raise ValueError(f'Record provenance mismatch: {mid}')
            seen.add(mid);rows.append(row)
    return rows

def worker(job):
    torch.set_num_threads(1)
    root=Path(job['results']);out=Path(job['output'])/f"worker-{job['worker']}";out.mkdir()
    entries=json.loads((root/'agents.json').read_text())['agents'];api,_=load_sdk();table,attacks=metadata_tables(api)
    network=PolicyNetwork(table,attacks).eval();load_checkpoint(root/'selected.pt',network)
    rows=[];retained=0;started=time.monotonic()
    for a,b,index,seat in job['jobs']:
        seed=job['seed']+a*100000+b*10000+index
        torch.manual_seed(seed);random.seed(seed)
        left=NeuralAgent(network,entries[a]['deck']);right=NeuralAgent(network,entries[b]['deck'])
        agents=[left,right] if seat==0 else [right,left]
        match_id=f'cross-{a}-{b}-{index}';trace=[]
        write_json(out/'active.json',dict(match_id=match_id,started_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
        def record(obs,action):
            if len(trace)<1500:trace.append(dict(actor=obs['current']['yourIndex'],round=obs['current']['round'],observation={k:v for k,v in obs.items() if k in ('current','select')},action=action))
        row,_=play_bo3(agents,match_id,out,metadata=dict(teacher=a,opponent=b,learner_seat=seat,checkpoint_sha256=job['sha256'],deck_sha256=hashlib.sha256(json.dumps(entries[a]['deck']).encode()).hexdigest(),opponent_deck_sha256=hashlib.sha256(json.dumps(entries[b]['deck']).encode()).hexdigest(),scheduler_seed=job['seed'],policy_seed=seed,inference_temperature=1.0,possible_interruption_retry=match_id in job.get('interrupted_ids',[])),trace_callback=record if match_id in job['sample_ids'] else None)
        rows.append(row)
        (out/'active.json').unlink()
        if trace:
            payload=json.dumps(dict(summary=row,trace=trace),separators=(',',':')).encode()
            if retained+len(payload)<=job['trace_cap']:
                folder=out/'sampled-visible-traces';folder.mkdir(exist_ok=True);(folder/f'{match_id}.json').write_bytes(payload);retained+=len(payload)
        write_json(out/'progress.json',dict(completed=len(rows),planned=len(job['jobs']),failures=sum(r['status']!='DONE' for r in rows),wall_seconds=time.monotonic()-started,updated_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    if digest(root/'selected.pt')!=job['sha256']:raise AssertionError('Checkpoint changed')
    return rows

def publish(report,entries,path):
    lines=['# Cross-deck shared-BC round robin','',f"Status: {report['status']}. {report['matches']} attempted BO3, {report['failures']} execution failures. Wall time {report['wall_seconds']/3600:.2f} hours.",'','Both players use the same fixed checkpoint with separate recurrent state and deck knowledge. Each player uses its own copied deck. This measures the combined effect of deck matchup and how the shared policy plays each deck.','', '| A | B | A first-game win rate (95% CI) | A BO3 win rate (95% CI) | BO3 draws |','|---|---|---|---|---:|']
    def fmt(v):
        if v['estimate'] is None:return 'Unavailable'
        lo,hi=v['interval_95'];return f"{100*v['estimate']:.1f}% ({100*lo:.1f}–{100*hi:.1f}%)"
    for m in report['matchups']:lines.append(f"| {entries[m['teacher']]['source_team']} | {entries[m['opponent']]['source_team']} | {fmt(m['first_game_interval'])} | {fmt(m['bo3_interval'])} | {m['bo3'].get('draw',0)} |")
    lines+=['','Each pairing balances game-one chooser rights exactly. Actual first player is chosen by the policy. Intervals are pointwise equal-chooser-weight Wilson-MOVER; draws count as nonwins. Native shuffles and coins are unseeded.']
    if report.get('interruptions'):lines+=['',f"Resumed after {len(report['interruptions'])} interruption(s). Completed records were retained. Potential in-flight attempts are recorded in resumption metadata; their unsaved game outcomes cannot be recovered. Linked retries count once per planned trial."]
    Path(path).write_text('\n'.join(lines)+'\n')

def run(results,output,count=400,workers=4,seed=904088,publish_windows=False,resume=False):
    root=Path(results).resolve();out=Path(output).resolve()
    if not resume and out.exists() and any(out.iterdir()):raise ValueError('Use a fresh experiment directory; previous attempts must be retained')
    out.mkdir(parents=True,exist_ok=True);entries=json.loads((root/'agents.json').read_text())['agents'];sha=digest(root/'selected.pt')
    if len(entries)!=4 or any(e['sha256']!=sha for e in entries):raise ValueError('Checkpoint provenance mismatch')
    jobs=[]
    for a,b in itertools.combinations(range(4),2):jobs.extend((a,b,i,s) for i,s in enumerate(balanced_schedule(count,seed+a*4+b)))
    rng=random.Random(seed);rng.shuffle(jobs);samples=rng.sample(jobs,min(100,len(jobs)));ids=[f'cross-{a}-{b}-{i}' for a,b,i,s in samples]
    existing=[];previous_wall=0;interrupted_ids=[];segment=out
    if resume:
        registration=json.loads((out/'registration.json').read_text())
        for key,value in dict(checkpoint_sha256=sha,agents=entries,count_per_pair=count,seed=seed,workers=workers).items():
            if registration[key]!=value:raise ValueError(f'Resume configuration mismatch: {key}')
        existing=completed_rows(out,jobs,sha,seed,entries)
        seen={r['match_id'] for r in existing}
        events=sorted(out.glob('resume-*/resumption.json'))
        segment=out/f'resume-{len(events)+1:03d}';segment.mkdir()
        last=out if not events else events[-1].parent
        prior_progress=[json.loads(p.read_text()) for p in last.glob('worker-*/progress.json')]
        previous_wall=max((v['wall_seconds'] for v in prior_progress),default=0)
        if events:previous_wall+=json.loads(events[-1].read_text())['previous_wall_seconds']
        # Old workers did not persist active IDs. The next scheduled ID may have
        # been in flight; retain that uncertainty rather than inventing outcomes.
        if not events:
            for i in range(workers):
                path=out/f'worker-{i}'/'matches.jsonl'
                done=len(path.read_text().splitlines()) if path.exists() else 0
                shard=jobs[i::workers]
                if done<len(shard):
                    a,b,index,_=shard[done];interrupted_ids.append(f'cross-{a}-{b}-{index}')
        else:
            interrupted_ids=[json.loads(p.read_text())['match_id'] for p in last.glob('worker-*/active.json')]
        event=dict(status='RESUMING',created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),retained_records=len(existing),remaining_matches=len(jobs)-len(existing),previous_wall_seconds=previous_wall,previous_status=json.loads((out/'status.json').read_text()),possible_interrupted_match_ids=interrupted_ids,interruption_note='Original results retained. In-flight game outcomes not recoverable; linked retries count once per planned trial. WSL shutdown observed at [removed] for first interruption.')
        write_json(segment/'resumption.json',event)
    else:
        write_json(out/'registration.json',dict(status='REGISTERED',created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),checkpoint_sha256=sha,agents=entries,matches=len(jobs),count_per_pair=count,workers=workers,seed=seed,chooser_balance='Half per pairing, shuffled',native_rng_seeded=False,interval_method='Pointwise equal-chooser-weight Wilson-MOVER 95%',sampled_traces=len(samples),diagnostic_cap_bytes=2*1024**3,scope='Cross-deck shared-network evaluation; no learning'))
    pending_jobs=[j for j in jobs if f'cross-{j[0]}-{j[1]}-{j[2]}' not in {r['match_id'] for r in existing}]
    retained_bytes=sum(p.stat().st_size for p in out.rglob('sampled-visible-traces/*.json'))
    start=time.monotonic();write_json(out/'status.json',dict(stage='RUNNING',completed=len(existing),planned_matches=len(jobs),resumed=resume))
    work=[dict(results=str(root),output=str(segment),worker=i,jobs=pending_jobs[i::workers],seed=seed,sha256=sha,sample_ids=ids,interrupted_ids=interrupted_ids,trace_cap=max(0,2*1024**3-retained_bytes)//workers) for i in range(workers)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(worker,w) for w in work];pending=set(futures)
        while pending:
            _,pending=concurrent.futures.wait(pending,timeout=30,return_when=concurrent.futures.FIRST_COMPLETED)
            progress=[json.loads(p.read_text()) for p in segment.glob('worker-*/progress.json')]
            completed=len(existing)+sum(v['completed'] for v in progress);failed=sum(r['status']!='DONE' for r in existing)+sum(v['failures'] for v in progress)
            write_json(out/'status.json',dict(stage='RUNNING',completed=completed,planned_matches=len(jobs),failures=failed,wall_seconds=previous_wall+time.monotonic()-start,updated_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
            if publish_windows:
                text=f'# Cross-deck shared-BC round robin\n\nRunning: {completed:,} / {len(jobs):,} BO3 completed; {failed} execution failures.\n\nSix different-deck pairings, 400 BO3 each. First-game chooser rights balanced 200/200 per pairing. Four CPU workers; no GPU usage. Final first-game and BO3 win rates with 95% intervals will appear here after completion.\n'
                if resume:text+='\nResumed after WSL shut down at [removed]. All previously recorded results were retained; possible interrupted attempts are recorded separately and their linked retries count once per planned trial.\n'
                Path('/path/to/ptcg-agent/reports/bc-cross-deck-round-robin.md').write_text(text)
        rows=existing+[r for f in futures for r in f.result()]
    metrics=[dict(teacher=a,opponent=b,**summarize([r for r in rows if r['teacher']==a and r['opponent']==b])) for a,b in itertools.combinations(range(4),2)]
    failures=sum(r['status']!='DONE' for r in rows);violations=sum(m['chooser_protocol_violations'] for m in metrics)
    interruptions=[json.loads(p.read_text()) for p in out.glob('resume-*/resumption.json')]
    report=dict(interruptions=interruptions,status='COMPLETE' if not failures and not violations else 'COMPLETED_WITH_FAILURES',matches=len(rows),failures=failures,chooser_protocol_violations=violations,wall_seconds=previous_wall+time.monotonic()-start,checkpoint_sha256=sha,matchups=metrics)
    write_json(out/'report.json',report);publish(report,entries,out/'report.md');write_json(out/'status.json',dict(stage=report['status'],matches=len(rows),failures=failures))
    if publish_windows:
        publish(report,entries,Path('reports/bc-cross-deck-round-robin.md'));publish(report,entries,Path('/path/to/ptcg-agent/reports/bc-cross-deck-round-robin.md'))
        with Path('docs/STATUS.md').open('a') as f:f.write(f"\nCross-deck round robin finished: {len(rows)} BO3, {failures} failures, {violations} chooser violations. See reports/bc-cross-deck-round-robin.md.\n")
    print(json.dumps(report),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',required=True);p.add_argument('--output',required=True);p.add_argument('--count',type=int,default=400);p.add_argument('--workers',type=int,default=4);p.add_argument('--publish-windows',action='store_true');p.add_argument('--resume',action='store_true');a=p.parse_args()
    try:run(a.results,a.output,a.count,a.workers,publish_windows=a.publish_windows,resume=a.resume)
    except Exception as e:
        folder=Path(a.output)
        if folder.exists():write_json(folder/'failure.json',dict(error=f'{type(e).__name__}: {e}'))
        raise
