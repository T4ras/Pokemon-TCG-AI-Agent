"""Time-bounded, balanced extension of the frozen decoder comparison."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import datetime
import fcntl
import gzip
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import random
import shutil
import time
import traceback

import torch

from . import decoder_matrix as base
from .live import play_bo3
from .rl_stage1 import load_model, tensor_sha, write_json


ROOT = Path('experiments/rl-decoder-matrix-until-1000-001')
REPORT = Path('reports/rl-decoder-matrix-until-1000.md')
SEED = 904114
DEADLINE = '[removed]'
MODELS = {}
INITIAL_TENSORS = {}
PROVENANCE = None


def utcnow():
    return datetime.datetime.now(datetime.timezone.utc)


def remaining(deadline, at=None):
    return (datetime.datetime.fromisoformat(deadline)-(at or utcnow())).total_seconds()


def admission_guard(durations):
    # Leave time for a complete balanced block and writing/verifying artifacts.
    # No outcome information enters this rule.
    return max(120., 1.5*max(durations[-32:], default=60.)+30.)


def admit_block(deadline, durations, at=None):
    return remaining(deadline, at)>admission_guard(durations)


def make_block(index, seed=SEED):
    if index<0:
        raise ValueError('Negative block index')
    jobs=[]
    ordinal=0
    for cell, own, other in base.CELLS:
        for label in base.LABELS:
            for seat in (0,1):
                jobs.append(dict(match_id=f'ext-b{index:05d}-{cell}-{label}-c{seat}',
                    block=index,cell=cell,learner_label=label,learner_seat=seat,
                    learner_mode=own,opponent_mode=other,scheduler_seed=seed+index,
                    policy_seed=10000000000+24*index+ordinal))
                ordinal+=1
    random.Random(seed+index).shuffle(jobs)
    return jobs


def verify(registration):
    if base.current_provenance()!=registration['provenance']:
        raise ValueError('Frozen decoder/checkpoint/SDK provenance changed')
    if base.sha(__file__)!=registration['extension_source_sha256']:
        raise ValueError('Registered extension runner changed')
    if base.sha(base.ROOT/'matches.jsonl.gz')!=registration['initial_matches_sha256']:
        raise ValueError('Original match archive changed')
    if base.sha(base.ROOT/'registration.json')!=registration['initial_registration_sha256']:
        raise ValueError('Original registration changed')


def initial_rows(registration):
    original=base.read(base.ROOT/'registration.json')
    rows=base.load_existing(base.ROOT,original['jobs'],registration['provenance'])
    if len(rows)!=2400:
        raise ValueError('Original benchmark must have exactly2400 matches')
    return rows


def load_rows(root=ROOT):
    path=Path(root)/'matches.jsonl'
    if not path.exists():
        return []
    rows=[]
    seen=set()
    for line in path.read_text().splitlines():
        row=json.loads(line)
        if row['match_id'] in seen:
            raise ValueError('Duplicate extension match')
        seen.add(row['match_id'])
        rows.append(row)
    return rows


def validate_rows(rows, provenance):
    by_block={}
    seen=set()
    for row in rows:
        index=row['block']
        planned=by_block.setdefault(index,{j['match_id']:j for j in make_block(index)})
        mid=row['match_id']
        if mid not in planned or mid in seen:
            raise ValueError('Unknown or duplicate extension trial')
        if any(row.get(k)!=v for k,v in base.row_metadata(planned[mid],provenance).items()):
            raise ValueError('Extension result provenance mismatch')
        seen.add(mid)


def initializer(provenance):
    global MODELS, INITIAL_TENSORS, PROVENANCE
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    PROVENANCE=provenance
    sources=dict(provenance['learners'],BC=provenance['opponent'])
    for label,entry in sources.items():
        if base.sha(entry['path'])!=entry['sha256']:
            raise ValueError('Worker input checkpoint changed')
        MODELS[label]=load_model(entry['path']).requires_grad_(False)
    INITIAL_TENSORS={label:tensor_sha(model) for label,model in MODELS.items()}


def execute(job, deadline, interrupted=False):
    metadata=base.row_metadata(job,PROVENANCE)
    metadata['possible_interruption_retry']=interrupted
    seconds=remaining(deadline)-10.
    if seconds<=0:
        return dict(metadata,match_id=job['match_id'],status='NOT_STARTED_DEADLINE',
            games=[],wall_seconds=0.,bo3_started=False,recorded_at_utc=base.now())
    torch.manual_seed(job['policy_seed'])
    random.seed(job['policy_seed'])
    own=base.EvaluationAgent(MODELS[job['learner_label']],PROVENANCE['deck'],job['learner_mode'])
    other=base.EvaluationAgent(MODELS['BC'],PROVENANCE['deck'],job['opponent_mode'])
    agents=[own,other] if job['learner_seat']==0 else [other,own]
    row,_=play_bo3(agents,job['match_id'],max_seconds=min(300.,seconds),metadata=metadata)
    # Both the in-memory tensors and immutable checkpoint files are checked after
    # each match; CPU model hashing costs much less than loading a new model.
    for label in (job['learner_label'],'BC'):
        if tensor_sha(MODELS[label])!=INITIAL_TENSORS[label]:
            raise ValueError('Frozen tensors changed in a persistent worker')
    row.update(bo3_started=True,learner_decisions=own.decisions,
        opponent_decisions=other.decisions,
        greedy_rng_checks=own.greedy_rng_checks+other.greedy_rng_checks,
        immutable_tensors_verified=True,recorded_at_utc=base.now())
    return row


def difference(a,b):
    if a['estimate'] is None or b['estimate'] is None:
        return dict(estimate=None,interval_95=None)
    d=a['estimate']-b['estimate']
    lo=d-math.sqrt((a['estimate']-a['interval_95'][0])**2+
                   (b['interval_95'][1]-b['estimate'])**2)
    hi=d+math.sqrt((a['interval_95'][1]-a['estimate'])**2+
                   (b['estimate']-b['interval_95'][0])**2)
    return dict(estimate=d,interval_95=[lo,hi],method='Approximate independent-cell MOVER difference')


def table(cells):
    lines=['| RL decoding | BC decoding | BO3 completed | W/L/D | BO3 WR (95% CI) | First-game WR (95% CI) | Failures |',
           '|---|---|---:|---|---|---|---:|']
    for c in cells:
        counts=c['bo3']
        wdl='/'.join(str(counts.get(k,0)) for k in ('win','loss','draw'))
        lines.append(f"| {c['learner_mode']} | {c['opponent_mode']} | {c['completed_bo3']} | {wdl} | {base.percent(c['bo3_interval'])} | {base.percent(c['first_game_interval'])} | {c['failures']} |")
    return lines


def publish(registration, old_rows, rows, stage, wall_seconds, blocks):
    attempted=[r for r in rows if r.get('bo3_started',True)]
    extension=base.cell_summaries(attempted)
    combined=base.cell_summaries(old_rows+attempted)
    comparisons=[]
    for scope,cells in [('extension_only',extension),('cumulative',combined)]:
        by_cell={c['cell']:c for c in cells}
        for opponent,a,b in [('sampled','AS','SS'),('argmax','AA','SA')]:
            comparisons.append(dict(scope=scope,opponent_mode=opponent,
                **difference(by_cell[a]['bo3_interval'],by_cell[b]['bo3_interval'])))
    result=dict(status=stage,deadline_utc=registration['deadline_utc'],deadline_moscow='[removed]',
        updated_at_utc=base.now(),new_attempted=len(attempted),new_completed=sum(c['completed_bo3'] for c in extension),
        initial_attempted=len(old_rows),cumulative_attempted=len(old_rows)+len(attempted),
        failures=sum(c['failures'] for c in extension),chooser_protocol_violations=sum(c['chooser_protocol_violations'] for c in extension),
        not_started_deadline=sum(not r.get('bo3_started',True) for r in rows),
        extension_cells=extension,cumulative_cells=combined,argmax_minus_sampled=comparisons,
        blocks_completed=blocks,wall_seconds=wall_seconds,
        provenance=registration['provenance'],pooled_learner_labels=base.LABELS,
        native_rng_seeded=False,cpu_only=True)
    write_json(ROOT/'report.json',result)
    status={k:result[k] for k in ['status','deadline_utc','updated_at_utc','new_attempted','new_completed','cumulative_attempted','failures','chooser_protocol_violations','not_started_deadline','blocks_completed','wall_seconds']}
    write_json(ROOT/'status.json',status)
    lines=['# Extended argmax versus sampling benchmark', '',
        f"Status: {stage}. Updated UTC: {result['updated_at_utc']}. Cutoff: **[removed], [removed] ([removed])**.", '',
        f"Additional BO3: {result['new_completed']} completed / {len(attempted)} attempted. "
        f"Cumulative attempted including the original2400: {result['cumulative_attempted']}. "
        f"CPU wall time: {wall_seconds/3600:.2f} hours. No GPU or training.", '',
        'Every complete24-match block has six matches per decoding cell: G/R1/R2 round30 each get both chooser allocations. '
        'Assignments are shuffled and a learner stays fixed within its BO3. All reported win rates pool the learner identities. '
        'The opponent is the exact BC epoch34 from the lambda/entropy experiment, with the same OceanMix deck on both sides. '
        'Sampled means temperature1; argmax is conditional greedy pointer/STOP decoding.', '',
        '## Cumulative results', '']
    lines+=table(combined)
    lines+=['', '## New extension only', '']+table(extension)
    lines+=['', '## Argmax minus sampled learner', '',
        '| Data | BC opponent | Difference in BO3 WR | 95% interval for difference |',
        '|---|---|---:|---|']
    for c in comparisons:
        if c['estimate'] is None:point=interval='pending'
        else:
            point=f"{100*c['estimate']:+.2f} points"
            interval=f"{100*c['interval_95'][0]:+.2f} to {100*c['interval_95'][1]:+.2f} points"
        lines.append(f"| {c['scope']} | {c['opponent_mode']} | {point} | {interval} |")
    lines+=['', f"Failures: {result['failures']}; chooser violations: {result['chooser_protocol_violations']}; "
        f"deadline-skipped unstarted trials: {result['not_started_deadline']}.", '',
        'Stopping is based on the registered clock cutoff and elapsed block durations, never on win rates or significance. '
        'The controller leaves a conservative completion guard before admitting another balanced block; in-flight matches '
        'have a wall-time cap bounded by the remaining time. All completed or failed attempts are retained. '
        'A failure or interruption can leave a partial block; such a remainder is reported explicitly, not silently discarded.', '',
        'Pointwise95% win-rate intervals use equal-chooser Wilson-MOVER; draws are nonwins. '
        'Difference intervals combine uncertainty from both cells. Intervals condition on these three fixed checkpoints. '
        'Because the extension was requested after inspecting the original results, extension-only results provide the '
        'cleaner new check; cumulative intervals are descriptive under this data-dependent extension. '
        'Time-based stopping can also introduce small duration-related effects. Native shuffle/coin randomness is unseeded.', '',
        'Only compact match/game records are retained; no successful full replays or detailed failure traces. '
        'The local CPU controller writes this report automatically while running. The original benchmark and report remain preserved.', '']
    REPORT.write_text('\n'.join(lines))
    for src in [REPORT,ROOT/'status.json',ROOT/'report.json',ROOT/'registration.json']:
        dst=base.WIN/src
        dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(src,dst)
    return result


def prepare():
    ROOT.mkdir(parents=True,exist_ok=True)
    if remaining(DEADLINE)<=0:
        raise ValueError('[removed] deadline has passed; do not roll it to another day')
    original=base.read(base.ROOT/'registration.json')
    base.verify(original)
    if base.read(base.ROOT/'status.json')['status']!='COMPLETE':
        raise ValueError('Original benchmark not complete')
    config=dict(run_id=ROOT.name,deadline_utc=DEADLINE,workers=8,seed=SEED,
        provenance=original['provenance'],extension_source_sha256=base.sha(__file__),
        initial_matches_sha256=base.sha(base.ROOT/'matches.jsonl.gz'),
        initial_registration_sha256=base.sha(base.ROOT/'registration.json'),
        initial_matches=2400,block_matches=24,new_matches_limit=None,
        admission_rule='UTC remaining > max(120s,1.5*max(last32 completed block durations, default60s)+30s)',
        worker_match_cap='min(300s, UTC remaining minus10s); do not start at/after deadline minus10s',
        stopping_depends_on_outcomes=False,model_selection='Random balanced equal G/R1/R2 per block, fixed throughout BO3',
        chooser_balance='Each frozen learner/cell has both chooser allocations in every complete block',
        reporting='Pooled learner WR per decoding cell; extension-only plus cumulative results',
        uncertainty='Pointwise equal-chooser Wilson-MOVER95%; independent-cell MOVER difference intervals',
        native_rng_seeded=False,cpu_only=True,training=False,leaderboard_submission=False,
        successful_full_replays_retained=0,detailed_failure_traces_retained=0,
        preflight_matches=24,approval='User: run as many BO3 as possible evenly across all matchups till10AM; [removed] Europe/Moscow')
    path=ROOT/'registration.json'
    if path.exists():
        if {k:v for k,v in base.read(path).items() if k!='created_at_utc'}!=config:
            raise ValueError('Registered deadline experiment changed')
    else:
        write_json(path,dict(config,created_at_utc=base.now()))
        with Path('docs/STATUS.md').open('a') as f:
            f.write('\nOctober16: registered balanced CPU decoder-matrix extension until[removed] ([removed]), '
                '24-match blocks/eight persistent workers; fixed original BC34 and G/R1/R2 round30. '
                'No result-based stopping, no GPU/training/submission. Initial2400 outcomes preserved; '
                'extension-only and cumulative pooled results reported. Preflight pending.\n')
    reg=base.read(path)
    publish(reg,initial_rows(reg),[],'REGISTERED',0.,0)
    return reg


def preflight():
    reg=base.read(ROOT/'registration.json')
    verify(reg)
    folder=ROOT/'preflight'
    if (ROOT/'preflight.json').exists():
        if base.read(ROOT/'preflight.json')['status']!='PASSED':
            raise ValueError('Prior preflight failed')
        return
    if folder.exists():
        raise ValueError('Partial preflight preserved; inspect before retry')
    folder.mkdir()
    jobs=make_block(1000000)
    # Smoke uses a separate namespace and is excluded from primary results.
    deadline=(utcnow()+datetime.timedelta(minutes=10)).isoformat()
    write_json(folder/'registration.json',dict(jobs=jobs,deadline_utc=deadline,created_at_utc=base.now()))
    with ProcessPoolExecutor(max_workers=8,mp_context=multiprocessing.get_context('spawn'),
            initializer=initializer,initargs=(reg['provenance'],)) as pool:
        rows=list(pool.map(execute,jobs,[deadline]*len(jobs)))
    write_json(folder/'results.json',rows)
    validate_rows(rows,reg['provenance'])
    cells=base.cell_summaries(rows)
    if len(rows)!=24 or any(r['status']!='DONE' for r in rows) or any(c['chooser_protocol_violations'] for c in cells):
        raise ValueError('Native deadline-extension smoke failed')
    result=dict(status='PASSED',native_bo3=24,greedy_rng_checks=sum(r['greedy_rng_checks'] for r in rows),
        immutable_models_verified=all(r['immutable_tensors_verified'] for r in rows),completed_at_utc=base.now())
    write_json(ROOT/'preflight.json',result)
    print(json.dumps(result),flush=True)


def run():
    reg=base.read(ROOT/'registration.json')
    verify(reg)
    if base.read(ROOT/'preflight.json')['status']!='PASSED' or base.read(ROOT/'pytest-validation.json')['status']!='PASSED':
        raise ValueError('Preflight and tests required')
    old=initial_rows(reg)
    rows=load_rows()
    validate_rows(rows,reg['provenance'])
    completed=base.read(ROOT/'blocks.json') if (ROOT/'blocks.json').exists() else []
    durations=[b['wall_seconds'] for b in completed]
    previous=base.read(ROOT/'status.json').get('wall_seconds',0.)
    start=time.monotonic()
    next_block=0
    while sum(r['block']==next_block for r in rows)==24:
        next_block+=1
    seen={r['match_id'] for r in rows}
    interrupted=[]
    if (ROOT/'active-block.json').exists():
        interrupted=[j['match_id'] for j in base.read(ROOT/'active-block.json')['jobs'] if j['match_id'] not in seen]
    publish(reg,old,rows,'RUNNING',previous,len(completed))
    if admit_block(reg['deadline_utc'],durations):
        with ProcessPoolExecutor(max_workers=reg['workers'],mp_context=multiprocessing.get_context('spawn'),
                initializer=initializer,initargs=(reg['provenance'],)) as pool:
            while admit_block(reg['deadline_utc'],durations):
                jobs=[j for j in make_block(next_block) if j['match_id'] not in seen]
                write_json(ROOT/'active-block.json',dict(block=next_block,jobs=jobs,started_at_utc=base.now()))
                bt=time.monotonic()
                waiting={pool.submit(execute,j,reg['deadline_utc'],j['match_id'] in interrupted):j for j in jobs}
                while waiting:
                    done,_=wait(waiting,timeout=15,return_when=FIRST_COMPLETED)
                    for future in done:
                        job=waiting.pop(future)
                        row=future.result()
                        if row['match_id']!=job['match_id'] or row['match_id'] in seen:
                            raise ValueError('Worker returned duplicate/wrong trial')
                        with (ROOT/'matches.jsonl').open('a') as f:
                            f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
                        rows.append(row);seen.add(row['match_id'])
                    if not done:
                        publish(reg,old,rows,'RUNNING',previous+time.monotonic()-start,len(completed))
                duration=time.monotonic()-bt
                durations.append(duration)
                completed.append(dict(block=next_block,wall_seconds=duration,completed_at_utc=base.now()))
                write_json(ROOT/'blocks.json',completed)
                (ROOT/'active-block.json').unlink()
                result=publish(reg,old,rows,'RUNNING',previous+time.monotonic()-start,len(completed))
                print(json.dumps({k:result[k] for k in ['status','new_completed','cumulative_attempted','failures','blocks_completed','wall_seconds']}),flush=True)
                next_block+=1
                if any(r['status']!='DONE' for r in rows[-len(jobs):]):
                    # Keep every failed/unfinished attempt. An execution failure
                    # holds the run for inspection rather than generating retries.
                    raise RuntimeError('Retained non-DONE match; no automatic replacement')
    verify(reg)
    validate_rows(rows,reg['provenance'])
    attempted=[r for r in rows if r.get('bo3_started',True)]
    complete=all(r['status']=='DONE' for r in rows) and len(rows)%24==0
    stage='COMPLETE' if complete else 'COMPLETE_WITH_PARTIAL_BLOCK'
    result=publish(reg,old,rows,stage,previous+time.monotonic()-start,len(completed))
    for name,data in [('matches.jsonl.gz',rows),('cumulative-matches.jsonl.gz',old+attempted)]:
        with gzip.open(ROOT/name,'wt',compresslevel=6) as f:
            for row in data:f.write(json.dumps(row)+'\n')
        shutil.copy2(ROOT/name,base.WIN/ROOT/name)
    with Path('docs/STATUS.md').open('a') as f:
        f.write('\nDeadline CPU decoder-matrix extension '+stage+' at '+base.now()+': '
            +str(result['new_completed'])+' new BO3, '+str(result['cumulative_attempted'])+
            ' cumulative, '+str(result['failures'])+' failures. Stop rule based only on[removed] cutoff and block timing. '
            'See reports/rl-decoder-matrix-until-1000.md. No GPU used.\n')
    shutil.copy2('docs/STATUS.md',base.WIN/'reports/STATUS.md')
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','preflight','run'])
    args=parser.parse_args();ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'controller.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('Deadline evaluator already running')
        try:
            if args.mode=='prepare':prepare()
            elif args.mode=='preflight':preflight()
            else:run()
        except Exception:
            write_json(ROOT/f'{args.mode}-failure-{time.time_ns()}.json',dict(error=traceback.format_exc(),at_utc=base.now()))
            if args.mode=='run' and (ROOT/'registration.json').exists():
                reg=base.read(ROOT/'registration.json')
                publish(reg,initial_rows(reg),load_rows(),'FAILED',base.read(ROOT/'status.json').get('wall_seconds',0.),len(base.read(ROOT/'blocks.json')) if (ROOT/'blocks.json').exists() else 0)
            raise


if __name__=='__main__':
    main()
