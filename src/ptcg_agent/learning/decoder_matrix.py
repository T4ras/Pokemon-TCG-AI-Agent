"""Registered CPU decoding comparison with a fixed, balanced learner mixture."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import datetime
import fcntl
import gzip
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import random
import shutil
import time
import traceback

import torch

from .benchmark import summarize
from .live import NeuralAgent, play_bo3
from .observations import encode
from .rl_stage1 import load_model, tensor_sha, write_json
from ..engine import SDK
from ..policies import validate_action


ROOT = Path('experiments/rl-decoder-matrix-001')
REPORT = Path('reports/rl-decoder-matrix.md')
WIN = Path('/path/to/ptcg-agent')
CELLS = [('AA', 'argmax', 'argmax'), ('AS', 'argmax', 'sampled'),
         ('SA', 'sampled', 'argmax'), ('SS', 'sampled', 'sampled')]
LABELS = ['G', 'R1', 'R2']
SEED = 904113


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def make_jobs(count=600, seed=SEED):
    if count < 6 or count % 6:
        raise ValueError('Count must be a positive multiple of six')
    # Random assignment without replacement gives exactly equal learner weights
    # and exactly half chooser rights for each frozen learner in every cell.
    base = [(label, seat) for label in LABELS for seat in (0, 1)
            for _ in range(count // 6)]
    random.Random(seed).shuffle(base)
    jobs = []
    for cell_index, (cell, learner_mode, opponent_mode) in enumerate(CELLS):
        for index, (label, seat) in enumerate(base):
            jobs.append(dict(match_id=f'{cell}-{index:04d}', cell=cell,
                learner_label=label, learner_seat=seat,
                learner_mode=learner_mode, opponent_mode=opponent_mode,
                scheduler_seed=seed,
                policy_seed=seed + 1000000 * cell_index + 17 * index))
    random.Random(seed + 1).shuffle(jobs)
    return jobs


class EvaluationAgent(NeuralAgent):
    def __init__(self, network, deck, mode):
        if mode not in ('argmax', 'sampled'):
            raise ValueError('Unknown decoding mode')
        self.greedy = mode == 'argmax'
        self.decisions = 0
        self.greedy_rng_checks = 0
        super().__init__(network, deck)

    def __call__(self, obs):
        row = (encode(obs, self.deck, self.knowledge)
               if getattr(self.network, "observation_schema", "visible-v1") == "visible-v1"
               else encode(obs, self.deck, self.knowledge, schema=self.network.observation_schema))
        before = torch.get_rng_state() if self.greedy else None
        with torch.no_grad():
            stats, self.hidden = self.network.decision(
                [row], self.hidden, greedy=self.greedy, temperature=1.)
        if before is not None:
            if not torch.equal(before, torch.get_rng_state()):
                raise AssertionError('Argmax inference consumed random numbers')
            self.greedy_rng_checks += 1
        action = stats[0][0]
        validate_action(obs['select'], action)
        self.decisions += 1
        return action


def current_provenance():
    stage = Path('experiments/rl-lambda-entropy-stage3-001')
    qualification = read(stage / 'qualification.json')
    expected = {r['run']: r['checkpoint_sha256'] for r in qualification['runs']}
    sources = {}
    baseline = None
    deck = None
    for label in LABELS:
        prior = read(stage / 'evaluation' / f'{label}-round-030' / 'report.json')['provenance']
        checkpoint = Path(prior['checkpoint'])
        if sha(checkpoint) != prior['checkpoint_sha256'] or sha(checkpoint) != expected[label]:
            raise ValueError('Final learner checkpoint differs from confirmation')
        opponent = Path(prior['opponent'])
        if sha(opponent) != prior['opponent_sha256']:
            raise ValueError('Frozen BC checkpoint changed')
        entry = dict(path=str(checkpoint.resolve()), sha256=sha(checkpoint), round=30)
        sources[label] = entry
        candidate = dict(path=str(opponent.resolve()), sha256=sha(opponent), epoch=34)
        if baseline is not None and candidate != baseline:
            raise ValueError('Original runs used different BC opponents')
        if deck is not None and prior['deck'] != deck:
            raise ValueError('Original runs used different decks')
        baseline, deck = candidate, prior['deck']
    paths = [Path(__file__), Path('src/ptcg_agent/learning/model.py'),
             Path('src/ptcg_agent/learning/observations.py'),
             Path('src/ptcg_agent/learning/live.py'),
             Path('src/ptcg_agent/learning/benchmark.py'),
             Path('src/ptcg_agent/learning/rl_stage1.py'),
             Path('src/ptcg_agent/engine.py'), Path('configs/engine.json')]
    return dict(learners=sources, opponent=baseline, deck=deck, deck_name='OceanMix',
        deck_sha256=hashlib.sha256(json.dumps(deck).encode()).hexdigest(),
        source_sha256={str(p): sha(p) for p in paths},
        native_library_sha256=sha(SDK / 'cg/libcg.so'),
        native_rng_seeded=False, inference_temperature=1.,
        argmax='Highest legal conditional decoder logit at each pointer/STOP step')


def prepare(count=600, workers=8):
    ROOT.mkdir(parents=True, exist_ok=True)
    config = dict(run_id=ROOT.name, matches_per_cell=count, workers=workers,
        seed=SEED, jobs=make_jobs(count), provenance=current_provenance(),
        learner_assignment='Randomly shuffled equal G/R1/R2 mixture; fixed within BO3',
        chooser_balance='Half per learner and cell; actual first player is chosen by policy',
        reporting='Pooled learner results only, per decoding cell; identities retained for provenance',
        interval_method='Pointwise equal-chooser Wilson-MOVER 95%; draws are nonwins',
        interval_scope='Conditional on these three fixed checkpoints and their equal mixture; not new training-run uncertainty',
        historical_matches_reused=False, preflight_bo3=8,
        successful_full_replays_retained=0, detailed_failure_traces_retained=0,
        failure_retention='Every compact failed attempt, including valid completed first games',
        cpu_only=True, training=False, leaderboard_submission=False)
    path = ROOT / 'registration.json'
    if path.exists():
        old = read(path)
        if {k:v for k,v in old.items() if k != 'created_at_utc'} != config:
            raise ValueError('Registered experiment changed')
    else:
        write_json(path, dict(config, created_at_utc=now()))
        with Path('docs/STATUS.md').open('a') as f:
            f.write('\nOctober16: registered rl-decoder-matrix-001, four argmax/sampled cells, '
                f'{count} fresh CPU BO3 each. Randomized equal mixture of G/R1/R2 round30, '
                'fixed original BC34 OceanMix opponent. Pooled win rates, balanced chooser rights '
                'within every learner/cell, no training/GPU/submission. Preflight pending.\n')
    publish(read(path), [], 'REGISTERED', 0.)
    return read(path)


def verify(registration):
    if current_provenance() != registration['provenance']:
        raise ValueError('Registered source/SDK/checkpoint provenance changed')
    if make_jobs(registration['matches_per_cell'], registration['seed']) != registration['jobs']:
        raise ValueError('Registered schedule changed')


def row_metadata(job, provenance):
    return dict({k:v for k,v in job.items() if k != 'match_id'},
        checkpoint_sha256=provenance['learners'][job['learner_label']]['sha256'],
        opponent_sha256=provenance['opponent']['sha256'],
        deck_sha256=provenance['deck_sha256'], inference_temperature=1.)


def load_existing(root, jobs, provenance):
    planned = {j['match_id']: j for j in jobs}
    seen = set()
    rows = []
    for p in sorted(Path(root).glob('segment-*/worker-*/matches.jsonl')):
        for line in p.read_text().splitlines():
            row = json.loads(line)
            mid = row['match_id']
            if mid not in planned or mid in seen:
                raise ValueError('Unknown or duplicate planned match')
            if any(row.get(k) != v for k,v in row_metadata(planned[mid], provenance).items()):
                raise ValueError('Completed match provenance differs from registration')
            seen.add(mid)
            rows.append(row)
    return rows


def worker(work):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    out = Path(work['output'])
    out.mkdir(parents=True, exist_ok=True)
    provenance = work['provenance']
    sources = dict(provenance['learners'], BC=provenance['opponent'])
    models = {label:load_model(entry['path']).requires_grad_(False)
              for label,entry in sources.items()}
    before = {label:tensor_sha(model) for label,model in models.items()}
    started = time.monotonic()
    for index, job in enumerate(work['jobs']):
        torch.manual_seed(job['policy_seed'])
        random.seed(job['policy_seed'])
        own = EvaluationAgent(models[job['learner_label']], provenance['deck'], job['learner_mode'])
        other = EvaluationAgent(models['BC'], provenance['deck'], job['opponent_mode'])
        agents = [own,other] if job['learner_seat']==0 else [other,own]
        write_json(out/'active.json',dict(match_id=job['match_id'], started_at_utc=now()))
        metadata = row_metadata(job, provenance)
        metadata['possible_interruption_retry'] = job['match_id'] in work['interrupted_ids']
        row, _ = play_bo3(agents, job['match_id'], max_seconds=300, metadata=metadata)
        row['learner_decisions'] = own.decisions
        row['opponent_decisions'] = other.decisions
        row['greedy_rng_checks'] = own.greedy_rng_checks + other.greedy_rng_checks
        with (out/'matches.jsonl').open('a') as f:
            f.write(json.dumps(row)+'\n')
            f.flush()
            os.fsync(f.fileno())
        (out/'active.json').unlink()
        write_json(out/'progress.json', dict(completed=index+1, planned=len(work['jobs']),
            wall_seconds=time.monotonic()-started, updated_at_utc=now()))
    for label, model in models.items():
        if tensor_sha(model) != before[label] or sha(sources[label]['path']) != sources[label]['sha256']:
            raise ValueError('Frozen model changed during evaluation')
    write_json(out/'completion.json', dict(matches=len(work['jobs']), immutable_models_verified=True,
        wall_seconds=time.monotonic()-started, completed_at_utc=now()))


def cell_summaries(rows):
    return [dict(cell=cell, learner_mode=own, opponent_mode=other,
                 **summarize([r for r in rows if r['cell']==cell]))
            for cell,own,other in CELLS]


def percent(metric):
    if metric['estimate'] is None:
        return 'pending'
    lo, hi = metric['interval_95']
    return f"{100*metric['estimate']:.1f}% ({100*lo:.1f}–{100*hi:.1f}%)"


def publish(registration, rows, stage, wall_seconds):
    cells = cell_summaries(rows)
    result = dict(status=stage, attempted=len(rows), planned=len(registration['jobs']),
        failures=sum(r['status']!='DONE' for r in rows),
        chooser_protocol_violations=sum(c['chooser_protocol_violations'] for c in cells),
        wall_seconds=wall_seconds, updated_at_utc=now(), cells=cells,
        provenance=registration['provenance'], pooled_learner_labels=LABELS,
        learner_allocation_counts=dict(Counter(r['learner_label'] for r in rows)),
        interval_method=registration['interval_method'], interval_scope=registration['interval_scope'])
    write_json(ROOT/'report.json', result)
    write_json(ROOT/'status.json', {k:result[k] for k in ['status','attempted','planned','failures','chooser_protocol_violations','wall_seconds','updated_at_utc']})
    lines = ['# Argmax versus sampling: pooled RL learners against frozen BC', '',
        f"Status: {stage}. Updated UTC: {result['updated_at_utc']}.", '',
        f"{len(rows)}/{result['planned']} fresh CPU BO3 attempted; {registration['matches_per_cell']} planned per cell. "
        'For every matchup, G/R1/R2 round30 are assigned by a randomly shuffled equal mixture, '
        'with chooser rights half per learner. One learner and decoding mode remain fixed throughout each BO3. '
        'Results below pool all three learner identities, as requested.', '',
        'Opponent: the exact frozen BC epoch34 checkpoint used by the lambda/entropy experiment '
        f"(`{registration['provenance']['opponent']['sha256']}`). Both decks are the original OceanMix deck. "
        'Sampled means temperature1; argmax means the highest legal conditional pointer/STOP logit. '
        'Legal deck memory and game resets are unchanged. No training, GPU usage, or leaderboard submission.', '',
        '| RL decoding | BC decoding | BO3 completed | BO3 W/L/D | BO3 WR (95% CI) | First-game WR (95% CI) | Failures |',
        '|---|---|---:|---|---|---|---:|']
    for c in cells:
        counts=c['bo3'];wdl='/'.join(str(counts.get(k,0)) for k in ['win','loss','draw'])
        lines.append(f"| {c['learner_mode']} | {c['opponent_mode']} | {c['completed_bo3']} | {wdl} | {percent(c['bo3_interval'])} | {percent(c['first_game_interval'])} | {c['failures']} |")
    lines += ['', 'Intervals are pointwise equal-chooser Wilson-MOVER95%; draws count as nonwins. '
        'They describe the fixed three-checkpoint mixture, not uncertainty over training another model. '
        'All games of a BO3 remain one match for inference; first-game outcomes are reported separately. '
        'Native shuffles/coins are unseeded, so common schedules do not imply identical deals. '
        'The learner mixture is exactly balanced; policy starting-order choices are not forced.', '',
        f"Chooser violations: {result['chooser_protocol_violations']}. CPU wall time: {wall_seconds/60:.1f} minutes.", '',
        'Compact match/game outcomes, identities, hashes, chooser choices and actual starters are retained. '
        'No full successful replays or detailed failure traces are saved. Failed attempts are retained and not silently replaced.', '']
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text('\n'.join(lines))
    for src,dst in [(REPORT,WIN/REPORT), (ROOT/'report.json',WIN/ROOT/'report.json'),
                    (ROOT/'status.json',WIN/ROOT/'status.json'),
                    (ROOT/'registration.json',WIN/ROOT/'registration.json')]:
        dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(src,dst)
    return result


def preflight():
    registration=read(ROOT/'registration.json')
    verify(registration)
    jobs=[]
    for index,(cell,own,other) in enumerate(CELLS):
        for seat in (0,1):
            jobs.append(dict(match_id=f'smoke-{cell}-{seat}',cell=cell,
                learner_label=LABELS[(2*index+seat)%3],learner_seat=seat,
                learner_mode=own,opponent_mode=other,scheduler_seed=SEED+200,
                policy_seed=SEED+200+31*len(jobs)))
    out=ROOT/'preflight'/'segment-001'/'worker-0'
    if (ROOT/'preflight.json').exists():
        if read(ROOT/'preflight.json')['status']!='PASSED':
            raise RuntimeError('Prior preflight failed; preserve and inspect it')
        return read(ROOT/'preflight.json')
    if out.exists():
        raise RuntimeError('Partial preflight retained; inspect before running again')
    write_json(ROOT/'preflight-registration.json',dict(jobs=jobs,created_at_utc=now()))
    worker(dict(output=str(out),provenance=registration['provenance'],jobs=jobs,interrupted_ids=[]))
    rows=load_existing(ROOT/'preflight',jobs,registration['provenance'])
    if len(rows)!=8 or any(r['status']!='DONE' for r in rows) or any(c['chooser_protocol_violations'] for c in cell_summaries(rows)):
        raise RuntimeError('Native matrix preflight failed')
    result=dict(status='PASSED',native_bo3=8,learner_labels_checked=sorted({r['learner_label'] for r in rows}),
        greedy_rng_checks=sum(r['greedy_rng_checks'] for r in rows), completed_at_utc=now())
    write_json(ROOT/'preflight.json',result)
    print(json.dumps(result),flush=True)
    return result


def run():
    registration=read(ROOT/'registration.json')
    verify(registration)
    if read(ROOT/'preflight.json')['status']!='PASSED' or read(ROOT/'pytest-validation.json')['status']!='PASSED':
        raise RuntimeError('Native preflight and test suite must pass before full evaluation')
    existing=load_existing(ROOT,registration['jobs'],registration['provenance'])
    seen={r['match_id'] for r in existing}
    pending=[j for j in registration['jobs'] if j['match_id'] not in seen]
    interrupted=[]
    for p in ROOT.glob('segment-*/worker-*/active.json'):
        mid=read(p)['match_id']
        if mid not in seen:interrupted.append(mid)
    previous=read(ROOT/'status.json').get('wall_seconds',0.)
    started=time.monotonic()
    publish(registration,existing,'RUNNING' if pending else 'COMPLETE',previous)
    if pending:
        segment=ROOT/f"segment-{len(list(ROOT.glob('segment-*')))+1:03d}"
        segment.mkdir()
        write_json(segment/'resumption.json',dict(retained=len(existing),pending=len(pending),
            possible_interrupted_match_ids=interrupted,created_at_utc=now()))
        work=[dict(output=str((segment/f'worker-{i}').resolve()),provenance=registration['provenance'],
            jobs=pending[i::registration['workers']],interrupted_ids=interrupted)
            for i in range(registration['workers'])]
        with ProcessPoolExecutor(max_workers=registration['workers'],mp_context=multiprocessing.get_context('spawn')) as pool:
            futures=[pool.submit(worker,w) for w in work]
            waiting=set(futures)
            while waiting:
                done,waiting=wait(waiting,timeout=15,return_when=FIRST_COMPLETED)
                for future in done:future.result()
                rows=load_existing(ROOT,registration['jobs'],registration['provenance'])
                result=publish(registration,rows,'RUNNING',previous+time.monotonic()-started)
                print(json.dumps({k:result[k] for k in ['status','attempted','planned','failures','wall_seconds']}),flush=True)
    rows=load_existing(ROOT,registration['jobs'],registration['provenance'])
    verify(registration)
    status='COMPLETE' if len(rows)==len(registration['jobs']) and all(r['status']=='DONE' for r in rows) and not any(c['chooser_protocol_violations'] for c in cell_summaries(rows)) else 'COMPLETED_WITH_FAILURES'
    result=publish(registration,rows,status,previous+time.monotonic()-started)
    with gzip.open(ROOT/'matches.jsonl.gz','wt',compresslevel=6) as f:
        for row in sorted(rows,key=lambda r:r['match_id']):f.write(json.dumps(row)+'\n')
    shutil.copy2(ROOT/'matches.jsonl.gz',WIN/ROOT/'matches.jsonl.gz')
    with Path('docs/STATUS.md').open('a') as f:
        f.write('\nPooled RL decoder matrix '+status+' at '+now()+': '+str(len(rows))+
            ' CPU BO3, '+str(result['failures'])+' failures. Frozen G/R1/R2 equal mixture against original BC34 OceanMix, four decoding cells. See reports/rl-decoder-matrix.md.\n')
    shutil.copy2('docs/STATUS.md',WIN/'reports/STATUS.md')
    print(json.dumps(dict(kind='decoder_matrix_complete',status=status,cells=result['cells'])),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['prepare','preflight','run'])
    parser.add_argument('--count',type=int,default=600)
    parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args()
    ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'controller.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('Matrix evaluator already running')
        try:
            if args.mode=='prepare':prepare(args.count,args.workers)
            elif args.mode=='preflight':preflight()
            else:run()
        except Exception:
            write_json(ROOT/f'{args.mode}-failure-{time.time_ns()}.json',dict(error=traceback.format_exc(),at_utc=now()))
            if args.mode=='run':
                old=read(ROOT/'status.json')
                write_json(ROOT/'status.json',dict(old,status='FAILED',updated_at_utc=now()))
            raise


if __name__=='__main__':
    main()
