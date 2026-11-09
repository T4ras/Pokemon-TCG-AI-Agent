"""One official BO3 per fresh subprocess; keep every result, including failures."""
import collections
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import resource
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from .engine import ROOT, SDK, load_sdk, sample_deck, sha256, validate_deck, official_environment
from .policies import make_policy, validate_action

def timestamp():
    return datetime.now(timezone.utc).isoformat()

def save(path, value):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2))
    tmp.replace(path)

class ObservedPolicy:
    def __init__(self, policy):
        self.policy = policy
        self.contexts = collections.Counter()
        self.latencies = []
        self.exceptions = []
        self.actions = 0
        self.failing_observation = None

    def __call__(self, obs, config=None):
        start = time.perf_counter()
        try:
            action = self.policy(obs)
            if obs.get('select') is not None:
                validate_action(obs['select'],action)
                self.contexts[str(obs['select']['context'])] += 1
            self.actions += 1
            return action
        except Exception:
            self.exceptions.append(traceback.format_exc())
            self.failing_observation = {k:v for k,v in obs.items() if k != "search_begin_input"}
            raise
        finally:
            self.latencies.append(time.perf_counter()-start)

    def diagnostics(self):
        times = sorted(self.latencies)
        return {'actions':self.actions,'contexts':dict(self.contexts),
            'unsupported_contexts':sorted(getattr(self.policy,'unsupported',[])),
            'exceptions':self.exceptions,'inference_seconds':sum(times),
            'latency_p50_seconds':times[len(times)//2] if times else None,
            'latency_p95_seconds':times[min(len(times)-1,int(len(times)*.95))] if times else None,
            'latency_max_seconds':max(times,default=0)}

def worker(request, output):
    if hasattr(os, 'sched_setaffinity'):
        os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])
    started, cpu = time.perf_counter(), time.process_time()
    record = {'status':'running','started_at':timestamp(),'cpu_affinity':sorted(os.sched_getaffinity(0)),**request}
    save(output / 'result.json',record)
    policies = []
    env = None
    try:
        api, game = load_sdk()
        deck = sample_deck()
        cards = {c.cardId:c for c in api.all_card_data()}
        attacks = {a.attackId:a for a in api.all_attack()}
        validate_deck(deck,cards)
        policies = [ObservedPolicy(make_policy(n,deck,request['seed']+i,cards,attacks)) for i,n in enumerate(request['policies'])]
        env = official_environment(bo=3,runTimeout=request['timeout_seconds'])
        env.run(policies)
        replay = env.toJSON()
        replay_path = output / 'replay.json'
        save(replay_path,replay)
        states = [{'status':s.status,'reward':s.reward} for s in env.state]
        games = getattr(env,'result',None)
        failures = [s for s in states if s['status'] != 'DONE']
        record.update({'status':'failed' if failures else 'complete','final':states,
            'game_results':games,'steps':len(env.steps),
            'replay':str(replay_path.relative_to(ROOT)),'replay_sha256':sha256(replay_path),
            'environment_errors':[step[0]['error'] for step in env.steps if 'error' in step[0]]})
        if not games or sum(games) not in (2,3):
            record['status'] = 'failed'
            record['failure_reason'] = 'No complete BO3 game results'
    except Exception:
        record.update(status='failed',failure_reason=traceback.format_exc())
    if record['status'] == 'failed' and env is not None:
        save(output / 'partial-replay.json',env.toJSON())
    for i,p in enumerate(policies):
        if p.failing_observation is not None:
            save(output / f'failing-observation-{i}.json',p.failing_observation)
    record.update(finished_at=timestamp(),wall_seconds=time.perf_counter()-started,
        cpu_seconds=time.process_time()-cpu,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        policy_diagnostics=[p.diagnostics() for p in policies])
    save(output / 'result.json',record)

def wilson(wins, n):
    if not n:
        return [0,1]
    z = 1.959963984540054
    p = wins/n
    center = (p+z*z/(2*n))/(1+z*z/n)
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [center-half,center+half]

def summarize(records, candidate):
    groups = {}
    for r in records:
        opponent = next(n for n in r['policies'] if n != candidate)
        g = groups.setdefault(opponent,{'attempted':0,'complete':0,'wins':0,'losses':0,'draws':0,'failures':0,'games_won':0,'games_lost':0,'games_drawn':0,'seats':{}})
        g['attempted'] += 1
        if r['status'] != 'complete':
            g['failures'] += 1
            continue
        i = r['policies'].index(candidate)
        reward = r['final'][i]['reward']
        key = 'wins' if reward > 0 else 'losses' if reward < 0 else 'draws'
        g[key] += 1
        g['complete'] += 1
        seat = g['seats'].setdefault(str(i),{'wins':0,'losses':0,'draws':0})
        seat[key] += 1
        g['games_won'] += r['game_results'][i]
        g['games_lost'] += r['game_results'][1-i]
        g['games_drawn'] += r['game_results'][2]
    for g in groups.values():
        g['match_win_rate'] = g['wins']/g['complete'] if g['complete'] else None
        g['match_win_rate_95pct_wilson'] = wilson(g['wins'],g['complete'])
    return groups

def source_provenance():
    paths = sorted(p for directory in ('src','tests','configs') for p in (ROOT/directory).rglob('*') if p.is_file() and '__pycache__' not in str(p))
    hashes = {str(p.relative_to(ROOT)):sha256(p) for p in paths}
    return {'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'git_status':subprocess.check_output(['git','status','--porcelain'],text=True),
        'source_sha256':hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest(),
        'source_files':hashes,'dependencies':subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True).splitlines(),
        'python':platform.python_version(),'platform':platform.platform(),
        'framework':importlib.metadata.version('kaggle-environments'),
        'engine':json.loads((ROOT/'configs/engine.json').read_text()),
        'binary_sha256':sha256(SDK/'cg/libcg.so'),
        'card_data_sha256':sha256(ROOT/'data/sdk/extracted/EN_Card_Data_R2_full.csv'),
        'deck_sha256':sha256(SDK/'deck.csv')}

def experiment(config_path, run_id):
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text())
    if not run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in run_id):
        raise ValueError('Run ID must use letters, digits, hyphens or underscores')
    run_dir = ROOT/'experiments'/run_id
    run_dir.mkdir(parents=True,exist_ok=False)
    artifact_dir = ROOT/'artifacts'/run_id
    artifact_dir.mkdir(parents=True,exist_ok=False)
    metadata = {'schema_version':1,'run_id':run_id,'status':'running','started_at':timestamp(),
        'hypothesis':config['hypothesis'],'configuration':config,'provenance':source_provenance(),
        'determinism':'Policy seeds controlled; native RNG not exposed. Seat swaps use independent shuffles.',
        'compute':{'provider':'local WSL CPU','gpu_hours':0,'timeout_per_match_seconds':config['timeout_seconds']},
        'artifacts':str(artifact_dir.relative_to(ROOT))}
    save(run_dir/'run.json',metadata)  # Register before any match.
    records = []
    try:
        for opponent in config['opponents']:
            for pair in range(config['seat_pairs_per_opponent']):
                for seat in (0,1):
                    index = len(records)
                    output = artifact_dir/f'match-{index:04d}'
                    output.mkdir()
                    names = [config['candidate'],opponent] if seat == 0 else [opponent,config['candidate']]
                    request = {'match_id':index,'pair':pair,'candidate_seat':seat,'policies':names,
                        'seed':config['seed']+index*2,'timeout_seconds':config['timeout_seconds']}
                    save(output/'request.json',request)
                    start = time.perf_counter()
                    try:
                        with (output/'worker.log').open('w') as log:
                            proc = subprocess.run([sys.executable,'-m','ptcg_agent.arena','worker',str(output)],
                                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=config['timeout_seconds'])
                        record = json.loads((output/'result.json').read_text()) if (output/'result.json').exists() else {**request,'status':'failed','failure_reason':'Worker produced no record'}
                        if proc.returncode != 0:
                            record.update(status='failed',failure_reason=f'Worker exit {proc.returncode}; inspect worker.log')
                    except subprocess.TimeoutExpired:
                        record = {**request,'status':'timeout','failure_reason':'Hard subprocess timeout'}
                    record['process_wall_seconds'] = time.perf_counter()-start
                    save(output/'result.json',record)
                    records.append(record)
                    with (run_dir/'metrics.jsonl').open('a') as stream:
                        stream.write(json.dumps(record)+'\n')
                    print(f"{run_id} match {index+1}: {record['status']} {record.get('game_results')}",flush=True)
        groups = summarize(records,config['candidate'])
        summary = {'schema_version':1,'matchups':groups,'attempted_matches':len(records),
            'failures':sum(r['status']!='complete' for r in records),
            'process_wall_seconds':sum(r['process_wall_seconds'] for r in records),
            'peak_rss_kib':max((r.get('peak_rss_kib',0) for r in records),default=0),
            'cpu_seconds':sum(r.get('cpu_seconds',0) for r in records),
            'gpu_hours':0,'contexts':{},'unsupported_contexts':[], 'max_match_policy_inference_seconds':0, 'max_decision_seconds':0}
        contexts, unsupported = collections.Counter(), set()
        for r in records:
            for d in r.get('policy_diagnostics',[]):
                summary['max_match_policy_inference_seconds'] = max(summary['max_match_policy_inference_seconds'],d['inference_seconds'])
                summary['max_decision_seconds'] = max(summary['max_decision_seconds'],d['latency_max_seconds'])
                contexts.update(d['contexts'])
                unsupported.update(d['unsupported_contexts'])
        summary['contexts'], summary['unsupported_contexts'] = dict(contexts),sorted(unsupported)
        save(run_dir/'summary.json',summary)
        metadata.update(status='complete' if not summary['failures'] else 'complete_with_failures',finished_at=timestamp())
        save(run_dir/'run.json',metadata)
        report = ['# First heuristic experiment','',config['hypothesis'],'',
            '| Opponent | BO3 W / L / D | Matches | Win rate | 95% Wilson interval | Failures |',
            '|---|---:|---:|---:|---:|---:|']
        for name,g in groups.items():
            lo,hi = g['match_win_rate_95pct_wilson']
            rate = g['match_win_rate']
            report.append(f"| {name} | {g['wins']} / {g['losses']} / {g['draws']} | {g['complete']} | {rate:.1%} | {lo:.1%}–{hi:.1%} | {g['failures']} |" if rate is not None else f'| {name} | No completed matches | 0 | — | — | {g["failures"]} |')
        report += ['', '## Interpretation', '',
            'Retest: this is a small local pilot on one deck, not evidence of competitive leaderboard strength. ',
            'V2 changes attack valuation only. V1 is the fixed nontrivial comparator; random tests basic usefulness.',
            'Intervals use whole BO3 matches. Native randomness is uncontrolled, and seat swaps are not paired identical deals.',
            'The energy-density estimate averages over unknown deck/prize assignments; it never reads face-down prizes.',
            '', '## Reproduction and artifacts','',
            f'`ptcg arena --config {config_path.relative_to(ROOT)} --run-id NEW_UNIQUE_ID`',
            f'Private full replays, requests and worker logs: `{artifact_dir.relative_to(ROOT)}`.',
            'Each match records seat, seeds, game outcomes, action contexts, failures, latency, CPU time and peak RSS.',
            f"Total subprocess wall time: {summary['process_wall_seconds']:.2f}s; peak worker RSS: {summary['peak_rss_kib']/1024:.1f} MiB. GPU use: 0 hours.",
            '', '## Research note — [removed] (Moscow)','',
            'We established a reproducible real simulator baseline, then tested an attack-scoring hypothesis.',
            'Expected damage is a decision aid, not a guarantee: random discards can miss, and high damage can consume the deck.',
            'Next: increase held-out matchup and deck coverage before imitation learning or spending GPU quota.']
        (run_dir/'report.md').write_text('\n'.join(report)+'\n')
        print(json.dumps(summary,indent=2))
    except BaseException:
        metadata.update(status='failed',finished_at=timestamp(),failure_reason=traceback.format_exc())
        save(run_dir/'run.json',metadata)
        raise

if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == 'worker':
        output = Path(sys.argv[2])
        worker(json.loads((output/'request.json').read_text()),output)
    else:
        raise SystemExit('Use ptcg arena')
