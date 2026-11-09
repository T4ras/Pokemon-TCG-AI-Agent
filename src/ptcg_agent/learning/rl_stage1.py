"""Registered single-game-reward trials and resumable frozen CPU mirror evaluation."""
import argparse, copy, datetime, gc, gzip, hashlib, json, multiprocessing, random, shutil, time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
import torch
from .model import PolicyNetwork, schema_from_state
from .rl_timing import TimedAgent, apply_returns
from .live import balanced_schedule, play_bo3
from .benchmark import summarize
from .training import save_checkpoint, load_checkpoint
from .ppo_batched import update
from .rl_full import atomic_save, training_outcomes, should_start_round


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2));temporary.replace(path)
def load_model(path,device='cpu'):
    state=torch.load(path,map_location='cpu',weights_only=False)['model']
    network=PolicyNetwork(state['static_cards'].tolist(),state['attack.weight'].shape[0],observation_schema=schema_from_state(state)).to(device)
    network.load_state_dict(state);network.eval();return network

def tensor_sha(network):
    h=hashlib.sha256()
    for name,tensor in sorted(network.state_dict().items()):h.update(name.encode());h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def evaluate_worker(job):
    torch.set_num_threads(1);out=Path(job['output']);out.mkdir(parents=True,exist_ok=True)
    own=load_model(job['checkpoint']).requires_grad_(False);other=load_model(job['opponent']).requires_grad_(False)
    started=time.monotonic();rows=[]
    for item in job['jobs']:
        torch.manual_seed(item['policy_seed']);random.seed(item['policy_seed'])
        a=TimedAgent(own,job['deck']);b=TimedAgent(other,job['deck'])
        agents=[a,b] if item['learner_seat']==0 else [b,a]
        write_json(out/'active.json',dict(match_id=item['match_id'],started_at_utc=now()))
        row,_=play_bo3(agents,item['match_id'],out,max_seconds=300,metadata=dict(**{k:v for k,v in item.items() if k!='match_id'},checkpoint_sha256=job['checkpoint_sha256'],opponent_sha256=job['opponent_sha256'],deck_sha256=job['deck_sha256'],inference_temperature=1.0,possible_interruption_retry=item['match_id'] in job['interrupted_ids']))
        rows.append(row);(out/'active.json').unlink()
        write_json(out/'progress.json',dict(completed=len(rows),failures=sum(r['status']!='DONE' for r in rows),wall_seconds=time.monotonic()-started,updated_at_utc=now()))
    if sha(job['checkpoint'])!=job['checkpoint_sha256'] or sha(job['opponent'])!=job['opponent_sha256']:raise ValueError('Frozen evaluation checkpoint changed')
    return rows

def evaluate(checkpoint,opponent,deck,output,count=200,seed=904091,workers=4):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    provenance=dict(checkpoint=str(Path(checkpoint).resolve()),opponent=str(Path(opponent).resolve()),checkpoint_sha256=sha(checkpoint),opponent_sha256=sha(opponent),deck=deck,deck_sha256=hashlib.sha256(json.dumps(deck).encode()).hexdigest(),count=count,seed=seed,workers=workers,inference_temperature=1.0,native_rng_seeded=False)
    jobs=[dict(match_id=f'mirror-{i:04d}',learner_seat=seat,scheduler_seed=seed,policy_seed=seed+i*17) for i,seat in enumerate(balanced_schedule(count,seed))]
    registration=out/'registration.json'
    if registration.exists():
        old=json.loads(registration.read_text())
        if old['provenance']!=provenance or old['jobs']!=jobs:raise ValueError('Evaluation resume provenance mismatch')
    else:write_json(registration,dict(status='REGISTERED',created_at_utc=now(),provenance=provenance,jobs=jobs,interval_method='Pointwise equal-chooser Wilson-MOVER 95%',successful_full_replays_retained=0))
    planned={j['match_id']:j for j in jobs};existing=[];seen=set();interrupted=[]
    for p in out.glob('segment-*/worker-*/matches.jsonl'):
        for line in p.read_text().splitlines():
            row=json.loads(line);mid=row['match_id']
            if mid not in planned or mid in seen:raise ValueError('Unknown or duplicate evaluation result')
            if any(row.get(k)!=v for k,v in planned[mid].items()):raise ValueError('Evaluation result provenance mismatch')
            for k in ['checkpoint_sha256','opponent_sha256','deck_sha256','inference_temperature']:
                if row.get(k)!=provenance[k]:raise ValueError('Evaluation artifact provenance mismatch')
            seen.add(mid);existing.append(row)
    for p in out.glob('segment-*/worker-*/active.json'):
        mid=json.loads(p.read_text())['match_id']
        if mid not in seen:interrupted.append(mid)
    pending=[j for j in jobs if j['match_id'] not in seen]
    started=time.monotonic();previous=json.loads((out/'status.json').read_text()).get('wall_seconds',0) if (out/'status.json').exists() else 0
    if pending:
        segment=out/f"segment-{len(list(out.glob('segment-*')))+1:03d}";segment.mkdir()
        write_json(segment/'resumption.json',dict(retained=len(existing),remaining=len(pending),possible_interrupted_match_ids=interrupted,created_at_utc=now()))
        work=[dict(**provenance,output=str(segment/f'worker-{i}'),jobs=pending[i::workers],interrupted_ids=interrupted) for i in range(workers)]
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            futures=[pool.submit(evaluate_worker,w) for w in work];waiting=set(futures)
            while waiting:
                _,waiting=wait(waiting,timeout=15,return_when=FIRST_COMPLETED)
                progress=[json.loads(p.read_text()) for p in segment.glob('worker-*/progress.json')]
                write_json(out/'status.json',dict(stage='RUNNING',completed=len(existing)+sum(p['completed'] for p in progress),planned=count,failures=sum(r['status']!='DONE' for r in existing)+sum(p['failures'] for p in progress),wall_seconds=previous+time.monotonic()-started,updated_at_utc=now()))
            rows=existing+[r for f in futures for r in f.result()]
    else:rows=existing
    result=summarize(rows);variance=0.
    for stratum in result['bo3_interval']['strata']:
        n=stratum['n'];wins=round((stratum['win_rate'] or 0)*n);a=wins+.5;b=n-wins+.5
        variance+=a*b/((a+b)**2*(a+b+1))/4
    result.update(status='COMPLETE' if result['failures']==0 and result['chooser_protocol_violations']==0 and len(rows)==count else 'COMPLETED_WITH_FAILURES',measurement_variance=variance,measurement_standard_error=variance**.5,wall_seconds=previous+time.monotonic()-started,provenance=provenance,training_seed_uncertainty_included=False,completed_at_utc=now())
    write_json(out/'report.json',result);write_json(out/'status.json',dict(stage=result['status'],completed=len(rows),planned=count,failures=result['failures'],wall_seconds=result['wall_seconds'],updated_at_utc=now()))
    print(json.dumps(dict(kind='evaluation_complete',output=str(out),status=result['status'],bo3=result['bo3_interval'])),flush=True);return result

def run(inputs,output,config,require_gpu=True,script_started=None,resume=False):
    inputs=Path(inputs);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    if not resume and (out/'report.json').exists():raise ValueError('Existing run: explicit resume required')
    if require_gpu and not torch.cuda.is_available():raise RuntimeError('Kaggle GPU required')
    device='cuda' if require_gpu else 'cpu';torch.set_num_threads(2)
    started=script_started if script_started is not None else time.monotonic();target=started+config['runtime_seconds'];finish=target+config['finish_margin_seconds']
    manifest=json.loads((inputs/'checkpoints.json').read_text());source=inputs/'epoch-34.pt'
    if sha(source)!=manifest['sha256']['34']:raise ValueError('BC checkpoint digest mismatch')
    deck=manifest['decks'][2];fixed=load_model(source,device).requires_grad_(False);initial=tensor_sha(fixed)
    if resume:
        report=json.loads((out/'report.json').read_text())
        if report['config']!=config or report['initial_tensor_sha256']!=initial:raise ValueError('Training resume mismatch')
    else:report=dict(status='RUNNING',config=config,initial_tensor_sha256=initial,opponent_sha256=sha(source),deck=deck,started_at_utc=now(),hardware=dict(device=device,gpu=torch.cuda.get_device_name(0) if require_gpu else None,torch=torch.__version__),trials={},checkpoints=[],native_rng_seeded=False,objective='game',post_training_evaluation='local CPU after verified retrieval')
    def emit(kind,**values):
        event=dict(kind=kind,elapsed_seconds=time.monotonic()-started,**values)
        with (out/'events.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
        print(json.dumps(event),flush=True)
    def persist():report['wall_seconds']=time.monotonic()-started;write_json(out/'report.json',report)
    emit('initialized',**report['hardware'],trial_order=[t['id'] for t in config['trials']]);persist()
    for trial in config['trials']:
        key=trial['id'];folder=out/key;folder.mkdir(exist_ok=True)
        state=report['trials'].setdefault(key,dict(parameters=trial,completed_rounds=0,rounds=[],status='RUNNING'))
        random.seed(config['seed']);torch.manual_seed(config['seed'])
        learner=copy.deepcopy(fixed).requires_grad_(True)
        optimizer=torch.optim.AdamW([{'params':[p for n,p in learner.named_parameters() if not n.startswith('value.')],'lr':config['actor_lr']},{'params':list(learner.value.parameters()),'lr':config['critic_lr']}],weight_decay=config['weight_decay'])
        if tensor_sha(learner)!=initial or optimizer.state:raise AssertionError('Different initial model or optimizer')
        latest=folder/'latest.pt'
        if resume and latest.exists():
            metadata=load_checkpoint(latest,learner,optimizer,restore_rng=True);state['completed_rounds']=metadata['round']
        emit('trial_started',trial=key,parameters=trial,initial_tensor_sha256=initial,completed_rounds=state['completed_rounds'])
        for ri in range(state['completed_rounds'],config['rounds']):
            active=folder/'active-rollouts.pt';saved=None
            if resume and active.exists():
                saved=torch.load(active,map_location=device,weights_only=False)
                if saved['round']!=ri+1 or saved['config']!=config or saved['trial']!=trial:raise ValueError('Active rollout recovery mismatch')
            previous=[r['round_wall_seconds'] for t in report['trials'].values() for r in t['rounds'] if 'round_wall_seconds' in r]
            if not should_start_round(target-time.monotonic(),previous,config['round_seconds_initial_estimate']):
                report['status']='BUDGET_INCOMPLETE';state['status']='PAUSED_AT_COMPLETE_ROUND';persist();emit('budget_stop',trial=key,completed_rounds=state['completed_rounds']);return report
            rt=time.monotonic();timings={};rows=[]
            if saved:
                learner.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);random.setstate(saved['python_rng']);torch.set_rng_state(saved['torch_rng'].cpu())
                if require_gpu:torch.cuda.set_rng_state_all([x.cpu() for x in saved['cuda_rng']])
                games=saved['games'];record=saved['record'];timings=record['timings_seconds']
            else:
                games=[];ct=time.monotonic();decisions=0
                for index,seat in enumerate(balanced_schedule(config['matches_per_round'],config['seed']+ri)):
                    a=TimedAgent(learner,deck,fixed,True);b=TimedAgent(fixed,deck)
                    agents=[a,b] if seat==0 else [b,a]
                    summary,rollouts=play_bo3(agents,f'{key}-r{ri+1}-{index+1}',folder,max_seconds=min(config['match_seconds'],max(1,finish-time.monotonic())),metadata=dict(trial=key,round_index=ri+1,collecting_seats={'game':seat},learner_seat=seat,policy_version=ri,scheduler_seed=config['seed']+ri,inference_temperature=1.0,opponent_sha256=report['opponent_sha256']))
                    rows.append(summary)
                    if summary['status']!='DONE' or summary['games'][0]['chooser']!=0:
                        report.update(status='FAILED_COLLECTION',failure=summary);state['partial_outcomes']=training_outcomes(rows)['game'];persist();raise RuntimeError('Failed match or chooser violation; partial results retained')
                    for agent in agents:
                        for metric in ('observation_encoding','policy_inference','reference_inference'):timings[metric]=timings.get(metric,0)+agent.timing[metric]
                    game_rows=apply_returns(rollouts[seat],summary['result'],seat,'game',lam=trial['lambda']);games.extend(game_rows);decisions+=sum(map(len,game_rows))
                    if decisions>config['rollout_cap_decisions']:raise RuntimeError('Registered rollout cap exceeded')
                    if (index+1)%8==0:emit('collection_progress',trial=key,round=ri+1,matches=index+1,decisions=decisions)
                if sum(r['learner_seat']==0 for r in rows)!=len(rows)//2:raise AssertionError('Chooser balance failed')
                timings['collection_total']=time.monotonic()-ct
                record=dict(round=ri+1,matches=len(rows),learner_decisions=decisions,timings_seconds=timings,training_outcomes=training_outcomes(rows)['game'],status='COLLECTED')
                t=time.monotonic();atomic_save(active,dict(round=ri+1,games=games,record=record,config=config,trial=trial,model=learner.state_dict(),optimizer=optimizer.state_dict(),python_rng=random.getstate(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if require_gpu else []))
                if active.stat().st_size>config['rollout_buffer_max_bytes']:raise RuntimeError('Registered rollout storage cap exceeded')
                with (folder/'matches.jsonl').open('rb') as src,gzip.open(folder/f'matches-round-{ri+1:03d}.jsonl.gz','wb',compresslevel=3) as dst:shutil.copyfileobj(src,dst)
                (folder/'matches.jsonl').unlink();timings['recovery_and_logging']=time.monotonic()-t
            emit('collection_complete',trial=key,**record);t=time.monotonic()
            metrics=update(learner,optimizer,games,config['batch_sequences'],epochs=config['ppo_epochs'],seed=config['seed']+ri,entropy_coefficient=trial['entropy_coefficient'],bc_kl_coefficient=config['bc_kl_coefficient'],value_coefficient=config['value_coefficient'],target_kl=config['old_policy_kl_stop'],deadline=finish,allow_oom_fallback=True,collect_diagnostics=True,component_gradients=True,callback=lambda m:emit('update_progress',trial=key,round=ri+1,**m))
            if require_gpu:torch.cuda.synchronize()
            timings['ppo']=time.monotonic()-t
            if any(m['stop_reason']=='deadline' for m in metrics):raise TimeoutError('Incomplete PPO update: recovery rollout retained')
            if not all(torch.isfinite(p).all() for p in learner.parameters()):raise FloatingPointError('Nonfinite model')
            record.update(ppo=metrics,status='UPDATED');t=time.monotonic()
            file=folder/f'game-round-{ri+1:03d}.pt';save_checkpoint(file,learner,optimizer,dict(round=ri+1,objective='game',trial=trial,config=config,deck=deck,record=record,initial_tensor_sha256=initial))
            tmp=folder/'latest.tmp';shutil.copy2(file,tmp);tmp.replace(latest)
            item=dict(file=str(file.relative_to(out)),trial=key,round=ri+1,sha256=sha(file),bytes=file.stat().st_size)
            if sum(p.stat().st_size for p in out.rglob('*.pt'))>config.get('max_checkpoint_bytes',8*1024**3):raise RuntimeError('Checkpoint storage cap exceeded')
            if not any(i['file']==item['file'] for i in report['checkpoints']):report['checkpoints'].append(item)
            active.unlink();timings['checkpoint']=time.monotonic()-t;record['round_wall_seconds']=time.monotonic()-rt
            state['rounds']=[r for r in state['rounds'] if r['round']!=ri+1]+[record];state['completed_rounds']=ri+1
            optimizer.zero_grad(set_to_none=True);del games,saved;gc.collect()
            if require_gpu:torch.cuda.empty_cache()
            persist();emit('round_complete',trial=key,**record)
        state['status']='COMPLETE'
        if tensor_sha(fixed)!=initial:raise AssertionError('Frozen BC changed')
        persist();del learner,optimizer;gc.collect()
        if require_gpu:torch.cuda.empty_cache()
    report.update(status='COMPLETE',completed_at_utc=now(),completed_rounds=sum(t['completed_rounds'] for t in report['trials'].values()),total_training_bo3=config['matches_per_round']*sum(t['completed_rounds'] for t in report['trials'].values()))
    persist();emit('complete',completed_rounds=report['completed_rounds'],total_training_bo3=report['total_training_bo3']);return report

def cuda_gate(inputs):
    """Real BC tensors: finite CUDA forward/backward for both entropy settings."""
    from .model import D
    from .ppo_batched import loss_batched
    fixed=load_model(Path(inputs)/'epoch-34.pt','cuda');results=[]
    for coefficient in [.005,.02]:
        network=copy.deepcopy(fixed).requires_grad_(True);hidden=torch.zeros(1,D,device='cuda');rows=[]
        for i in range(3):
            row=dict(ids=[0],kinds=[0],features=[[0.]*32],option_ids=[0,0,0],targets=[0,0,0],attack_ids=[0,0,0],option_features=[[0.]*32]*3,minimum=1,maximum=1,reset=i==0)
            before=hidden.detach().cpu().clone()
            with torch.no_grad():stats,hidden=network.decision([row],hidden,actions=[[0]])
            row.update(action=[0],hidden=before,old_logp=float(stats[1]),old_value=float(stats[3]),reference=[r.cpu() for r in stats[-1]],advantage=.2,**{'return':1.});rows.append(row)
        loss,n,kl=loss_batched(network,[(rows,0,3)],0.,1.,entropy_coefficient=coefficient)
        (loss/n).backward();norm=torch.nn.utils.clip_grad_norm_(network.parameters(),.5,error_if_nonfinite=True)
        if not torch.isfinite(loss) or not torch.isfinite(kl):raise FloatingPointError('CUDA gate failed')
        results.append(dict(entropy_coefficient=coefficient,loss=float(loss.detach()/n),gradient_norm=float(norm),decisions=n))
        del network,rows,loss,kl;gc.collect();torch.cuda.empty_cache()
    del fixed;gc.collect();torch.cuda.empty_cache();return dict(status='PASSED',results=results)

def main():
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='mode',required=True)
    train=s.add_parser('train');train.add_argument('--inputs',required=True);train.add_argument('--output',required=True);train.add_argument('--config',required=True);train.add_argument('--cpu-smoke',action='store_true');train.add_argument('--resume',action='store_true')
    ev=s.add_parser('evaluate');ev.add_argument('--checkpoint',required=True);ev.add_argument('--opponent',required=True);ev.add_argument('--inputs',required=True);ev.add_argument('--output',required=True);ev.add_argument('--count',type=int,default=200);ev.add_argument('--workers',type=int,default=4);ev.add_argument('--seed',type=int,default=904091)
    a=p.parse_args()
    if a.mode=='train':run(a.inputs,a.output,json.loads(Path(a.config).read_text()),not a.cpu_smoke,resume=a.resume)
    else:evaluate(a.checkpoint,a.opponent,json.loads((Path(a.inputs)/'checkpoints.json').read_text())['decks'][2],a.output,a.count,a.seed,a.workers)
if __name__=='__main__':main()
