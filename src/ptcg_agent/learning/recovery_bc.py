"""CUDA-gated exact-state recovery with an explicitly restarted LR schedule."""
import copy,hashlib,json,shutil,time
from pathlib import Path
import torch
from .full_bc import FullTrial
from .lr_search import atomic_save
from .model import PolicyNetwork,metadata_tables
from .training import load_sequences,sequence_loss
from .ocean_ablation import model_digest
from ..engine import load_sdk

def checkpoint_state(checkpoint,config,teacher_best):
    state=copy.deepcopy(checkpoint['metadata'])
    if state['epoch']!=7 or state['updates']!=1351 or state['next_chunk']!=0:
        raise ValueError('Wrong recovery boundary')
    state['parent_cosine']=state.pop('cosine',None)
    state['parent_budget_adjustment']=state.pop('budget_adjustment',None)
    state['planned_epochs']=config['total_epochs']
    state['cosine']=dict(peak=config['learning_rate'],floor=config['min_lr'],warmup=0,
        total=config['total_epochs']*config['updates_per_epoch'],
        anchor=dict(update=state['updates'],lr=config['learning_rate']))
    state['teacher_best']=teacher_best
    state['best_macro_action_nll']=state['validation']['macro_action_nll']
    return state

def cuda_gate(inputs,output,table,attacks):
    if not torch.cuda.is_available():raise RuntimeError('CUDA gate requires GPU')
    batch=torch.load(inputs/'failed-cohort.pt',map_location='cpu',weights_only=False)['cohort'][8:16]
    results={}
    for name in ['final.pt','best.pt']:
        ck=torch.load(inputs/name,map_location='cpu',weights_only=False)
        net=PolicyNetwork(table,attacks).cuda();net.load_state_dict(ck['model']);net.train()
        torch.manual_seed(1);torch.cuda.manual_seed_all(1)
        with torch.autocast('cuda',dtype=torch.float16):loss,_,count=sequence_loss(net,batch)
        loss.backward()
        finite=bool(torch.isfinite(loss)) and all(bool(torch.isfinite(p.grad).all()) for p in net.parameters() if p.grad is not None)
        norm=float(torch.nn.utils.clip_grad_norm_(net.parameters(),1.))
        results[name]=dict(loss=float(loss.detach()),gradient_norm=norm,finite=finite,decisions=count)
        del net,ck;torch.cuda.empty_cache()
        if not finite or not __import__('math').isfinite(norm):raise FloatingPointError('CUDA precision gate failed')
    result=dict(kind='cuda_gate',status='PASSED',gpu=torch.cuda.get_device_name(0),results=results)
    (output/'cuda-gate.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    return result

def run(processed,output,config):
    started=time.monotonic();output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if any(output.iterdir()):raise ValueError('Fresh recovery output required')
    torch.set_num_threads(2);sdk,_=load_sdk();table,attacks=metadata_tables(sdk)
    inputs=Path(config['resume_checkpoint']).parent
    gate=cuda_gate(inputs,output,table,attacks)
    source=inputs/'best.pt';checkpoint=torch.load(source,map_location='cpu',weights_only=False)
    teacher_best={}
    for t in range(4):
        p=inputs/f'best-teacher-{t}.pt';ck=torch.load(p,map_location='cpu',weights_only=False)
        teacher_best[str(t)]=ck['metadata']['validation']['teacher'][str(t)]['action_nll']
        shutil.copy2(p,output/p.name)
    state=checkpoint_state(checkpoint,config,teacher_best)
    shutil.copy2(source,output/'initial.pt');shutil.copy2(source,output/'best.pt')
    manifest=json.loads((Path(processed)/'manifest.json').read_text())
    if manifest['failures']:raise ValueError('Replay failures')
    report=dict(status='RUNNING',config=config,resume_epoch=7,resume_updates=1351,
        resume_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),cuda_gate=gate,
        data_counts=manifest['counts'],device='cuda',gpu=torch.cuda.get_device_name(0),torch_version=torch.__version__)
    (output/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(dict(kind='recovery_start',resume_epoch=7,resume_updates=1351,lr=config['learning_rate'])),flush=True)
    train=load_sequences(processed,'train');val={t:load_sequences(processed,'validation',t) for t in range(4)}
    target=started+config['gpu_target_seconds']
    trial=FullTrial(table,attacks,config,output,torch.device('cuda'),train,val,source,state_override=state,wall_target=target)
    if trial.updates_per_epoch!=193 or not trial.optimizer.state:raise ValueError('Recovery state mismatch')
    initial=model_digest(trial.network)
    report['initial_model_sha256']=initial;report['optimizer_restored']=True
    result=trial.train('recovery',config['learning_rate'],warmup_epochs=0,finish_epoch=config['total_epochs'],deadline=target+config['finish_grace_seconds']-60)
    atomic_save(output/'final.pt',trial.network,trial.optimizer,trial.scaler,trial.state)
    shutil.copy2(output/'best.pt',output/'selected.pt');shutil.copy2(Path(processed)/'manifest.json',output/'data-manifest.json')
    def agents(checkpoint):
        sha=hashlib.sha256((output/checkpoint).read_bytes()).hexdigest()
        return dict(agents=[dict(name=f'shared-deck-{i}',role='shared-bc-deck-instance',source_submission_id=t['submission_id'],source_team=t['name'],deck=t['deck'],checkpoint='selected.pt',sha256=sha,sampling_temperature=1.) for i,t in enumerate(manifest['teachers'])])
    (output/'agents.json').write_text(json.dumps(agents('selected.pt'),indent=2))
    files=sorted(output.glob('epoch-*.pt'))+[output/'final.pt',output/'selected.pt']
    report.update(status='COMPLETE' if result['status']=='COMPLETE' else ('COMPLETE_BUDGET' if result['status']=='INCOMPLETE_BUDGET' else result['status']),trial=result,
        planned_epochs=trial.state['planned_epochs'],completed_epochs=trial.state['epoch'],updates=trial.state['updates'],budget_adjustment=trial.state.get('budget_adjustment'),
        checkpoints=[dict(file=f.name,sha256=hashlib.sha256(f.read_bytes()).hexdigest(),agents=agents(f.name)) for f in files],teacher_best=trial.teacher_best,wall_seconds=time.monotonic()-started)
    (output/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(dict(kind='recovery_end',status=report['status'],epochs=report['completed_epochs'])),flush=True)
    return report
