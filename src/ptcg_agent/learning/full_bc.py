"""Continuous budgeted shared BC, epoch snapshots and smooth cosine schedule."""
import hashlib,json,math,random,shutil,time
from pathlib import Path
import torch
from .lr_search import Trial,atomic_save,make_scaler,new_state,emit
from .ocean_ablation import model_digest
from .model import PolicyNetwork,metadata_tables
from .training import load_sequences
from ..engine import load_sdk

def cosine_lr(peak,floor,warmup,total,update,anchor=None):
    if update<warmup:return peak*(update+1)/max(1,warmup)
    start=warmup;value=peak
    if anchor:start,value=anchor['update'],anchor['lr']
    progress=max(0.,min(1.,(update-start)/max(1,total-1-start)))
    return floor+.5*(value-floor)*(1+math.cos(math.pi*progress))

class FullTrial(Trial):
    def __init__(self,*args,wall_target=None,**kwargs):
        super().__init__(*args,**kwargs)
        self.wall_target=wall_target;self.epoch_started=time.monotonic();self.epoch_costs=[];self.teacher_best=dict(self.state.get("teacher_best",{}))
        self.best=self.state.get("best_macro_action_nll",self.state.get("validation",{}).get("macro_action_nll",float("inf")))
        if self.state.get('validation') and not (self.output/'best.pt').exists():
            checkpoint=kwargs.get('checkpoint',args[7] if len(args)>7 else None)
            if checkpoint is None:raise ValueError('Resume checkpoint required')
            shutil.copy2(checkpoint,self.output/'best.pt')
        self.state.setdefault('planned_epochs',self.config['total_epochs'])
        self.state.setdefault('cosine',dict(peak=self.config['learning_rate'],floor=self.config['min_lr'],warmup=self.config['warmup_epochs']*self.updates_per_epoch,total=self.state['planned_epochs']*self.updates_per_epoch))
    def learning_rate(self,rate,warmup_updates,relative_updates):
        return cosine_lr(**self.state['cosine'],update=self.state['updates'])
    def validate(self,label,rate):
        metrics=super().validate(label,rate)
        for teacher,v in metrics['teacher'].items():
            if v['action_nll']<self.teacher_best.get(teacher,float('inf')):
                self.teacher_best[teacher]=v['action_nll']
                atomic_save(self.output/f'best-teacher-{teacher}.pt',self.network,self.optimizer,self.scaler,self.state)
        self.state["teacher_best"]=dict(self.teacher_best)
        self.state["best_macro_action_nll"]=self.best
        return metrics
    def on_epoch_end(self):
        now=time.monotonic();self.epoch_costs.append(now-self.epoch_started)
        epoch=self.state['epoch']
        if epoch==self.config.get('budget_calibration_epoch',3) and self.wall_target is not None and not self.state.get('budget_adjustment'):
            cost=sum(self.epoch_costs)/len(self.epoch_costs)*1.02
            remaining=max(0,self.wall_target-now-180)
            planned=max(epoch+1,min(self.config['max_epochs'],epoch+int(remaining/cost)))
            old=self.state['planned_epochs'];schedule=self.state['cosine'];anchor=dict(update=self.state['updates'],lr=self.learning_rate(0,0,0))
            schedule.update(total=planned*self.updates_per_epoch,anchor=anchor);self.state['planned_epochs']=planned
            adjustment=dict(kind='budget_adjustment',epoch=epoch,previous_epochs=old,planned_epochs=planned,mean_epoch_seconds=cost/1.02,remaining_seconds=remaining,cosine_anchor=anchor)
            self.state['budget_adjustment']=adjustment;emit(self.output/'budget.jsonl',adjustment)
        atomic_save(self.output/'last.pt',self.network,self.optimizer,self.scaler,self.state)
        if epoch%self.config['save_every_epochs']==0:
            atomic_save(self.output/f'epoch-{epoch:03d}.pt',self.network,self.optimizer,self.scaler,self.state)
            emit(self.output/'snapshots.jsonl',dict(kind='snapshot',epoch=epoch,updates=self.state['updates'],file=f'epoch-{epoch:03d}.pt'))
        self.epoch_started=time.monotonic()

def run(processed,output,config,require_gpu=True):
    started=time.monotonic();output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if any(output.iterdir()):raise ValueError('Fresh output required; retain failed attempts')
    torch.set_num_threads(2);random.seed(config['seed']);torch.manual_seed(config['seed'])
    if require_gpu and not torch.cuda.is_available():raise RuntimeError('GPU required')
    device=torch.device('cuda' if require_gpu else 'cpu');sdk,_=load_sdk();table,attacks=metadata_tables(sdk)
    network=PolicyNetwork(table,attacks).to(device);optimizer=torch.optim.AdamW(network.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    initial=model_digest(network);atomic_save(output/'initial.pt',network,optimizer,make_scaler(device),new_state(config['seed']));del network,optimizer
    manifest=json.loads((Path(processed)/'manifest.json').read_text())
    if manifest['failures']:raise ValueError('Replay failures')
    report=dict(status='RUNNING',config=config,initial_model_sha256=initial,initial_optimizer_empty=True,data_counts=manifest['counts'],device=str(device),gpu=torch.cuda.get_device_name(0) if require_gpu else None,torch_version=torch.__version__)
    (output/'report.json').write_text(json.dumps(report,indent=2))
    train=load_sequences(processed,'train');val={t:load_sequences(processed,'validation',t) for t in range(4)}
    target=started+config['gpu_target_seconds']
    trial=FullTrial(table,attacks,config,output,device,train,val,output/'initial.pt',wall_target=target if require_gpu else None)
    if trial.updates_per_epoch!=config['updates_per_epoch'] or model_digest(trial.network)!=initial or trial.optimizer.state:raise ValueError('Fresh initialization/epoch mismatch')
    result=trial.train('shared',config['learning_rate'],warmup_epochs=config['warmup_epochs'],finish_epoch=config['total_epochs'],deadline=target+config['finish_grace_seconds']-60)
    atomic_save(output/'final.pt',trial.network,trial.optimizer,trial.scaler,trial.state)
    shutil.copy2(output/'best.pt',output/'selected.pt');shutil.copy2(Path(processed)/'manifest.json',output/'data-manifest.json')
    def agents(checkpoint):
        sha=hashlib.sha256((output/checkpoint).read_bytes()).hexdigest()
        return dict(agents=[dict(name=f'shared-deck-{i}',role='shared-bc-deck-instance',source_submission_id=t['submission_id'],source_team=t['name'],deck=t['deck'],checkpoint='selected.pt',sha256=sha,sampling_temperature=1.) for i,t in enumerate(manifest['teachers'])])
    (output/'agents.json').write_text(json.dumps(agents('selected.pt'),indent=2))
    files=sorted(output.glob('epoch-*.pt'))+[output/'final.pt',output/'selected.pt']
    items=[dict(file=f.name,sha256=hashlib.sha256(f.read_bytes()).hexdigest(),agents=agents(f.name)) for f in files]
    report.update(status='COMPLETE' if result['status']=='COMPLETE' else ('COMPLETE_BUDGET' if result['status']=='INCOMPLETE_BUDGET' else result['status']),trial=result,planned_epochs=trial.state['planned_epochs'],completed_epochs=trial.state['epoch'],updates=trial.state['updates'],budget_adjustment=trial.state.get('budget_adjustment'),checkpoints=items,teacher_best=trial.teacher_best,wall_seconds=time.monotonic()-started)
    (output/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ['checkpoints','data_counts']}),flush=True);return report
