"""Controlled shared-BC warmup trials and exact-state continuation branches."""
import copy, hashlib, json, math, os, random, shutil, time
from pathlib import Path
import torch
from .model import PolicyNetwork, metadata_tables
from .training import chunks, sequence_loss, evaluate_nll, load_sequences, load_checkpoint
from ..engine import load_sdk

def emit(path,record):
    with Path(path).open("a") as f:f.write(json.dumps(record,allow_nan=False)+"\n")
    print(json.dumps(record,allow_nan=False),flush=True)

def epoch_order(data,seed,epoch,target_per_teacher=None):
    groups={}
    for c in data:groups.setdefault(c[0][0]["teacher"],[]).append(c)
    rng=random.Random(seed+epoch*1009);target=max(map(len,groups.values()));result=[]
    if target_per_teacher is not None:
        if target_per_teacher<=0:raise ValueError('Positive fixed epoch size required')
        for teacher in sorted(groups):
            local=random.Random(seed+epoch*1009+teacher*100003)
            result.extend(local.choices(groups[teacher],k=target_per_teacher))
        rng.shuffle(result)
        return result
    for teacher in sorted(groups):
        group=groups[teacher]
        result.extend(rng.choices(group,k=target) if len(group)<target else group)
    rng.shuffle(result);return result

def target_lr(rate,warmup_updates,updates):
    return rate*min(1.,(updates+1)/max(1,warmup_updates)) if warmup_updates else rate

def evaluate_teachers(network,validation,batch_sequences):
    metrics={str(t):evaluate_nll(network,validation[t],batch_sequences) for t in sorted(validation)}
    if not all(v["decisions"]>0 and math.isfinite(v["action_nll"]) for v in metrics.values()):raise ValueError("Invalid teacher validation")
    return dict(teacher=metrics,macro_action_nll=sum(v["action_nll"] for v in metrics.values())/len(metrics))

def atomic_save(path,network,optimizer,scaler,state):
    payload=dict(model=network.state_dict(),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),metadata=copy.deepcopy(state),python_rng=random.getstate(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
    path=Path(path);temp=path.with_suffix(".tmp");torch.save(payload,temp);temp.replace(path)

def restore(path,network,optimizer,scaler):
    payload=torch.load(path,map_location=next(network.parameters()).device,weights_only=False)
    network.load_state_dict(payload["model"]);optimizer.load_state_dict(payload["optimizer"])
    if payload.get("scaler"):scaler.load_state_dict(payload["scaler"])
    random.setstate(payload["python_rng"]);torch.set_rng_state(payload["torch_rng"].cpu())
    if torch.cuda.is_available() and next(network.parameters()).is_cuda:
        torch.cuda.set_rng_state_all([r.cpu() for r in payload["cuda_rng"]])
    return copy.deepcopy(payload["metadata"])

def new_state(seed):
    return dict(epoch=0,next_chunk=0,updates=0,decisions=0,order_seed=seed,epoch_loss_sum=0.,epoch_nll_sum=0.,epoch_decisions=0)

def make_scaler(device):return torch.amp.GradScaler("cuda",enabled=device.type=="cuda")

class Trial:
    def __init__(self,table,attacks,config,output,device,train,validation,checkpoint,state_override=None):
        self.config=config;self.output=Path(output);self.output.mkdir(parents=True,exist_ok=True)
        self.device=device;self.network=PolicyNetwork(table,attacks,observation_schema=config.get("observation_schema","visible-v1")).to(device)
        self.optimizer=torch.optim.AdamW(self.network.parameters(),lr=1e-4,weight_decay=config["weight_decay"])
        self.scaler=make_scaler(device)
        previous=restore(checkpoint,self.network,self.optimizer,self.scaler)
        self.state=copy.deepcopy(state_override) if state_override is not None else previous
        self.data=chunks(train);self.validation=validation
        self.updates_per_epoch=math.ceil(math.ceil(len(epoch_order(self.data,self.state["order_seed"],0,self.config.get("chunks_per_teacher")))/config["batch_sequences"])/config["accumulation"])
        self.precision="float16" if device.type=="cuda" else "float32"
        self.best=float("inf");self.measurements=[];self.update_seconds=[]
    def validate(self,label,rate):
        before=time.monotonic();metrics=evaluate_teachers(self.network,self.validation,self.config["batch_sequences"])
        if self.config.get('selection_teachers'):
            metrics['all_teacher_macro_action_nll']=metrics['macro_action_nll']
            metrics['macro_action_nll']=sum(metrics['teacher'][str(t)]['action_nll'] for t in self.config['selection_teachers'])/len(self.config['selection_teachers'])
            metrics['selection_teachers']=self.config['selection_teachers']
        record=dict(kind="validation",trial=label,target_lr=rate,epoch=self.state["epoch"],next_chunk=self.state["next_chunk"],updates=self.state["updates"],decisions=self.state["decisions"],wall_seconds=time.monotonic()-before,**metrics)
        emit(self.output/"validation.jsonl",record);self.measurements.append(record)
        self.state["validation"]=metrics;self.state["label"]=label;self.state["target_lr"]=rate
        atomic_save(self.output/"last.pt",self.network,self.optimizer,self.scaler,self.state)
        if metrics["macro_action_nll"]<self.best:
            self.best=metrics["macro_action_nll"];atomic_save(self.output/"best.pt",self.network,self.optimizer,self.scaler,self.state)
        return metrics
    def learning_rate(self,rate,warmup_updates,relative_updates):
        return target_lr(rate,warmup_updates,relative_updates)
    def on_epoch_end(self):
        pass
    def train(self,label,rate,warmup_epochs=0,finish_epoch=None,max_updates=None,deadline=None):
        origin_updates=self.state["updates"];warmup_updates=warmup_epochs*self.updates_per_epoch
        start=time.monotonic();self.validate(label,rate);status="COMPLETE";failure=None;seen_overflows=0;failure_context=None
        try:
            while finish_epoch is None or self.state["epoch"]<finish_epoch:
                order=epoch_order(self.data,self.state["order_seed"],self.state["epoch"],self.config.get("chunks_per_teacher"))
                while self.state["next_chunk"]<len(order):
                    if max_updates is not None and self.state["updates"]-origin_updates>=max_updates:break
                    if deadline is not None and time.monotonic()>deadline:status="INCOMPLETE_BUDGET";break
                    began=time.monotonic();begin=self.state["next_chunk"]
                    end=min(len(order),begin+self.config["batch_sequences"]*self.config["accumulation"])
                    batch_count=math.ceil((end-begin)/self.config["batch_sequences"])
                    lr=self.learning_rate(rate,warmup_updates,self.state["updates"]-origin_updates)
                    for group in self.optimizer.param_groups:group["lr"]=lr
                    self.network.train();self.optimizer.zero_grad(set_to_none=True);total_loss=0.;nll=0.;decisions=0
                    for offset in range(begin,end,self.config["batch_sequences"]):
                        batch=order[offset:min(end,offset+self.config["batch_sequences"])]
                        if self.config.get('capture_numerical_failure',False):
                            failure_context=dict(update=self.state['updates'],epoch=self.state['epoch'],chunk_offset=offset,batch=batch,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if self.device.type=='cuda' else [],python_rng=random.getstate())
                        with torch.autocast(device_type=self.device.type,dtype=torch.float16,enabled=self.device.type=="cuda"):
                            loss,likelihood,count=sequence_loss(self.network,batch)
                        if not torch.isfinite(loss):raise FloatingPointError("Nonfinite BC loss")
                        self.scaler.scale(loss/batch_count).backward()
                        total_loss+=float(loss.detach())/batch_count;nll+=float(likelihood);decisions+=count
                    self.scaler.unscale_(self.optimizer)
                    norm=torch.nn.utils.clip_grad_norm_(self.network.parameters(),self.config["clip_grad_norm"])
                    if not torch.isfinite(norm):
                        if self.scaler.is_enabled():
                            self.scaler.step(self.optimizer);self.scaler.update();seen_overflows+=1
                            emit(self.output/"updates.jsonl",dict(kind="gradient_overflow",trial=label,attempt=seen_overflows,updates=self.state["updates"],lr=lr))
                            if seen_overflows<=3:continue
                        raise FloatingPointError("Persistent nonfinite gradients")
                    seen_overflows=0
                    previous=[p.detach().clone() for p in self.network.parameters()]
                    self.scaler.step(self.optimizer);self.scaler.update()
                    delta=float(torch.stack([(p.detach()-old).square().sum() for p,old in zip(self.network.parameters(),previous)]).sum().sqrt())
                    weight=float(torch.stack([p.square().sum() for p in previous]).sum().sqrt())
                    if not all(torch.isfinite(p).all() for p in self.network.parameters()):raise FloatingPointError("Nonfinite model parameter")
                    self.state["updates"]+=1;self.state["decisions"]+=decisions;self.state["next_chunk"]=end
                    self.state["epoch_loss_sum"]+=total_loss;self.state["epoch_nll_sum"]+=nll;self.state["epoch_decisions"]+=decisions
                    seconds=time.monotonic()-began;self.update_seconds.append(seconds)
                    record=dict(kind="update",trial=label,epoch=self.state["epoch"],updates=self.state["updates"],next_chunk=end,lr=lr,loss=total_loss,action_nll=nll/max(1,decisions),decisions=decisions,gradient_norm=float(norm),clipped=bool(norm>self.config["clip_grad_norm"]),parameter_update_norm=delta,relative_update_norm=delta/max(weight,1e-12),seconds=seconds)
                    with (self.output/"updates.jsonl").open("a") as f:f.write(json.dumps(record)+"\n")
                    if self.state["updates"]%10==0:print(json.dumps(record),flush=True)
                    if self.state["updates"]%self.config["validation_interval_updates"]==0:self.validate(label,rate)
                    elif self.state["updates"]%self.config["periodic_checkpoint_updates"]==0:atomic_save(self.output/"last.pt",self.network,self.optimizer,self.scaler,self.state)
                if self.state["next_chunk"]>=len(order):
                    emit(self.output/"epochs.jsonl",dict(kind="epoch",trial=label,completed_epoch=self.state["epoch"]+1,updates=self.state["updates"],decisions=self.state["epoch_decisions"],action_nll=self.state["epoch_nll_sum"]/max(1,self.state["epoch_decisions"])))
                    self.state.update(epoch=self.state["epoch"]+1,next_chunk=0,epoch_loss_sum=0.,epoch_nll_sum=0.,epoch_decisions=0)
                if status!="COMPLETE" or (max_updates is not None and self.state["updates"]-origin_updates>=max_updates):break
                self.validate(label,rate)
                self.on_epoch_end()
                finish_epoch=self.state.get("planned_epochs",finish_epoch)
        except (FloatingPointError,RuntimeError,ValueError) as exc:
            if failure_context is not None:
                torch.save(failure_context,self.output/'failed-microbatch.pt')
            numerical=isinstance(exc,FloatingPointError) or "parameter logits" in str(exc)
            status="FAILED_NUMERICAL" if numerical else "FAILED_RUNTIME"
            failure=f"{type(exc).__name__}: {str(exc)[:300]}"
            (self.output/"failure.json").write_text(json.dumps(dict(status=status,error=failure,state=self.state),indent=2))
        if status in ("COMPLETE","INCOMPLETE_BUDGET"):self.validate(label,rate)
        result=dict(label=label,target_lr=rate,status=status,error=failure,completed_epochs=self.state["epoch"],updates=self.state["updates"]-origin_updates,best_macro_action_nll=self.best,checkpoint=str(self.output/"best.pt"),last_checkpoint=str(self.output/"last.pt"),wall_seconds=time.monotonic()-start,updates_per_epoch=self.updates_per_epoch,mean_update_seconds=sum(self.update_seconds)/max(1,len(self.update_seconds)),precision=self.precision)
        (self.output/"trial.json").write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True);return result

def run(processed,starting,output,config,require_gpu=True):
    entered=time.monotonic();output=Path(output);output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);device=torch.device("cuda" if require_gpu else "cpu")
    if require_gpu and not torch.cuda.is_available():raise RuntimeError("GPU required for registered search")
    data=json.loads((Path(processed)/"manifest.json").read_text())
    train=load_sequences(processed,"train");validation={t:load_sequences(processed,"validation",t) for t in range(4)}
    api,_=load_sdk();table,attacks=metadata_tables(api)
    base_network=PolicyNetwork(table,attacks).to(device);base_optimizer=torch.optim.AdamW(base_network.parameters(),lr=1e-4,weight_decay=config["weight_decay"])
    load_checkpoint(starting,base_network,base_optimizer,restore_rng=True)
    base_scaler=make_scaler(device);baseline=evaluate_teachers(base_network,validation,config["batch_sequences"])
    base_state=new_state(config["seed"]);base_state["validation"]=baseline
    atomic_save(output/"initial.pt",base_network,base_optimizer,base_scaler,base_state)
    del base_network,base_optimizer
    report=dict(status="RUNNING",config=config,baseline=baseline,data_counts=data["counts"],initial_trials=[],refinement_trials=[],device=str(device),gpu=torch.cuda.get_device_name(0) if require_gpu else None,torch_version=torch.__version__,native_rng_seeded=False)
    def persist():
        report["wall_seconds"]=time.monotonic()-entered
        tmp=output/"report.tmp";tmp.write_text(json.dumps(report,indent=2));tmp.replace(output/"report.json")
    persist();target=entered+config["gpu_target_seconds"];safety=target+config["finish_grace_seconds"]
    for i,rate in enumerate(config["initial_lrs"]):
        trial=Trial(table,attacks,config,output/f"initial-{i}",device,train,validation,output/"initial.pt")
        result=trial.train(f"initial-{i}",rate,warmup_epochs=config["warmup_epochs"],finish_epoch=config["initial_epochs"],deadline=safety-90)
        report["initial_trials"].append(result);persist();del trial
        if require_gpu:torch.cuda.empty_cache()
        if result["status"]=="INCOMPLETE_BUDGET":break
    completed=[r for r in report["initial_trials"] if r["status"]=="COMPLETE" and r["completed_epochs"]==config["initial_epochs"]]
    if len(report["initial_trials"])!=len(config["initial_lrs"]) or not completed:
        report["status"]="INCOMPLETE_INITIAL_SEARCH";persist();return report
    winner=min(completed,key=lambda r:r["best_macro_action_nll"]);report["initial_winner"]=winner
    report["boundary_warning"]=winner["target_lr"] in (min(config["initial_lrs"]),max(config["initial_lrs"]))
    remaining=max(0.,target-time.monotonic()-90)
    # All three receive the same target update count, estimated from observed training
    # and validation throughput. Periodic state allows a safe budget-limited exit.
    cost=max(winner["wall_seconds"]/max(1,winner["updates"]),winner["mean_update_seconds"])*1.05
    per_updates=max(0,int(remaining/(3*cost)))
    report["refinement_update_budget"]=per_updates;persist()
    candidates=[winner]
    for i,factor in enumerate(config["refinement_factors"]):
        if not per_updates:break
        trial=Trial(table,attacks,config,output/f"refine-{i}",device,train,validation,winner["checkpoint"])
        result=trial.train(f"refine-{i}",winner["target_lr"]*factor,max_updates=per_updates,deadline=safety-90)
        report["refinement_trials"].append(result);persist();del trial
        if require_gpu:torch.cuda.empty_cache()
        if result["status"]=="COMPLETE":candidates.append(result)
        if time.monotonic()>safety-90:break
    # Incumbent remains eligible; incomplete or failed branches never win selection.
    best=min(candidates,key=lambda r:r["best_macro_action_nll"])
    shutil.copy2(best["checkpoint"],output/"selected.pt")
    report["selected"]=dict(best,checkpoint="selected.pt",sha256=hashlib.sha256((output/"selected.pt").read_bytes()).hexdigest())
    entries=[dict(name=f"shared-deck-{i}",role="shared-bc-deck-instance",deck=t["deck"],source_submission_id=t["submission_id"],source_team=t["name"],checkpoint="selected.pt",sha256=report["selected"]["sha256"],sampling_temperature=1.0) for i,t in enumerate(data["teachers"])]
    (output/"agents.json").write_text(json.dumps(dict(agents=entries,scope="One shared network, four decks; no specialist fine-tuning"),indent=2))
    shutil.copy2(Path(processed)/"manifest.json",output/"data-manifest.json")
    complete=len(report["refinement_trials"])==3 and all(r["status"]=="COMPLETE" for r in report["refinement_trials"])
    report["status"]="COMPLETE" if complete else "PARTIAL_REFINEMENT";persist();return report
