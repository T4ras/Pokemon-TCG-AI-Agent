"""Registered, paired fresh-initialization OceanMix replacement experiment."""
import copy,datetime,hashlib,json,random,shutil,time
from pathlib import Path
import torch
from .lr_search import Trial,atomic_save,make_scaler,new_state
from .training import load_sequences
from .model import PolicyNetwork,metadata_tables
from ..engine import load_sdk

def model_digest(network):
    digest=hashlib.sha256()
    for key,value in sorted(network.state_dict().items()):digest.update(key.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()

def run(data_root,output,config,require_gpu=True):
    started=time.monotonic();output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if any(output.iterdir()):raise ValueError('Fresh output required; retain earlier attempts')
    torch.set_num_threads(2)
    if require_gpu and not torch.cuda.is_available():raise RuntimeError('Kaggle GPU required')
    device=torch.device('cuda' if require_gpu else 'cpu');random.seed(config['seed']);torch.manual_seed(config['seed'])
    api,_=load_sdk();table,attacks=metadata_tables(api)
    network=PolicyNetwork(table,attacks).to(device);optimizer=torch.optim.AdamW(network.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    initial_digest=model_digest(network);atomic_save(output/'initial.pt',network,optimizer,make_scaler(device),new_state(config['seed']))
    del network,optimizer
    report=dict(status='RUNNING',config=config,initial_model_sha256=initial_digest,initial_optimizer_empty=True,common_validation_teachers=config['common_teachers'],arms={},device=str(device),gpu=torch.cuda.get_device_name(0) if require_gpu else None,native_rng_seeded=False)
    def persist():
        report['wall_seconds']=time.monotonic()-started
        p=output/'report.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2));tmp.replace(p)
    persist();deadline=started+config['gpu_target_seconds']+config['finish_grace_seconds']
    for arm in config['arms']:
        processed=Path(data_root)/arm/'processed';manifest=json.loads((processed/'manifest.json').read_text())
        if manifest['failures']:raise ValueError('Unresolved replay validation failures')
        train=load_sequences(processed,'train');validation={t:load_sequences(processed,'validation',t) for t in range(4)}
        trial=Trial(table,attacks,config,output/arm,device,train,validation,output/'initial.pt')
        if model_digest(trial.network)!=initial_digest or trial.optimizer.state:raise ValueError('Initialization or optimizer differs between arms')
        if trial.updates_per_epoch!=config['updates_per_epoch']:raise ValueError('Fixed epoch update budget mismatch')
        result=trial.train(arm,config['learning_rate'],warmup_epochs=config['warmup_epochs'],finish_epoch=config['initial_epochs'],deadline=deadline)
        report['arms'][arm]=dict(result,initial_model_sha256=initial_digest,data_counts=manifest['counts'])
        shutil.copy2(processed/'manifest.json',output/arm/'data-manifest.json')
        if result['status']=='COMPLETE':
            if result['updates']!=config['updates_per_epoch']*config['initial_epochs']:raise ValueError('Unequal training budget')
            shutil.copy2(output/arm/'best.pt',output/arm/'selected.pt');sha=hashlib.sha256((output/arm/'selected.pt').read_bytes()).hexdigest()
            entries=[dict(name=f'{arm}-deck-{i}',role='shared-bc-deck-instance',source_submission_id=t['submission_id'],source_team=t['name'],deck=t['deck'],checkpoint='selected.pt',sha256=sha,sampling_temperature=1.0) for i,t in enumerate(manifest['teachers'])]
            (output/arm/'agents.json').write_text(json.dumps(dict(agents=entries,selection='Best unchanged-three-deck validation action NLL; no test/gameplay selection'),indent=2))
        persist();del trial,train,validation
        if require_gpu:torch.cuda.empty_cache()
        if result['status']!='COMPLETE':report['status']='INCOMPLETE';persist();return report
    report['status']='COMPLETE';persist();print(json.dumps(report),flush=True);return report
