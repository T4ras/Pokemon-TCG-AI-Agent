"""Preparation-only bounded historical workload, no new games/candidate fits."""
from .common import *
from . import features
from .features import target_arrays
from .models import make_neural,neural_loss
import numpy as np,resource,sys

def main():
 resource.setrlimit(resource.RLIMIT_CPU,(600,605));start=time.process_time();items=read(OUT/'reuse-primary.json');chosen=min([x for x in items if x['origin']=='legacy-belief'],key=lambda x:stable(SEED,'offline-probe',x['id']));features.check_time=lambda:None
 hist=gzip_read(R/chosen['corpus']['legal_path']);targets=gzip_read(R/chosen['corpus']['truth_path']);f,info=features.encode(hist,chosen['corpus']['selection'],True);t=target_arrays(targets,f,info['points'],chosen['opponent_list']);npwrite(PRIVATE/'probe'/'features.npz',**f)
 import torch
 torch.set_num_threads(3);torch.set_num_interop_threads(1);timings={};sizes={}
 for family in ('MLP','GRU'):
  model=make_neural(family,11);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);tick=time.process_time();loss=neural_loss(model,f,t);loss.backward();optimizer.step();timings[family]=time.process_time()-tick;sizes[family]=sum(p.numel() for p in model.parameters());del model,optimizer
 # Faithful response association for imported screen histories, without simulation.
 screen=min([x for x in items if x['origin'].startswith('retained-screen')],key=lambda x:stable(SEED,'actor-prefix',x['id']));source=gzip_read(R/screen['corpus']['legal_path']);sys.path.insert(0,str(POLICY));from frozen_unown.runtime import load
 actor,model,_=load(POLICY);actor=type(actor)(model,prior()[anchor()]['cards'],sample=False,collect=False,seed=11)
 for e in source:
  if actor.act(e['observation'])!=e['action']:raise InvariantFailure('Imported observation/response association violates faithful actor-prefix parity')
 result=dict(at_utc=iso(),passed=True,historical_probe_id=chosen['id'],callbacks=len(hist),labelled_decisions=len(f['selected']),queried_cards=f['A'].shape[1],count_classes=f['A'].shape[-1],feature_CPU_seconds=info['CPU_seconds'],feature_bytes=(PRIVATE/'probe'/'features.npz').stat().st_size,fit_workload_CPU_seconds=timings,neural_parameters=sizes,model_workload='complete BO3 losses over all retained decisions, queries and count classes; full current-game GRU prefixes, forward/backward/AdamW on disposable copies',probe_models_used_in_experiment=False,screen_prefix_parity=dict(id=screen['id'],callbacks=len(source),passed=True),CPU_seconds=time.process_time()-start,new_native_matches=0,locked_test_accessed=False)
 write(OUT/'probe.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
