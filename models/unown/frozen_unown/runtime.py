"""Inference-only assembly of the unchanged current-pool Unown components."""
from pathlib import Path
import collections, copy, functools, gzip, hashlib, importlib, inspect, json, math, os, random, sys, textwrap, types
from collections import Counter, defaultdict
from dataclasses import dataclass
import numpy as np
import torch

def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,default=str))

def load(root):
    root=Path(root).resolve();sys.path.insert(0,str(root))
    torch.set_num_threads(1);torch.manual_seed(904092)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    from ptcg.rl import model as model_module, encoder, buffers
    from ptcg.tracker import InfoTracker
    from frozen_unown.packets import Layout, Packet
    sd=torch.load(root/'weights.pt',map_location='cpu',weights_only=True)
    # Constructor metadata comes directly from frozen tensors. No SDK rebuild or
    # random extension can alter them; strict state loading follows construction.
    model_module.N_CARDS=sd['card_id_emb.weight'].shape[0]
    model_module.N_ATTACKS=sd['attack_id_emb.weight'].shape[0]
    model_module.N_SKILLS=sd['skill_id_emb.weight'].shape[0]
    model_module.build_tables=lambda:tuple(sd[k].numpy() for k in ('card_static','card_attacks','card_skills','att_static'))
    model_module.load_text_tables=lambda:tuple(sd[k].numpy() for k in ('card_text','att_text','skill_text'))
    net=model_module.PTCGTransformer(**model_module.arch_from_state_dict(sd,heads=8),critic='blind')
    net.load_state_dict(sd,strict=True);net.eval()
    base=dict(globals(),Layout=Layout,Packet=Packet,SEED=904092,OUT=root/'runtime-cache',PRIVATE=root/'runtime-cache',
        PREVIOUS=root/'identity',PREVIOUS_PRIVATE=root/'current',PARENT=root/'original',UP=root/'released',
        reference=lambda:(None,model_module,encoder,buffers,InfoTracker))
    def unit(name,**injected):
        path=root/'frozen_unown/units'/(name+'.py');m=types.ModuleType('frozen_unown.loaded_'+name)
        m.__dict__.update(base);m.__dict__.update(injected);m.__file__=str(path);sys.modules[m.__name__]=m
        exec(compile(path.read_text(),str(path),'exec'),m.__dict__);return m
    rt=unit('rt');runtime=unit('runtime',rt=rt)
    net.__class__=type('FrozenCurrentPoolUnown',(rt.DynamicMixin,type(net)),{})
    prefix=unit('prefix',runtime=runtime);prefix.install(net);runtime.freeze_inference(net)
    order=unit('order');context=dict(candidate_model=net,retain_history=False,actual_starter=None)
    actor_type=order.actor_class(runtime,context)
    deck=[int(x) for x in (root/'deck.csv').read_text().split()]
    actor=actor_type(net,deck,sample=False,seed=904092,collect=False)
    for parameter in net.parameters():parameter.requires_grad_(False)
    return actor,net,rt
