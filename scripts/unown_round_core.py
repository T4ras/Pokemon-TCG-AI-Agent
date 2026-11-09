"""Isolated diagnostic wiring; historical sources/checkpoints remain immutable."""
import unown_corrected_ppo_core as core
from unown_round_common import *
import collections,copy,gzip,random
import numpy as np,torch
batch=core.batch;runtime=core.runtime;legacy=core.legacy
for m in [core,batch,batch.scalar,legacy,runtime,runtime.rt]:m.OUT=OUT;m.PRIVATE=PRIVATE
load_model=core.load_model
NormalMemory=runtime.rt.MatchTracker
class ClearedMemory(NormalMemory):
 def reset_game(self,game):
  self.lower_bounds.clear();self.provenance.clear();super().reset_game(game)
def use_memory(clear):runtime.rt.MatchTracker=ClearedMemory if clear else NormalMemory
def outcome(r,g=None):
 x=r['result'] if g is None else r['games'][g]['result'];return float(x==r['learner_seat'])
def bootstrap(rows,keys,seed=SEED,rounds=(0,1)):
 rng=np.random.default_rng(seed);arrays={};boots={};cells={}
 for key in keys:
  selected=[r for r in rows if r.get('group',r.get('cell'))==key];samples=[];summary=[]
  for seat in [0,1]:
   group=[r for r in selected if r['learner_seat']==seat and r['status']=='DONE'];a=np.asarray([[outcome(r),*[outcome(r,g) for g in rounds]] for r in group]);assert len(a)
   bs=a[rng.integers(0,len(a),size=(20000,len(a)))].mean(1);samples.append(bs);summary.append(a.mean(0));arrays[key,seat]=a
  boots[key]=np.mean(samples,axis=0);cells[key]=np.mean(summary,axis=0).tolist()
 return cells,boots
def interval(x):return np.quantile(x,[.025,.975]).tolist()
def rounds_analysis():
 rows=[]
 for key in ['initial','final']:
  for line in (SOURCE/f'evaluation-{key}/matches.jsonl').read_text().splitlines():
   r=read_line(line);r['group']=key;rows.append(r)
 cells,boots=bootstrap(rows,['initial','final']);delta=np.array(cells['final'])-cells['initial'];b=boots['final']-boots['initial'];roles=[]
 for label in ['initial','final']:
  for game in [0,1,2]:
   for first in [False,True]:
    rs=[r for r in rows if r['group']==label and len(r['games'])>game and (r['games'][game]['first_player']==r['learner_seat'])==first]
    roles.append(dict(checkpoint=label,game=game+1,learner_first=first,n=len(rs),wins=sum(outcome(r,game) for r in rs),draws=sum(r['games'][game]['result']==2 for r in rs)))
 result=dict(exploratory=True,component_rates={k:dict(zip(['BO3','G1','G2'],v)) for k,v in cells.items()},G1_gain=float(delta[1]),G2_gain=float(delta[2]),interaction=float(delta[1]-delta[2]),CI95=interval(b[:,1]-b[:,2]),starting_role_cells=roles,attempts=len(rows),failures=sum(r['status']!='DONE' for r in rows),method=read(OUT/'registration.json')['bootstrap'],native_deals='Independent; later-game roles descriptive')
 write(OUT/'round-interaction.json',result);return result
def read_line(x):import json;return json.loads(x)
