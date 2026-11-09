"""Frozen final embedding-adapted Unown, original Hydrapple, conditional argmax."""
import os,sys
from pathlib import Path

def _root():
    if '__file__' in globals():
        p=Path(__file__).resolve().parent
        if (p/'frozen-candidate.json').exists():return p
    for entry in [os.getcwd(),'/kaggle_simulations/agent']+list(sys.path):
        p=Path(entry)
        if (p/'frozen-candidate.json').exists():return p
    raise RuntimeError('Frozen candidate package directory unavailable')

ROOT=_root();sys.path.insert(0,str(ROOT))
from frozen_unown.runtime import load
_actor,_network,_encoder=load(ROOT)

def agent(observation,configuration=None):
    return _actor.act(observation)
