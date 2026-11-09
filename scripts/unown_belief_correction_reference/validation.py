"""Bounded exact-enumeration and saved-state checks; no native games."""
from .common import *
from .deadline_hooks import install
import unown_belief_full_training.baseline as baseline

def rare_path():
 groups=[((('D','H','P'),()),collections.Counter({i:1})) for i in range(6)]
 capacity=dict(D=35,H=5,P=6,L=0);calls=[];old=baseline.check_time
 try:
  baseline.check_time=lambda:calls.append(1)
  expected=baseline.assignments(groups,capacity)
 finally:baseline.check_time=old
 assert len(calls)>=2 and len(expected)>256
 install();actual=baseline.assignments(groups,capacity)
 assert actual==expected
 return dict(layouts=len(actual),deadline_checks_exercised=len(calls),exact_layout_weight_parity=True,native_matches=0)
