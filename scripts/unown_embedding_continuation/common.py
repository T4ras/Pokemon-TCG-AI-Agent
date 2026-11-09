"""Frozen scope and ordinary file/clock helpers; no model API dependencies."""
from pathlib import Path
import collections,copy,datetime as dt,hashlib,json,math,os,random,time
from unown_runner.context import now,iso,parse,sha,read,write,append,exclusive,records,FIXED_RECIPE,RunContext,Execution,implementation_hashes
ROOT=Path(__file__).resolve().parents[2]
RUN='astra-unown-new-card-embedding-continuation-001';OUT=ROOT/'experiments'/RUN;PRIVATE=ROOT/'data/private'/RUN
REPORT=ROOT/'reports/astra-unown-new-card-embedding-continuation.md'
INITIAL_TENSOR='72b4ad7e65e154aa5736ec9facec5d43ffb8f39dc8c7041dd6cebe23fcc9ef78'
INITIAL_FILE='718d9c2346f17e4e324ede41b2be360c99f9b58ccbfc7e6b3b5833c166037193'
SEED=904103;TARGET=500000;MAX_COHORTS=255;SESSION_CAP=5063;PROVIDER_CAP=4943;PROCESS_CAP=4823;ABSOLUTE_CAP=5003;EXPORT_RESERVE=300
KERNEL='portfolio-owner/ptcg-unown-embedding-resume';DATASET='portfolio-owner/ptcg-unown-embedding-resume-inputs'
IMAGE='gcr.io/kaggle-private-byod/python@sha256:9a6bcfda6d03b2cd9aa347ee8c1bae62043328ed0f4afe9bda014e4f8bbf9435'
if os.name!='nt':
 import numpy as np
 import torch

def context(root=ROOT,mode='learning'):
 private=root/'data/private'/RUN;assets={k:str(root/v) for k,v in read(private/'asset-paths.json').items()}
 return RunContext(root,RUN,assets,Execution('bucket64'),SEED,dict(FIXED_RECIPE,lr=1e-4),
  dict(max_cohorts=MAX_COHORTS,native_attempts=MAX_COHORTS*16,session_seconds=SESSION_CAP,export_reserve_seconds=EXPORT_RESERVE,storage_bytes=3*1024**3,log_bytes=256*1024**2),mode)

def progress(stage,**fields):
 value=dict(fields,at_utc=iso(),stage=stage,pid=os.getpid());write(OUT/'progress.json',value);print(json.dumps(value),flush=True)

def embedding_lr(fresh):
 fraction=min(1,max(0,fresh/TARGET));return 1e-5+.5*(1e-4-1e-5)*(1+math.cos(math.pi*fraction))
