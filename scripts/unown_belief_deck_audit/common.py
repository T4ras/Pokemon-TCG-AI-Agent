from unown_belief_full_training.common import *
RUN='astra-unown-belief-deck-head-regression-audit-001'
OUT=R/'experiments'/RUN
PRIVATE=R/'data/private'/RUN
AUDIT_REPORT=R/'reports/astra-unown-belief-deck-head-regression-audit.md'
CPU_CAP=14400
ELAPSED_CAP=21600
FIT_CAP=3600
CURATED_CAP=2*1024**3
PRIVATE_ROOTS=(PRIVATE,)
SEED=904098

def deadline():return dt.datetime.fromisoformat(read(OUT/'lifecycle.json')['deadline_utc'])
def source_check():
 for name,v in read(OUT/'source-manifest.json')['files'].items():
  if sha(R/name)!=v['sha256']:raise InvariantFailure('Frozen audit input changed: '+name)
def guard():
 if (OUT/'STOP.json').exists():raise RuntimeError('Audit stop: '+read(OUT/'STOP.json')['reason'])
 if now()>=deadline():raise TimeoutError('Registered six-hour audit deadline')

FULL=R/'experiments/astra-unown-belief-classifier-full-training-001'
SUBSET=R/'experiments/astra-unown-belief-learning-dataset-v1-source-repair-005'
