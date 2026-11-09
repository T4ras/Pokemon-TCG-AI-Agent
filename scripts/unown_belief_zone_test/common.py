from unown_belief_full_training.common import *
RUN='astra-unown-belief-hidden-zone-test-001'
OUT=R/'experiments'/RUN
PRIVATE=R/'data/private'/RUN
REPORT=R/'reports/astra-unown-belief-hidden-zone-brier-test.md'
CPU_CAP=14400
ELAPSED_CAP=10800
FIT_CAP=0
PRIVATE_ROOTS=(PRIVATE,)
CURATED_CAP=2*1024**3
SEED=900002

def deadline():return dt.datetime.fromisoformat(read(OUT/'continuation-window.json')['deadline_utc'])
def guard():
 if (OUT/'STOP.json').exists():raise RuntimeError('Registered stop')
 if now()>=deadline():raise TimeoutError('Three-hour UTC test-evaluation cap')
def source_check():
 for name,v in read(OUT/'source-manifest.json')['files'].items():
  if sha(R/name)!=v['sha256']:raise InvariantFailure('Frozen input/source changed: '+name)
