from unown_belief_full_training.common import *
import functools
RUN='astra-unown-belief-population-sampling-repair-001'
OUT=R/'experiments'/RUN
PRIVATE=R/'data/private'/RUN
REPORT=R/'reports/astra-unown-belief-population-sampling-repair.md'
CPU_CAP=14400
ELAPSED_CAP=21600
FIT_CAP=0
CURATED_CAP=2*1024**3
PRIVATE_ROOTS=(PRIVATE,)
CHECKPOINTS=(0,2334,4668,7002,9336,11670,14004)
UPDATES=14004
PREP_CHARGE=600
SEED=900003

def deadline():return dt.datetime.fromisoformat(read(OUT/'continuation-window.json')['deadline_utc'])
def guard():
 if (OUT/'STOP.json').exists():raise RuntimeError('Registered resource stop')
 if now()>=deadline():raise TimeoutError('Six-hour UTC experiment cap')
def source_check():
 for name,v in read(OUT/'source-manifest.json')['files'].items():
  if '/locked-test/' in name or 'hidden-zone-test' in name and name.startswith('data/'):raise PermissionError('Test payload in source manifest')
  if sha(R/name)!=v['sha256']:raise InvariantFailure('Frozen source/input changed: '+name)
@functools.lru_cache(maxsize=1)
def inputs():return {x['id']:x for x in read(OUT/'cache-manifest.json')}
def load(ident):
 if ident not in inputs() or inputs()[ident]['partition'] not in ('training','validation'):raise PermissionError('Only frozen train/validation caches authorized')
 from unown_belief_full_training.preprocess import load_features
 return load_features(ident)
def entries(partition):
 if partition not in ('training','validation'):raise PermissionError('Test excluded')
 return [x for x in read(OUT/'split-manifest.json')['entries'] if x['partition']==partition]
def coefficient(item):return 1. if item.get('unit_population_weight') else item['neural_loss_weight']
def choose_secondary(rows):
 # Exact ties go to earlier checkpoints; fixed-final endpoint is never changed.
 return min(rows,key=lambda r:(r['summary']['populations']['full_field']['metrics']['objective_NLL'],r['updates']))
