from unown_belief_full_training.common import *
RUN='astra-unown-belief-correction-ppo-pilot-001'
OUT=R/'experiments'/RUN
PRIVATE=R/'data/private'/RUN
REPORT=R/'reports/astra-unown-belief-correction-ppo-pilot.md'
BELIEF_DIM=1479
POLICY_TENSOR='5b43fff134b12aef91361b18d504a04e9baa408c36d5fea89caf493566905cb8'
SEED=904096

def guard():
 if (OUT/'STOP.json').exists():raise RuntimeError(read(OUT/'STOP.json')['reason'])
 if now()>=dt.datetime.fromisoformat(read(OUT/'lifecycle.json')['deadline_utc']):raise TimeoutError('36hourworkflowcap')
def tensor_digest(state):
 h=hashlib.sha256()
 for k,v in sorted(state.items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
