"""Local recovery invariant checks; never launches native/GPU work."""
from .common import *
from .train import build_cohorts
import torch

def run():
 prior=ROOT/'experiments/astra-unown-belief-correction-ppo-execution-003';assert read(prior/'state-training-completion.json')['candidate_eligible']
 assert sha(PRIVATE/'state-final.pt')==read(OUT/'registration.json')['state_checkpoint_sha256']
 saved=torch.load(PRIVATE/'initial-correction.pt',map_location='cpu',weights_only=False);assert saved['step']==0 and not saved['optimizer']['state'];assert saved['extra']['runner_state']=='BOUNDARY' and saved['extra']['cohort_reference'] is None
 assert sha(OUT/'training-schedule.json')==sha(prior/'training-schedule.json') and sha(OUT/'evaluation-schedule.json')==sha(prior/'evaluation-schedule.json')
 jobs=build_cohorts('belief',read(OUT/'training-schedule.json')['cohorts']);assert len(jobs)==127 and len(jobs)*16+16==2048
 assert read(OUT/'exact-cache-parity-timing.json')['passed'] and read(OUT/'recovery-forecast.json')['fits'];assert SESSION+361==19800 and SESSION+9741<=32400
 progress('RECOVERY_PREPARED_PENDING_APPROVAL',fresh_decisions=0,prior_attempts=16,new_attempt_cap=2032,state_frozen=True)
if __name__=='__main__':run()
