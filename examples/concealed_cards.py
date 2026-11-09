"""Two fictional deck hypotheses using the actual retained Bayesian classifier."""
import json,sys
import numpy as np
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from unown_belief_bayes_repair.deck import fit_theta,deck_weights
theta=fit_theta([[3,1],[2,2],[5,0],[4,1]],[0,0,1,1],np.array([[30,30],[45,15]],dtype=float),alpha=30)
w=deck_weights([3,0],[True,True],[.5,.5],theta,beta=.5)
print(json.dumps({"synthetic":True,"deck_hypothesis_probabilities":w.tolist(),"note":"Fictional inputs; not a competition result or trained checkpoint."},indent=2))
