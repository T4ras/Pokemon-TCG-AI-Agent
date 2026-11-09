"""Synthetic subset of tests/test_unown_belief_bayes_repair.py; no private fixtures or native matches."""
import itertools,collections,copy,json

import numpy as np

import pytest

from unown_belief_bayes_repair.common import *

from unown_belief_bayes_repair.baseline import Facts,baseline_one

from unown_belief_bayes_repair.allocation import allocation_one

from unown_belief_bayes_repair.deck import fit_theta,deck_weights,mix_allocations,catalogue

def test_theta_matches_registered_formula_and_no_labelled_impossible_card():
 decks=np.array([[30,30],[60,0]],dtype=float);x=np.array([[3,1],[0,2],[4,0]],dtype=float);y=[0,0,1];theta=fit_theta(x,y,decks,30);assert theta[0]==pytest.approx([(3+15)/36,(3+15)/36]);assert theta[1]==pytest.approx([1,0])
 with pytest.raises(InvariantFailure):fit_theta([[0,1]],[1],decks,30)

def test_weight_beta_zero_and_stable_large_ledger():
 theta=np.array([[.9,.1],[.1,.9]]);prior=np.array([.3,.7]);assert deck_weights([0,0],[True,True],prior,theta,1)==pytest.approx(prior)
 assert deck_weights([50,0],[True,True],prior,theta,0)==pytest.approx(prior)
 w=deck_weights([50,0],[True,True],prior,theta,.25);assert np.isfinite(w).all() and sum(w)==pytest.approx(1) and w[0]>.999
 assert deck_weights([1,0],[False,True],prior,theta,1)==pytest.approx([0,1])

def test_total_variance_includes_both_terms():
 a=np.zeros((2,1,7));a[0,0,1]=0;a[1,0,1]=2;a[:,:,4]=.5;v=mix_allocations([.5,.5],a);assert v[0,1]==1 and v[0,4]==pytest.approx(1.5)
