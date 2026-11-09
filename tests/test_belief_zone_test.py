import numpy as np
import pytest
from unown_belief_zone_test.evaluate import zone_scores,METRICS,test_component as read_test_component

def fixture():
 p=np.zeros((2,2,3,3));p[...,0]=.7;p[...,1]=.3
 f=dict(A=p.copy(),known=np.zeros((2,2),int),games=np.array([1,2]));t=dict(counts=np.zeros((2,2,3),int));return f,t,p

def test_zone_presence_and_multiclass_definitions():
 f,t,p=fixture();t['counts'][:,:,1]=1
 m,c=zone_scores(f,t,p)
 assert m['presence_hand']==pytest.approx(.09)
 assert m['presence_draw_pile']==pytest.approx(.49)
 assert m['presence_prizes']==pytest.approx(.09)
 assert m['count_multiclass_draw_pile']==pytest.approx(.98)
 assert m['uncertain_presence_draw_pile']==pytest.approx(.49)
 assert c['uncertain_queries']==[4,4,4]

def test_known_hand_clamps_actual_presence():
 f,t,p=fixture();f['known'][:]=1;m,_=zone_scores(f,t,p)
 assert m['total_hand_presence']==0 and m['presence_hand']==pytest.approx(.09)

def test_uncertain_filter_uses_baseline_not_model():
 f,t,p=fixture();f['A'][...,0]=1;f['A'][...,1]=0
 m,_=zone_scores(f,t,p)
 assert m['uncertain_presence_hand'] is None
 assert m['presence_hand']==pytest.approx(.09)

def test_invalid_model_distribution_rejected():
 f,t,p=fixture();p[...,0]=2
 with pytest.raises(AssertionError):zone_scores(f,t,p)

def test_access_bridge_rejects_unregistered_ID():
 with pytest.raises(PermissionError):read_test_component(dict(id='not-registered'),'targets')

def test_equal_game_not_equal_decision_weights():
 f,t,p=fixture();f={k:np.concatenate([v[:1],v[:1],v[1:]],axis=0) for k,v in f.items()};t={k:np.concatenate([v[:1],v[:1],v[1:]],axis=0) for k,v in t.items()};p=np.concatenate([p[:1],p[:1],p[1:]],axis=0);t['counts'][2,:,0]=1
 m,_=zone_scores(f,t,p);assert m['presence_hand']==pytest.approx((.09+.49)/2)
