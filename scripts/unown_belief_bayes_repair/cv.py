"""Nested balanced CV; outer labels never enter parameter selection."""
from .common import *
from .deck import catalogue,fit_theta,deck_weights,GRID
import numpy as np,gzip,time

def mean_by_game(points,values):
 groups=collections.defaultdict(list)
 for p,v in zip(points,values):groups[p['game']].append(float(v))
 return sum(sum(v)/len(v) for v in groups.values())/len(groups)

def load_features(records):return {r['id']:json.load(gzip.open(R/r['feature_path'],'rt')) for r in records}
def fit_on(ids,features,labels,deck_counts,alpha):
 return fit_theta([features[i]['final_ledger'] for i in ids],[labels[i] for i in ids],deck_counts,alpha)

def validation_loss(ids,features,labels,theta,beta,prevalence,decks):
 p=prior();nums=collections.defaultdict(list)
 for ident in ids:
  y=labels[ident];ps=features[ident]['points'];loss=[]
  for point in ps:
   mask=np.zeros(len(decks),dtype=bool);mask[point['compatible']]=True;w=deck_weights(point['lower'],mask,prevalence,theta,beta);loss.append(-np.log(max(w[y],np.finfo(float).tiny)))
  nums[y].append(mean_by_game(ps,loss))
 return sum(p[decks[y]]['prior']*sum(v)/len(v) for y,v in nums.items())

def select_parameters(outer_fold,features,labels,folds,check=lambda:None):
 decks,cards,counts,prevalence=catalogue();training=sorted(i for f,ids in enumerate(folds) if f!=outer_fold for i in ids);held=set(folds[outer_fold]);scores=[];start=time.process_time()
 for alpha,beta in GRID:
  losses=[]
  for j,valid in enumerate(folds):
   if j==outer_fold:continue
   check();inner_train=sorted(set(training)-set(valid));assert len(inner_train)==148 and not set(inner_train)&held and not set(valid)&held
   theta=fit_on(inner_train,features,labels,counts,alpha);losses.append(validation_loss(valid,features,labels,theta,beta,prevalence,decks))
  scores.append(dict(alpha=alpha,beta=beta,inner_log_loss=float(np.mean(losses)),inner_losses=losses))
 best=scores[0]
 for row in scores[1:]:
  if row['inner_log_loss']<best['inner_log_loss']-1e-12:best=row
 check();theta=fit_on(training,features,labels,counts,best['alpha']);model_path=PRIVATE/'models'/f'outer-{outer_fold}.npz';model_path.parent.mkdir(parents=True,exist_ok=True)
 with model_path.open('wb') as f:np.savez_compressed(f,theta=theta)
 result=dict(outer_fold=outer_fold,alpha=best['alpha'],beta=best['beta'],training_ids=training,excluded_ids=folds[outer_fold],grid=scores,model_path=str(model_path.relative_to(R)),model_sha256=sha(model_path),CPU_seconds=time.process_time()-start,inner_population='full-field n/100; one mean per BO3; balanced chooser',outer_labels_used=False);write(OUT/'model-selection'/f'outer-{outer_fold}.json',result);return theta,result
