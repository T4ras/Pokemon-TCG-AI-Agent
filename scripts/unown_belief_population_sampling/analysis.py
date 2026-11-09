"""Shared whole-match bootstrap; primary final and secondary selection stay distinct."""
from .common import *
from .metrics import ALL_METRICS
import numpy as np

def mean_records(a,b):
 b={r['id']:r for r in b};return [dict(id=r['id'],opponent_list=r['opponent_list'],chooser=r['chooser'],metrics={k:(r['metrics'][k]+b[r['id']]['metrics'][k])/2 if r['metrics'][k] is not None and b[r['id']]['metrics'][k] is not None else None for k in ALL_METRICS}) for r in a]
def difference(a,b,point,draws,keys):
 i,j=keys.index(a),keys.index(b);delta=point[i]-point[j];boot=draws[:,i]-draws[:,j]
 return {k:dict(difference=float(delta[n]),CI95=np.quantile(boot[:,n],[.025,.975]).tolist()) for n,k in enumerate(ALL_METRICS)}
def run(check=lambda:None,replicates=10000):
 methods={'A':read(OUT/'baseline-validation.json')['records']};selection={};jobs=read(OUT/'jobs.json')
 for job in jobs:
  rows=[]
  for u in CHECKPOINTS:
   check();row=read(OUT/'validation'/f'{job["key"]}-u{u:05d}.json');assert row['BO3']==592 and row['updates']==u;methods[f'{job["key"]}-u{u:05d}']=row['records'];rows.append(row)
  selected=choose_secondary(rows);selection[job['key']]=dict(updates=selected['updates'],n100_combined_NLL=selected['summary']['populations']['full_field']['metrics']['objective_NLL'],initialization_eligible=True,primary_endpoint_unchanged=14004);methods[job['key']+'-secondary-selected']=selected['records']
 for arm in ('control','population'):
  for suffix in [f'u{u:05d}' for u in CHECKPOINTS]+['secondary-selected']:
   methods[f'{arm}-seedmean-{suffix}']=mean_records(methods[f'{arm}-s11-{suffix}'],methods[f'{arm}-s29-{suffix}'])
 keys=list(methods);indexed={k:{r['id']:r for r in rows} for k,rows in methods.items()};valid=entries('validation');assert len(valid)==592;shape=(len(keys),len(ALL_METRICS));B=replicates;rng=np.random.default_rng(SEED);point={p:np.zeros(shape) for p in ('full_field','eligible')};den={p:np.zeros(shape) for p in point};boot={p:np.zeros((B,*shape)) for p in point};strata=[]
 for deck,desc in sorted(prior().items()):
  for chooser in (0,1):
   check();ids=sorted(r['id'] for r in valid if r['opponent_list']==deck and r['chooser']==chooser);assert len(ids)==8
   values=np.array([[[indexed[m][i]['metrics'][k] if indexed[m][i]['metrics'][k] is not None else np.nan for k in ALL_METRICS] for m in keys] for i in ids]);complete=np.isfinite(values).all(0);avg=np.nan_to_num(values).mean(0)*complete;counts=rng.multinomial(8,np.full(8,1/8),size=B)/8;sample=(counts@np.nan_to_num(values).reshape(8,-1)).reshape(B,*shape)*complete
   for pop in point:
    weight=desc['entries']/(100 if pop=='full_field' else 82)/2 if pop=='full_field' or desc['group']=='passing' else 0
    point[pop]+=weight*avg;den[pop]+=weight*complete;boot[pop]+=weight*sample
   strata.append(dict(deck=deck,chooser=chooser,BO3=8))
 result=dict(at_utc=iso(),primary_updates=14004,validation_BO3=592,test_access=False,selection=selection,bootstrap_replicates=B,bootstrap_seed=SEED,interval_scope='Paired wholeBO3 within deck/chooser; conditional on these fitted models. Training-seed and secondary checkpoint-selection uncertainty not fully captured. Seedmean is meanmetrics, not an ensemble.',populations={},strata=strata)
 for pop in point:
  check();coverage=den[pop];safe=np.where(coverage>0,coverage,1);est=point[pop]/safe;draws=boot[pop]/safe;summary={}
  for i,key in enumerate(keys):summary[key]={k:dict(value=float(est[i,n]) if coverage[i,n] else None,CI95=np.quantile(draws[:,i,n],[.025,.975]).tolist() if coverage[i,n]>=1-1e-8 else None,coverage=float(coverage[i,n])) for n,k in enumerate(ALL_METRICS)}
  contrasts={}
  for seed in ('s11','s29','seedmean'):
   contrasts[f'population_minus_control_{seed}']=difference(f'population-{seed}-u14004',f'control-{seed}-u14004',est,draws,keys)
   for arm in ('control','population'):
    final=f'{arm}-{seed}-u14004';contrasts[f'{arm}_minus_initial_{seed}']=difference(final,f'{arm}-{seed}-u00000',est,draws,keys);contrasts[f'{arm}_minus_A_{seed}']=difference(final,'A',est,draws,keys)
   contrasts[f'secondary_population_minus_control_{seed}']=difference(f'population-{seed}-secondary-selected',f'control-{seed}-secondary-selected',est,draws,keys)
  result['populations'][pop]=dict(methods=summary,contrasts=contrasts)
 primary=result['populations']['full_field']['contrasts'];m=primary['population_minus_control_seedmean']['deck_log_loss'];each=[primary['population_minus_control_'+s]['deck_log_loss']['difference'] for s in ('s11','s29')];improved=m['CI95'][1]<0 and all(x<0 for x in each);conflict=min(each)<0<max(each)
 result['conclusion']=dict(primary_mean_NLL_difference=m,seed_NLL_differences=each,classification='Conflicting seed results; mechanism inconclusive.' if conflict else 'Treatment improves fixed-final deckNLL over matched control on these fitted seeds.' if improved else 'Fixed-final improvement over matched control is not established.',less_regression_not_replacement=True)
 for ref in ('initial','A'):
  c=primary[f'population_minus_{ref}_seedmean']['deck_log_loss'];result['conclusion']['treatment_vs_'+ref]=dict(**c,improvement_established=c['CI95'][1]<0 and all(primary[f'population_minus_{ref}_{s}']['deck_log_loss']['difference']<0 for s in ('s11','s29')))
 zones=['count_NLL']+['count_NLL_'+z for z in ('hand','draw_pile','prizes')]+[f'{prefix}_{z}' for prefix in ('presence','uncertain_presence') for z in ('hand','draw_pile','prizes')]
 result['conclusion']['hidden_zone_comparisons']={ref:{k:primary[f'population_minus_{ref}_seedmean'][k] for k in zones} for ref in ('control','initial','A')};result['conclusion']['scope']='Validation mechanism evidence only; fixed final remains primary. No deployment, test confirmation or automatic followup.'
 write(OUT/'analysis.json',result);plot(result);return result

def plot(a):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(2,2,figsize=(11,7));metrics=['deck_log_loss','count_NLL','uncertain_presence_hand','objective_NLL'];titles=['Deck NLL (primary at14,004)','Aggregate count NLL','Uncertain-hand presence Brier','Combined objective']
 for ax,k,title in zip(axes.flat,metrics,titles):
  ax.axhline(a['populations']['full_field']['methods']['A'][k]['value'],color='black',linestyle=':',label='A')
  for arm,color in [('control','C0'),('population','C1')]:
   for seed,style in [(11,'-'),(29,'--')]:
    values=[a['populations']['full_field']['methods'][f'{arm}-s{seed}-u{u:05d}'][k]['value'] for u in CHECKPOINTS];ax.plot(CHECKPOINTS,values,style,color=color,label=f'{arm}, seed{seed}')
  ax.set_title(title);ax.set_xlabel('Optimizer updates');ax.grid(alpha=.2)
 axes[0,0].legend(fontsize=8);fig.suptitle('Frozen sampling comparison: full-field n/100 validation');fig.tight_layout();fig.savefig(OUT/'learning-curves.png',dpi=150);plt.close(fig)
