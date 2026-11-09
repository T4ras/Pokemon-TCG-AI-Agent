"""Readable full-population benchmark report; raw evidence stays in manifests."""
from .common import *
import shutil

def f(x,n=5):return '—' if x is None else f'{x:.{n}f}'
def link(n):return f'[{n}](../experiments/{RUN}/{n})'
def delta(c):return '—' if not c else f'{c["difference"]:+.6f} [{c["CI95"][0]:+.6f}, {c["CI95"][1]:+.6f}]'
def params(v):
 if 'trainable_parameters' in v:return str(v['trainable_parameters'])+' trainable'
 if 'theta_values' in v:return str(v['theta_values'])+' theta values'
 return str(v.get('trees',0))+' trees; '+str(v.get('fitted_numeric_parameters',0))+' numeric values/thresholds'

def total(k):return 2334 if '-nfull-' in k else 592 if '-n16-' in k else 296 if '-n8-' in k else 0

def render(state,reason,mirror=None):
 audit=read(OUT/'manifest-audit.json');reg=read(OUT/'registration.json');completion=read(OUT/'completion.json') if (OUT/'completion.json').exists() else {};fits=[read(p) for p in (OUT/'fit-results').glob('*.json')];a=read(OUT/'analysis.json') if (OUT/'analysis.json').exists() else None
 text=f'# Belief classifiers trained on all available histories\n\nState: **{state}**. Updated {iso()}.\n\n{reason}\n\n'
 text+=f'This separate experiment uses **all2,334 valid training BO3**, the unchanged592 validation BO3 and zero locked-test evaluations. New fits complete: **{sum(r["state"]=="COMPLETE" for r in fits)}/7**. No new native matches, replacements, actor changes, GPU, paid model APIs, uploads, PPO or submission. Historical experiments and four missing histories remain unchanged.\n\n'
 text+='## Frozen population and objective\n\nThe committed archive has2,334 training,592 validation and592 locked-test records. Whole match IDs and non-test legal histories are unique and partitions are disjoint. All2,926 non-test feature/target/metadata caches were checked by hash. No additional training features were needed. The locked-test payload is never parsed by this benchmark; integrity checks hash opaque committed files only. There is no downsampling.\n\n'
 text+=f'Training composition: {audit["origin_counts"]}. Deck totals range {audit["deck_count_min"]}–{audit["deck_count_max"]}; deck/chooser cells range {audit["stratum_count_min"]}–{audit["stratum_count_max"]}. The full point has unequal counts, unlike the balanced296/592 subsets. '+link('full-training-manifest.json')+' freezes every training ID, origin and weight.\n\n'
 text+='The intended objective is sum over decks of n/100 prevalence × one-half per chooser × mean BO3 loss in that stratum, followed by the unchanged game/decision/query/zone averages. Neural epochs uniformly permute all2,334 records once, with per-record loss factor2,334×p(deck)/(2×stratum count); weights average1. There is no weighted sampling. Trees retain32 stages with4 records per deck/chooser and the existing37×p(deck) batch loss weight; no second inverse-count correction is applied. Bayesian fitting uses one final reveal ledger per BO3, equal chooser influence within each class via n(deck)/(2×stratum count), and the unchanged prevalence prior; alpha smoothing retains the actual per-deck effective sample count.\n\n'
 text+='| Deck / family | Chooser0 | Chooser1 | Total |\n|---|---:|---:|---:|\n'
 counts=read(OUT/'stratum-counts.json')
 for d in sorted(prior()):
  ns=[next(r['BO3'] for r in counts if r['deck']==d and r['chooser']==s) for s in (0,1)];text+=f'| {d[:10]} / {prior()[d]["archetype"]} | {ns[0]} | {ns[1]} | {sum(ns)} |\n'
 text+='\n## Unchanged methods and coverage\n\nArchitectures, residual logits around A, legal masks, count support0–17, legal observer, exact A marginals, query universe and full-prefix GRU resets match005. Trees and MLP receive the same information; the GRU retains every required callback, including forced observations. No deck-head repair, count-total projection, history change or new representation was introduced. Neural AdamW0.001/weight decay0.0001, clip1, six maximum epochs, patience3 and minimum validation-NLL improvement1e-5 are unchanged. Both seeds11/29 are retained. Bayesian alpha30/120 and beta0.25/0.5/1 selection still uses validation deck log loss, ties to lower beta then higher alpha.\n\n'
 for seed in (11,29):
  t=read(OUT/f'tree-schedule-s{seed}.json');text+=f'Trees seed{seed}:32 stages, {t["unique_BO3"]} unique training BO3, {t["draws"]} total exposures, per-history exposure range {t["minimum"]}–{t["maximum"]}. The original coverage rule holds; no schedule amendment. '+link(f'tree-schedule-s{seed}.json')+'.\n\n'
 text+='More data at the same epoch count permits more optimizer updates. This is a population-and-compute scaling comparison, not a controlled equal-update comparison. Recovery checkpoints cannot substitute for incomplete fits.\n\n'
 if a:
  text+='## Validation learning curve\n\nThe592 validation BO3 and original A-defined uncertain-query mask are unchanged. These are model-selection results, not final test evidence. Nominal95% intervals use20,000 paired whole-BO3 resamples within deck/chooser strata, identical across all methods/sizes/seeds. Seeds and queries are not independent matches; intervals condition on fitted models and omit retraining uncertainty. Seed means average metrics inside each BO3 and are not evaluated ensembles.\n\n![Learning curves](../experiments/'+RUN+'/learning-curves.png)\n\n'
  for pop,label in [('eligible','Primary30-list n/82'),('full_field','Secondary37-list n/100')]:
   point=a['point'][pop];text+=f'### {label}\n\n| Model / training BO3 | Uncertain Brier | Minus A [95%] | Deck accuracy | Deck NLL | Count NLL | Combined NLL |\n|---|---:|---|---:|---:|---:|---:|\n'
   for k,v in point.items():text+='| '+k+' / '+str(total(k))+' | '+f(v.get('uncertain_brier'))+' | '+delta(a['minus_A'][pop].get(k,{}).get('uncertain_brier'))+' | '+f(v.get('deck_accuracy'))+' | '+f(v.get('deck_log_loss'))+' | '+f(v.get('count_NLL'))+' | '+f(v.get('objective_NLL'))+' |\n'
   text+='\n| Full model | Brier minus same family/seed at592 [95%] | Deck NLL change [95%] | Combined NLL change [95%] |\n|---|---|---|---|\n'
   for k,v in a['minus_592'][pop].items():text+='| '+k+' | '+delta(v.get('uncertain_brier'))+' | '+delta(v.get('deck_log_loss'))+' | '+delta(v.get('objective_NLL'))+' |\n'
   text+='\n| Full model | All-query Brier | ECE | H/D/P count MAE | Deck multiclass Brier | Impossible mass | Coverage |\n|---|---:|---:|---|---:|---:|---:|\n'
   for k,v in point.items():
    if '-nfull-' not in k:continue
    text+='| '+k+' | '+f(v.get('all_brier'))+' | '+f(v.get('ECE'))+' | '+' / '.join(f(v.get(m)) for m in ['hand_MAE','deck_MAE','prize_MAE'])+' | '+f(v.get('deck_Brier'))+' | '+f(v.get('support_mass_violation'),8)+' | '+f(a['coverage'][pop][k].get('uncertain_brier'),3)+' |\n'
   text+='\n| Full model | H/D/P expected-total MAE | Card-total violation MAE / fraction |\n|---|---|---|\n'
   for k,v in point.items():
    if '-nfull-' in k:text+='| '+k+' | '+' / '.join(f(v.get(m)) for m in ['hand_total_MAE','deck_total_MAE','prize_total_MAE'])+' | '+f(v.get('card_total_violation_MAE'))+' / '+f(v.get('card_total_violation_fraction'))+' |\n'
   text+='\n'
  text+='## Interpretation and candidate\n\n'
  point=a['point']['eligible'];available=[]
  for family in ('Bayes','Trees','MLP','GRU'):
   k='Bayes-nfull-s0' if family=='Bayes' else family+'-nfull-seedmean';c=a['minus_592']['eligible'].get(k,{}).get('uncertain_brier')
   if not c:text+=f'{family}: full-data comparison incomplete; no negative scientific conclusion.\n\n';continue
   meaning='a validation improvement supported by the paired interval' if c['CI95'][1]<0 else 'a validation regression supported by the paired interval' if c['CI95'][0]>0 else 'an uncertain change at this resolution';text+=f'{family}: {meaning}; Brier change versus592 is {delta(c)}.\n\n';available.append((point[k]['uncertain_brier'],family))
  if available:
   available.sort();text+=f'{available[0][1]} has the smallest full-data family-mean point Brier. This ranking alone does not establish statistically significant superiority over the other families.\n\n'
  for family in ('MLP','GRU'):
   k=family+'-nfull-seedmean';v=point.get(k);old=point.get(family+'-n16-seedmean')
   if v and old:text+=f'{family} deck/count/combined NLL: full {f(v["deck_log_loss"])}/{f(v["count_NLL"])}/{f(v["objective_NLL"])};592 {f(old["deck_log_loss"])}/{f(old["count_NLL"])}/{f(old["objective_NLL"])};A {f(point["A"]["deck_log_loss"])}/{f(point["A"]["count_NLL"])}/{f(point["A"]["objective_NLL"])}. Deck accuracy full {f(v["deck_accuracy"])} versus592 {f(old["deck_accuracy"])} and A {f(point["A"]["deck_accuracy"])}. A lower hand Brier does not conceal a worse deck head or inconsistent count totals.\n\n'
  promising=[]
  for family in ('Bayes','Trees','MLP','GRU'):
   for n in ('16','full'):
    k=f'Bayes-n{n}-s0' if family=='Bayes' else f'{family}-n{n}-seedmean';c=a['minus_A']['eligible'].get(k,{}).get('uncertain_brier');v=point.get(k,{})
    if c and c['CI95'][1]<0 and a['coverage']['eligible'][k].get('uncertain_brier',0)>=1-1e-8:promising.append((v['uncertain_brier'],family,n))
  if state!='COMPLETE':text+='The seven-fit task is not complete, so no final candidate is selected. Incomplete fits and recovered checkpoints are excluded from scientific comparisons.\n\n'
  elif promising:
   promising.sort();_,family,n=promising[0];candidate=f'Bayes-n{n}-s0' if family=='Bayes' else f'{family}-n{n}-s11';text+=f'The registered selection rule nominates **{candidate}** for a separately authorized locked-test confirmation of belief prediction. The family/size uses the smallest qualifying validation point Brier; seed11 was fixed in advance, not chosen as the better seed. Tradeoffs above remain relevant. No test evaluation or deployment follows automatically.\n\n'
  else:text+='No candidate qualifies under the registered validation rule. No automatic test, collection, PPO or follow-up.\n\n'
  text+='## Per-deck and game/stage results\n\n| Deck / family | Bayes | Trees seed mean | MLP seed mean | GRU seed mean |\n|---|---:|---:|---:|---:|\n';chosen=['Bayes-nfull-s0']+[x+'-nfull-seedmean' for x in ('Trees','MLP','GRU')]
  for row in a['per_deck']:text+='| '+row['deck'][:10]+' / '+row['archetype']+' | '+' | '.join(f(row['methods'].get(k,{}).get('uncertain_brier')) for k in chosen)+' |\n'
  text+='\n| Game/stage; primary population | Bayes | Trees | MLP | GRU |\n|---|---|---|---|---|\n'
  for g,v in a['breakdowns'].items():text+='| '+g+' | '+' | '.join(f(v['eligible'].get(k,{}).get('point',{}).get('uncertain_brier'))+' / coverage '+f(v['eligible'].get(k,{}).get('coverage',{}).get('uncertain_brier'),2) for k in chosen)+' |\n'
  text+='\nLater-game/stage participation is outcome-dependent. Full seed-specific per-deck and game/stage metrics remain in '+link('analysis.json')+'.\n\n'
 text+='## Compute, artifacts and online cost\n\nCaps:20 additional CPU-core hours,24 UTC elapsed hours from launch,10 fitting core hours, four compute threads,8GiB additional private artifacts. Individual fit caps:Bayes10min;trees45min each;MLP90min each;GRU120min each. Private raw caches are referenced in place. New common online features are measured once on the same16 frozen validation histories; incremental model timing processes all legal callbacks and genuine readouts. There is zero training/timing overlap and timing is not predictive evaluation.\n\n'
 text+='| Fit | State | CPU / elapsed min | Epochs/stages / selected epoch | Unique BO3 / updates | Cache-load CPU min | Parameters |\n|---|---|---|---|---|---:|---|\n'
 charges={r['key']:r for r in rows(OUT/'fit-cost-ledger.jsonl')}
 for r in sorted(fits,key=lambda x:x['key']):text+='| ['+r['key']+'](../experiments/'+RUN+'/fit-results/'+r['key']+'.json) | '+r['state']+' | '+f(charges.get(r['key'],r)['CPU_seconds']/60,2)+' / '+f(charges.get(r['key'],r).get('elapsed_seconds',0)/60,2)+' | '+str(r['epochs_or_stages'])+' / '+str(r.get('best_epoch'))+' | '+str(r.get('unique_training_BO3_seen'))+' / '+str(r.get('optimizer_updates'))+' | '+f(r.get('cache_loading',{}).get('CPU_seconds',0)/60,2)+' | '+params(r.get('parameter_details',{}))+' |\n'
 text+='\nActual tree exposure counts, parameter definitions, recovery artifacts and full SHA256s for each model are in '+link('fit-results')+' and the per-fit records. Trees report split thresholds and leaf-value scalars rather than neural trainable weights. Bayesian theta values and neural trainable parameter counts are explicit.\n\n'
 text+='| Full model | Observer/A CPU s per BO3 | Incremental model CPU s | Complete online CPU s | Validation/export CPU min | Peak process MiB |\n|---|---:|---:|---:|---:|---:|\n'
 for p in sorted((OUT/'validation-results').glob('*nfull*.json')):
  r=read(p);t=r['full_online_timing'];increment=t['incremental_model_CPU_seconds_per_BO3'];text+='| '+r['key']+' | '+f(t['complete_recurring_CPU_seconds_per_BO3']-increment,3)+' | '+f(increment,3)+' | '+f(t['complete_recurring_CPU_seconds_per_BO3'],3)+' | '+f(r['CPU_seconds']/60,2)+' | '+f(r['peak_RSS_kib']/1024,1)+' |\n'
 if completion:text+=f'\nFinal controller-accounted CPU: {completion.get("final_CPU_seconds",completion["CPU_seconds"])/3600:.3f} core hours; measured execution {completion["measured_execution_CPU_seconds"]/3600:.3f}; measured preparation {completion["measured_preparation_CPU_seconds"]/3600:.3f}. The preparation reserve is a conservative budget charge, not invented actual compute. Fit subprocess CPU including startup/export: {completion["fit_CPU_seconds"]/3600:.3f} hours. Elapsed {completion.get("final_elapsed_seconds",completion["elapsed_seconds"])/3600:.3f} hours. Added private artifacts {completion["additional_private_bytes"]/1024**3:.3f}GiB. Native/GPU/paidAPI/test scores all0.\n'
 text+='\nEvidence: '+', '.join(link(n) for n in ['registration.json','manifest-audit.json','full-training-manifest.json','stratum-counts.json','model-config.json','historical-results-manifest.json','preprocessing-cost.json','baseline-current-timing.json','source-manifest.json','analysis.json','completion.json'])+'. Historical reports and results were not rewritten.\n'
 FULL_REPORT.write_text(text)
 status=R/'docs/STATUS.md';old=status.read_text(encoding='utf-8-sig');begin='<!-- BELIEF-FULL-TRAINING -->';end='<!-- BELIEF-FULL-TRAINING-END -->'
 if begin in old:i=old.index(begin);j=old.index(end,i)+len(end);old=old[:i]+old[j:].lstrip('\n')
 status.write_text(begin+f'\n## Full-data belief classifier benchmark\n\n{iso()}: **{state}**. {reason} All2334 valid training histories retained;7 planned fits; locked-test scores0. [Report](../reports/astra-unown-belief-classifier-full-training.md).\n'+end+'\n\n'+old)
 mirror=mirror or (Path('C:/path/to/ptcg-agent') if os.name=='nt' else Path('/path/to/ptcg-agent'))
 for source in [FULL_REPORT,status]+[p for p in OUT.iterdir() if p.is_file() and p.suffix in ('.json','.jsonl','.md','.png')]:
  target=mirror/source.relative_to(R);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
 for folder in ('fit-results','validation-results'):
  for source in (OUT/folder).glob('*.json'):
   target=mirror/source.relative_to(R);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
