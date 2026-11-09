from .common import *
import shutil

def fmt(x,n=6):return 'unavailable' if x is None else f'{x:.{n}f}'
def cell(x):
 ci=x.get('CI95');return fmt(x.get('value'))+(f' [{fmt(ci[0])}, {fmt(ci[1])}]' if ci else '')
def diff(x):return f'{x["difference"]:+.6f} [{x["CI95"][0]:+.6f}, {x["CI95"][1]:+.6f}]'
def render(state,reason,mirror=None):
 a=read(OUT/'analysis.json') if (OUT/'analysis.json').exists() else None;c=read(OUT/'completion.json') if (OUT/'completion.json').exists() else None;reg=read(OUT/'registration.json');forecast=read(OUT/'forecast.json');fits=[read(p) for p in sorted((OUT/'fit-results').glob('*.json'))]
 text=f'# Population-based sampling and neural deck-head regression\n\nState: **{state}**. Updated {iso()}.\n\n{reason}\n\n'
 text+='This registered comparison changes only how population prevalence enters one-BO3 updates. It uses all 2,334 training histories as the eligible pool and the same 592 validation histories. The 592 previously opened test histories and their predictions are excluded; they are no longer an untouched test. No native matches, GPU, PPO, paid model APIs, uploads, submissions, tuning or automatic follow-up. Historical evidence is preserved.\n\n'
 text+='## Frozen comparison\n\nBoth seeds (11 and 29) use the existing 57,691-parameter zero-residual MLP. Within each seed, control and treatment load exactly the same initial model, fresh AdamW state and Torch RNG state. Initialization seeds and sampling seeds are recorded separately. AdamW remains lr 0.001, weight decay 0.0001, betas (0.9,0.999), epsilon 1e-8; global gradient clipping stays 1. One complete BO3 supplies each update. Features, masks and the original loss are unchanged: deck CE plus the mean of hand/draw-pile/prize count CE, with query mean, decision mean within game and equal game weight within a match.\n\n'
 text+='Control makes six shuffled passes through all histories, multiplying the match loss by N p(deck)/(2 n(deck,chooser)). Treatment independently samples a deck by frozen n/100 prevalence, a chooser with probability 1/2, then a history uniformly with replacement within that stratum; its match weight is one. All histories remain eligible. No second prevalence correction is applied.\n\n'
 text+='For any fixed per-history loss L, both expected scalar objectives equal Σ_d p(d)/2 Σ_c mean[L | d,c]. The coefficient and outcome-independent sampler checks are in [sampler-check.json](../experiments/'+RUN+'/sampler-check.json). This identity does not imply equivalent updates through AdamW and clipping.\n\n'
 text+='Each of four runs receives exactly 14,004 updates, with checkpoints at 0, 2,334, 4,668, 7,002, 9,336, 11,670 and 14,004. No early stopping. The primary endpoint is **full-field n/100 validation deck NLL at 14,004**, treatment minus matched control; negative is better. The n/82 passing-opponent population is separately labelled.\n\n'
 text+=f'The prelaunch conservative forecast was {forecast["conservative_CPU_seconds"]/3600:.3f} CPU-core hours against the four-hour cap, including historical worst epoch/evaluation timings, stationary replays and overhead. Six UTC elapsed hours and four compute threads remain hard caps; the runner retains failures and stops rather than reducing the protocol.\n\n'
 if fits:
  text+='## Execution and exposure\n\n| Run | State | Updates | Unique / eligible histories | CPU minutes |\n|---|---|---:|---:|---:|\n'
  for f in fits:text+='| '+f['job']['key']+' | '+f['state']+' | '+str(f['updates'])+' | '+str(f['unique_histories'])+' / 2334 | '+fmt(f['CPU_seconds']/60,2)+' |\n'
  text+='\nRealized exposure counts and every deck/chooser frequency, including histories with zero exposure, are retained in [exposures](../experiments/'+RUN+'/exposures). Control visits every history six times when complete; treatment coverage is a measured outcome of its fixed sampling schedule, not a requirement to visit each record.\n\n'
  text+='| Run | Chooser 0 / 1 updates | Maximum absolute deck-frequency deviation from p(deck) |\n|---|---:|---:|\n'
  for f in fits:
   e=read(OUT/'exposures'/(f['job']['key']+'.json'));counts=e['deck_chooser_counts'];n=e['total_updates'];chooser=[sum(v for k,v in counts.items() if k.endswith(':'+str(s))) for s in (0,1)];deviation=max(abs(sum(counts.get(deck+':'+str(s),0) for s in (0,1))/n-info['prior']) for deck,info in prior().items()) if n else None;text+='| '+f['job']['key']+' | '+str(chooser[0])+' / '+str(chooser[1])+' | '+fmt(deviation)+' |\n'
  text+='\nControl exposure frequencies follow the nearly uniform collection; its loss weights represent prevalence. Treatment exposure frequencies directly represent prevalence. These frequencies are therefore expected to differ.\n\n'
 if a:
  text+='## Primary fixed-final comparison\n\nPaired whole-BO3 bootstrap draws are shared across methods within deck×chooser strata (10,000 replicates). Intervals condition on these fitted models, do not cover all training-seed uncertainty, and are not multiplicity-adjusted simultaneous claims. Seed means average scores within BO3; they are not ensembles or extra independent matches.\n\n'
  text+='| Population | Seed | Treatment − control deck NLL, 95% interval |\n|---|---|---:|\n'
  for pop,label in [('full_field','Full field n/100 — primary'),('eligible','Passing opponents n/82')]:
   for seed in ('s11','s29','seedmean'):text+='| '+label+' | '+seed+' | '+diff(a['populations'][pop]['contrasts']['population_minus_control_'+seed]['deck_log_loss'])+' |\n'
  text+='\n'+a['conclusion']['classification']+'\n\n'
  text+='### Treatment versus initialization and A\n\n| Reference | Seed | Final treatment − reference deck NLL (n/100), 95% interval |\n|---|---|---:|\n'
  for ref in ('initial','A'):
   for seed in ('s11','s29','seedmean'):text+='| '+ref+' | '+seed+' | '+diff(a['populations']['full_field']['contrasts'][f'population_minus_{ref}_{seed}']['deck_log_loss'])+' |\n'
  text+='\nA treatment that only regresses less than control is optimization evidence, not a successful replacement for A. Gains against control, initialization and A must be assessed separately. Conflicting seeds and intervals containing zero remain unresolved.\n\n'
  for pop,title in [('full_field','Full field n/100'),('eligible','Passing opponents n/82')]:
   methods=a['populations'][pop]['methods'];names=['A']+[f'{arm}-{seed}-u{u:05d}' for u in (0,14004) for arm in ('control','population') for seed in ('s11','s29','seedmean')]
   text+='## '+title+': initialization and fixed final\n\n| Model/checkpoint | Deck NLL | Deck accuracy | Aggregate count NLL | Combined objective |\n|---|---:|---:|---:|---:|\n'
   for name in names:text+='| '+name+' | '+' | '.join(cell(methods[name][k]) for k in ('deck_log_loss','deck_accuracy','count_NLL','objective_NLL'))+' |\n'
   text+='\n'
   for prefix,heading in [('count_NLL','Count NLL'),('presence','All-card presence Brier'),('uncertain_presence','Fixed A-defined uncertain presence Brier')]:
    text+='### '+heading+'\n\n| Model/checkpoint | Unknown hand | Draw pile | Prizes |\n|---|---:|---:|---:|\n'
    for name in names:text+='| '+name+' | '+' | '.join(cell(methods[name][prefix+'_'+z]) for z in ('hand','draw_pile','prizes'))+' |\n'
    text+='\n'
   text+='### Final legality and count consistency\n\n| Final model | Support mass violation | Hand / draw-pile / prize total MAE | Card-total violation MAE / fraction |\n|---|---:|---|---|\n'
   for name in [f'{arm}-{seed}-u14004' for arm in ('control','population') for seed in ('s11','s29','seedmean')]:
    v=methods[name];text+='| '+name+' | '+fmt(v['support_mass_violation']['value'],9)+' | '+' / '.join(fmt(v[k]['value']) for k in ('hand_total_MAE','deck_total_MAE','prize_total_MAE'))+' | '+' / '.join(fmt(v[k]['value']) for k in ('card_total_violation_MAE','card_total_violation_fraction'))+' |\n'
   text+='\nMarginal count consistency is diagnostic; no projection or behavior correction was introduced. Hand targets exclude legally known hand cards; those cards are added back when checking zone totals. Fixed uncertain subsets use A presence probability in [0.1,0.9] separately per zone and are identical across models.\n\n'
  text+='## Learning curves\n\n![Frozen checkpoint learning curves](../experiments/'+RUN+'/learning-curves.png)\n\nEvery registered checkpoint is retained. Full metrics and intervals for all seven checkpoints, both arms and seeds are in analysis.json; primary comparisons always use update 14,004.\n\n'
  text+='## Secondary selection with initialization eligible\n\n| Run | Selected update | n/100 combined validation objective |\n|---|---:|---:|\n'
  for key,x in a['selection'].items():text+='| '+key+' | '+str(x['updates'])+' | '+fmt(x['n100_combined_NLL'])+' |\n'
  text+='\nThis secondary rule minimizes the existing combined validation objective, with exact ties favoring the earlier checkpoint. It does not replace the primary endpoint. It reuses validation for selection and description, so selected-checkpoint results are not independent confirmation; conditional bootstrap intervals do not include selection uncertainty.\n\n'
  text+='## What is established and unresolved\n\n'+a['conclusion']['classification']+' The matched comparison changes sampling and scalar weighting together as registered; it tests their practical implementation of the same expected objective. It does not isolate AdamW from clipping, establish universal superiority across seeds, or prove that shared-head interference is absent.\n\n'
  for ref in ('initial','A'):
   x=a['conclusion']['treatment_vs_'+ref];text+='Treatment improvement over '+ref+': '+('supported on these fitted seeds under the registered comparison' if x['improvement_established'] else 'not established')+'; difference '+diff(x)+'.\n\n'
  text+='Hidden-zone preservation must be read separately from deck identification. The following are final treatment minus reference seed-mean differences in n/100 (negative is better); intervals crossing zero remain inconclusive, and a tradeoff is not labelled preservation.\n\n| Reference | Metric | Difference, 95% interval |\n|---|---|---:|\n'
  for ref,values in a['conclusion']['hidden_zone_comparisons'].items():
   for k,x in values.items():text+='| '+ref+' | '+k+' | '+diff(x)+' |\n'
  text+='\nNo deployment, test confirmation or follow-up experiment is launched.\n\n'
 stations=list((OUT/'stationary-training').glob('*.json'))
 if stations:
  text+='## Stationary training deck NLL\n\n| Run/checkpoint | Intended n/100 | Passing n/82 | Uniform collection |\n|---|---:|---:|---:|\n'
  for p in sorted(stations):
   x=read(p);text+='| '+p.stem+' | '+' | '.join(fmt(x['summary']['populations'][pop]['metrics']['deck_log_loss']) for pop in ('full_field','eligible','collection'))+' |\n'
  text+='\nThese are stationary replays at initialization/final, not online losses collected while parameters changed.\n\n'
 curves=list((OUT/'curves').glob('*.json'))
 if curves:
  text+='## Initialization parity\n\n| Run | Max deck probability difference from A | Max count difference from floored/masked A | Max count difference from raw A |\n|---|---:|---:|---:|\n'
  for p in sorted(curves):
   rows=read(p)
   if rows:
    x=rows[0]['initial_parity'];text+='| '+p.stem+' | '+' | '.join(fmt(x[k],9) for k in ('max_deck_vs_A','max_counts_vs_floored_A','max_counts_vs_raw_A'))+' |\n'
  text+='\nCount initialization deliberately applies the historical 1e-5 floor inside legal count support, then normalizes. This quantified difference from raw A is retained, not called an error. Deck parity tolerance is 2e-7 absolute; count parity is 2e-6 absolute plus 2e-5 relative. All original model/loss tests and new sampling/control-path checks precede training.\n\n'
 if c:text+=f'## Resource accounting\n\nAccounted CPU {c["CPU_seconds"]/3600:.3f} core hours; measured execution {c["measured_execution_CPU_seconds"]/3600:.3f}. Preparation includes a conservative 600-second charge, not invented actual measurement. UTC elapsed {c["elapsed_seconds"]/3600:.3f} hours. Completed optimizer updates {c["optimizer_updates"]}/56,016. Test payload/model access, GPU, native matches, PPO, paid APIs, uploads and submissions: zero.\n\n'
 failures=list((OUT/'failures').glob('*.json'))
 if failures:text+='Failures are retained in [failures](../experiments/'+RUN+'/failures); partial runs are not treated as equal completed comparisons.\n\n'
 text+='Evidence: [registration](../experiments/'+RUN+'/registration.json), [provenance](../experiments/'+RUN+'/source-manifest.json), [forecast](../experiments/'+RUN+'/forecast.json), [fixed schedules](../experiments/'+RUN+'/schedules.json), [initialization](../experiments/'+RUN+'/initialization.json), [sampler checks](../experiments/'+RUN+'/sampler-check.json), [analysis](../experiments/'+RUN+'/analysis.json), [fit receipts](../experiments/'+RUN+'/fit-results), [cost receipt](../experiments/'+RUN+'/completion.json). Private checkpoints remain outside Git.\n'
 REPORT.parent.mkdir(parents=True,exist_ok=True);REPORT.write_text(text);status=R/'docs/STATUS.md';old=status.read_text();begin='<!-- BELIEF-POPULATION-SAMPLING -->';end='<!-- BELIEF-POPULATION-SAMPLING-END -->';block=begin+'\n## Population-sampling repair experiment\n\n'+iso()+': **'+state+'**. '+reason+' [Report](../reports/'+REPORT.name+'). Fixed 14,004 updates per arm/seed; previously opened test excluded; no automatic follow-up.\n'+end+'\n'
 if begin in old:old=old[:old.index(begin)]+old[old.index(end)+len(end):].lstrip('\n')
 status.write_text(block+'\n'+old)
 if mirror is None:mirror=Path('/path/to/ptcg-agent') if os.name!='nt' else Path('C:/path/to/ptcg-agent')
 for p in (REPORT,status):
  target=mirror/p.relative_to(R);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
 dest=mirror/'experiments'/RUN;dest.mkdir(parents=True,exist_ok=True)
 for p in list(OUT.glob('*.json'))+list(OUT.glob('*.png')):shutil.copy2(p,dest/p.name)
