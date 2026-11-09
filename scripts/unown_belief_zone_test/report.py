from .common import *
import shutil

def render(state,reason,mirror=None):
 a=read(OUT/'analysis.json') if (OUT/'analysis.json').exists() else None;c=read(OUT/'completion.json') if (OUT/'completion.json').exists() else None
 text=f'# Hidden-zone Brier scores: frozen belief classifiers\n\nState: **{state}**. Updated {iso()}.\n\n{reason}\n\n'
 text+='This is a user-authorized first scoring of the previously sealed 592 test BO3. All 21 historical validation-selected fits plusA are frozen before test access. No training, checkpoint selection, native match, GPU, paidAPI, upload or submission. Testresults must not be used to describe this testset as untouched in later development.\n\n'
 text+='Binary presence Brier is (P(at leastone copy inzone) − actual0/1presence)², averaged across 143 card IDs, then decisions within game, equally across games within BO3, and by frozen deck/chooser population weights. Lower is better. Hand means the unknown portion of the hand, matching historical benchmark semantics; total-hand presence including legally known cards is separately retained. Draw pile and prizes use full zone counts. Certain/absent cards remain in all-card averages.\n\n'
 text+='Uncertain-only scores use the same fixed A-defined0.1–0.9 presence subset per zone for every model; subsets differ across zones. Seedmean is mean of scores, not an ensemble. Multiclass copy-count Brier (sum across 18 categories, then mean across 143 card IDs) is in analysis.json and is a different metric from binary presence. Allselected fits at 296, 592 and 2,334 training BO3 are included; no intermediate/last/probe checkpoints.\n\n'
 if a:
  text+=f'Completed {a["completed_BO3"]}/592 test BO3; failures/missing {len(a["failures"])}. Intervals use 10,000 whole-match bootstrap draws within deck×chooser strata, shared acrossmethods; conditional on frozen models. No multiplicity-adjusted winner claim.\n\n'
  for pop,title in [('eligible','Primary30-list n/82 population'),('full_field','Full37-list n/100 population')]:
   text+='## '+title+'\n\n'
   for prefix,subtitle in [('presence','All 143 card identities'),('uncertain_presence','Fixed uncertain subset')]:
    text+='### '+subtitle+'\n\n| Model | Hand | Draw pile | Prizes |\n|---|---:|---:|---:|\n'
    for key,metrics in a['populations'][pop].items():
     vals=[]
     for z in ('hand','draw_pile','prizes'):
      x=metrics[prefix+'_'+z];v=x['value'];ci=x['CI95'];vals.append('unavailable' if v is None else f'{v:.6f}'+(f' [{ci[0]:.6f}, {ci[1]:.6f}]' if ci else ' (incomplete interval coverage)'))
     text+='| '+key+' | '+' | '.join(vals)+' |\n'
    text+='\n'
 if c:text+=f'## Cost\n\nAggregate accountedCPU {c["CPU_seconds"]/3600:.3f}core hours (includes conservative preparation reserve); UTC elapsed {c["elapsed_seconds"]/3600:.3f}hours. Native matches/GPU/training/paid APIs: 0.\n\n'
 text+='Evidence: [registration](../experiments/'+RUN+'/registration.json), [frozenmodels](../experiments/'+RUN+'/models.json), [testmanifest](../experiments/'+RUN+'/test-manifest.json), [source/checkpoint hashes](../experiments/'+RUN+'/source-manifest.json), [analysis](../experiments/'+RUN+'/analysis.json), [completion](../experiments/'+RUN+'/completion.json). Private features, labels and raw histories remain outside Git.\n'
 REPORT.parent.mkdir(parents=True,exist_ok=True);REPORT.write_text(text)
 status=R/'docs/STATUS.md';old=status.read_text();begin='<!-- BELIEF-ZONE-TEST -->';end='<!-- BELIEF-ZONE-TEST-END -->';block=begin+'\n## Hidden-zone test Brier evaluation\n\n'+iso()+': **'+state+'**. '+reason+' [Report](../reports/'+REPORT.name+'). The 592 test BO3 are now authorized for fixed scoring; no future claim of an untouched test after access. No training/native/GPU.\n'+end+'\n'
 if begin in old:old=old[:old.index(begin)]+old[old.index(end)+len(end):].lstrip('\n')
 status.write_text(block+'\n'+old)
 if mirror is None:mirror=Path('/path/to/ptcg-agent') if os.name!='nt' else Path('C:/path/to/ptcg-agent')
 for p in (REPORT,status):
  target=mirror/p.relative_to(R);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
 dest=mirror/'experiments'/RUN;dest.mkdir(parents=True,exist_ok=True)
 for p in OUT.glob('*.json'):shutil.copy2(p,dest/p.name)
