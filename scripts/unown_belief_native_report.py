"""Compact human report; machine manifests hold hashes, never pasted dumps."""
from unown_belief_native_common import *

def render(state,reason,final=False):
 records=rows(OUT/'matches.jsonl');intents=list((OUT/'intents').glob('*.json'));cost=read(OUT/'completion.json') if (OUT/'completion.json').exists() else {};analysis=read(OUT/'held-out-analysis.json') if (OUT/'held-out-analysis.json').exists() else None
 text=f'# Unown belief-tracker validation\n\nState: **{state}**. Updated {iso()}.\n\n{reason}\n\n'
 text+='This is the new native-observer run002. The previous incomplete run001 is preserved, with **zero native attempts**. Its report is archived in the readiness directory. This observer does not change actions or learn: both players retain the exact frozen submission57834515, conditional argmax, legal knowledge and game/match reset rules.\n\n'
 text+=f'New allocation: **{len(intents)}/592 attempted**; {sum(r.get("status")=="DONE" for r in records)} complete records, {sum(r.get("status")!="DONE" for r in records)} failed records. Development296 and held-out296, eight/list/partition, four per initial chooser assignment. No replacement matches. Native deals are independent; policy seeds do not seed native shuffles.\n\n'
 text+='The full-field prior remains n_d/100 across37 lists:30 passing lists/82 entries, six rejected/13 entries, anchor/5 entries. Eligible results use n_d/82 separately. The original schedules, query set and16 stability history IDs are byte-for-byte copies of run001. Original policy assets remain unchanged and private.\n\n'
 text+='## Implemented method and limitations\n\nThe compiler tracks physical serials revealed legally, concealed Basic setup, failed opening attempts/redeals, hand/deck/prize/looking/public/resolving zones, plays, attachment/evolution, shuffles and ordering. Public relocations within one callback do not invent a new draw. An opaque native serialization never enters the observer. Strategic selections use positive availability and uniform latent matching-copy choices, without learned action likelihoods or absence inference. Unresolved effect/order semantics explicitly fall back.\n\n'
 text+='Reconstruction is guided SMC over the complete current-game prefix. Joint finite-population proposals condition future chance-draw slots and source availability, with exact combinatorial proposal corrections. Within-list resampling carries the product of incremental normalizers; recovered list evidence multiplies its game-start mass. No compatible list is discarded because allocations died; no prior is restored. Normalized finite-particle estimates remain approximate. The production population has a512-state cap and four/list floor; auxiliary recovery uses256 proposals (1024 at2048).\n\n'
 text+='BaselineA applies the frozen prior and hard public composition/location facts;B substitutes valid current full-tracker deck probabilities into the same simple current-allocation model. The location model exactly marginalizes ambiguous known-serial domains, joint subset/location-count bounds and opening/hidden Basic constraints under its declared uniform allocation assumptions. Full depletion/unsupported predictions useA with a flag. B is also labeled B-plus-fallback when current full masses are unavailable; these are never described as authentic current B predictions. Supported-only results are separate. Conditional order that cannot be represented faithfully causes an explicit implementation stop rather than invented labels.\n\n'
 text+='Truth is collected only after actors finish. The pinned wrapper’s frame.obs is one selection behind; labels instead join validated pre-action native keys, explicit game boundaries, both public counts and own hand serials. Every selected point has a separate private label. At most24 genuine decisions/game are selected by the lowest hash(matchID,game,decisionIndex). All forced observations still update inference. Unknown-copy labels exclude certified known hand copies. Queries, tactical groups and A∈[.10,.90] uncertain subset are frozen.\n\n'
 text+='Scores average queries→decisions→games→whole BO3→equal chooser strata→population weights.20,000 paired whole-BO3 bootstrap samples use shared indices within list/chooser. Missing metrics retain explicit population coverage; absent strata never establish a complete-field gain. Stability uses16 development histories,512/2048 particles and seeds11/29/47, with no new games;2048 is not ground truth.\n\n'
 if analysis:
  text+='## Held-out predictive results\n\n'
  for population in ('full_field','eligible'):
   point=analysis['point'][population];text+=f'### {population.replace("_"," ")}\n\n| Scope/query group | Method | Brier | Hand count MAE | Deck MAE | Prize MAE | ECE |\n|---|---|---:|---:|---:|---:|---:|\n'
   for scope in ('all_in','supported_only'):
    for group in ('all','uncertain'):
     for method in ('A','B_plus_fallback','Full_plus_fallback'):
      p=f'{scope}/{group}/{method}';vals=[point.get(p+'/'+k) for k in ('brier','hand_MAE','deck_MAE','prize_MAE','ECE')];fmt=['—' if v is None else f'{v:.5f}' for v in vals];text+=f'| {scope}/{group} | {method} | '+' | '.join(fmt)+' |\n'
   text+='\n'
   for k,v in point.items():
    if '_minus_Full_brier' in k:text+=f'- {k}: {v:+.5f}; paired95% interval {analysis["CI95"][population].get(k)}.\n'
   text+='\n'
  text+='### Every list\n\n| Hash | Family | Held-out attempts/completed | Full supported point fraction | A Brier | Full-plus-fallback Brier |\n|---|---|---:|---:|---:|---:|\n'
  for d,v in prior().items():
   strata=analysis['per_list'][d];ms=[s.get('metrics',{}) for s in strata.values()]
   def avg(k):
    xs=[m[k] for m in ms if k in m];return '—' if not xs else f'{sum(xs)/len(xs):.5f}'
   text+=f'| {d[:10]} | {v["archetype"]} | {sum(s["attempts"] for s in strata.values())}/{sum(s["completed"] for s in strata.values())} | {avg("coverage/full_supported")} | {avg("all_in/all/A/brier")} | {avg("all_in/all/Full_plus_fallback/brier")} |\n'
  text+='\nCategory scores, all calibration bins, every chooser stratum and exact metric coverage are retained in held-out-analysis.json.\n\n'
 if (OUT/'stability-summary.json').exists():
  s=read(OUT/'stability-summary.json');text+=f'Offline stability: {s["replays"]} replays on16 histories; mean absolute probability discrepancy {s["mean_absolute_probability_discrepancy"]:.5f}, maximum history/pair {s["maximum_history_pair_discrepancy"]:.5f}. These compare tracker-plus-fallback outputs; support/recovery counts are retained separately.\n\n'
 if (OUT/'practical-gates.json').exists():
  gate=read(OUT/'practical-gates.json');text+='## Practical research gates\n\n'
  for name,passed in gate['checks'].items():text+=f'- {name}: {"PASS" if passed else "FAIL"}.\n'
  text+='\n**Recommendation:** '+gate['recommendation']+'\n\n'
 text+='## Execution and decision\n\n'
 if cost:text+=f'Actual local costs: UTC run wall {cost.get("UTC_wall_seconds",0)/3600:.3f}h; measured worker CPU {cost.get("worker_CPU_seconds",0)/3600:.3f} core-hours; storage {cost.get("storage_bytes",0)/1024**2:.2f}MiB. The fresh registration/deadline includes implementation/preflight as well as collection and analysis. GPU0, training0, submissions0, paid model API requests0. At most four workers and one Torch thread each. No power-plan edit or sleep inhibition.\n\n'
 if final and state=='COMPLETE' and analysis:
  pt=analysis['point']['full_field'];ci=analysis['CI95']['full_field'];a='all_in/uncertain/A_minus_Full_brier';b='all_in/uncertain/B_plus_fallback_minus_Full_brier';coverage=pt.get('coverage/full_supported',0);passed=coverage>=.9 and pt.get(a,0)>=.005 and ci.get(a,[-1])[0]>0 and pt.get(b,0)>0 and ci.get(b,[-1])[0]>0
  # All registered engineering gates remain necessary; never promote from
  # predictive scores alone. A complete detailed gate record is separate.
  text+=f'Predictive gate subset passed: {passed}; full supported coverage {coverage:.3%}. **Do not launch PPO automatically.** '+('A separately reviewed research proposal still needs all calibration/stability/latency/memory gates verified.' if passed else 'Repair guided reconstruction if support is low; otherwise retain the simpler baseline unless a clear paired location gain is established.')+'\n\n'
 elif final:text+='**Do not proceed to PPO.** Repair the named failure recorded in runner-failure.json; preserve all attempts and do not replace them automatically.\n\n'
 else:text+='Validation is running. No predictive conclusion or PPO recommendation is available yet. The ordinary controller writes final analyses and costs without language-model monitoring.\n\n'
 text+='Evidence: [fresh registration](../experiments/'+RUN+'/registration.json), [source manifest](../experiments/'+RUN+'/source-manifest.json), [readiness report](astra-unown-belief-native-observer-readiness.md), [compact matches](../experiments/'+RUN+'/matches.jsonl), [held-out analysis](../experiments/'+RUN+'/held-out-analysis.json), [cost/completion](../experiments/'+RUN+'/completion.json). Historical run001 and its source/evidence remain intact.\n'
 REPORT.write_text(text,encoding='utf-8')
 status=R/'docs/STATUS.md';body=status.read_text(encoding='utf-8-sig');start='<!-- BELIEF-NATIVE-002-START -->';end='<!-- BELIEF-NATIVE-002-END -->';block=f'{start}\n## Native belief observer validation\n\n{iso()}: **{state}**. {len(intents)}/592 attempts; zero GPU/training. {reason} [Validation report](../reports/astra-unown-belief-tracker-validation.md); [readiness](../reports/astra-unown-belief-native-observer-readiness.md). Historical run001 retained with zero attempts.\n{end}\n\n'
 if start in body:a=body.index(start);b=body.index(end,a)+len(end);body=body[:a]+body[b:].lstrip('\n')
 status.write_text(block+body,encoding='utf-8')
 import shutil
 mirror=Path('C:/path/to/ptcg-agent') if os.name=='nt' else Path('/path/to/ptcg-agent')
 sources=[REPORT,READY_REPORT,status]+[OUT/f for f in ('registration.json','source-manifest.json','native-gate.json','inference-amendment.md','lifecycle.json','progress.json','launch.json','startup.json','completion.json')]
 if final:sources +=[p for p in OUT.iterdir() if p.is_file() and p.suffix in ('.json','.jsonl','.md')]
 for source in sources:
  if source.exists():target=mirror/source.relative_to(R);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)

