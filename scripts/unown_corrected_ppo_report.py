"""Autonomous terminal report and cost accounting; no experiments launched."""
import sys,time
import numpy as np
from unown_corrected_ppo_common import *
def optional(p):return read(p) if p.exists() else {}
def lines(p):return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()] if p.exists() else []
def pct(x):return 'unavailable' if x is None else f'{100*x:.2f}%'
def interval(x):return 'unavailable' if not x else '['+', '.join(pct(v) for v in x)+']'
def finalize(state,reason):
 t=time.monotonic();cpu=time.process_time();train=optional(OUT/'training-completion.json');evaluation=optional(OUT/'evaluation-analysis.json');gpu=optional(OUT/'gpu-costs.json');stages=lines(OUT/'local-costs.jsonl');controller=optional(OUT/'controller-costs.json');manifest=read(OUT/'manifest.json');updates=lines(OUT/'updates.jsonl');cohorts=lines(OUT/'cohorts.jsonl');diags=lines(OUT/'initialization-diagnostics.jsonl')
 accepted=[u for u in updates if u['accepted']];windows=[]
 for label,items in [('first16',accepted[:16]),('last16',accepted[-16:])]:
  metrics={}
  for k in ['policy_loss','value_loss','entropy','clip_fraction','gradient_norm']:
   vals=[u['metrics'][k] for u in items if k in u.get('metrics',{})];metrics[k]=float(np.mean(vals)) if vals else None
  windows.append(dict(window=label,accepted_updates=len(items),means=metrics))
 samples=optional(OUT/'retrieval.json').get('receipt',{});timing=train.get('timing',{});elapsed_remote=train.get('elapsed_seconds');private_files=[p for p in PRIVATE.iterdir() if p.is_file()] if PRIVATE.exists() else []
 counts={s:len(list((OUT/s/'intents').glob('*.json'))) for s in ['compatibility','training','evaluation-initial','evaluation-final']}
 costs=dict(at_utc=iso(),elapsed_since_registration_seconds=(now()-parse(read(OUT/'registration.json')['local_execution_start_utc'])).total_seconds(),local_elapsed_excluding_bounded_remote_seconds=(now()-parse(read(OUT/'registration.json')['local_execution_start_utc'])).total_seconds()-optional(OUT/'remote-clock.json').get('excluded_seconds',0),local_stages=stages,measured_local_CPU_core_hours=sum(s['CPU_core_hours'] for s in stages),controller=controller,remote_training_CPU_core_hours=train.get('remote_CPU_core_hours'),GPU=gpu,native_attempts=counts,
  Pipeline_Fixed=dict(preflight_seconds=train.get('preflight_seconds'),diagnostics_seconds=timing.get('diagnostic_seconds'),checkpoint_seconds=timing.get('checkpoint_seconds'),transaction_snapshot_rollback_seconds=timing.get('transaction_snapshot_rollback_overhead_seconds'),other='Local preparation/tests/cold extraction/transfer/startup/export/fixed evaluation/analysis. Full cloud startup/export measured by receipt; public record'),
  Agent_Variable=dict(collection_seconds=timing.get('collection_seconds'),PPO_gradient_optimizer_seconds=timing.get('update_seconds')),paid_model_API_requests=0,paid_model_API_cost_USD=0,competition_submissions=0,measurement_limits='Measured process CPU/stage wall time; interactive source implementation/reading before registration and some Windows/child preparation CPU are not fully metered. API quota account-wide, conservative session wall includes queue. Remote wrapper export/provisioning are fixed overhead, not attributed to training. Missing measurements are unknown, not zero.')
 write(OUT/'costs.json',costs)
 finals={}
 for name in ['latest.pt','phase-input.pt','first-rollout.pt','last-rollout.pt','partial-final-rollout.pt','rolling-rejection-before.pt','rolling-rejection-rejected.pt','final.pt']:
  p=PRIVATE/name
  if p.exists():finals[name]=dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size)
 write(OUT/'recovery-manifest.json',dict(at_utc=iso(),files=finals,hourly=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in sorted(PRIVATE.glob('hourly-*.pt'))],no_automatic_resume=True))
 table=['| Checkpoint | Attempted/completed/failed | Equal-chooser BO3 rate | First-game rate |','|---|---:|---:|---:|']
 for label,v in evaluation.get('checkpoints',{}).items():table.append(f"| {label} | {v['attempted']}/{v['completed']}/{v['failures']} | {pct(v['metrics']['BO3'].get('equal_chooser_win_rate'))} | {pct(v['metrics']['first_game'].get('equal_chooser_win_rate'))} |")
 diff=evaluation.get('difference',{}).get('BO3',{});point=diff.get('final_minus_initial');decision='Preparation/execution incomplete; no learning-strength conclusion'
 if evaluation.get('status')=='SKIPPED':decision='No retained model change' if not train.get('model_tensors_changed') else 'Strength evaluation skipped because correctness failure invalidates this run'
 elif evaluation.get('status')=='COMPLETE':
  ci=diff['bootstrap_CI95'];decision='Fixed comparison supports improved greedy Hydrapple performance; no promotion, external/deck transfer remains untested' if ci[0]>0 else 'Fixed comparison supports a regression; stop this recipe on this result' if ci[1]<0 else 'No demonstrated improvement at the registered resolution; report uncertainty and actual experience, no automatic sweep'
  if diff.get('degenerate_bootstrap'):decision='Bootstrap degenerate; use conservative exact bounds, no claim of certainty'
 primary='unavailable' if point is None else f"{100*point:+.2f} pp;95% chooser-stratified whole-match bootstrap {interval(diff.get('bootstrap_CI95'))}; conservative exact95% {interval(diff.get('CI95_conservative'))}"
 curves=[]
 for i in range(0,len(cohorts),8):
  block=cohorts[i:i+8];wdl=collections_counter(block);curves.append(dict(cohorts=[b['phase'] for b in block],completed_BO3=sum(b['BO3'] for b in block),learner_decisions=sum(b['decisions'] for b in block),accepted_updates=sum(b['accepted_updates'] for b in block),sampled_training_WDL=wdl))
 write(OUT/'training-summary.json',dict(loss_windows=windows,training_curves=curves,diagnostics=diags,decision=decision))
 report=f'''# Corrected Hydrapple PPO six-hour experiment

Updated {iso()}. State: **{state}**. {reason}

**Decision: {decision}.** Changed weights or elapsed GPU time do not establish competence. No current global ranking is asserted. No deployment, competition submission, added seed, opponent, feature mapping, LR sweep or longer training is launched.

## Frozen initialization and protocol

Initial checkpoint SHA256 `{INITIAL_SHA}`,tensor hash `{INITIAL_TENSOR}`. Fresh AdamW state, no earlier PPO continuation. Both actors use the exact broad-validated starting-order implementation SHA256 `{manifest['corrected_implementation_sha256']}`: physical seat while public starting order is unknown; then0 actual first/1 actual second; reset each game. Physical seats, legal observations/actions and reward bookkeeping stay unchanged. Collection stores the same corrected packets used for sampling, without changing recorded old action probabilities/log probabilities. PPO recomputation, the frozen greedy corrected initialization opponent, evaluation and diagnostics all consume corrected encoding.

Original ordered Hydrapple deck hash `{manifest['ordered_deck_sha256']}`; one fixed opponent only. Learner training uses the existing temperature1 legal conditional sampler; evaluated policies and opponent use existing conditional argmax. Text/static tables fixed; oracle/unavailable auxiliary objectives disabled. Recipe is constant AdamW1e-5,betas(.95,.999),epsilon1e-8,weight decay0,no scheduler; one PPO epoch,logical128/remainder actual-size averaging,gamma1,GAE.95,clip.2,blind-value coefficient2,entropy coefficient.01,global gradient clip1.5. Existing per-game learner-relative terminal±1/0 rewards and terminal bootstrap masks are retained; whole BO3 policies remain fixed. No anchor loss.

Registration/manifests/source hashes: experiments/{RUN}. Exact unchanged optimized core SHA256 `{manifest['optimized_core_sha256']}`; reused CPU/CUDA arithmetic proof is in reused-numerics.json. The new storage/stop wiring has focused regressions; fresh2-BO3 CPU compatibility cohort was disposable and excluded from learning. CPU parity/checkpoint-resume and the exact extracted upload archive are checked without checkout fallback. GPU startup checks128 corrected compatibility packets against scalar inference; every actual CUDA training cohort passes the unchanged chosen-LP1e-5/old-policy-KL1e-6 bounds before learning. No probability rebasing, truncation or relaxed tolerance. Kernel/payload SHA, pinned SDK/image, new public-first semantics and fixed schedules are retained separately.

## Revised stopping and actual learning

Phase forward categorical KL(old phase||current)>.02 before a minibatch ends optimization of that rollout. Post-update phase KL>.03 persists rejected diagnostics, restores model/optimizer/gradients/mode/execution and all relevant RNG exactly, and ends that rollout. Earlier accepted updates remain. Remaining optimization work is discarded; next cohort is fresh on-policy. Three consecutive complete fresh cohorts with zero accepted updates stop the run; numerical/mask/nonfinite correctness failures stop with preserved evidence. Initialization KL is diagnostic only—no0.10 stop and no hidden replacement threshold. Startup near-zero reference check is a correctness check on identical initialization, not a cumulative drift gate.

Training status/stop reason: `{train.get('status','not launched')}` / `{train.get('stop_reason','not launched')}`. Completed BO3 {train.get('completed_BO3',0)},optimized-cohort BO3 {train.get('optimized_cohort_BO3',0)},complete cohorts {train.get('completed_cohorts',0)},learner conditional decisions {train.get('learner_decisions',0)}. Retained accepted updates {train.get('accepted_updates',0)},post-update rejections {train.get('rejected_updates',0)},rollout early stops `{train.get('rollout_early_stops',{})}`,zero-update streak {train.get('zero_cohort_streak',0)}. Incomplete final cohort: `{train.get('incomplete_final_cohort')}`. Each complete16-match cohort has8/8 shuffled initial choosers. Incomplete experience is saved/reported and never optimized as a complete cohort.

Final tensor hash `{train.get('final_tensor_hash','unavailable')}`,checkpoint SHA256 `{train.get('checkpoint_sha256','unavailable')}`. Recoverable latest/phase-input/current+first rollout/rejected snapshots/hourly/final states are private; exact checksums and paths are in recovery-manifest.json. No intermediate checkpoint was evaluated for winner selection.

Loss/gradient windows (unweighted raw component values; value coefficient2 and entropy coefficient-.01 applied in the verified loss): `{windows}`. Every update records pre/post KL,full conditional clipping/entropy,meaningful-action entropy (more than one legal masked option),policy/value losses and gradient norm. Training curves in training-summary.json group complete cohorts in blocks of8; they are sampled-policy outcomes against a fixed opponent while learner weights change, not stationary greedy strength estimates. Curves: `{curves}`.

Fixed corrected diagnostic set: `{optional(OUT/'diagnostic-freeze.json').get('states')}` legally observed states selected and packet-hashed before training,not training experience. Reference distributions recomputed under corrected initialization on actual backend. Startup,roughly hourly cohort-boundary and endpoint full-distribution KL,tail,TV,action-agreement and entropy records: `{diags}`. No full initialization sweep occurs after every minibatch. Diagnostic inputs are retained off-policy legal histories; agreement/entropy changes alone do not establish strength.

Local preparation repair: the released main silently set Torch CPU threads to2 after the registered1-thread setup. A disposable resume check detected different subsequent arithmetic despite exact restored parameters/optimizer/RNG. The new loader enforces the registered backend after construction; focused regression and bitwise uninterrupted/resumed update checks pass. Earlier local check failures and component differences are retained, no tolerances relaxed. The2 original native compatibility matches were replayed from both retained learner histories with exact sampler actions; repair added0 native matches. Probe weights/optimizer/experience were excluded from training. Details: preparation-repair.json and CPU-resume-details.json.

## Fixed fresh evaluation

Evaluation status: `{evaluation.get('status','not run')}`. Exactly256 fresh CPU BO3 per fixed checkpoint,128/128 initial chooser assignments,against frozen greedy corrected initialization. Both evaluated actors greedy. Native deals independent,not paired by inference seeds. No previous-test pooling,no intermediate selection,no outcome-based extensions. Strength evaluation is skipped for no accepted surviving change/identical tensors; correctness-failed runs do not receive a misleading strength conclusion.

{chr(10).join(table)}

Primary final-minus-initial equal-chooser BO3 difference: **{primary}**. Primary bootstrap uses20000 independent whole-match resamples within checkpoint/chooser strata; exact conservative difference uses four98.75% CP stratum intervals. Draws count as nonwins. Any degenerate bootstrap is flagged,never certainty. First-game results and chooser strata: `{json.dumps({k:v['metrics']['first_game'] for k,v in evaluation.get('checkpoints',{}).items()})}`. Compact evaluation records retain every game's actual starter,chooser/outcome and failures. Later-game participation/conversion are descriptive conditional subsets rather than independent game trials. Detailed game2/3 and conversion reconstruction is saved in evaluation-extra.json. Preselected first2 matches perchooser/checkpoint retain both actors' full legal histories through later games; bounded failures retained. Hidden diagnostic data never enters actor input.

This is one sustained time-bounded arm,not a controlled comparison with another training recipe. Any observed gain applies to these frozen corrected Hydrapple policies/opponent,not leaderboard/cross-deck superiority. A negative/inconclusive result remains valid; accepted updates and weight change are not learning-strength evidence by themselves. No follow-up is authorized automatically. A positive result warrants a separately registered broader fixed-opponent confirmation before investment; a negative/uncertain result warrants stopping and interpreting the reported actual experience/diagnostics,not automatically increasing time or weakening safeguards.

Per-game/chooser/actual-starter WDL and first-game conversion/recovery (with whole-match denominators and intervals): `{optional(OUT/'evaluation-extra.json')}`.

## Costs and execution

GPU record: `{gpu}`. Maximum six GPU-session hours,one job,including startup,checks,training and export. Provider timeout20700s,child20100s,training19500s plus absolute push deadline21000s leave export/provisioning margins inside21600s. No automatic replacements, resets or extensions. Provider quota delta,conservative push-to-terminal session hours,allocatedT4x2,visible/used device0,allocated versus used device-hours are distinct. Account-wide quota may be confounded by separately started account work; terminal verification is explicit. Private upload bundle `{optional(OUT/'bundle.json')}`; authorization is the current human request,necessary private assets only. No credentials or unrelated/old PPO assets uploaded.

Measured phase timings: `{timing}`. Optimizer timing separates gradient/step from phase diagnostics; snapshot/rollback overhead is fixed pipeline cost. Actual remote script elapsed {elapsed_remote}s; completed useful BO3 per script hour `{(train.get('completed_BO3',0)/elapsed_remote*3600) if elapsed_remote else None}` including recorded startup/checks/checkpoint/diagnostics (provider provision/export additionally in receipt). Pipeline/Fixed covers preparation,tests,cold extraction,transfers,startup,checks,diagnostics,snapshots/checkpoints/export,fixed evaluation and analysis; Agent/Variable covers native collection and PPO gradient/optimizer updates. Full accounting: costs.json,local-costs.jsonl,controller-costs.json,gpu-costs.json and remote receipt.

Measured local CPU {costs['measured_local_CPU_core_hours']:.4f} core-hours,remote trainer CPU {train.get('remote_CPU_core_hours')},controller CPU {controller.get('CPU_core_hours')}. UTC total elapsed {costs['elapsed_since_registration_seconds']:.1f}s;local elapsed excluding only bounded remote window {costs['local_elapsed_excluding_bounded_remote_seconds']:.1f}s,ceiling14400s. Preparation implementation/read CPU before measured stages is not fully metered; missing measurements are unknown. Native attempts `{counts}`. Private files are outside Git; histories512MiB/recovery2GiB caps and compressed export bound storage. GPU paid-model API requests0/cost$0,model credential searches0,public publication0,competition submissions0.
'''
 REPORT.write_text(report,encoding='utf-8');status=ROOT/'docs/STATUS.md';text=status.read_text(encoding='utf-8');marker='## Corrected Hydrapple PPO six-hour experiment';start=text.find(marker)
 if start>=0:
  end=text.find('\n## ',start+len(marker));text=text[:start]+(text[end+1:] if end>=0 else '')
 status.write_text(marker+'\n\n'+iso()+f': **{state}**. {decision}. BO3{train.get("completed_BO3",0)},accepted updates{train.get("accepted_updates",0)},stop {train.get("stop_reason","not launched")}. No automatic training/deployment/submission. [Report](../reports/astra-unown-corrected-hydrapple-ppo-6h.md).\n\n'+text,encoding='utf-8')
 append(OUT/'local-costs.jsonl',dict(stage='final-report',at_utc=iso(),elapsed_seconds=time.monotonic()-t,CPU_core_hours=(time.process_time()-cpu)/3600));costs['local_stages']=lines(OUT/'local-costs.jsonl');costs['measured_local_CPU_core_hours']=sum(s['CPU_core_hours'] for s in costs['local_stages']);write(OUT/'costs.json',costs)
def collections_counter(items):
 import collections
 result=collections.Counter()
 for r in items:result.update(r['training_WDL'])
 return dict(result)
if __name__=='__main__':finalize(sys.argv[1],sys.argv[2])
