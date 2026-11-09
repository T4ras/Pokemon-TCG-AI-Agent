"""Readable fixed-checkpoint CPU evidence; never pool earlier checkpoint tests."""
from .common import *

def render_benchmark():
 def get(name):return read(OUT/name) if (OUT/name).exists() else {}
 registration=get('checkpoint-benchmark-registration.json')
 if not registration:return 'No additional strength evaluation has been allocated for this continuation.'
 analysis=get('evaluation-analysis.json');completion=get('evaluation-final-completion.json')
 pct=lambda v:'pending' if v is None else f'{100*v:.2f}%'
 pp=lambda v:f'{100*v:+.2f} pp'
 contrasts=[];cells=[]
 for metric,views in analysis.get('contrasts',{}).items():
  for name,v in views.items():
   if v.get('status')=='COMPLETE':contrasts.append(f"| {metric} | {name} | {pct(v['initial'])} | {pct(v['final'])} | {pp(v['final_minus_initial'])} | [{pp(v['interval95'][0])}, {pp(v['interval95'][1])}] |")
 for cell in analysis.get('cells',[]):
  strata=cell['strata'];total=lambda key:sum(s[key] for s in strata)
  def estimate(key):
   data=cell.get(key,{})
   return f"{pct(data['rate'])} [{pct(data['interval95'][0])}, {pct(data['interval95'][1])}]" if data else 'pending'
  cells.append(f"| {cell['checkpoint']} | {cell['deck']} | {cell['opponent']} | {total('attempts')}/{total('completions')}/{total('failures')} | {total('W')}/{total('D')}/{total('L')} | {estimate('BO3')} | {total('first_W')}/{total('first_D')}/{total('first_L')} | {estimate('first_game')} |")
 primary=analysis.get('contrasts',{}).get('BO3',{}).get('primary',{})
 if primary.get('status')=='COMPLETE':
  conclusion='The primary95% interval overlaps zero; this benchmark does not establish an overall BO3 improvement.' if primary['interval95'][0]<=0<=primary['interval95'][1] else 'The primary95% interval is positive in this local panel.' if primary['interval95'][0]>0 else 'The primary95% interval is negative in this local panel.'
 else:conclusion='The fixed-checkpoint strength comparison is pending.'
 return f'''## Completed-checkpoint CPU benchmark

{conclusion}

The user separately authorized **512 fresh CPU BO3** after training reached502,384 collected decisions and15,996 accepted updates. The exact final checkpoint is frozen before the first match. The original512 initialization attempts are reused unchanged. This adds512 native attempts to the earlier1,024, for1,536 total; no new baseline matches are run. D10/D04 × Hydro/D02/D01/D03 ×64, balanced32/32 by initial chooser; both sides use conditional argmax. Forced-observation memory, game resets and legal cross-game knowledge use the existing adapter. No extra native diagnostics, replacement matches, outcome-based extension, checkpoint selection or new training.

Hydro/D02 are seen decks; D01/D03 are held-out training decklists with the same frozen policy family. These are local Unown policies, not leaderboard agents. Native deals are independent and unpaired. Reused schedule/inference seeds do not control native shuffles. The earlier stopped-checkpoint benchmark remains separate and is not pooled into this comparison.

This additional evaluation follows an earlier checkpoint evaluation. Intervals are pointwise; no family-wise or sequential error-control claim is made across the project's repeated looks. An empirical all-loss bootstrap interval is not proof of a zero true win probability.

Final attempts/completed/failures: **{completion.get('attempts',0)}/{completion.get('completed',0)}/{completion.get('failures',0)}**. Recorded evaluation elapsed seconds: **{completion.get('UTC_elapsed_seconds','pending')}**. Additional GPU use and training updates: **zero**.

| Metric | Aggregate | Initialization | Completed checkpoint | Gain |95% interval |
|---|---|---:|---:|---:|---|
{chr(10).join(contrasts) or '| Pending | | | | | |'}

| Checkpoint | Learner | Opponent | Attempts/done/fail | BO3 W/D/L | BO3 WR [95%] | First-game W/D/L | First WR [95%] |
|---|---|---|---:|---:|---|---:|---|
{chr(10).join(cells) or '| Pending | | | | | | | |'}

The primary contrast equally weights all8 cells and both chooser assignments.20,000 independent whole-match bootstrap draws stay within checkpoint/deck/opponent/chooser strata. First-game analysis keeps valid first games after later failures. Per-chooser W/D/L, actual starters, draws, failures and first-game denominators are in [evaluation-analysis.json](../experiments/{RUN}/evaluation-analysis.json). [Registration](../experiments/{RUN}/checkpoint-benchmark-registration.json), [compact final match records](../experiments/{RUN}/evaluation-final-records.jsonl), [completion receipt](../experiments/{RUN}/checkpoint-benchmark-completion.json).
'''
