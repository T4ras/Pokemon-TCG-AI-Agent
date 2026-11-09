"""Frozen, CPU-only Unown Gradiant package comparison state."""
from pathlib import Path
import datetime as dt,hashlib,json,os
REPO=Path(__file__).resolve().parent.parent
RUN='astra-unown-gradiant-benchmark-001'
ROOT=REPO/'experiments'/RUN
PRIVATE=REPO/'data/private'/RUN
COMMIT='a176778d99da74a44b495bf19299b4795faf12f2'
UP=PRIVATE/'upstream'/COMMIT
CURRENT=PRIVATE/'current-environment'
REPORT=REPO/'reports/astra-unown-gradiant-benchmark.md'
START='[removed]'
DEADLINE='[removed]'
SEED=900007
BOOT=900006
LABELS=['BC34-Hydrapple','League32-Hydrapple','G30-OceanMix']
def now():return dt.datetime.now(dt.timezone.utc)
def iso():return now().isoformat()
def parse(s):return dt.datetime.fromisoformat(s.replace('Z','+[removed]'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+f'.{os.getpid()}.tmp');q.write_text(json.dumps(v,indent=2,default=str));q.replace(p)
def exclusive(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(v,f,indent=2,default=str);f.flush();os.fsync(f.fileno())
def append(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('a') as f:f.write(json.dumps(v,default=str)+'\n');f.flush()
def optional(name):
 p=ROOT/name;return read(p) if p.exists() else {}
def progress(stage,**kw):write(ROOT/'progress.json',dict(at_utc=iso(),stage=stage,**kw))
def normalize(x):
 if isinstance(x,str):return x.strip()
 if isinstance(x,list):return [normalize(v) for v in x]
 if isinstance(x,dict):return {k:normalize(v) for k,v in x.items()}
 return x
def rows(component):
 return [json.loads(l) for p in sorted((ROOT/component).glob('worker-*/outcomes.jsonl')) for l in p.read_text().splitlines()]
def mirror():
 import shutil
 win=Path(r'C:\path\to\ptcg-agent') if os.name=='nt' else Path('/path/to/ptcg-agent')
 for p in [REPORT,REPO/'docs/STATUS.md']+list(ROOT.rglob('*')):
  if p.is_file():
   q=win/p.relative_to(REPO);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)

def report(state,reason=None):
 m=optional('manifest.json');c=optional('compatibility.json');a=optional('analysis.json');cost=optional('costs.json');checks=rows('official');bench=rows('strength')
 text=['# Unown Gradiant frozen-package benchmark','',f'Updated {iso()}. State: **{state}**.', '',
  'This is an inference/compatibility experiment comparing complete deck-policy packages. Previous third place does not establish current Playground strength. No training, weight adaptation, deck search, uploads, submissions or GPU usage.', '',
  f'Upstream: [dipamc/kaggle-ptcg-ai-battle](https://github.com/dipamc/kaggle-ptcg-ai-battle/tree/{COMMIT}), commit `{COMMIT}`. MIT License, copyright [removed] Dipam Chakraborty; notice retained with the isolated private source. Exact submitted package 55559155, original Hydrapple/Meganium deck, heads8, temperature0, cfg=None; original rejection guard and time valve retained. The newer top-level runtime is excluded from inference.', '',
  '## Frozen identities', '',
  f'Candidate/checkpoint/deck/table/dependency hashes: manifest.json and upstream-manifest.json. Current official environment: `{m.get("environment_commit","pending")}`, version2, BO3. Native deals are independent and not seedable by our inference seed.', '',
  '| Package | Checkpoint SHA256 | Deck SHA256 |','|---|---|---|']
 for label,p in m.get('packages',{}).items():text.append(f'| {label} | `{p.get("sha256")}` | `{p.get("deck_sha256")}` |')
 text+=['','## Compatibility and adaptation','',
  'The release has 1,267 original cards and 1,556 attacks; Playground has 1,431 cards and 1,755 attacks. Model tables have 1,284 card rows and 1,573 attack rows, including reserved slack, which is not a learned representation of the 164 new cards or 199 new attacks. Fixed-pool support is checked by identity, not merely numeric ID.', '',
  'The old official library already installed in kaggle-environments supplies original static metadata without starting games. The original competition download endpoint returned403; this alternative is verified against all checkpoint static and text buffers. API bindings are from the locally held official SDK; the legacy metadata-only binding sets AllCard/AllAttack return types. Current engine execution is a separate process/library namespace and always uses Playground v2.', '',
  'Candidate inference runs in an isolated child using the unmodified submitted Python main.py and ptcg runtime. The adapter resets the original tracker, forced-run counter and rejection state at every game/match; it invents no cross-game knowledge. Our comparators retain their original legal deck-memory rules and forced-observation recurrent updates. Only actor-visible observations cross the pipe. The privileged oracle block is zero; the actor has no path from it. Original static table/PCA buffers and all weights remain fixed.', '',
  'There are 63 scored option slots plus STOP and 160 state-token slots. Non-forced option overflow, token overflow and unverified card/attack identities raise an explicit compatibility error rather than silently dropping options or clamping unknown identities. These guards leave supported decisions unchanged. Original sequential pick/STOP, equivalence-class deduplication, forced tracking, rejection recovery and panic valve are retained. Panic/fallback/errors disqualify the runtime gate.', '',
  f'Compatibility evidence: `{json.dumps(c,default=str)}`.', '',
  '## Official checks and admission forecast','',f'Official attempts recorded: {len(checks)}/6. Strength attempts recorded: {len(bench)}/288. No replacements and no native diagnostics outside the294-attempt allocation.', '',
  f'Admission: `{json.dumps(optional("admission.json"),default=str)}`.', '',
  '| Check | Opponent | Candidate chooser | Status | BO3 result | Match seconds |','|---|---|---:|---|---:|---:|']
 for r in checks:text.append(f'| {r["match_id"]} | {r["opponent"]} | {r["candidate_seat"]==0} | {r["status"]} | {r.get("result")} | {r.get("wall_seconds",0):.2f} |')
 text+=['','## Strength results','',
  'Primary: released package versus final league32 Hydrapple. Secondary: BC34 Hydrapple and G30 OceanMix. For each pairing,96 independent fresh BO3,48 per initial chooser stratum. Draws are nonwins. Twenty thousand whole-match bootstrap replicates independently within pairing/initial-chooser strata; the same resampled match indices produce BO3 and first-game intervals. Later-game and actual-starter summaries are descriptive. Failed later rounds retain any valid first game. Missing outcomes are shown rather than replaced. No checkpoint/temperature selection or outcome-based extensions.', '',
  '| Opponent | BO3 W/L/D/failed | BO3 win rate,95% interval | First-game W/L/D/failed | First-game win rate,95% interval |','|---|---|---|---|---|']
 for label,v in a.get('cells',{}).items():
  def fmt(x):return 'unavailable' if x.get('estimate') is None else f'{100*x["estimate"]:.2f}% [{100*x["interval_95"][0]:.2f},{100*x["interval_95"][1]:.2f}]'
  text.append(f'| {label} | {v["bo3_counts"]} | {fmt(v["bo3"])} | {v["first_counts"]} | {fmt(v["first_game"])} |')
 text+=['',f'Recommendation: **{a.get("recommendation", "Pending compatibility/execution; no strength conclusion")}.**',f'Primary gate: {a.get("primary_gate","pending")}. Requires at least60% wins,95% lower bound above50%, and no correctness/runtime failure. Passing establishes only the requested local matchup, not architecture/deck causality or broad diversity benefit.', '',
  'Compact complete outcomes, all opponent/chooser cells, actual starters, draws, finish reasons and failures are in official/ and strength/ and matches.jsonl.gz when complete. Detailed successful histories are retained only for the six official checks; bounded first failure examples stay private. Official checks are excluded from strength statistics.', '',
  '## Runtime, limits and costs','',
  'Current official wrapper timing behavior is recorded rather than assumed. Cold startup/model load, per-decision and match inference, IPC time and parent/child RSS are measured. In addition to wrapper checks, require full BO3 candidate inference plus cold startup below570 seconds, respecting a conservative shared600-second allowance and the original30-second valve. Only two CPUs per match worker; candidate keeps the submitted two Torch threads. Exact hosted container digest is unavailable; this is a local official-environment test, not hosted validation.', '',
  f'Costs: `{json.dumps(cost,default=str)}`. Fixed/Pipeline costs cover acquisition, metadata/adapter compatibility and the six checks. Variable/Agent costs cover only the288-match benchmark. CPU core-hours are measured process CPU seconds divided by3600; wall time includes preparation from {START}. Absolute deadline {DEADLINE}. GPU zero, training zero, uploads/submissions zero.', '',
  f'Failure/blocker: {reason or optional("failure.json").get("reason") or "none currently recorded"}.', '',
  'The ordinary local runner enforces deadlines and generates this report without model calls. Existing policies, checkpoints, active submissions and observations are preserved. No further training or experiment is launched automatically.']
 REPORT.write_text('\n'.join(text)+'\n')
 p=REPO/'docs/STATUS.md';s=p.read_text();marker='\n<!-- UNOWN-GRADIANT-BENCHMARK -->\n'
 if marker in s:s=s.split(marker)[0]
 s+=marker+f'\n## Unown Gradiant compatibility / benchmark\n\n{iso()}: **{state}**. Official checks {len(checks)}/6; strength {len(bench)}/288. {reason or a.get("recommendation", "Frozen CPU inference comparison underway")}. No training, GPU, upload or submission. See [report](../reports/astra-unown-gradiant-benchmark.md) and experiments/{RUN}/progress.json.\n'
 p.write_text(s)
