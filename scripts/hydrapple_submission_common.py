"""Shared, standard-library state for the one authorized league submission."""
from pathlib import Path
import datetime as dt
import hashlib,json,os

REPO=Path(__file__).resolve().parent.parent
RUN='astra-hydrapple-league-submission-001'
ROOT=REPO/'experiments'/RUN
PRIVATE=REPO/'data/private'/RUN
ARCH=REPO/'artifacts'/RUN/'hydrapple-league-phase032-argmax.tar.gz'
REPORT=REPO/'reports/astra-hydrapple-league-submission.md'
SLUG='the-pokemon-company-ptcg-ai-battle-challenge-playground'
BC=57436919
G=57239424
DESCRIPTION='Hydrapple league phase-32 after 2048 BO3 | frozen CPU argmax | external evaluation'
CPHASH='1830c1a854e3865c9af26243727c6182d96b4b60a23ca1e62cd87e65875bac37'
TENSOR='10686fc80ec5c75d00794f079b9c3a812b940d17505641240ec228c138a2a905'
OLDHASH='216154d472274b6359f822d45137863a2ed127496e189c7292c1a5fa0c43d692'
START='[removed]'

def now():return dt.datetime.now(dt.timezone.utc)
def iso():return now().isoformat()
def parse(s):
    x=dt.datetime.fromisoformat(s.replace('Z','+[removed]'))
    assert x.tzinfo is not None,'Timestamp must have timezone'
    return x.astimezone(dt.timezone.utc)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_name(p.name+f'.{os.getpid()}.tmp')
    q.write_text(json.dumps(v,indent=2,default=str),encoding='utf-8');q.replace(p)
def exclusive(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,default=str);f.flush();os.fsync(f.fileno())
def optional(name,default=None):
    p=ROOT/name
    return read(p) if p.exists() else ({} if default is None else default)
def progress(stage,**kw):write(ROOT/'progress.json',dict(at_utc=iso(),stage=stage,**kw))

def slot_guard(submissions,active,remaining):
    assert remaining>=1,'No submission allowance remains'
    assert {int(x['id']) for x in active}=={BC,G},'Unexpected active submissions; stop before upload'
    order=sorted(submissions,key=lambda s:s['date'],reverse=True)
    assert len(order)>=2 and [int(x['ref']) for x in order[:2]]==[BC,G],'Upload would displace a protected or unexpected submission'
    assert not any(x.get('description')==DESCRIPTION for x in submissions),'Existing matching submission: reconcile, never duplicate'
    return dict(preserved=BC,leaving=G,before_active_ids=[BC,G],expected_after_ids=['NEW_LEAGUE',BC])

def window(start,sid,episode):
    start=parse(start);end=start+dt.timedelta(days=7);early=start+dt.timedelta(days=1)
    msk=dt.timezone(dt.timedelta(hours=3))
    return dict(frozen_at_utc=iso(),submission_id=sid,baseline_submission_id=BC,validation_episode_id=episode,
        window_start_utc=start.isoformat(),window_end_exclusive_utc=end.isoformat(),manual_early_health_check_utc=early.isoformat(),
        window_start_moscow=start.astimezone(msk).isoformat(),window_end_exclusive_moscow=end.astimezone(msk).isoformat(),
        manual_early_health_check_moscow=early.astimezone(msk).isoformat(),duration_seconds=604800,field_results_interpreted=False,automatic_monitoring=False)

def report(state):
    archive=optional('archive.json');m=optional('package-manifest.json');receipt=optional('submission-receipt.json');val=optional('validation.json');w=optional('observation-window.json');amend=optional('stage-i-amendment.json');cost=optional('costs.json');failure=optional('failure.json')
    resolution=optional('failure-resolution.json')
    if failure and resolution.get('resolved_failure_sha256')==sha(ROOT/'failure.json'):failure={}
    rows=[read(p) for p in sorted((ROOT/'official').glob('L-package-?.json'))]
    lines=['# Frozen league Hydrapple submission','',f'Updated {iso()}. State: **{state}**.','',
      'The selected candidate is the league arm final phase-32 endpoint after 2,048 BO3. Its direct local result was promising; broad superiority and a league advantage remain unconfirmed. This is an external evaluation, not a promotion based on proven strength. No weights, deck, decoder, rewards or policy behavior were changed.', '',
      f'Checkpoint SHA256: `{CPHASH}`. Model tensor SHA256: `{TENSOR}`.',
      f'Archive: `{archive.get("path","not built")}`. SHA256: `{archive.get("sha256","pending")}`. Bytes: {archive.get("bytes","pending")}.',
      f'Deck SHA256: `{m.get("deck_list_sha256","pending")}`. Visible-v1; CPU FP32 conditional argmax; forced observations update GRU; per-game GRU reset; legal revealed deck lower bounds carry between games and reset each match.', '',
      'The inference-only archive contains main.py, deck.csv, weights.pt, package manifest and the existing Python runtime/model/encoder. It excludes optimizer/RNG training state, trajectories, credentials and SDK/native binaries. Exact file, dependency, source and training identity records are in package-manifest.json and manifest.json.', '',
      f'Official packaging BO3 completed: **{len(rows)}/6**. Three existing opponent deck families, both initial chooser assignments; packaging checks only, no strength inference. Native randomness is independent; inference seeds do not seed native deals.', '',
      '| Match | Opponent | Learner seat | Games | Actions checked | Hidden max error | Runtime seconds | Peak RSS MiB |', '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:lines.append(f'| {r["match_id"]} | {r["opponent"]} | {r["learner_seat"]} | {r["official_result"]} | {r["source_packaged_action_matches"]} | {r["max_hidden_absolute_error"]} | {r["run_UTC_seconds"]:.2f} | {r["peak_worker_RSS_kib"]/1024:.1f} |')
    lines += ['', 'Each completed check retains legal actor-visible full histories privately and checks exact source/package actions, hidden state at atol 1e-6/rtol 0, action legality, game and match resets, forced recurrent updates, legal memory parity, unchanged tensors and time/RSS headroom. Offline forced/reset checks add zero native matches.', '',
      'Current official requirements and active-slot prediction are recorded immediately before upload in preflight.json and official-requirements.json. Local checks reuse the pinned official environment; exact hosted image/Torch build is not exposed, so hosted validation is required.', '',
      f'Submission ID: **{receipt.get("ref","not submitted/confirmed")}**. Receipt time: {receipt.get("response_received_at_utc","unavailable")}. Hosted result: **{val.get("validation_result","not checked")}**. Active IDs: {val.get("active_ids","not yet checked after upload")}.',
      f'Error: {val.get("error_description") or failure.get("reason") or "none recorded"}.', '',
      'Exactly one upload is authorized. An exclusive intent prevents repeat calls after interruption; ambiguous responses are reconciled through records and never blindly resubmitted. Hosted failure does not trigger repairs or another upload.', '',
      'The initial pre-upload workflow stopped because the authenticated API FAQ contains unresolved competition-variable placeholders. The initial failure, report, logs and original workflow sources are preserved in recovery-001. Recovery verifies the fresh rendered official FAQ against the current API template. It reuses the exact verified archive and six packaging results, without new native matches, policy changes or rebuilds. See recovery-001/registration.json for the operational-source amendment.', '',
      '## Stage I amendment', '',
      'The original Stage I seven-day plan and all historical evidence are preserved. The user now authorizes G 57239424 to leave while BC34 Hydrapple 57436919 remains active. The original G-versus-BC34 comparison is intentionally curtailed at the observed slot transition; it must not be described as a completed seven-day comparison.',
      f'Actual transition evidence: {json.dumps(amend,default=str) if amend else "Not yet observed; no transition claimed."}', '',
      '## New external observation protocol', '',
      'The frozen protocol is external-protocol.json. After official validation completes, the seven-UTC-day window tracks unchanged BC34 Hydrapple 57436919 and the new league package. Include every available field attempt created in [start,end), including errors/unfinished episodes; exclude validation self-play. Retain exact opponent versions, timestamps, outcomes and available first-game/chooser fields; explicitly record missing data. Dated publicScore snapshots do not expose rating uncertainty.', '',
      'Compare contemporaneous common-opponent coverage only where supported; report exact opponent identities, date/chooser support and coverage fractions. Adaptive matchmaking, field drift and unavailable records limit inference. Raw win rates or score differences do not prove causal improvement. No tuning, extension, automatic monitoring, training or reupload.', '',
      '## Manual handoff dates', '']
    if w:
        for k in ('window_start_utc','window_end_exclusive_utc','manual_early_health_check_utc','window_start_moscow','window_end_exclusive_moscow','manual_early_health_check_moscow'):lines.append(f'- {k}: **{w[k]}**')
    else:lines.append('Pending official hosted validation completion; dates will be written automatically if validation succeeds. No proxy completion timestamp is invented.')
    lines += ['', '## Costs and recovery', '',f'GPU usage: **zero**. Training: **zero**. Local accounting starts {START}; local checks capped at 90 minutes. Hosted polling capped at two hours, then PENDING and clean exit.',f'Actual costs: `{json.dumps(cost,default=str)}`.', '',
      'Private histories, logs and checkpoint/package assets remain ignored. Public recovery: progress.json, lifecycle.json, completion.json, failure.json when applicable, immutable submission-intent.json, receipt and validation evidence. Later status checks read these summaries and use read-only reconciliation; they do not rerun completed work.']
    REPORT.parent.mkdir(exist_ok=True);REPORT.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    status=REPO/'docs/STATUS.md';s=status.read_text(encoding='utf-8');marker='\n<!-- HYDRAPPLE-LEAGUE-SUBMISSION -->\n'
    if marker in s:s=s.split(marker)[0]
    s+=marker+f'\n## Frozen league Hydrapple external evaluation\n\n{iso()}: **{state}**. Submission {receipt.get("ref","not yet confirmed")}; hosted status {val.get("validation_result","pending checks")}. BC34 Hydrapple 57436919 is protected; G 57239424 may leave. No training or GPU use. See [report](../reports/astra-hydrapple-league-submission.md) and experiments/{RUN}/progress.json.\n'
    if w:s+=f'\nManual health: {w["manual_early_health_check_utc"]} ({w["manual_early_health_check_moscow"]} Moscow). Window end: {w["window_end_exclusive_utc"]} ({w["window_end_exclusive_moscow"]} Moscow). No automatic monitoring.\n'
    status.write_text(s,encoding='utf-8')

def mirror():
    import shutil
    win=Path(r'C:\path\to\ptcg-agent') if os.name=='nt' else Path('/path/to/ptcg-agent')
    for source in [REPORT,REPO/'docs/STATUS.md',REPO/'reports/astra-stage-i-hydrapple-submission.md']+list(ROOT.rglob('*')):
        if source.is_file():
            target=win/source.relative_to(REPO);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
