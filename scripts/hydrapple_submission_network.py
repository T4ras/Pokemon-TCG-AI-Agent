"""Windows-network worker. One upload call, then strictly read-only reconciliation."""
import sys,os,time,hashlib,zipfile,tarfile,io
from pathlib import Path
from hydrapple_submission_common import *
os.chdir(REPO)
sys.path.append(str(REPO/'.venv/lib/python3.12/site-packages'))
import requests
_send=requests.Session.send
def bounded(self,request,**kw):
    if not kw.get('timeout'):kw['timeout']=(10,45)
    return _send(self,request,**kw)
requests.Session.send=bounded

def api():
    from kagglesdk.kaggle_creds import KaggleCredentials
    if os.name=='nt':
        KaggleCredentials.DEFAULT_CREDENTIALS_FILE=r'\\wsl.localhost\Ubuntu\home\taras\.kaggle\credentials.json'
        save=KaggleCredentials.save
        KaggleCredentials.save=lambda self,file_path=None:save(self,KaggleCredentials.DEFAULT_CREDENTIALS_FILE)
    from kaggle.api.kaggle_api_extended import KaggleApi
    a=KaggleApi();a.authenticate();return a

def resource_limits(pages,rendered,checked_at):
    """Resolve documented API templates using fresh, retained rendered-page evidence."""
    faq=next(p['content'] for p in pages if p['name']=='Frequently Asked Questions')
    expected={'archive_limit_MiB':197.7,'RAM_GiB':12.2,'disk_GiB':11.8,'vCPUs':2,'daily_submission_limit':5,'active_submission_limit':2}
    assert rendered['competition']==SLUG
    assert rendered['url']=='https://www.kaggle.com/competitions/'+SLUG+'/overview/description'
    assert rendered['source']=='Live rendered Kaggle page via browser DOM snapshot'
    age=(parse(checked_at)-parse(rendered['checked_at_utc'])).total_seconds()
    assert 0<=age<=1800,'Rendered resource evidence stale; refresh the official page'
    assert hashlib.sha256(faq.encode()).hexdigest()==rendered['API_FAQ_template_sha256'],'Official FAQ template changed'
    for key,value in expected.items():assert rendered[key]==value,'Official resource limit changed: '+key
    if '${competition.' in faq:
        for field in ['SubmissionSizeLimit','AgentRam','AgentCpuCores','AgentDisk','NumScoredSubmissions','MaxDailySubmissions']:
            assert '${competition.'+field+'}' in faq,'Unexpected FAQ placeholder structure'
    else:assert '197.7' in faq and '12.2' in faq,'Numeric FAQ disagrees with rendered evidence'
    return dict(expected,verification='Fresh rendered official FAQ plus matching current API template',rendered_at_utc=rendered['checked_at_utc'])

def requirements(a):
    checked=iso();pages=[p.to_dict() for p in a.competition_list_pages(SLUG)]
    write(PRIVATE/'official-pages.json',dict(at_utc=checked,pages=pages))
    evaluation=next(p['content'] for p in pages if p['name']=='Evaluation')
    assert 'latest 2' in evaluation and '5 agents' in evaluation,'Official active-slot/allowance rule changed'
    limits=resource_limits(pages,read(ROOT/'rendered-requirements.json'),checked)
    from kagglesdk.competitions.types.competition_api_service import ApiGetCompetitionRequest
    with a.build_kaggle_client() as k:
        q=ApiGetCompetitionRequest();q.competition_name=SLUG
        competition=k.competitions.competition_api_client.get_competition(q).to_dict()
    write(PRIVATE/'current-competition.json',dict(at_utc=iso(),competition=competition))
    assert competition['id']==168019 and competition['userHasEntered'] and competition['evaluationMetric']=='cabt_bo3'
    assert not competition.get('submissionsDisabled',False) and competition['maxDailySubmissions']==limits['daily_submission_limit']
    assert parse(competition['deadline'])>now(),'Competition submission deadline passed'
    old=read(REPO/'experiments/astra-stage-h-hydrapple-readiness-001/official-requirements.json')
    reviewed=read(ROOT/'reviewed-official-differences.json') if (ROOT/'reviewed-official-differences.json').exists() else {}
    current={}
    for name,v in old['current_files'].items():
        response=requests.get(v['url'],timeout=(10,35));response.raise_for_status();data=response.content
        digest=hashlib.sha256(data).hexdigest();write(PRIVATE/('official-'+name+'.json'),dict(url=v['url'],sha256=digest,checked_at_utc=iso()))
        (PRIVATE/('official-'+name)).write_bytes(data)
        expected=reviewed.get(name,{}).get('sha256',v['sha256'])
        assert digest==expected,'Current official wrapper/image recipe changed: '+name
        current[name]=dict(url=v['url'],sha256=digest,stage_h_sha256=v['sha256'],reviewed_difference=reviewed.get(name))
    files=[];token=None
    for _ in range(10):
        result=a.competition_list_files(SLUG,page_token=token,page_size=1000).to_dict();files+=result['files'];token=result.get('nextPageToken')
        if not token:break
    assert not token,'Truncated official data listing'
    write(PRIVATE/'official-data-files.json',dict(at_utc=iso(),files=files))
    libraries=[x for x in files if x['name'].endswith('/cg/libcg.so')]
    folder=PRIVATE/'current-sdk';folder.mkdir(exist_ok=True)
    if libraries:
        native=[]
        for x in libraries:
            assert x['totalBytes']<4*1024**2
            before={p for p in folder.iterdir() if p.is_file()}
            a.competition_download_file(SLUG,x['name'],path=str(folder),force=True,quiet=True)
            paths=[p for p in folder.iterdir() if p.is_file() and p.name.startswith('libcg.so')]
            assert len(paths)==1,'Cannot identify downloaded native library'
            raw=paths[0].read_bytes()
            if zipfile.is_zipfile(paths[0]):
                with zipfile.ZipFile(paths[0]) as z:
                    matches=[n for n in z.namelist() if n.endswith('libcg.so')];assert len(matches)==1;raw=z.read(matches[0])
            digest=hashlib.sha256(raw).hexdigest();assert digest==old['latest_native_library_sha256'],'Official native SDK changed'
            native.append(dict(name=x['name'],sha256=digest))
    else:
        archives=[x for x in files if 'sample_submission' in x['name'].lower() and x['name'].lower().endswith('.zip')]
        assert len(archives)==1,'Cannot identify current official SDK native library'
        x=archives[0];assert x['totalBytes']<100*1024**2
        a.competition_download_file(SLUG,x['name'],path=str(folder),force=True,quiet=True)
        archive=next(folder.glob('*.zip'))
        with zipfile.ZipFile(archive) as z:
            names=[n for n in z.namelist() if n.endswith('/cg/libcg.so')];assert len(names)==1
            digest=hashlib.sha256(z.read(names[0])).hexdigest()
        assert digest==old['latest_native_library_sha256'],'Official SDK changed'
        native=[dict(name=x['name'],sha256=digest,download_sha256=sha(archive))]
    result=dict(checked_at_utc=checked,destination='https://www.kaggle.com/competitions/'+SLUG+'/submissions',pages_sha256=sha(PRIVATE/'official-pages.json'),current_files=current,native=native,resource_limits=limits,rendered_evidence_sha256=sha(ROOT/'rendered-requirements.json'),archive_limit_MiB=limits['archive_limit_MiB'],RAM_GiB=limits['RAM_GiB'],vCPUs=limits['vCPUs'],latest_active=limits['active_submission_limit'],official_page_sources=['https://www.kaggle.com/competitions/'+SLUG+'/overview/description'],unresolved='Hosted image digest/exact Torch build not exposed. Pinned wrapper is stricter on cumulative clock; current official wrapper/source hashes match inspected Stage H evidence. Hosted validation is mandatory.')
    write(ROOT/'official-requirements.json',result)

def preflight(a):
    requirements(a)
    # The mutable allowance/slot checks are deliberately last, immediately before upload.
    limits=a.competition_get_submission_limits(SLUG)
    subs=[s.to_dict() for s in a.competition_submissions(SLUG,page_size=100)]
    assert len(subs)<100,'Submission history truncated'
    episodes=a.competition_list_episodes(BC)
    teams={agent.team_id for e in episodes for agent in e.agents if agent.submission_id==BC};assert len(teams)==1
    team=teams.pop();assert team==16992760,'Unexpected account/team'
    active=[x.to_dict() for x in a.competition_team_submissions(team)]
    predicted=slot_guard(subs,active,limits.num_allowed_now)
    write(PRIVATE/'preflight-submissions-full.json',dict(at_utc=iso(),submissions=subs,active=active))
    pre=dict(at_utc=iso(),team_id=team,limits=dict(num_today=limits.num_today,num_total=limits.num_total,num_allowed_now=limits.num_allowed_now,limited_by_total=limits.limited_by_total),active=active,slot_prediction=predicted,archive_sha256=sha(ARCH),requirements_sha256=sha(ROOT/'official-requirements.json'))
    write(ROOT/'preflight.json',pre);return pre

def submit():
    assert not (ROOT/'submission-intent.json').exists(),'Prior upload intent; reconciliation only'
    local=read(ROOT/'local-completion.json');assert local['status']=='PASSED' and local['official_BO3']==6
    assert sha(ARCH)==local['archive_sha256']==read(ROOT/'archive.json')['sha256']
    assert (now()-parse(START)).total_seconds()<5400,'Local verification/upload preparation budget exceeded'
    a=api();pre=preflight(a)
    assert sha(ARCH)==pre['archive_sha256']
    assert (now()-parse(pre['at_utc'])).total_seconds()<30,'Slot preflight stale'
    assert (now()-parse(START)).total_seconds()<5400,'Local budget exceeded before upload'
    exclusive(ROOT/'submission-intent.json',dict(at_utc=iso(),description=DESCRIPTION,archive_sha256=sha(ARCH),bytes=ARCH.stat().st_size,competition=SLUG,attempt_number=1,preflight_sha256=sha(ROOT/'preflight.json'),protected_submission_id=BC,allowed_displacement=G,protocol_sha256=sha(ROOT/'external-protocol.json')))
    progress('SINGLE_UPLOAD_IN_PROGRESS',submission_occurred='intent saved; response pending')
    try:
        response=a.competition_submit(str(ARCH),DESCRIPTION,SLUG,quiet=True)
        receipt=response.to_dict();receipt.update(response_received_at_utc=iso(),archive_sha256=sha(ARCH),description=DESCRIPTION)
        write(ROOT/'submission-receipt.json',receipt)
        assert receipt.get('ref'),'Response lacks submission ID'
    except Exception as err:
        write(ROOT/'ambiguous-upload-response.json',dict(at_utc=iso(),error_type=type(err).__name__,action='Read-only reconciliation; no upload retry'))
        raise SystemExit(2)

def reconcile():
    assert (ROOT/'submission-intent.json').exists(),'No upload intent to reconcile'
    if optional('submission-receipt.json').get('ref'):return
    intent=read(ROOT/'submission-intent.json');a=api()
    records=[s.to_dict() for s in a.competition_submissions(SLUG,page_size=100)]
    write(PRIVATE/'upload-reconciliation.json',dict(at_utc=iso(),records=records))
    matches=[x for x in records if x.get('description')==DESCRIPTION and parse(x['date'])>=parse(intent['at_utc'])-dt.timedelta(minutes=2)]
    if len(matches)==1:
        x=matches[0];write(ROOT/'submission-receipt.json',dict(ref=x['ref'],response_received_at_utc=iso(),archive_sha256=intent['archive_sha256'],reconciled=True,official_submission_date=x['date'],description=DESCRIPTION))
    else:write(ROOT/'reconciliation-status.json',dict(at_utc=iso(),status='AMBIGUOUS',matching_records=len(matches),retry_permitted=False))

def amendment(active,at):
    ids={x['id'] for x in active};receipt=optional('submission-receipt.json');sid=receipt.get('ref')
    if ids=={BC,G}:return
    pre=read(ROOT/'preflight.json')
    if not (ROOT/'stage-i-amendment.json').exists():
        record=dict(recorded_at_utc=at,previous_active_ids=[BC,G],observed_active_ids=sorted(ids),new_submission_id=sid,BC34_preserved=BC in ids,G_left=G not in ids,transition_after_utc=pre['at_utc'],transition_at_or_before_utc=at,exact_server_slot_transition_time='Not exposed; bracketed by last preflight and first observed changed active snapshot',original_window_preserved=True,original_G_comparison='Intentionally curtailed, not a completed seven-day comparison',authorization='[removed] user explicitly permits G to leave and requires57436919 remain')
        exclusive(ROOT/'stage-i-amendment.json',record)
        p=REPO/'reports/astra-stage-i-hydrapple-submission.md'
        with p.open('a',encoding='utf-8') as f:f.write('\n\n## [removed] protocol amendment: league Hydrapple external evaluation\n\nThe user authorized a new frozen league phase-32 submission, permitting original G 57239424 to leave while preserving BC34 Hydrapple 57436919. The G-versus-BC34 observation period is intentionally curtailed; it is not a completed seven-day comparison. All earlier records and the original plan remain intact. Observed transition: '+json.dumps(record)+'. See [new receipt and protocol](astra-hydrapple-league-submission.md).\n')

def poll():
    a=api();receipt=read(ROOT/'submission-receipt.json');sid=int(receipt['ref']);at=iso()
    items=a.competition_submissions(SLUG,page_size=100);s=next(x for x in items if x.ref==sid)
    episodes=a.competition_list_episodes(sid);active=[x.to_dict() for x in a.competition_team_submissions(read(ROOT/'preflight.json')['team_id'])]
    state=dict(checked_at_utc=at,submission_id=sid,submission_status=s.status.name,error_description=s.error_description,active_ids=[x['id'] for x in active],BC34_preserved=BC in [x['id'] for x in active],validation_result='PENDING')
    write(PRIVATE/('validation-'+str(time.time_ns())+'.json'),dict(at_utc=at,submission=s.to_dict(),episodes=[e.to_dict() for e in episodes],active=active))
    amendment(active,at)
    vals=[e for e in episodes if e.type.name=='EPISODE_TYPE_VALIDATION'];state['validation_episodes']=[e.to_dict() for e in vals]
    failed=bool(s.error_description or s.status.name in ['ERROR','CANCELLED'] or any(e.state.name in ['ERRORED','NEVER_STARTED'] for e in vals))
    if BC not in state['active_ids'] or any(x not in [BC,G,sid] for x in state['active_ids']):
        state.update(validation_result='SLOT_CONFLICT',error_description='Unexpected active transition or protected BC34 missing; no further upload')
    elif failed:state['validation_result']='FAILED'
    elif s.status.name=='COMPLETE':
        completed=[e for e in vals if e.state.name=='COMPLETED' and e.to_dict().get('endTime') and all(x.submission_id==sid for x in e.agents)]
        if completed:
            e=min(completed,key=lambda e:parse(e.to_dict()['endTime']));w=window(e.to_dict()['endTime'],sid,e.id)
            w['protocol_sha256']=sha(ROOT/'external-protocol.json');w['archive_sha256']=receipt['archive_sha256']
            if (ROOT/'observation-window.json').exists():
                old=read(ROOT/'observation-window.json');assert old['window_start_utc']==w['window_start_utc'] and old['submission_id']==sid
            else:exclusive(ROOT/'observation-window.json',w)
            state.update(validation_result='PASSED',validation_completed_at_utc=w['window_start_utc'],validation_episode_id=e.id)
            # The window is now frozen, before field metadata or ratings are interpreted.
            write(PRIVATE/'initial-candidate-episodes.json',dict(at_utc=at,submission_id=sid,episodes=[x.to_dict() for x in episodes]))
            try:write(PRIVATE/'initial-BC34-episodes.json',dict(at_utc=iso(),submission_id=BC,episodes=[x.to_dict() for x in a.competition_list_episodes(BC)]))
            except Exception as err:state['baseline_metadata_retrieval_error']=type(err).__name__
            write(ROOT/'dated-rating-snapshot.json',dict(at_utc=at,source='Official team public submissions API',records=[dict(submission_id=x['id'],publicScore=x.get('publicScore'),rating_uncertainty=None,uncertainty_status='Not exposed; no mu/sigma inferred') for x in active],field_results_interpreted=False))
            if set(state['active_ids'])!={BC,sid}:state.update(validation_result='SLOT_CONFLICT',error_description='Validation completed but expected active pair not confirmed')
        else:state['validation_result']='COMPLETE_WAITING_FOR_VALIDATION_TIMESTAMP'
    write(ROOT/'validation.json',state)
    if failed:
        for e in vals:
            for agent in e.agents:
                if agent.submission_id==sid:
                    try:a.competition_episode_agent_logs(e.id,agent.index,path=str(PRIVATE),quiet=True)
                    except Exception as err:write(ROOT/f'log-retrieval-error-{e.id}-{agent.index}.json',dict(at_utc=iso(),error_type=type(err).__name__))
    print(json.dumps(state),flush=True)

if __name__=='__main__':
    try:{'submit':submit,'reconcile':reconcile,'poll':poll,'preflight':lambda:preflight(api())}[sys.argv[1]]()
    except Exception as err:
        write(ROOT/'network-error.json',dict(at_utc=iso(),operation=sys.argv[1],error_type=type(err).__name__,reason=str(err)[:700] if isinstance(err,AssertionError) else 'Read private log; no secrets in public summary'))
        raise
