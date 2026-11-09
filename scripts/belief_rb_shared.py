from unown_belief_dev.common import *
import gzip,shutil,re,traceback,time,copy
run='astra-unown-belief-rao-blackwellization-001';out=R/'experiments'/run;pkg=R/'scripts/unown_belief_rb';pkg.mkdir(exist_ok=True)
for name in ('__init__.py','common.py','locations.py','adapter.py','state.py','proposal.py','filter.py','baseline.py','scoring.py'):
 source=R/'scripts/unown_belief_dev'/name
 (pkg/name).write_text(source.read_text().replace('unown_belief_dev','unown_belief_rb'),encoding='utf-8')
p=pkg/'common.py';s=p.read_text().replace("RUN='astra-unown-belief-development-repair-003'", "RUN='astra-unown-belief-rao-blackwellization-001'");s=s.replace("REPORT=R/'reports/astra-unown-belief-development-repair.md'", "REPORT=R/'reports/astra-unown-belief-rao-blackwellization.md'");p.write_text(s)
for name in ('native-effect-profiles.json','query-set.json','native-area-semantics.json'):shutil.copy2(OUT/name,out/name)
for name in ('adapter.py','baseline.py'):
 p=pkg/name;s=p.read_text();s=s.replace('self.effect=None;', 'self.effect=None;self.crispin_pending=False;')
 if name=='adapter.py':
  s=s.replace("if t==0:self.emit(dict(op='shuffle'),out)","if t==0:self.crispin_pending=False;self.emit(dict(op='shuffle'),out)")
  s=s.replace("elif t==10:self.emit(dict(op='move',src='H',dst='R',kind='available',cid=cid,sid=sid),out)","elif t==10:\n    self.crispin_pending=cid==1198\n    self.emit(dict(op='move',src='H',dst='R',kind='available',cid=cid,sid=sid),out)")
  s=s.replace("src=self.known.get(sid,(None,'H'))[1]", "src='D' if t==11 and self.crispin_pending else self.known.get(sid,(None,'H'))[1]\n    if t==11 and self.crispin_pending:self.crispin_pending=False")
  s=s.replace('profiles={86:4,1094:7,1122:7}', 'profiles={86:4,120:2,1094:7,1122:7}')
 else:
  s=s.replace("if t==0:self.top=[];self.bottom=[];self.bottom_blocks=[]", "if t==0:self.crispin_pending=False;self.top=[];self.bottom=[];self.bottom_blocks=[]")
  s=s.replace("elif t==10:self.transfer('H','R',cid,sid)", "elif t==10:\n    self.crispin_pending=cid==1198\n    self.transfer('H','R',cid,sid)")
  s=s.replace("domain=self.copies.get(sid,(None,{'H'}))[1];src=next(iter(domain)) if len(domain)==1 else 'H';self.transfer(src,'B',cid,sid);self.setup_basic=None", "domain=self.copies.get(sid,(None,{'H'}))[1];src='D' if t==11 and self.crispin_pending else next(iter(domain)) if len(domain)==1 else 'H'\n    if t==11 and self.crispin_pending:self.crispin_pending=False\n    self.transfer(src,'B',cid,sid);self.setup_basic=None")
 p.write_text(s)
from unown_belief_rb.baseline import Facts,baseline
from unown_belief_rb.adapter import Compiler
res=[];started=time.process_time()
for item in read(out/'development-inputs.json'):
 h=json.load(gzip.open(R/item['corpus']['legal_path'],'rt'));f=Facts(metadata()['cards']);c=Compiler(metadata()['cards']);unsupported=None;error=None
 for i,e in enumerate(h):
  try:
   f.observe(e['observation']);f.known_hand()
   if not unsupported:
    try:c.compile(e['observation'])
    except UnsupportedEvent as x:unsupported=dict(index=i,message=str(x))
  except Exception as x:error=dict(index=i,type=type(x).__name__,message=str(x),traceback=traceback.format_exc());break
 r=dict(id=item['id'],callbacks=len(h),processed=i+1,shared_error=error,compiler_unsupported=unsupported);res.append(r);print(json.dumps(r),flush=True)
write(out/'shared-readiness.json',dict(at_utc=iso(),all_shared_pass=all(not x['shared_error'] for x in res),CPU_seconds=time.process_time()-started,records=res,new_native_matches=0,held_out_accessed=False))
