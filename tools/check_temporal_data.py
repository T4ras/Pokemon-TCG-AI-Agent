"""Reject saved temporal literals in this public export; print paths, not values."""
from pathlib import Path
import io,json,re,tokenize,sys,os
ROOT=Path(__file__).resolve().parents[1]
MONTH=r'(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)'
RULES={
 'calendar date':re.compile(r'(?<![A-Za-z0-9])(?:19|20)\d{2}[-_/](?:0?[1-9]|1[0-2])[-_/](?:0?[1-9]|[12]\d|3[01])(?:[Tt _]\d\d:\d\d(?::\d\d)?(?:\.\d+)?(?:Z|[+-]\d\d:?\d\d)?)?(?!\d)'),
 'numeric calendar date':re.compile(r'(?<![A-Za-z0-9])(?:0?[1-9]|[12]\d|3[01])[-/.](?:0?[1-9]|1[0-2])[-/.](?:19|20)\d{2}(?!\d)'),
 'named calendar date':re.compile(r'\b'+MONTH+r'[ .-]*\d{1,2}(?:st|nd|rd|th)?(?:,?[ .-]*(?:19|20)\d{2})?',re.I),
 'reverse named date':re.compile(r'\b\d{1,2}(?:st|nd|rd|th)?[ .-]*'+MONTH+r'(?:,?[ .-]*(?:19|20)\d{2})?\b',re.I),
 'month and year':re.compile(r'\b'+MONTH+r'[ .-]+(?:19|20)\d{2}\b',re.I),
 'date-derived number':re.compile(r'(?<![A-Za-z0-9_.])20[12]\d\d{4,10}(?![A-Za-z0-9_])'),
 'calendar year':re.compile(r'(?<![A-Za-z0-9])20[12]\d(?![A-Za-z0-9])'),
 'embedded calendar year':re.compile(r'(?<=[A-Za-z_-])20[12]\d(?=[A-Za-z_.-])'),
}
CLOCK=re.compile(r'(?<![\d:])(?:[01]?\d|2[0-3]):[0-5]\d(?::[0-5]\d)?(?:\.\d+)?(?:\s*(?:UTC|GMT|MSK|Moscow|AM|PM|Z))?(?![\d:])',re.I)
DROP=re.compile(r'(?i)^(?:.*(?:_at_utc|_at_msk|_timestamp|_datetime)|at_utc|utc|timestamp|datetime|registered_date|prepared_date|created_at|updated_at|started_at|ended_at|finished_at|deadline_utc|original_UTC_deadline|absolute_deadline|original_deadline|deadline)$')
issues=[];count=0
def check(text,name,start=1,clock=True):
 for kind,pattern in RULES.items():
  for m in pattern.finditer(text):issues.append(dict(file=name,kind=kind,line=start+text[:m.start()].count('\n')))
 if clock:
  for m in CLOCK.finditer(text):issues.append(dict(file=name,kind='clock value',line=start+text[:m.start()].count('\n')))
def export_files():
 for base,dirs,names in os.walk(ROOT):
  dirs[:]=[d for d in dirs if d not in {'.git','.venv','__pycache__','.pytest_cache'} and not d.endswith('.egg-info')]
  for name in names:
   if not name.endswith(('.pyc','.pyo')):yield Path(base)/name
for p in sorted(export_files()):
 if not p.is_file():continue
 name=p.relative_to(ROOT).as_posix();count+=1
 check(name,name,clock=False)
 t=p.read_text(encoding='utf-8')
 if p.suffix=='.py':
  for tok in tokenize.generate_tokens(io.StringIO(t).readline):
   if tok.type in (tokenize.STRING,tokenize.COMMENT) or tokenize.tok_name.get(tok.type)=='FSTRING_MIDDLE':check(tok.string,name,tok.start[0])
   elif tok.type==tokenize.NUMBER:check(tok.string,name,tok.start[0],clock=False)
 elif p.suffix=='.json':
  def walk(x):
   if isinstance(x,dict):
    for k,v in x.items():
     if DROP.fullmatch(k):issues.append(dict(file=name,kind='saved temporal metadata field'))
     # Hashes and byte counts are not temporal representations.
     if k not in ('sha256','bytes'):walk(v)
   elif isinstance(x,list):
    for v in x:walk(v)
   elif isinstance(x,str):check(x,name)
   elif isinstance(x,int) and not isinstance(x,bool):check(str(x),name,clock=False)
  walk(json.loads(t))
 else:check(t,name)
print(json.dumps(dict(files=count,issues=issues,passed=not issues,scope='Saved date/clock literals and temporal JSON fields in names and text; source time-measurement functions and filesystem metadata are not saved historical date values.'),indent=2))
sys.exit(bool(issues))

