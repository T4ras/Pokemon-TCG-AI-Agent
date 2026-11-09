"""Audit this source-only publication folder without displaying secret values."""
from pathlib import Path
import ast,json,re,sys,os
ROOT=Path(__file__).resolve().parents[1]
FORBIDDEN={'.pt','.pth','.ckpt','.safetensors','.onnx','.npy','.npz','.pkl','.pickle','.bin','.so','.dll','.zip','.tar','.gz','.log','.jsonl','.xml','.pid','.pyc'}
PATTERNS={
 'private key':r'-----BEGIN (?:RSA |EC |OPENSSH |DSA |ENCRYPTED )?PRIVATE KEY-----',
 'GitHub credential':r'(?:github_pat_[A-Za-z0-9_]{20,}|gh[pousr]_[A-Za-z0-9]{20,})',
 'cloud access key':r'(?:AKIA|ASIA)[0-9A-Z]{16}',
 'model API credential':r'\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{20,}',
 'Kaggle credential':r'KGAT_[A-Za-z0-9_-]{12,}',
 'Google API credential':r'AIza[A-Za-z0-9_-]{30,}',
 'Slack credential':r'xox[baprs]-[A-Za-z0-9-]{15,}',
 'password hash':r'\$2[aby]\$[0-9]{2}\$[./A-Za-z0-9]{53}|\$argon2(?:id|i|d)\$|pbkdf2_sha256\$[0-9]+\$',
 'JWT':r'\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}',
 'generic credential assignment':r'''(?i)["']?(?:KAGGLE_KEY|KAGGLE_API_TOKEN|OPENAI_API_KEY|api[_-]?key|api[_-]?token|access[_-]?token|refresh[_-]?token|client[_-]?secret|password|passwd)["']?\s*[:=]\s*["']([^"'\r\n]{8,})["']''',
}
def export_files():
 for base,dirs,names in os.walk(ROOT):
  dirs[:]=[d for d in dirs if d not in {'.git','.venv','__pycache__','.pytest_cache'} and not d.endswith('.egg-info')]
  for name in names:
   if not name.endswith(('.pyc','.pyo')):yield Path(base)/name
issues=[];count=0;py=0
for p in sorted(export_files()):
 if not p.is_file():continue
 rel=p.relative_to(ROOT);name=rel.as_posix()
 if any(x in rel.parts for x in ['.git','.venv','data','artifacts','private','__pycache__','.pytest_cache']):
  issues.append(dict(file=name,kind='excluded directory'));continue
 count+=1
 if p.suffix.lower() in FORBIDDEN or p.name in {'.env','kaggle.json','auth.json'}:
  issues.append(dict(file=name,kind='excluded file type'));continue
 try:t=p.read_text(encoding='utf-8')
 except UnicodeDecodeError:
  issues.append(dict(file=name,kind='unexpected binary'));continue
 if p.suffix=='.py':
  py+=1
  try:ast.parse(t,filename=name)
  except SyntaxError as e:issues.append(dict(file=name,kind='Python syntax',line=e.lineno))
 for kind,pattern in PATTERNS.items():
  for m in re.finditer(pattern,t):
   # Historical redaction regression fixtures deliberately use this dummy.
   if kind=='generic credential assignment' and m.group(1)=='secretvalue' and name=='tests/test_experiment_supervisor.py':continue
   issues.append(dict(file=name,kind=kind,line=t[:m.start()].count('\n')+1))
 if p.suffix=='.md':
  for m in re.finditer(r'\[[^\]\n]+\]\(([^)\n]+)\)',t):
   dest=m.group(1).strip().strip('<>')
   if dest.startswith(('http:','https:','mailto:','#')):continue
   if not (p.parent/dest.split('#',1)[0]).exists():issues.append(dict(file=name,kind='missing local link',target=dest))
# Missing source links in machine-readable experiment index are also errors.
idx=ROOT/'experiments/index.json'
if idx.exists():
 for e in json.loads(idx.read_text()):
  for dest in [e['report']]+e['implementation']+([e['registration']] if 'registration' in e else []):
   if not (ROOT/dest).exists():issues.append(dict(file='experiments/index.json',kind='missing indexed target',target=dest))
import subprocess
temporal=subprocess.run([sys.executable,str(ROOT/'tools/check_temporal_data.py')],capture_output=True,text=True)
if temporal.returncode:
 temporal_result=json.loads(temporal.stdout)
 issues.extend(temporal_result['issues'])
result=dict(files=count,python_sources=py,issues=issues,passed=not issues,scope='Static source, file-type, local-link, saved-temporal-data and recognizable-credential-pattern checks; not proof that arbitrary credentials cannot exist.')
print(json.dumps(result,indent=2))
sys.exit(bool(issues))

