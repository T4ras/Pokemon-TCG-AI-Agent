"""Small crash-safe storage primitives; no credential-file access."""
import contextlib, datetime as dt, hashlib, json, os, re, tempfile
from pathlib import Path

def utc(): return dt.datetime.now(dt.timezone.utc).isoformat()
def canonical(v): return json.dumps(v, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()
def digest(v): return hashlib.sha256(canonical(v)).hexdigest()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sync_dir(p):
 if os.name!='nt':
  fd=os.open(p,os.O_RDONLY)
  try: os.fsync(fd)
  finally: os.close(fd)
def atomic(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 fd,name=tempfile.mkstemp(prefix='.'+p.name+'.',dir=p.parent)
 try:
  with os.fdopen(fd,'wb') as f: f.write(canonical(v)+b'\n');f.flush();os.fsync(f.fileno())
  os.replace(name,p);sync_dir(p.parent)
 finally:
  if os.path.exists(name): os.unlink(name)
def exclusive(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 # Publish a fully fsynced file without replacing an existing receipt.
 # Windows rename is non-replacing and works on WSL UNC paths; Windows
 # hard-link creation on those paths is unsupported. POSIX uses link().
 fd,name=tempfile.mkstemp(prefix='.intent-',dir=p.parent)
 try:
  with os.fdopen(fd,'wb') as f:f.write(canonical(v)+b'\n');f.flush();os.fsync(f.fileno())
  if os.name=='nt':os.rename(name,p)
  else:os.link(name,p)
  sync_dir(p.parent)
 finally:
  if os.path.exists(name):os.unlink(name)
def relative(root,name):
 root=Path(root).resolve();p=root/name
 if p.is_symlink() or any(x.is_symlink() for x in p.parents if x!=root.parent):raise ValueError('Symlink in evidence/source path')
 p=p.resolve()
 if not p.is_relative_to(root):raise ValueError('Path escapes registered root')
 return p
def manifest(root):
 root=Path(root);out={}
 for p in sorted(root.rglob('*')):
  if p.is_symlink():raise ValueError('Symlink in execution')
  if p.is_file() and not any(x in ('__pycache__','.pytest_cache','.git') for x in p.relative_to(root).parts):out[p.relative_to(root).as_posix()]=sha(p)
 return out
def scrub(text,limit=8192):
 text=str(text)
 text=re.sub(r'(?i)(authorization\s*:\s*bearer\s+|(?:api[_-]?key|access[_-]?token|refresh[_-]?token)\s*[=:]\s*)\S+',r'\1[REDACTED]',text)
 text=re.sub(r'''(?i)((?:api[_-]?key|access[_-]?token|refresh[_-]?token|id[_-]?token)["']?\s*[:=]\s*["']?)[^"',\s}]+''',r'\1[REDACTED]',text)
 text=re.sub(r'\bsk-[A-Za-z0-9_-]+','[REDACTED]',text)
 return text[-limit:]
class Lock:
 """OS-held lock: automatic release on process death; use local disk on Windows."""
 def __init__(self,p):self.p=Path(p)
 def __enter__(self):
  self.p.parent.mkdir(parents=True,exist_ok=True);self.f=self.p.open('a+b')
  if self.f.seek(0,2)==0:self.f.write(b'0');self.f.flush()
  self.f.seek(0)
  try:
   if os.name=='nt':
    import msvcrt;msvcrt.locking(self.f.fileno(),msvcrt.LK_NBLCK,1)
   else:
    import fcntl;fcntl.flock(self.f,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except OSError:self.f.close();raise RuntimeError('Another supervisor owns this experiment')
  return self
 def __exit__(self,*a):
  if os.name=='nt':
   import msvcrt;self.f.seek(0);msvcrt.locking(self.f.fileno(),msvcrt.LK_UNLCK,1)
  self.f.close()
