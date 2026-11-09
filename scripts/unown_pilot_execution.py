"""Small, testable execution contracts. No network or GPU work at import."""
import re,time,os,sys,subprocess
from urllib.parse import urlparse

class StatusUnavailable(RuntimeError):pass

def validate_budget(prior_seconds,session_seconds,allocated_devices):
 if prior_seconds<0 or session_seconds<=0 or allocated_devices!=2 or prior_seconds+session_seconds*allocated_devices>7200:
  raise ValueError('Continuation exceeds original7200-second aggregate allowance')

def check_experiment_receipt(receipt,diagnostic):
 if receipt.get('state')!='COMPLETE':
  raise RuntimeError('Remote experiment FAILED: '+str(diagnostic.get('error',receipt.get('error')))+'; provider COMPLETE only means the wrapper exported evidence')

def canonical_launch(receipt,expected_owner):
 """The returned receipt, never the requested slug, identifies the job."""
 refs=[]
 if receipt.get('ref'):
  value=receipt['ref'].removeprefix('/code/').strip('/')
  if not re.fullmatch(r'[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+',value):raise ValueError('Invalid receipt ref')
  refs.append(value)
 if receipt.get('url'):
  url=urlparse(receipt['url'])
  if url.scheme!='https' or url.hostname not in ['www.kaggle.com','kaggle.com'] or not url.path.startswith('/code/'):raise ValueError('Invalid receipt URL')
  refs.append(url.path.removeprefix('/code/').strip('/'))
 if not refs or any(not re.fullmatch(r'[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+',v) for v in refs) or len(set(refs))!=1 or refs[0].split('/')[0]!=expected_owner:raise ValueError('Missing/conflicting receipt identity')
 version=receipt.get('versionNumber');kid=receipt.get('kernelId')
 if not isinstance(version,int) or version<1 or not isinstance(kid,int) or kid<1:raise ValueError('Missing version/kernel ID')
 return dict(kernel=refs[0],version=version,kernel_id=kid)

def validate_kernel_metadata(meta):
 slug=re.sub(r'[^a-z0-9]+','-',meta['title'].lower()).strip('-')
 if meta['id'].split('/')[-1]!=slug:raise ValueError('Kernel title/slug mismatch')
 if meta.get('machine_shape')!='NvidiaTeslaT4':raise ValueError('Explicit supported T4 allocation required')
 if meta.get('is_private') is not True or meta.get('enable_internet') is not False:raise ValueError('Private offline job required')
 if '@sha256:' not in meta.get('docker_image',''):raise ValueError('Pinned image required')

def wait_for_terminal(api,identity,cap_seconds,emit,clock=time.monotonic,sleep=time.sleep,interval=15,max_errors=3):
 started=clock();errors=0
 while clock()-started<cap_seconds:
  try:
   response=api.kernels_status(identity['kernel']).to_dict();status=response.get('status','').upper()
  except Exception as error:
   errors+=1;emit('monitor-error',dict(kernel=identity['kernel'],error_type=type(error).__name__,consecutive=errors))
   if errors>=max_errors:raise StatusUnavailable(f'Status unavailable after{errors} attempts for receipt job {identity["kernel"]}; no retry launch') from None
  else:
   errors=0;emit('gpu-status',dict(kernel=identity['kernel'],version=identity['version'],response=response))
   if status in ['COMPLETE','ERROR','CANCELLED','CANCELED']:return response
   if status not in ['QUEUED','RUNNING']:raise StatusUnavailable('Unknown provider status: '+status)
  sleep(min(interval,max(0,cap_seconds-(clock()-started))))
 raise TimeoutError('Receipt job did not terminate within the registered supervision cap')

def capture_cuda(torch,save,probe=None,physical=None):
 """Persist details BEFORE rejecting a runtime, including unavailable CUDA."""
 result=dict(python=sys.version,torch_version=str(torch.__version__),CUDA_build=torch.version.cuda,
  visible_device_environment=os.environ.get('CUDA_VISIBLE_DEVICES'),status='CHECKING')
 save(result.copy())
 try:
  result['CUDA_available']=bool(torch.cuda.is_available());result['visible_device_count']=int(torch.cuda.device_count())
  result['devices']=[dict(index=i,name=torch.cuda.get_device_name(i),capability=list(torch.cuda.get_device_capability(i))) for i in range(result['visible_device_count'])]
  result['compiled_architectures']=list(torch.cuda.get_arch_list())
  if physical is not None:result['physical_allocation']=physical()
  save(result.copy())
  if not result['CUDA_available']:raise RuntimeError('CUDA unavailable; see gpu-runtime.json')
  if result['visible_device_count']!=1:raise RuntimeError(f'Expected one visible device, found{result["visible_device_count"]}; see gpu-runtime.json')
  if 'T4' not in result['devices'][0]['name']:raise RuntimeError('Unexpected GPU model; registered T4 required')
  if probe is not None:probe()
  result.update(status='PASS',CUDA_operation_probe='PASS');save(result.copy());return result
 except BaseException as error:
  result.update(status='FAILED',error_type=type(error).__name__,error=str(error));save(result.copy());raise

def physical_inventory():
 try:
  result=subprocess.run(['nvidia-smi','--query-gpu=index,name,memory.total,driver_version','--format=csv,noheader'],capture_output=True,text=True,timeout=10)
  return dict(returncode=result.returncode,stdout=result.stdout[:4000],stderr=result.stderr[:1000])
 except Exception as error:return dict(error_type=type(error).__name__)

def cuda_startup(torch,save):
 def probe():
  # Fixed integers: verifies a real CUDA kernel without advancing policy RNG.
  with torch.no_grad():
   cpu=torch.arange(256,dtype=torch.float32).reshape(16,16);gpu=cpu.to('cuda')
   actual=(gpu@gpu.T).cpu();torch.cuda.synchronize()
   torch.testing.assert_close(actual,cpu@cpu.T,atol=0,rtol=0)
 return capture_cuda(torch,save,probe,physical_inventory)
