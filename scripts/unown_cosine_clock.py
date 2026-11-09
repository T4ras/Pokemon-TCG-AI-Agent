"""Read-only Windows awake clock: no power settings or sleep prevention."""
import threading
from unown_cosine_common import *
def projected_awake(sample,utc,last):
 age=max(0,(utc-parse(sample['at_utc'])).total_seconds());interpolate=age if age<=2 else 0.
 return max(last,float(sample['awake_seconds'])+interpolate)
class AwakeClock:
 def __init__(self):self.last=0.;self.loaded_at=-1.;self.sample=None
 def __call__(self):
  if time.monotonic()-self.loaded_at>.2:
   start=time.monotonic()
   while True:
    self.sample=read(OUT/'power-clock.json')
    if (now()-parse(self.sample['at_utc'])).total_seconds()<30:break
    if time.monotonic()-start>15:raise RuntimeError('Awakeclocktelemetryunavailable; preserveattempt,no silent timeout bypass')
    time.sleep(.2)
   self.loaded_at=time.monotonic()
  self.last=projected_awake(self.sample,now(),self.last);return self.last
class RuntimeClock:
 def __init__(self,clock):self.clock=clock
 def perf_counter(self):return self.clock()
 def monotonic(self):return self.clock()
 def __getattr__(self,key):return getattr(time,key)
def awake_windows():
 import ctypes
 value=ctypes.c_ulonglong();assert ctypes.windll.kernel32.QueryUnbiasedInterruptTime(ctypes.byref(value));return value.value/1e7
def power_windows():
 import ctypes
 class State(ctypes.Structure):_fields_=[('ACLineStatus',ctypes.c_ubyte),('BatteryFlag',ctypes.c_ubyte),('BatteryLifePercent',ctypes.c_ubyte),('SystemStatusFlag',ctypes.c_ubyte),('BatteryLifeTime',ctypes.c_ulong),('BatteryFullLifeTime',ctypes.c_ulong)]
 s=State();ok=ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(s));return {k:getattr(s,k) for k,_ in s._fields_} if ok else {'status':'unavailable'}
def telemetry(stop,fast=None,wakeup=None):
 previous=None;lastlog=None
 while not stop.is_set():
  try:
   import ctypes
   ctypes.windll.kernel32.GetTickCount64.restype=ctypes.c_ulonglong
   sample=dict(at_utc=iso(),awake_seconds=awake_windows(),uptime_seconds=ctypes.windll.kernel32.GetTickCount64()/1000,power=power_windows(),controller_pid=os.getpid());write(OUT/'power-clock.json',sample)
   gap=None
   if previous:gap=max(0,(sample['uptime_seconds']-previous['uptime_seconds'])-(sample['awake_seconds']-previous['awake_seconds']))
   if lastlog is None or (parse(sample['at_utc'])-lastlog).total_seconds()>=60 or gap and gap>2 or previous and sample['power']!=previous['power']:
    append(OUT/'power-context.jsonl',dict(sample,observed_OS_sleep_seconds_since_previous=gap));lastlog=parse(sample['at_utc'])
   previous=sample
  except Exception as e:
   try:append(OUT/'power-telemetry-errors.jsonl',dict(at_utc=iso(),type=type(e).__name__))
   except Exception:pass
  if wakeup:
   wakeup.wait(1 if fast and fast.is_set() else 30);wakeup.clear()
  else:stop.wait(1)
