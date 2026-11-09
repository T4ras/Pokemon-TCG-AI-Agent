"""Route reused online exact enumeration to the current registered deadline."""
from .common import guard,now,parse
import os,time
import unown_belief_full_training.baseline as baseline

def online_deadline():
 guard()
 absolute=os.environ.get('UNOWN_ABSOLUTE_DEADLINE')
 if absolute and now()>=parse(absolute):raise TimeoutError('Active belief session absolute deadline')
 start=os.environ.get('UNOWN_JOB_START')
 if start and time.monotonic()>=float(start)+43200-300:raise TimeoutError('Active belief session export reserve')

def install():
 baseline.check_time=online_deadline
 from unown_belief_correction.belief import OnlineBelief
 OnlineBelief.__init__.__defaults__=(online_deadline,)
