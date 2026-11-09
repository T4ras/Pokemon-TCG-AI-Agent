"""Verified replacement operations; ambiguous origin never becomes deck evidence."""
from .common import *

class UnknownProvenance(UnsupportedEvent):pass

# These effects exchange two original, same-owner physical copies.
PROFILES={106:('D','D'),1090:('B','B'),1415:('B','B')}
def validate_identity(cid,sid,known,allowed):
 if cid not in allowed:raise UnknownProvenance('OUTSIDE_LIBRARY_OR_UNMODELED_PROVENANCE: cannot attribute card '+str(cid)+' to the starting deck')
 if sid in known and known[sid][0]!=cid:raise UnknownProvenance('SAME_SERIAL_CHANGED_IDENTITY: explicit origin/transform support required')

def replacement(log,known,effect):
 if effect not in PROFILES:raise UnknownProvenance('UNRESOLVED_REPLACEMENT_SOURCE: effect '+str(effect))
 required=('cardIdBefore','serialBefore','cardIdAfter','serialAfter')
 if any(k not in log for k in required):raise UnknownProvenance('Incomplete native CHANGE provenance')
 before,bs,after,as_=(log[k] for k in required)
 if bs==as_:raise UnknownProvenance('CHANGE reused a physical serial; not the verified native replacement')
 if bs not in known or known[bs][0]!=before:raise UnknownProvenance('Replacement board source is not legally established')
 zone=known[bs][1]
 if not isinstance(zone,str):
  if len(zone)!=1:raise UnknownProvenance('Replacement board source is ambiguous')
  zone=next(iter(zone))
 if zone not in ('A','B'):raise UnknownProvenance('Replacement source is not on the known public board')
 src,dst=PROFILES[effect]
 if as_ in known:
  identity,where=known[as_];where={where} if isinstance(where,str) else set(where)
  if identity!=after or src not in where:raise UnknownProvenance('Replacement target contradicts its established source')
 # Public trash/bench/attachments share coarse B storage in this observer.
 # Keep the two physical references distinct; never rename a token.
 return [dict(op='move',src=zone,dst=dst,kind='available',cid=before,sid=bs,position='top' if dst=='D' else 'append'),dict(op='move',src=src,dst=zone,kind='available',cid=after,sid=as_,position='append')]
