"""Reusable declarations extracted from unown_start_order_actor.py; dependencies injected per run."""

@dataclass
class StartingOrder:
 game:object=None
 first:object=None
 seat:object=None
 def reset(self):self.game=None;self.first=None;self.seat=None
 def observe(self,raw):
  if not raw.get('current'):self.reset();return None
  cur=raw['current'];game=cur.get('round',1);seat=cur['yourIndex'];first=cur['firstPlayer']
  assert seat in [0,1] and first in [-1,0,1]
  new=game!=self.game;old=self.first
  if new:self.game=game;self.first=None;self.seat=seat
  assert self.seat==seat,'Physical actor seat changed inside BO3'
  if first in [0,1]:
   if self.first is not None:assert self.first==first,'Starting order changed inside game'
   self.first=first
  else:assert self.first is None,'Public starting order became unknown within same game'
  value=seat if self.first is None else int(seat!=self.first)
  return dict(game=game,seat=seat,public_first=first,known_first=self.first,feature=value,new_game=new,previous_game_first=old if new else None,
   transition='new_game_unknown' if new and first<0 else 'new_game_known' if new else 'order_established' if old is None and first>=0 else 'stable')


def replace_feature(packet,state):
 assert state.seat in [0,1]
 lo,_=packet.layout.offsets['global_f'];assert packet.row[lo+31]==state.seat,'Original physical-seat channel differs'
 value=state.seat if state.first is None else int(state.seat!=state.first)
 if value==state.seat:return packet
 out=copy.copy(packet);out.row=packet.row.copy();out.row[lo+31]=value
 return out


class PolicyView:
 def __init__(self,model,state,callback=None):self.original=model;self.state=state;self.callback=callback
 def __getattr__(self,name):return getattr(self.original,name)
 def core(self,packet):
  changed=replace_feature(packet,self.state)
  out=self.original.core(changed)
  if self.callback:self.callback(packet,changed,out)
  return out


def actor_class(runtime,context):
 base=runtime.PilotActor
 class OrderActor(base):
  def __init__(self,model,*a,**kw):
   super().__init__(model,*a,**kw);self.candidate=model is context['candidate_model'];self.order=StartingOrder();self.feature_rows=[];self.transitions=[];self.current=None;self.sequence=0;self.prefixes=[]
   self.handle=context['history_handle'](self) if context.get('retain_history') else None
   if self.candidate:self.model=PolicyView(model,self.order,self.packet_observed)
  def packet_observed(self,before,after,out):
   lo,_=before.layout.offsets['global_f'];detail=dict(original=float(before.row[lo+31]),used=float(after.row[lo+31]),changed=before is not after)
   if self.handle and before is not after and context.get('example_count',0)<16 and getattr(self.model.original,'diagnostic_forwards_allowed',True):
    import torch
    with torch.no_grad():old=self.model.original.core(before)[0]['logits'][0];new=out[0]['logits'][0]
    a=int(old.argmax());b=int(new.argmax());df,_=before.layout.offsets['dec_float'];self.prefixes.append(dict(detail,original_action=a,candidate_action=b,disagrees=a!=b,STOP=after.layout.stop,picked_count=int(round(float(before.row[df+5])*5))))
    if a!=b:context['example_count']=context.get('example_count',0)+1
  def act(self,raw):
   if not raw.get('select'):
    self.order.reset();answer=super().act(raw)
    if self.handle:
     import json
     self.handle.write(json.dumps(dict(observation={'select':None},action=answer))+'\n')
    return answer
   info=self.order.observe(raw);self.current=runtime.rt.actor_view(raw);sel=self.current['select'];n=len(sel['option']);forced=sel['minCount']==sel['maxCount']==n or n==1 and sel['minCount']>=1
   used=info['feature'] if self.candidate else info['seat'];self.prefixes=[]
   row=dict(info,candidate=self.candidate,used_feature=used,context=sel['context'],turn=self.current['current']['turn'],forced=forced,sequence=self.sequence)
   if info['transition']!='stable':self.transitions.append(row)
   self.feature_rows.append(row);answer=super().act(raw);policy_answer=answer
   if context.get('actual_starter') is not None and sel['context']==41:
    answer=[next(i for i,o in enumerate(sel['option']) if o['type']==(1 if info['seat']==context['actual_starter'] else 2))]
    row['setup_override']=True;row['unmodified_policy_answer']=policy_answer
   if self.handle:
    import json
    for pref in self.prefixes:pref['prefix']=answer[:pref['picked_count']]
    self.handle.write(json.dumps(dict(observation=self.current,action=answer,feature=row,prefixes=self.prefixes,memory_game=self.memory.game,deck_lower_bounds=dict(self.memory.lower_bounds)),default=str)+'\n')
   self.sequence+=1
   if context.get('failure_ring') is not None:context['failure_ring'].append(dict(candidate=self.candidate,observation=self.current,action=answer,feature=row))
   return answer
 return OrderActor

