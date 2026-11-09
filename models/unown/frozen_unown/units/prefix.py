"""Reusable declarations extracted from unown_repair_execution.py; dependencies injected per run."""

def host_length(packets):
 lengths=[]
 for packet in packets:
  ints=packet.row[slice(*packet.layout.offsets['tok_int'])].reshape(packet.layout.tokens,3+packet.layout.tools)
  valid=ints[:,1]>0;length=int(valid.sum())
  assert valid[:length].all() and not valid[length:].any(),'Noncontiguous legal token prefix'
  lengths.append(length)
 return max(lengths)


@functools.lru_cache(maxsize=256)
def compiled(layout,length):
 original=runtime.rt.functions(layout)[0]
 namespace=dict(original.__globals__);namespace['_host_length']=length
 source=textwrap.dedent(inspect.getsource(runtime.rt.reference()[1].PTCGTransformer._core))
 tools='    x = self.token_proj(torch.cat(['
 assert source.count(tools)==1
 source=source.replace(tools,'    for extra_tool in range(5, TOK_INT):\n        et = tok_i[:, :, extra_tool]\n        tool_vec = tool_vec + card_vec[et] * (et > 0).unsqueeze(-1)\n'+tools)
 needle='L = int(valid.sum(1).max().item())';assert source.count(needle)==1
 source=source.replace(needle,'L = _host_length')
 exec(compile(source,'<Unown-exact-host-prefix>','exec'),namespace)
 return namespace['_core']


def installed_core(model,packet):
 length=getattr(packet,'valid_length',None)
 if length is None:
  length=host_length([r['packet'] for r in packet.rows] if hasattr(packet,'rows') else [packet])
 tensor=packet.tensor(next(model.parameters()).device)
 return compiled(packet.layout,length)(model,tensor),tensor


def install(model):
 model.core=types.MethodType(installed_core,model)
 return model


class DevicePacket:
 def __init__(self,rows,device,dtype=torch.float32):
  self.rows=rows;self.layout=rows[0]['packet'].layout
  assert all(r['packet'].layout==self.layout for r in rows)
  self.valid_length=host_length([r['packet'] for r in rows])
  self.value=torch.as_tensor(np.stack([r['packet'].row for r in rows]),device=device,dtype=dtype)
 def tensor(self,device):
  assert self.value.device==torch.device(device);return self.value

