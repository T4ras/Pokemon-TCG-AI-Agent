"""Recurrent visible-state transformer and a joint-action pointer policy."""
import math
import torch
from torch import nn
from torch.distributions import Categorical
from .observations import WIDTH, SCHEMA, SEMANTIC_SCHEMA

D = 192

def collate(rows, device):
    schemas = {r.get("schema", SCHEMA) for r in rows}
    if len(schemas) != 1 or not schemas <= {SCHEMA, SEMANTIC_SCHEMA}:
        raise ValueError("Mixed or unknown observation schemas")
    semantic = schemas == {SEMANTIC_SCHEMA}
    batch=len(rows);S=max(len(r["ids"]) for r in rows);O=max(len(r["option_ids"]) for r in rows)
    result={}
    for key in ("ids","kinds"):
        a=torch.zeros(batch,S,dtype=torch.long)
        for i,r in enumerate(rows):a[i,:len(r[key])]=torch.tensor(r[key])
        result[key]=a.to(device)
    a=torch.zeros(batch,S,WIDTH)
    for i,r in enumerate(rows):a[i,:len(r["features"])]=torch.tensor(r["features"])
    result["features"]=a.to(device)
    result["state_mask"]=torch.arange(S,device=device)[None,:]>=torch.tensor([len(r["ids"]) for r in rows],device=device)[:,None]
    for key in ("option_ids","targets","attack_ids"):
        a=torch.zeros(batch,O,dtype=torch.long)
        for i,r in enumerate(rows):a[i,:len(r[key])]=torch.tensor(r[key])
        result[key]=a.to(device)
    a=torch.zeros(batch,O,WIDTH)
    for i,r in enumerate(rows):
        if r["option_features"]:a[i,:len(r["option_features"])]=torch.tensor(r["option_features"])
    result["option_features"]=a.to(device)
    for key in ("minimum","maximum","reset"):
        result[key]=torch.tensor([r[key] for r in rows],device=device)
    result["option_mask"]=torch.arange(O,device=device)[None,:]>=torch.tensor([len(r["option_ids"]) for r in rows],device=device)[:,None]
    result["schema_version"] = torch.full((batch,), 2 if semantic else 1, device=device)
    if semantic:
        from .semantics import MECHANIC_WIDTH
        for key, size in (("parents", S), ("sources", O), ("target_links", O)):
            a = torch.full((batch, size), -1, dtype=torch.long)
            for i, r in enumerate(rows):
                expected = len(r["ids"]) if key == "parents" else len(r["option_ids"])
                if len(r[key]) != expected:
                    raise ValueError(f"Invalid {key} length")
                if any(type(v) is not int or v < -1 or v >= len(r["ids"]) for v in r[key]):
                    raise ValueError(f"Invalid {key} entity link")
                if key == "parents" and any(v >= j for j, v in enumerate(r[key])):
                    raise ValueError("Parent must precede child")
                a[i, :expected] = torch.tensor(r[key])
            result[key] = a.to(device)
        a = torch.zeros(batch, O, MECHANIC_WIDTH)
        for i, r in enumerate(rows):
            if len(r["mechanics"]) != len(r["option_ids"]):
                raise ValueError("Invalid mechanics length")
            if r["mechanics"]:
                a[i, :len(r["mechanics"])] = torch.tensor(r["mechanics"])
        result["mechanics"] = a.to(device)
    return result

def schema_from_state(state):
    version = int(state.get("_schema_version", 1))
    if version not in (1, 2):
        raise ValueError(f"Unsupported checkpoint observation version: {version}")
    return SEMANTIC_SCHEMA if version == 2 else SCHEMA

class PolicyNetwork(nn.Module):
    def __init__(self, static_cards, attack_count, dropout=0.05, observation_schema=SCHEMA):
        super().__init__()
        if observation_schema not in (SCHEMA, SEMANTIC_SCHEMA):
            raise ValueError("Unknown observation schema")
        self.observation_schema = observation_schema
        static=torch.tensor(static_cards,dtype=torch.float32)
        self.register_buffer("static_cards",static)
        self.card=nn.Embedding(len(static_cards),D);self.attack=nn.Embedding(attack_count,D)
        self.static_projection=nn.Linear(static.shape[1],D)
        self.kind=nn.Embedding(32,D);self.features=nn.Linear(WIDTH,D)
        layer=nn.TransformerEncoderLayer(D,6,768,dropout,activation="gelu",batch_first=True,norm_first=True)
        self.transformer=nn.TransformerEncoder(layer,4,enable_nested_tensor=False)
        self.option_features=nn.Linear(WIDTH,D)
        self.option_cross=nn.MultiheadAttention(D,6,batch_first=True,dropout=dropout)
        self.option_norm=nn.LayerNorm(D)
        self.game_gru=nn.GRUCell(D,D);self.pick_gru=nn.GRUCell(D,D)
        self.query=nn.Linear(D,D);self.key=nn.Linear(D,D)
        self.stop=nn.Parameter(torch.randn(D)*0.02)
        self.value=nn.Sequential(nn.Linear(D,128),nn.GELU(),nn.Linear(128,1),nn.Tanh())
        if observation_schema == SEMANTIC_SCHEMA:
            from .semantics import MECHANIC_WIDTH
            self.register_buffer("_schema_version", torch.tensor(2))
            self.parent_projection = nn.Linear(D, D, bias=False)
            self.source_projection = nn.Linear(D, D, bias=False)
            self.target_projection = nn.Linear(D, D, bias=False)
            self.mechanics_projection = nn.Linear(MECHANIC_WIDTH, D, bias=False)
    def load_state_dict(self, state_dict, strict=True, assign=False):
        if schema_from_state(state_dict) != self.observation_schema:
            raise ValueError("Checkpoint/encoder schema mismatch; explicit migration required")
        return super().load_state_dict(state_dict, strict=strict, assign=assign)
    @staticmethod
    def linked(x, indices):
        gathered = x.gather(1, indices.clamp_min(0)[:, :, None].expand(-1, -1, x.shape[-1]))
        return gathered * (indices >= 0)[:, :, None]
    def card_embedding(self, ids):return self.card(ids)+self.static_projection(self.static_cards[ids])
    def encode_batch(self,b):
        version = 2 if self.observation_schema == SEMANTIC_SCHEMA else 1
        if not (b["schema_version"] == version).all():
            raise ValueError("Observation/checkpoint schema mismatch")
        x=self.card_embedding(b["ids"])+self.kind(b["kinds"])+self.features(b["features"])
        if version == 2:
            x = x + self.parent_projection(self.linked(x, b["parents"]))
        x=self.transformer(x,src_key_padding_mask=b["state_mask"])
        opts=self.card_embedding(b["option_ids"])+self.card_embedding(b["targets"])+self.attack(b["attack_ids"])+self.option_features(b["option_features"])
        if version == 2:
            opts = (opts + self.source_projection(self.linked(x, b["sources"]))
                    + self.target_projection(self.linked(x, b["target_links"]))
                    + self.mechanics_projection(b["mechanics"]))
        # Unnormalized encoder residuals can grow beyond FP16 range during
        # long runs. Keep Q/K/V projections and attention output in FP32;
        # casting an already-overflowed output afterward cannot recover it.
        with torch.autocast(device_type=x.device.type,enabled=False):
            cross,_=self.option_cross(opts.float(),x.float(),x.float(),key_padding_mask=b["state_mask"],need_weights=False)
            if not torch.isfinite(cross).all():
                raise FloatingPointError("Nonfinite FP32 cross-attention output")
            options=self.option_norm(opts.float()+cross)
        return x[:,0],options
    def recurrent(self,global_,hidden,reset):
        hidden=hidden*(~reset.bool()).float()[:,None]
        return self.game_gru(global_,hidden)
    def decode(self,options,hidden,b,actions=None,reference=None,greedy=False,temperature=1.0):
        if not math.isfinite(temperature) or temperature<=0:raise ValueError("Temperature must be finite and positive")
        B,O,_=options.shape
        choices=torch.cat([options,self.stop[None,None,:].expand(B,1,-1)],dim=1)
        keys=self.key(choices);query=hidden
        used=b["option_mask"].clone();done=torch.zeros(B,dtype=torch.bool,device=hidden.device)
        joint=hidden.sum(-1)*0;entropy=joint.clone();kl=joint.clone()
        selected=[[] for _ in range(B)];distributions=[]
        limit=int(b["maximum"].max().item())+1
        for step in range(limit):
            active=(~done)&(step<b["maximum"])
            if not active.any():break
            stop_mask=step<b["minimum"]
            mask=torch.cat([used,stop_mask[:,None]],dim=1)
            # Completed rows take a deterministic STOP and contribute no likelihood.
            mask=mask.clone();mask[~active,:]=True;mask[~active,O]=False
            logits=torch.einsum("bd,bod->bo",self.query(query),keys)/math.sqrt(D)
            if mask.all(-1).any():raise ValueError("No legal decoder choice")
            if not torch.isfinite(logits.masked_fill(mask,0.)).all():
                raise FloatingPointError("Nonfinite legal action logits")
            logits=(logits.float()/temperature).masked_fill(mask,float("-inf"))
            dist=Categorical(logits=logits);log_probs=dist.logits
            if actions is None:pick=logits.argmax(dim=-1) if greedy else dist.sample()
            else:
                pick=torch.tensor([a[step] if step<len(a) else O for a in actions],device=hidden.device)
                if ((pick!=O)&(~active)).any():raise ValueError("Target action longer than maximum")
                if mask.gather(1,pick[:,None])[active].any():raise ValueError("Masked target action")
            joint=joint+dist.log_prob(pick)*active
            entropy=entropy+dist.entropy()*active
            if reference is not None and step<len(reference):
                ref=reference[step].to(hidden.device)
                # Masks are identical because both policies use the same action prefix.
                diff=torch.where(mask,torch.zeros_like(log_probs),log_probs-ref)
                kl=kl+(dist.probs*diff).sum(-1)*active
            distributions.append(log_probs.detach())
            for i,v in enumerate(pick.tolist()):
                if active[i] and v<O:selected[i].append(v)
            done=done|((pick==O)&active)
            chosen=choices[torch.arange(B,device=hidden.device),pick]
            query=self.pick_gru(chosen,query)
            used=used.clone();used[torch.arange(B,device=hidden.device),pick.clamp(max=max(0,O-1))]|=pick<O
        if actions is not None and selected!=actions:raise ValueError("Teacher action was not reproduced")
        return selected,joint,entropy,self.value(hidden).squeeze(-1),kl,distributions
    def decision(self,rows,hidden,actions=None,reference=None,greedy=False,temperature=1.0):
        device=next(self.parameters()).device;b=collate(rows,device)
        global_,options=self.encode_batch(b);hidden=self.recurrent(global_,hidden,b["reset"])
        return self.decode(options,hidden,b,actions,reference,greedy=greedy,temperature=temperature),hidden

def metadata_tables(api):
    cards=api.all_card_data();attacks=api.all_attack()
    table=[[0.]*13 for _ in range(max(c.cardId for c in cards)+1)]
    for c in cards:
        table[c.cardId]=[c.cardType/6,c.hp/400,c.retreatCost/5,(c.weakness or 0)/11,(c.resistance or 0)/11,(c.energyType or 0)/11,float(c.basic),float(c.stage1),float(c.stage2),float(c.ex),float(c.megaEx),float(c.tera),float(c.aceSpec)]
    return table,max(a.attackId for a in attacks)+1


def initialize_semantic(legacy):
    """Explicit warm start for retraining, never an inference-only repair."""
    if legacy.observation_schema != SCHEMA:
        raise ValueError("Expected a visible-v1 source model")
    state = legacy.state_dict()
    network = PolicyNetwork(state["static_cards"].tolist(), state["attack.weight"].shape[0],
                            dropout=legacy.option_cross.dropout, observation_schema=SEMANTIC_SCHEMA)
    network.to(next(legacy.parameters()).device)
    migrated = network.state_dict()
    for name, value in state.items():
        if name not in migrated or migrated[name].shape != value.shape:
            raise ValueError(f"Incompatible shared parameter: {name}")
        migrated[name] = value.detach().clone()
    network.load_state_dict(migrated)
    return network
