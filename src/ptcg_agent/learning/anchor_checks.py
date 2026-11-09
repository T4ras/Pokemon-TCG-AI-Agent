"""Inference-only anchor diagnostics. No tolerance relaxation or optimizer access."""
from contextlib import contextmanager
from copy import deepcopy
import io,platform,time
from pathlib import Path
import torch
from .model import D
from .economical_collector import deferred_reference
from .rl_stage1 import tensor_sha,write_json

ATOL=5e-4
RTOL=1e-5

@contextmanager
def anchor_attention(backend):
    """Single-thread actor/reference process; restore settings even on failure."""
    if backend not in ("default","math"):
        raise ValueError("Unknown anchor attention backend")
    if backend=="default":
        yield
        return
    from torch.nn.attention import sdpa_kernel,SDPBackend
    previous=torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        with sdpa_kernel(SDPBackend.MATH):
            yield
    finally:
        torch.backends.mha.set_fastpath_enabled(previous)

@torch.no_grad()
def serial_reference(network,games,backend="default"):
    """Original per-observation decision path with all forced/reset updates."""
    result=deepcopy(games)
    device=next(network.parameters()).device
    network.eval()
    with anchor_attention(backend),torch.autocast(device_type=device.type,enabled=False):
        for game in result:
            hidden=torch.zeros(1,D,device=device)
            for row in game:
                stats,hidden=network.decision([row],hidden,actions=[row["action"]])
                row["reference"]=[d[0].cpu().clone() for d in stats[-1]]
    return result

def batched_reference(network,games,backend="math"):
    """Fixed backend shared by the validation gate and production anchor pass."""
    result=deepcopy(games)
    timing=fill_reference(network,result,backend)
    return result,timing

def fill_reference(network,games,backend="math"):
    device=next(network.parameters()).device
    with anchor_attention(backend),torch.autocast(device_type=device.type,enabled=False):
        return deferred_reference(network,games)

def compare_references(actual,expected):
    """Compare legal support and every log probability, retaining first failure."""
    summary=dict(passed=True,atol=ATOL,rtol=RTOL,distributions=0,
        failed_distributions=0,maximum_logp_error=0.,maximum_probability_error=0.,
        argmax_disagreements=0,first_failure=None)
    details=None
    def structural(reason,**where):
        summary.update(passed=False,first_failure=dict(reason=reason,**where))
        return summary,None
    if len(actual)!=len(expected):
        return structural("game_count",actual=len(actual),expected=len(expected))
    for gi,(game,old) in enumerate(zip(actual,expected)):
        if len(game)!=len(old):return structural("observation_count",game=gi,actual=len(game),expected=len(old))
        for ri,(row,ref) in enumerate(zip(game,old)):
            aa,bb=row["reference"],ref["reference"]
            if len(aa)!=len(bb):return structural("decoder_depth",game=gi,row=ri,actual=len(aa),expected=len(bb))
            for depth,(a,b) in enumerate(zip(aa,bb)):
                summary["distributions"]+=1
                if a.ndim!=1 or b.ndim!=1 or a.shape!=b.shape:
                    return structural("distribution_shape",game=gi,row=ri,depth=depth,actual=list(a.shape),expected=list(b.shape))
                a=a.detach().cpu();b=b.detach().cpu()
                af,bf=torch.isfinite(a),torch.isfinite(b)
                support=torch.equal(af,bf)
                valid=bool((af|torch.isneginf(a)).all() and (bf|torch.isneginf(b)).all())
                finite=af&bf
                error=float((a[finite]-b[finite]).abs().max()) if finite.any() else 0.
                probability=float((a.exp()-b.exp()).abs().max()) if valid else None
                summary["maximum_logp_error"]=max(summary["maximum_logp_error"],error)
                if probability is not None:summary["maximum_probability_error"]=max(summary["maximum_probability_error"],probability)
                if valid and af.any() and bf.any():summary["argmax_disagreements"]+=int(a.argmax()!=b.argmax())
                passed=valid and support and bool(af.any()) and torch.allclose(a,b,atol=ATOL,rtol=RTOL)
                if not passed:
                    summary["passed"]=False;summary["failed_distributions"]+=1
                    if summary["first_failure"] is None:
                        summary["first_failure"]=dict(game=gi,row=ri,depth=depth,
                            legal_support_equal=support,valid_log_probabilities=valid,maximum_logp_error=error,
                            options=len(row["option_ids"]),minimum=row["minimum"],maximum=row["maximum"],reset=bool(row["reset"]))
                        details=dict(location=summary["first_failure"],actual=a,expected=b)
    return summary,details

def runtime_info(network):
    device=next(network.parameters()).device
    return dict(torch=torch.__version__,python=platform.python_version(),device=str(device),
        dtype=str(next(network.parameters()).dtype),mha_fastpath=torch.backends.mha.get_fastpath_enabled(),
        matmul_precision=torch.get_float32_matmul_precision(),
        gpu=torch.cuda.get_device_name(device) if device.type=="cuda" else None,
        cuda=torch.version.cuda)

def diagnose_anchor(cpu,games,output,gpu=None,deadline=None,production="batched-math"):
    """Persist progress and all comparisons before returning or reporting failure."""
    if production not in ("cpu-live","batched-math") or production=="cpu-live" and gpu is not None:
        raise ValueError("Invalid production reference mode")
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if len(games)!=2 or any(not g or len(g)>64 or not g[0]["reset"] for g in games):
        raise ValueError("Exactly two complete observation prefixes, at most64 rows each")
    if gpu is not None and tensor_sha(cpu)!=tensor_sha(gpu):
        raise ValueError("CPU/GPU anchor tensors differ")
    hashes=dict(cpu=tensor_sha(cpu),gpu=tensor_sha(gpu) if gpu is not None else None)
    report=dict(status="RUNNING",production=production,runtime=dict(cpu=runtime_info(cpu),gpu=runtime_info(gpu) if gpu is not None else None),
        observations=sum(map(len,games)),variants={},comparisons={},training_updates=0)
    variants={"saved":deepcopy(games)}
    started=time.monotonic()
    def persist():
        report["seconds"]=time.monotonic()-started
        write_json(output/"anchor-checks.json",report)
    def check_budget():
        if deadline is not None and time.monotonic()>=deadline:raise TimeoutError("Anchor diagnostic deadline")
    def compare(name,left,right):
        summary,details=compare_references(variants[left],variants[right])
        report["comparisons"][name]=summary
        if details is not None:
            buffer=io.BytesIO();torch.save(details,buffer)
            if buffer.tell()>2*1024**2:raise RuntimeError("Failure tensor cap")
            file=output/(name+"-failure.pt");file.write_bytes(buffer.getvalue())
            summary["private_failure_tensors"]=file.name
        persist()
    persist()
    try:
        for device,model in (("cpu",cpu),("cuda",gpu)):
            if model is None:continue
            for backend in (("default",) if production=="cpu-live" else ("default","math")):
                for mode in (("serial",) if production=="cpu-live" else ("serial","batched")):
                    check_budget();key=f"{device}.{backend}.{mode}";t=time.monotonic()
                    if mode=="serial":variants[key]=serial_reference(model,games,backend)
                    else:variants[key],_=batched_reference(model,games,backend)
                    report["variants"][key]=dict(seconds=time.monotonic()-t,observations=sum(map(len,games)))
                    compare(key+"_vs_saved",key,"saved")
                if production!="cpu-live":
                    compare(f"{device}.{backend}.batch_vs_serial",f"{device}.{backend}.batched",f"{device}.{backend}.serial")
            if device=="cuda":
                for backend in ("default","math"):
                    compare(f"device.{backend}.serial",f"cuda.{backend}.serial",f"cpu.{backend}.serial")
        required=["cpu.default.serial_vs_saved"]
        if production=="batched-math":
            required+=["cpu.math.serial_vs_saved","cpu.math.batch_vs_serial"]
        if gpu is not None:
            required+=["cuda.math.batched_vs_saved","cuda.math.batch_vs_serial","device.math.serial"]
        report["required_checks"]=required
        report["required_checks_passed"]=all(report["comparisons"][k]["passed"] for k in required)
        report["status"]="COMPLETE"
        return report
    except Exception as exc:
        report.update(status="FAILED",error=type(exc).__name__+": "+str(exc))
        raise
    finally:
        report["anchor_tensors_unchanged"]=hashes==dict(cpu=tensor_sha(cpu),gpu=tensor_sha(gpu) if gpu is not None else None)
        persist()

def require_anchor_equivalence(report):
    if report["status"]!="COMPLETE" or not report["required_checks_passed"] or not report["anchor_tensors_unchanged"]:
        raise AssertionError("Anchor equivalence failed; detailed diagnostics persisted before abort")
