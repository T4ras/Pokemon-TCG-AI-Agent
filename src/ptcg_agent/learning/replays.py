"""Strict replay alignment and whole-BO3 splits; no visualization input features."""
import hashlib, json
from collections import Counter
from pathlib import Path
from .observations import DeckKnowledge, encode, SCHEMA, SEMANTIC_SCHEMA
from ..engine import load_sdk, validate_deck
from ..policies import validate_action

def split_for(episode, seed=0):
    bucket = int(hashlib.sha256(f"{seed}:{episode}".encode()).hexdigest()[:8],16)%10
    return "test" if bucket==0 else "validation" if bucket==1 else "train"

def game_results(replay):
    # The adapter stores completed-game outcomes only in visualization metadata.
    # Extract the scalar label here; never pass any visualized board to encode().
    results = []
    previous_terminal = False
    for frame in replay["steps"][0][0].get("visualize",[]):
        result = frame.get("current",{}).get("result",-1)
        if result >= 0 and not previous_terminal: results.append(result)
        previous_terminal = result>=0
    if not 1 <= len(results) <= 3: raise ValueError("Missing or invalid game outcome metadata")
    return results

def prepare(source, output, schema=SCHEMA):
    if schema not in (SCHEMA, SEMANTIC_SCHEMA):
        raise ValueError("Unknown observation schema")
    source, output = Path(source), Path(output)
    if schema == SEMANTIC_SCHEMA and output.exists() and any(output.iterdir()):
        raise ValueError("Fresh semantic output required")
    output.mkdir(parents=True,exist_ok=True)
    manifest = json.loads((source/"sources.json").read_text())
    api,_=load_sdk();cards={c.cardId:c for c in api.all_card_data()}
    decks={};counts=Counter();failures=[];splits={};files=[]
    selected={t["submission_id"]:i for i,t in enumerate(manifest["teachers"])}
    allowed={int(sid):set(map(str,ids)) for sid,ids in manifest.get('teacher_episode_allowlist',{}).items()}
    def eligible(sid,eid):return sid in selected and (sid not in allowed or str(eid) in allowed[sid])
    # Choose the first deterministic metadata-only hash seed with all-deck coverage.
    # This never examines actions, outcomes or model metrics.
    for split_seed in ([manifest["split_seed"]] if "split_seed" in manifest else range(1000)):
        coverage=Counter()
        for eid,meta in manifest["episodes"].items():
            for actor in meta["agents"]:
                if eligible(actor["submissionId"],eid):coverage[(selected[actor["submissionId"]],manifest.get("frozen_splits",{}).get(eid,split_for(eid,split_seed)))]+=1
        if all(coverage[(i,"train")]>=8 and coverage[(i,"validation")]>=1 and coverage[(i,"test")]>=1 for i in range(4)):break
    else:raise ValueError("Cannot construct grouped four-deck splits")
    for eid, metadata in manifest["episodes"].items():
        path=source/"raw"/f"episode-{eid}-replay.json"
        try:
            replay=json.loads(path.read_text())
            if replay["name"]!="cabt" or replay["configuration"].get("bo")!=3: raise ValueError("Wrong environment")
            if replay["statuses"]!=["DONE","DONE"]: raise ValueError("Unreliable teacher episode")
            results=game_results(replay);split=manifest.get("frozen_splits",{}).get(eid,split_for(eid,split_seed));splits[eid]=split
            for actor in metadata["agents"]:
                sid=actor["submissionId"]
                if not eligible(sid,eid):continue
                teacher=selected[sid];seat=actor.get("index",0)
                deck=replay["steps"][1][seat]["action"];validate_deck(deck,cards)
                if teacher in decks and Counter(decks[teacher])!=Counter(deck):raise ValueError("Submission uses multiple decks")
                decks[teacher]=deck
                knowledge=DeckKnowledge();sequence=[]
                for t,step in enumerate(replay["steps"][:-1]):
                    item=step[seat];obs=item["observation"]
                    if item["status"]!="ACTIVE" or obs.get("select") is None:continue
                    if obs["current"]["yourIndex"]!=seat:raise ValueError("Actor visibility mismatch")
                    if obs["current"]["players"][1-seat].get("hand") is not None:raise ValueError("Opponent hand leak")
                    action=replay["steps"][t+1][seat]["action"]
                    validate_action(obs["select"],action)
                    row=(encode(obs,deck,knowledge) if schema==SCHEMA else encode(obs,deck,knowledge,schema=schema));row["action"]=action
                    round_id=obs["current"]["round"]
                    result=results[round_id-1]
                    row["outcome"]=0 if result==2 else (1 if result==seat else -1)
                    row["round"]=round_id;row["teacher"]=teacher
                    row["trainable"]=not (row["minimum"]==row["maximum"]==len(row["option_ids"]) and len(action)<=1)
                    sequence.append(row)
                if not sequence:raise ValueError("No teacher decisions")
                dest=output/f"{eid}-{teacher}-{seat}.json"
                dest.write_text(json.dumps(dict(episode_id=int(eid),teacher=teacher,seat=seat,split=split,rows=sequence),separators=(",",":")))
                files.append(dest.name);counts[f"{teacher}/{split}/episodes"]+=1;counts[f"{teacher}/{split}/decisions"]+=len(sequence)
        except Exception as error:
            failures.append(dict(episode_id=eid,error=f"{type(error).__name__}: {error}"))
    report=dict(schema=schema,split_seed=split_seed,teachers=[dict(t,deck=decks[i]) for i,t in enumerate(manifest["teachers"])],counts=dict(counts),failures=failures,splits=splits,files=files,raw_bytes=sum(p.stat().st_size for p in (source/"raw").glob("*.json")),processed_bytes=sum((output/f).stat().st_size for f in files))
    (output/"manifest.json").write_text(json.dumps(report,indent=2))
    if report["processed_bytes"]>5*1024**3:raise ValueError("Processed data cap exceeded")
    if len(decks)!=4:raise ValueError("Exactly four compatible decks required")
    for teacher in range(4):
        if counts[f"{teacher}/train/episodes"]<8 or counts[f"{teacher}/validation/episodes"]<1 or counts[f"{teacher}/test/episodes"]<1:
            raise ValueError(f"Insufficient grouped split coverage for teacher {teacher}: {counts}")
    return report
