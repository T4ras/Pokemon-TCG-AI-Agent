import argparse
import contextlib
import importlib.metadata
import json
import platform
import subprocess
import time
from pathlib import Path
from .engine import ROOT, SDK, load_sdk, sample_deck, sha256, validate_deck, official_environment
from .policies import RandomPolicy

def smoke(output):
    output.mkdir(parents=True, exist_ok=False)
    api, game = load_sdk()
    deck = sample_deck()
    cards = {c.cardId:c for c in api.all_card_data()}
    validate_deck(deck, cards)
    obs, start = game.battle_start(deck, deck)
    if obs is None:
        raise RuntimeError(f"Native deck rejected: {start.errorType}")
    game.battle_finish()
    from kaggle_environments import make
    begin = time.perf_counter()
    env = official_environment(bo=3)
    env.run([RandomPolicy(deck, 1), RandomPolicy(deck, 2)])
    elapsed = time.perf_counter()-begin
    replay = env.toJSON()
    (output / "replay.json").write_text(json.dumps(replay))
    final = [{"status":x.status,"reward":x.reward} for x in env.state]
    result = {"elapsed_seconds":elapsed,"final":final,"steps":len(env.steps),
        "game_results":getattr(env, "result", None), "configuration":dict(env.configuration),"python":platform.python_version(),
        "environment":importlib.metadata.version("kaggle-environments"),
        "upstream_commit":subprocess.check_output(["git","-C",str(ROOT / "data/sdk/kaggle-environments"),"rev-parse","HEAD"],text=True).strip(),
        "environment_binary_sha256":sha256(ROOT / "data/sdk/kaggle-environments/kaggle_environments/envs/cabt/cg/libcg.so"),
        "sdk_sha256":sha256(SDK / "cg/libcg.so"),
        "archive_sha256":sha256(ROOT / "data/sdk/the-pokemon-company-ptcg-ai-battle-challenge-playground.zip"),
        "deck_sha256":sha256(SDK / "deck.csv"),
        "seed_limitation":"Policy RNG seeded; native shuffles/coins not controlled"}
    (output / "summary.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    if any(x["status"] != "DONE" for x in final):
        raise RuntimeError("Official match failed; inspect replay")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["smoke", "arena", "package"])
    p.add_argument("--output", type=Path)
    p.add_argument("--config", type=Path)
    p.add_argument("--run-id")
    args = p.parse_args()
    if args.command == "smoke":
        if args.output is None:
            p.error("smoke requires --output")
        smoke(args.output)
    elif args.command == "package":
        if args.output is None:
            p.error("package requires --output")
        from .packaging import package
        package(args.output)
    else:
        if args.config is None or not args.run_id:
            p.error("arena requires --config and --run-id")
        from .arena import experiment
        experiment(args.config,args.run_id)

if __name__ == "__main__":
    main()
