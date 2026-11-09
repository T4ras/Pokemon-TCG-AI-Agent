"""Use the private competition SDK without copying its assets into source."""
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SDK = ROOT / "data/sdk/extracted/sample_submission/sample_submission/sample_submission"

def load_sdk():
    if not (SDK / "cg/libcg.so").is_file():
        raise FileNotFoundError("Download/extract official resources; see README.md")
    sys.path.insert(0, str(SDK))
    from cg import api, game
    return api, game

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def sample_deck():
    return [int(x) for x in (SDK / "deck.csv").read_text().split()]

def validate_deck(deck, cards):
    from collections import Counter
    if len(deck) != 60 or any(x not in cards for x in deck):
        raise ValueError("Deck must contain 60 known cards")
    names = Counter(cards[x].name for x in deck if cards[x].cardType != 5)
    if max(names.values(), default=0) > 4:
        raise ValueError("More than four cards with one name")
    if sum(cards[x].aceSpec for x in deck) > 1:
        raise ValueError("More than one ACE SPEC")
    if not any(cards[x].basic and cards[x].cardType == 0 for x in deck):
        raise ValueError("No Basic Pokemon")

def official_environment(**configuration):
    import importlib.util
    from kaggle_environments import make, register
    import json, subprocess
    repo = ROOT / "data/sdk/kaggle-environments"
    expected = json.loads((ROOT / "configs/engine.json").read_text())["upstream_commit"]
    actual = subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip()
    if actual != expected or subprocess.check_output(["git","-C",str(repo),"status","--porcelain"],text=True).strip():
        raise RuntimeError("Official environment checkout differs from pinned clean source")
    path = repo / "kaggle_environments/envs/cabt"
    spec = importlib.util.spec_from_file_location("ptcg_official_cabt", path / "cabt.py", submodule_search_locations=[str(path)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    register("ptcg_official_cabt", {
        "interpreter":module.interpreter, "renderer":module.renderer,
        "html_renderer":module.html_renderer, "specification":module.specification,
        "agents":module.agents})
    return make("ptcg_official_cabt", configuration=configuration, debug=True)
