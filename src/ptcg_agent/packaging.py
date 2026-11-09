'''Build a private deployable archive; never submits it.'''
import json
import shutil
import zipfile
from .engine import ROOT, SDK, sha256

def package(output):
    output = output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    bundle = output/'bundle'
    bundle.mkdir()
    shutil.copy2(ROOT/'src/ptcg_agent/policies.py',bundle/'policies.py')
    shutil.copy2(SDK/'deck.csv',bundle/'deck.csv')
    cg = bundle/'cg'
    cg.mkdir()
    for name in ('__init__.py','api.py','game.py','sim.py','utils.py','libcg.so'):
        shutil.copy2(SDK/'cg'/name,cg/name)
    (bundle/'main.py').write_text('''from pathlib import Path
from cg.api import all_card_data, all_attack
from policies import HeuristicPolicy

ROOT = Path(__file__).resolve().parent
deck = [int(x) for x in (ROOT / 'deck.csv').read_text().split()]
policy = HeuristicPolicy(deck, 904117, {c.cardId:c for c in all_card_data()},
                         {a.attackId:a for a in all_attack()}, version=2)

def agent(obs, configuration=None):
    return policy(obs)
''')
    archive = output/'agent.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(bundle.rglob('*')):
            if p.is_file():
                z.write(p,p.relative_to(bundle))
    size = archive.stat().st_size
    if size >= 197.7*1024**2:
        raise ValueError('Package exceeds documented size ceiling')
    result = {'archive':str(archive.relative_to(ROOT)),'bytes':size,'sha256':sha256(archive),
        'contents':{str(p.relative_to(bundle)):sha256(p) for p in bundle.rglob('*') if p.is_file()},
        'private_assets':True,'competition_submitted':False,
        'limitations':'Local callable BO3 and archive structure only; Kaggle container runner unverified'}
    (output/'manifest.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
