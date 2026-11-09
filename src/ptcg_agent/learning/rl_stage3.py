"""Stage-three continuation without changing the qualified PPO implementation."""
import argparse
import copy
import json
import shutil
from pathlib import Path

import torch
from . import rl_stage1 as base
from .training import load_checkpoint

# Horizon/budget/run identifiers may change; the learning experiment may not.
FROZEN_KEYS = ['seed', 'matches_per_round', 'actor_lr', 'critic_lr', 'weight_decay',
    'batch_sequences', 'ppo_epochs', 'old_policy_kl_stop', 'bc_kl_coefficient',
    'value_coefficient', 'gamma', 'match_seconds', 'rollout_cap_decisions',
    'rollout_buffer_max_bytes', 'sequence_length', 'burn_in',
    'optimizer_target_decisions', 'learner_temperature', 'opponent_temperature',
    'initial_epoch', 'objective', 'learner_deck', 'no_early_pruning',
    'checkpoint_every_round']


def validate_parent(path, config, deck, initial_sha):
    continuation = config['continuation']
    if base.sha(path) != continuation['sha256']:
        raise ValueError('Continuation checkpoint digest mismatch')
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    metadata = checkpoint['metadata']
    if metadata['round'] != continuation['round'] or metadata['trial'] != config['trials'][0]:
        raise ValueError('Continuation round or lambda/entropy pair changed')
    if metadata['initial_tensor_sha256'] != initial_sha or metadata['deck'] != deck or metadata['objective'] != 'game':
        raise ValueError('Continuation BC initialization, deck or objective changed')
    if any(metadata['config'].get(k) != config.get(k) for k in FROZEN_KEYS):
        raise ValueError('Continuation learning settings changed')
    if not checkpoint.get('optimizer') or not checkpoint['optimizer']['state']:
        raise ValueError('Continuation optimizer state missing')
    if not all(k in checkpoint for k in ['python_rng', 'torch_rng', 'cuda_rng']):
        raise ValueError('Continuation RNG state missing')
    tensors = list(checkpoint['model'].values()) + [v for s in checkpoint['optimizer']['state'].values() for v in s.values() if isinstance(v, torch.Tensor)]
    if not all(torch.isfinite(t).all() for t in tensors):
        raise FloatingPointError('Nonfinite continuation model or optimizer')
    if continuation['round'] >= config['rounds']:
        raise ValueError('Continuation horizon does not extend checkpoint')
    return checkpoint


def prepare_continuation(inputs, output, config, require_gpu=True):
    """Seed an ordinary recovery folder, preserving the exact parent archive."""
    inputs, output = Path(inputs), Path(output)
    if (output / 'report.json').exists():
        raise ValueError('Existing continuation requires explicit resume')
    if len(config['trials']) != 1:
        raise ValueError('One continuation trial per job')
    manifest = json.loads((inputs / 'checkpoints.json').read_text())
    source = inputs / 'epoch-34.pt'
    if base.sha(source) != manifest['sha256']['34']:
        raise ValueError('BC source changed')
    fixed = base.load_model(source)
    initial = base.tensor_sha(fixed)
    deck = manifest['decks'][2]
    parent = Path(config['continuation']['checkpoint'])
    checkpoint = validate_parent(parent, config, deck, initial)
    trial = config['trials'][0]; folder = output / trial['id']; folder.mkdir(parents=True, exist_ok=True)
    original = folder / f"game-round-{config['continuation']['round']:03d}.pt"
    shutil.copy2(parent, original); shutil.copy2(parent, folder / 'latest.pt')
    if base.sha(original) != config['continuation']['sha256']:
        raise ValueError('Continuation copy changed')
    report = dict(status='RUNNING', config=config, initial_tensor_sha256=initial,
        opponent_sha256=base.sha(source), deck=deck, started_at_utc=base.now(),
        hardware=dict(device='cuda' if require_gpu else 'cpu',
            gpu=torch.cuda.get_device_name(0) if require_gpu else None, torch=torch.__version__),
        trials={trial['id']: dict(parameters=trial, completed_rounds=config['continuation']['round'], rounds=[], status='RUNNING')},
        checkpoints=[dict(file=str(original.relative_to(output)), trial=trial['id'],
            round=config['continuation']['round'], sha256=base.sha(original), bytes=original.stat().st_size, from_parent=True)],
        continuation=dict(config['continuation'], model_optimizer_rng_restoration_required=True),
        native_rng_seeded=False, objective='game', post_training_evaluation='local CPU; fixed endpoint round30')
    base.write_json(output / 'report.json', report)
    print(json.dumps(dict(kind='continuation_prepared', trial=trial['id'], starting_round=config['continuation']['round'],
        parent_sha256=base.sha(parent), optimizer_state_entries=len(checkpoint['optimizer']['state']),
        original_bc_tensor_sha256=initial)), flush=True)
    return report


def restoration_gate(inputs, config, device='cpu'):
    """Verify exact model/optimizer/RNG restoration before starting collection."""
    if not config.get('continuation'):
        return dict(mode='fresh', optimizer='new', passed=True)
    source = Path(inputs) / 'epoch-34.pt'; manifest = json.loads((Path(inputs) / 'checkpoints.json').read_text())
    parent = Path(config['continuation']['checkpoint'])
    network = base.load_model(source, device).requires_grad_(True)
    payload = validate_parent(parent, config, manifest['decks'][2], base.tensor_sha(network))
    optimizer = torch.optim.AdamW([
        {'params': [p for n,p in network.named_parameters() if not n.startswith('value.')], 'lr':config['actor_lr']},
        {'params': list(network.value.parameters()), 'lr':config['critic_lr']}], weight_decay=config['weight_decay'])
    load_checkpoint(parent, network, optimizer, restore_rng=True)
    for name, value in network.state_dict().items():
        if not torch.equal(value.detach().cpu(), payload['model'][name]):
            raise AssertionError('Restored model differs from parent')
    restored = optimizer.state_dict()
    if restored['param_groups'] != payload['optimizer']['param_groups']:
        raise AssertionError('Restored optimizer groups differ')
    for key, state in payload['optimizer']['state'].items():
        for field, value in state.items():
            actual = restored['state'][key][field]
            if isinstance(value, torch.Tensor):
                if not torch.equal(actual.cpu(), value):raise AssertionError('Restored optimizer tensor differs')
            elif value != actual:raise AssertionError('Restored optimizer value differs')
    import random
    if random.getstate() != payload['python_rng'] or not torch.equal(torch.get_rng_state(), payload['torch_rng']):
        raise AssertionError('Restored Python/Torch RNG differs')
    if device == 'cuda':
        if len(payload['cuda_rng']) != torch.cuda.device_count():
            raise ValueError('Continuation GPU count differs from parent RNG state')
        if not all(torch.equal(a.cpu(), b.cpu()) for a,b in zip(torch.cuda.get_rng_state_all(), payload['cuda_rng'])):
            raise AssertionError('Restored CUDA RNG differs')
    return dict(mode='continuation', starting_round=payload['metadata']['round'],
        model_optimizer_rng_exact=True, parent_sha256=base.sha(parent), passed=True)


def normalize_report(report):
    """Keep prior training exposure separate from this stage's new work."""
    report['endpoint_round'] = max(t['completed_rounds'] for t in report['trials'].values())
    report['completed_rounds'] = sum(len(t['rounds']) for t in report['trials'].values())
    report['total_training_bo3'] = report['config']['matches_per_round'] * report['completed_rounds']
    report['lifetime_training_bo3'] = report['config']['matches_per_round'] * report['endpoint_round']
    report['starting_round'] = report['config'].get('continuation', {}).get('round', 0)
    return report


def run(inputs, output, config, require_gpu=True, script_started=None, resume=False):
    output = Path(output)
    if require_gpu and not torch.cuda.is_available():
        raise RuntimeError('Kaggle GPU required')
    if config.get('continuation') and not resume:
        prepare_continuation(inputs, output, config, require_gpu)
        resume = True
    gate = restoration_gate(inputs, config, 'cuda' if require_gpu else 'cpu')
    print(json.dumps(dict(kind='restoration_gate', **gate)), flush=True)
    report = base.run(inputs, output, config, require_gpu, script_started, resume)
    normalize_report(report); base.write_json(output / 'report.json', report)
    print(json.dumps(dict(kind='stage3_training_complete', status=report['status'],
        new_rounds=report['completed_rounds'], endpoint_round=report['endpoint_round'],
        new_training_bo3=report['total_training_bo3'])), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--inputs', required=True)
    parser.add_argument('--output', required=True); parser.add_argument('--config', required=True)
    parser.add_argument('--cpu-smoke', action='store_true'); parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run(args.inputs, args.output, json.loads(Path(args.config).read_text()), not args.cpu_smoke, resume=args.resume)


if __name__ == '__main__':main()
