"""Preservation-only preparation for Unown schema ug-current-v1.

This is not an inference adapter or a checkpoint exporter. Missing semantic
coordinates must stop export before a checkpoint is written.
"""
import hashlib
import numpy as np

SCHEMA = 'ug-current-v1'
SEED = 904106


class ProjectionUnavailable(RuntimeError):
    pass


def require_projection_assets(assets):
    missing = []
    for kind, width in [('cards', 128), ('attacks', 64), ('skills', 64)]:
        for suffix, shape in [('mean', (3072,)), ('components', (width, 3072)),
                              ('scales', (width,))]:
            key = f'{kind}_{suffix}'
            if key not in assets or np.asarray(assets[key]).shape != shape:
                missing.append(key)
        if kind + '_parent_reconstruction_verified' not in assets or assets.get(kind + '_parent_reconstruction_verified') is not True:
            missing.append(kind + '_parent_reconstruction_verified')
    if missing:
        raise ProjectionUnavailable('Missing verified original coordinate assets: ' + ', '.join(missing))


def extend_rows(old, new_size, name, seed=SEED):
    """Independent named stream; copy every old row, including reserved rows."""
    old = np.asarray(old)
    if new_size < len(old):
        raise ValueError('Cannot shrink a learned table')
    out = np.empty((new_size, *old.shape[1:]), dtype=old.dtype)
    out[:len(old)] = old
    named_seed = int.from_bytes(hashlib.sha256(f'{seed}:{name}'.encode()).digest()[:8], 'little')
    rng = np.random.default_rng(named_seed)
    # Registered initialization: normal(0, .02), fp32 then cast to parent dtype.
    out[len(old):] = rng.normal(0, .02, out[len(old):].shape).astype(np.float32).astype(old.dtype)
    return out


def assign_rows(old_ids, current_ids, changed_ids, old_size, unknown_row):
    """Current engine IDs -> stable model rows, without consuming parent slack."""
    old_ids, changed_ids = set(old_ids), set(changed_ids)
    if unknown_row < old_size and unknown_row in old_ids:
        raise ValueError('Unknown aliases a real parent identity')
    next_row = max(old_size, unknown_row + 1)
    mapping = {0: 0}
    appended = []
    for ident in sorted(current_ids):
        if ident in old_ids and ident not in changed_ids:
            mapping[ident] = ident
        else:
            mapping[ident] = next_row
            appended.append(dict(engine_id=ident, model_row=next_row,
                                 category='changed' if ident in old_ids else 'new'))
            next_row += 1
    assert unknown_row not in mapping.values()
    return mapping, appended, next_row


def project_vectors(vectors, mean, components, scales):
    """Exact released whitening + row normalization formula; no PCA fitting."""
    z = ((np.asarray(vectors) - mean) @ components.T) / scales
    return z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-9)
