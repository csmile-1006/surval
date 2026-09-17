"""Frozen policy-conditioning reference and fixed-half/LSE1 cached scoring.

Only the selected reference checkpoint supplies features. Target checkpoints
supply predictions; their features are never compared to the reference index.
"""

import numpy as np

from .droid_dino import align_dino_cache
from .droid_long_tuning import AXES as LONG_AXES, hp_grid as long_grid
from .droid_tuning import chunk_errors, grouped_rows, scores_from_errors
from .local_threshold.config import LocalThresholdConfig
from .local_threshold.database import StateDatabase, StateRecords
from .rlds_cache import validate_row_coverage


AXES = {k: LONG_AXES[k] for k in ('ta', 'k', 'quantile')}
POLICY_OMN = 'Off_Manifold_Norm_PolicyReference'


def fixed_grid(axes=AXES):
    if set(axes) != set(AXES):
        raise ValueError('Only Ta, k and quantile may be tuned')
    candidates = long_grid({**axes, 'lse_tau': [1.]})
    return [{**h, 'chunk_top_frac': .5} for h in candidates
            if h['chunk_top_k'] == (h['ta']+1)//2]


def build_policy_reference(cache, reference_epoch=25):
    p = cache['provenance']
    if cache['epoch'] != reference_epoch:
        raise ValueError('Reference cache epoch does not match requested frozen encoder')
    if (not p['feature_source'].endswith('policy.obs_encoder')
            or p.get('feature_pool') != 'flatten_observation_horizon'):
        raise ValueError('Expected cached complete policy observation conditioning')
    if p.get('max_rows') is not None or p.get('max_demos') is not None:
        raise ValueError('Reference must cover the complete validation split')
    features = np.asarray(cache['obs_features'], dtype=np.float32)
    if features.ndim != 2 or features.shape[0] != len(cache['actions']) or not len(features):
        raise ValueError('Reference feature/GT rows are not aligned')
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms <= 1e-12):
        raise ValueError('Reference features must have finite nonzero norms')
    keys = list(zip(cache['demo_ids'].tolist(), cache['index_in_demo'].tolist()))
    validate_row_coverage(keys, keys)
    names, inverse = np.unique(cache['demo_ids'], return_inverse=True)
    cfg = LocalThresholdConfig.for_droid_action_space(cache['droid_action_space'],
        encoder_name=f'policy_cache_epoch{reference_epoch}:{p["feature_source"]}',
        encoder_pool=p['feature_pool'], encoder_device='cached',
        state_window_size=p['observation_horizon'], dataset_name=p['dataset_name'],
        # Proprio is already fused inside the policy feature; do not append it again.
        use_proprio=False, actions_from_cache=True, index_type='flat_ip')
    db = StateDatabase(cfg)
    db.attach(features/norms, StateRecords(cache['actions'][:, 0].copy(), inverse.astype(np.int32),
        cache['index_in_demo'].astype(np.int64), {i: str(name) for i, name in enumerate(names)}))
    metadata = dict(kind='frozen_policy_conditioning', reference_epoch=reference_epoch,
        reference_checkpoint=cache['checkpoint'], reference_run=p['run'], reference_run_timestamp=p['run_timestamp'],
        feature_source=p['feature_source'], feature_pool=p['feature_pool'],
        dataset_name=p['dataset_name'], dataset_files=p['dataset_files'],
        observation_horizon=p['observation_horizon'], action_start_offset=cache['action_start_offset'],
        droid_action_space=cache['droid_action_space'], split='val')
    return db, cache['actions'].copy(), metadata


def align_policy_reference(db, chunks, metadata, cache):
    if metadata.get('kind') != 'frozen_policy_conditioning':
        raise ValueError('Not a frozen policy reference')
    p = cache['provenance']
    if (p['run'], p['run_timestamp']) != (metadata['reference_run'], metadata['reference_run_timestamp']):
        raise ValueError('Reference and target must belong to the same task/training run')
    # This existing helper is representation-agnostic: only metadata, row keys and GT.
    return align_dino_cache(db, chunks, metadata, cache)


def score_fixed_grid(cache, tables, axes=AXES):
    """Reuse canonical half-chunk errors and existing tau1 survival reduction."""
    grid = fixed_grid(axes)
    errors = chunk_errors(cache, axes['ta'])
    groups = grouped_rows(cache)
    scales = np.stack([tables[k][:, :, qi] for k in axes['k'] for qi in range(len(axes['quantile']))])
    if scales.shape[1:] != errors[axes['ta'][0]].shape or not np.isfinite(scales).all():
        raise ValueError('Scale table must be finite [k*q,cache_rows,action_groups]')
    output = []
    for ta in axes['ta']:
        for start in range(0, len(scales), 32):
            output.extend(scores_from_errors(errors[ta][None], scales[start:start+32], groups))
    result = np.asarray(output, np.float64)
    if result.shape != (len(grid),) or not np.isfinite(result).all():
        raise ValueError('Incomplete or nonfinite score grid')
    return result
