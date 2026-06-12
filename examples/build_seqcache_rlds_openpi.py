"""Example: build a surval seqcache from an openpi (pi0/pi05) RLDS DROID
checkpoint, then score it with sequential-validation.

Why this differs from the robomimic/lerobot examples
-----------------------------------------------------
openpi's RLDS sequential loader already yields **pre-chunked rows** —
`(observation, action_chunk, demo_id, index_in_demo)` — with all observation
transforms applied. There is no raw per-episode trajectory to slice, so the
`surval.ingest` reader/chunking layer (which exists to turn raw episodes into
horizon rows) does not apply here. Instead we drive openpi's loader directly and
write rows with the **low-level** `surval.cache_io.write_seqcache_hdf5`
(the "§6b" path in the README). We additionally store `obs_features` (the
mean-pooled prefix embedding), which surval now requires.

Follows `surval_openpi/scripts/validate_rlds_loss.py` for the model/dataloader
plumbing, trimmed to a single checkpoint and the continuous-action (pi0/pi05)
case (no FAST-tokenizer decoding).

Requires the openpi env (JAX + openpi). Example:

    /path/to/surval_openpi/.venv/bin/python examples/build_seqcache_rlds_openpi.py \
        --config-name pi05_droid_block_rlds_finetune \
        --params-path  .../rise_droid_checkpoints/1600/params \
        --assets-dir   .../rise_droid_checkpoints/1600/assets \
        --rlds-data-dir .../rise_droid_data \
        --dataset-name stack_the_three_blocks_on_the_tray_train_100_valid_5 \
        --out /tmp/rlds_cache/seqcache_step_001600.hdf5 --step 1600 --num-samples 8

Scoring runs in-process if `surval.sequential_validate` imports (needs
tensorboardX); otherwise the cache is written and the score command is printed.

State-conditional thresholds: the `obs_features` written here are the model's
mean-pooled prefix (VLM) embedding, so the state DB is built directly from them.
After building the cache:

    python scripts/build_state_db_from_cache.py --cache-file <out> \
        --output-dir <db> --block-layout droid_joint
    python scripts/compute_local_thresholds.py --state-db-dir <db>
    # then score with: --block-scale-method state_inter --state-db-dir <db> --threshold-quantile 0.9
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys

import numpy as np

# Use the in-development surval for cache_io (numpy + h5py only; importing
# cache_io does NOT pull in tensorboardX, so it is safe in the openpi venv).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))
from surval.cache_io import write_seqcache_hdf5  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import openpi.models.model as _model  # noqa: E402
import openpi.training.config as _config  # noqa: E402
import openpi.training.data_loader as _data_loader  # noqa: E402


# The 7 DROID joint-velocity blocks (single arm; gripper dim 7 left unscored),
# copied from surval_openpi/scripts/sequential_validate_from_cache_droid.py.
DROID_ACTION_SPACE = {
    "action_dim": 8,
    "block_names": [f"joint_{i}" for i in range(7)],
    "block_slices": {f"joint_{i}": slice(i, i + 1) for i in range(7)},
    "block_dims": {f"joint_{i}": 1 for i in range(7)},
    "arm_pairs": [],
    "scale_groups": [{"blocks": [f"joint_{i}" for i in range(7)], "summary_key": "s_joint"}],
    "summary_scale_fields": [("ActionBlockScale_joint", "s_joint")],
    "block_types": {f"joint_{i}": "joint_velocity" for i in range(7)},
}


def make_action_unnormalizer(data_config):
    """Invert the action normalization so the cache stores raw physical units
    (mirrors validate_rlds_loss.py)."""
    stats = getattr(data_config, "norm_stats", None)
    stats = stats.get("actions") if stats else None
    if stats is None:
        return lambda x: x
    use_q = bool(getattr(data_config, "use_quantile_norm", False))
    mean, std = np.asarray(stats.mean, np.float32), np.asarray(stats.std, np.float32)
    q01 = np.asarray(stats.q01, np.float32) if use_q else None
    q99 = np.asarray(stats.q99, np.float32) if use_q else None

    def unnorm(x):
        x = np.asarray(x, np.float32)
        if x.size == 0:
            return x
        ad = x.shape[-1]
        if use_q:
            return (x + 1.0) / 2.0 * (q99[:ad] - q01[:ad] + 1e-6) + q01[:ad]
        return x * (std[:ad] + 1e-6) + mean[:ad]

    return unnorm


def make_steps(model, num_samples):
    """JIT steps for state features (mean-pooled prefix) and action sampling."""

    @jax.jit
    def features_step(observation):
        processed = _model.preprocess_observation(None, observation, train=False)
        tokens, mask, _ = model.embed_prefix(processed)
        m = mask[..., None].astype(tokens.dtype)
        return (tokens * m).sum(axis=1) / m.sum(axis=1).clip(min=1)  # [B, D]

    @jax.jit
    def sample_step(sample_rngs, observation):
        return jax.lax.map(lambda rng: model.sample_actions(rng, observation), sample_rngs)

    if not hasattr(model, "embed_prefix"):
        features_step = None
    return features_step, sample_step


def to_str(x):
    x = x.item() if hasattr(x, "item") else x
    return x.decode() if isinstance(x, bytes) else str(x)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config-name", required=True)
    ap.add_argument("--params-path", required=True, help="path to the checkpoint 'params' dir")
    ap.add_argument("--out", required=True, help="e.g. .../seqcache_step_001600.hdf5")
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--rlds-data-dir", required=True)
    ap.add_argument("--dataset-name", required=True)
    ap.add_argument("--assets-dir", default=None, help="local assets dir for norm_stats (avoids GCS)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-samples", type=int, default=8)
    ap.add_argument("--action-dim", type=int, default=8)
    ap.add_argument("--cache-ta", type=int, default=None, help="rows' horizon T (default: action_horizon)")
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument("--score", action="store_true", help="run seqval after building (needs surval+tensorboardX)")
    args = ap.parse_args()

    out_parent = os.path.dirname(os.path.abspath(args.out))
    if out_parent:
        os.makedirs(out_parent, exist_ok=True)

    config = _config.get_config(args.config_name)
    config = dataclasses.replace(config, exp_name="seqcache_example")
    data_factory = config.data
    if args.assets_dir:  # load norm_stats from the local checkpoint instead of GCS
        data_factory = dataclasses.replace(
            data_factory, assets=dataclasses.replace(data_factory.assets, assets_dir=args.assets_dir)
        )
    data_config = dataclasses.replace(data_factory, add_demo_id=True).create(config.assets_dirs, config.model)
    unnorm = make_action_unnormalizer(data_config)

    loader, n = _data_loader.create_rlds_sequential_data_loader(
        data_config, rlds_data_dir=args.rlds_data_dir, dataset_name=args.dataset_name,
        action_horizon=config.model.action_horizon, batch_size=args.batch_size,
        shuffle=False, drop_last=True,
    )
    print(f"[load] config={args.config_name} action_horizon={config.model.action_horizon} dataset_len={n}")

    model = config.model.load(_model.restore_params(args.params_path, dtype=jnp.bfloat16))
    features_step, sample_step = make_steps(model, args.num_samples)
    if features_step is None:
        raise RuntimeError("model has no embed_prefix; cannot produce required obs_features.")

    demo_ids, idx_in_demo = [], []
    gt_chunks, feat_chunks = [], []
    pred_chunks = [[] for _ in range(args.num_samples)]
    base_rng = jax.random.key(0)

    for bi, (obs, actions, demo_b, idx_b) in enumerate(loader):
        if args.max_batches is not None and bi >= args.max_batches:
            break
        B = int(actions.shape[0])
        base_rng, *rngs = jax.random.split(base_rng, args.num_samples + 1)
        preds = np.asarray(sample_step(jnp.stack(rngs), obs))  # [S, B, Th, 32]
        feats = np.asarray(features_step(obs)).astype(np.float32)  # [B, D]

        ta = preds.shape[2] if args.cache_ta is None else min(args.cache_ta, preds.shape[2])
        ad = args.action_dim
        gt = unnorm(np.asarray(actions)[:, :ta, :ad]).astype(np.float32)
        gt_chunks.append(gt)
        feat_chunks.append(feats)
        for s in range(args.num_samples):
            pred_chunks[s].append(unnorm(preds[s, :, :ta, :ad]).astype(np.float32))
        for i in range(B):
            demo_ids.append(to_str(demo_b[i]))
            idx_in_demo.append(int(idx_b[i].item() if hasattr(idx_b[i], "item") else idx_b[i]))
        if (bi + 1) % 10 == 0:
            print(f"[build] batch {bi + 1} rows={len(demo_ids)}")

    actions_arr = np.concatenate(gt_chunks, axis=0)
    pred_list = [np.concatenate(pc, axis=0) for pc in pred_chunks]
    obs_features = np.concatenate(feat_chunks, axis=0)
    write_seqcache_hdf5(
        args.out, demo_ids=np.array(demo_ids), index_in_demo=np.array(idx_in_demo, np.int64),
        actions=actions_arr, pred_actions_list=pred_list, obs_features=obs_features,
        checkpoint=args.params_path, step=args.step,
    )
    print(f"[build] wrote {args.out}: rows={actions_arr.shape[0]} T={actions_arr.shape[1]} "
          f"A={actions_arr.shape[2]} S={len(pred_list)} F={obs_features.shape[1]}")

    if args.score:
        try:
            import argparse as _ap
            from surval.sequential_validate import add_common_args, run, validate_common_args
        except ImportError as e:
            print(f"[score] skipped (surval.sequential_validate unavailable: {e}). "
                  f"Score it in a surval env, e.g.:\n"
                  f"  python scripts/score_seqcache.py --action-space droid "
                  f"--cache-dir {out_parent} --output-dir {out_parent}/out --block-scale-method inter")
            return
        import glob
        import json
        p = _ap.ArgumentParser(); add_common_args(p)
        a = p.parse_args(["--cache-dir", out_parent, "--output-dir", f"{out_parent}/out",
                          "--block-scale-method", "inter"])
        validate_common_args(a, num_blocks=len(DROID_ACTION_SPACE["block_names"]))
        run(a, DROID_ACTION_SPACE)
        s = json.load(open(glob.glob(f"{out_parent}/out/*group_summary.json")[0]))[0]
        print(f"[score] inter (state-free)  SurVAL = {s.get('mean_surval_score')}")


if __name__ == "__main__":
    main()
