"""Example: build a surval seqcache from a robomimic flow_policy checkpoint.

End-to-end demonstration of `surval.ingest` against a real policy + dataset:

    RobomimicHDF5Reader  ->  build_seqcache(predict_fn, feature_fn)  ->  canonical cache

`surval.ingest` owns the format-agnostic half (reading the dataset, slicing
ground-truth action chunks, writing the canonical HDF5). The *model-specific*
glue stays here in the example, because only your policy knows it:

  1. **Action normalization** — the dataset stores raw actions, but the policy
     predicts in a normalized [-1, 1] space. We read the checkpoint's
     `action_normalization_stats` and normalize the ground truth to match, so
     GT and predictions live in the same space.
  2. **Observation history** — the policy consumes `observation_horizon` stacked
     frames. We expand each per-step obs into a `[T_ep, To, ...]` stack in a
     reader wrapper so `build_seqcache` hands `predict_fn` a `[B, To, ...]` batch.
  3. **predict_fn / feature_fn** — thin wrappers over `model.get_action` (sampled
     `num_samples` times) and `model.get_state_features` (the obs_features).

Requires the robomimic fork that defines `flow_policy` on the PYTHONPATH (this is
NOT a surval dependency). Run inside that environment, e.g.:

    python examples/build_seqcache_robomimic.py \
        --ckpt  .../models/model_epoch_600.pth \
        --dataset .../two_arm_box_cleanup_processed.hdf5 \
        --out   /tmp/cache/seqcache_step_000600.hdf5 \
        --step 600 --num-samples 8

To then score the cache, run `scripts/score_seqcache.py --action-space dex
--cache-dir <out's dir>` (state-free `--block-scale-method inter` by default), or
call `surval.sequential_validate.run` with the matching ACTION_SPACE.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.torch_utils as TorchUtils
from robomimic.algo import algo_factory

from surval.ingest import Episode, RobomimicHDF5Reader, build_seqcache
from surval.sequential_validate import _load_cache_hdf5


def load_policy(ckpt_path, device):
    """Load a robomimic checkpoint into an eval-ready algo (the reference recipe)."""
    ckpt = FileUtils.load_dict_from_checkpoint(ckpt_path=ckpt_path, weights_only=False)
    algo_name, _ = FileUtils.algo_name_from_checkpoint(ckpt_dict=ckpt)
    config, _ = FileUtils.config_from_checkpoint(algo_name=algo_name, ckpt_dict=ckpt, verbose=False)
    ObsUtils.initialize_obs_utils_with_config(config)
    sm = ckpt["shape_metadata"]
    model = algo_factory(
        algo_name=config.algo_name, config=config,
        obs_key_shapes=sm["all_shapes"], ac_dim=sm["ac_dim"], device=device,
    )
    model.deserialize(ckpt["model"])
    model.set_eval()
    return model, config, ckpt, sm


def build_action_normalizer(ckpt, action_keys, ac_dim):
    """Return a fn mapping raw actions -> normalized [-1, 1] policy space.

    Uses the per-component offset/scale stored in the checkpoint, concatenated in
    the config's `action_keys` order (matches the flat 24-dim `actions` layout).
    """
    st = ckpt["action_normalization_stats"]
    offset = np.concatenate([np.asarray(st[k]["offset"]).reshape(-1) for k in action_keys]).astype(np.float32)
    scale = np.concatenate([np.asarray(st[k]["scale"]).reshape(-1) for k in action_keys]).astype(np.float32)
    assert offset.shape == (ac_dim,), (offset.shape, ac_dim)
    return lambda a: (a.astype(np.float32) - offset) / scale


class FrameStackedReader:
    """Wrap a reader: normalize GT actions and expand obs to a To-frame history.

    obs[k] becomes `[T_ep, To, ...]` where frame index t holds the window
    [t-To+1 .. t] (clamped at the start). `build_seqcache` then gathers
    `obs[k][starts] -> [B, To, ...]`, exactly the shape `model.get_action` wants.
    """

    def __init__(self, base, normalize_actions, To, max_demos):
        self.base, self.normalize_actions, self.To, self.max_demos = base, normalize_actions, To, max_demos

    def __iter__(self):
        for i, ep in enumerate(self.base):
            if self.max_demos is not None and i >= self.max_demos:
                break
            stacked = {}
            for k, v in ep.obs.items():
                v = np.asarray(v)
                frames = [np.concatenate([v[:1]] * j + [v[: len(v) - j]], axis=0) if j else v
                          for j in range(self.To - 1, -1, -1)]
                stacked[k] = np.stack(frames, axis=1)  # [T_ep, To, ...]
            yield Episode(demo_id=ep.demo_id, actions=self.normalize_actions(ep.actions),
                          obs=stacked, state=None)


def make_callbacks(model, device, num_samples):
    """predict_fn (sampled action chunks) + feature_fn (obs_features), both over
    the To-frame obs batch produced by FrameStackedReader."""

    def to_obs_dict(obs_batch):
        proc = ObsUtils.process_obs_dict({k: np.asarray(v) for k, v in obs_batch.items()})
        return {k: torch.as_tensor(v).float().to(device) for k, v in proc.items()}

    @torch.no_grad()
    def predict_fn(obs_batch):
        od = to_obs_dict(obs_batch)
        return np.stack(
            [model.get_action(obs_dict=od, goal_dict=None).cpu().numpy().astype(np.float32)
             for _ in range(num_samples)],
            axis=0,
        )  # [S, B, Ta, A]

    @torch.no_grad()
    def feature_fn(obs_batch):
        return model.get_state_features(to_obs_dict(obs_batch), None).cpu().numpy().astype(np.float32)

    return predict_fn, feature_fn


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True, help="e.g. .../seqcache_step_000600.hdf5")
    ap.add_argument("--step", type=int, required=True, help="training step/epoch (cache time index)")
    ap.add_argument("--split", default="valid", help="robomimic mask split (default: valid)")
    ap.add_argument("--num-samples", type=int, default=8, help="flow samples per row (S)")
    ap.add_argument("--horizon", type=int, default=None, help="row horizon T_h (default: action_horizon)")
    ap.add_argument("--max-demos", type=int, default=None)
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    model, config, ckpt, sm = load_policy(args.ckpt, device)
    To = config.algo.horizon.observation_horizon
    Ta = config.algo.horizon.action_horizon
    horizon = args.horizon or Ta
    action_keys = list(config.train.action_keys)
    obs_keys = list(sm["all_obs_keys"])
    print(f"[load] ac_dim={sm['ac_dim']} To={To} Ta={Ta} horizon={horizon} obs_keys={len(obs_keys)}")

    out_parent = os.path.dirname(os.path.abspath(args.out))
    if out_parent:
        os.makedirs(out_parent, exist_ok=True)

    normalize = build_action_normalizer(ckpt, action_keys, sm["ac_dim"])
    base = RobomimicHDF5Reader(args.dataset, split=args.split, obs_keys=obs_keys)
    reader = FrameStackedReader(base, normalize, To, args.max_demos)
    predict_fn, feature_fn = make_callbacks(model, device, args.num_samples)

    summary = build_seqcache(
        args.out, reader, predict_fn=predict_fn, feature_fn=feature_fn,
        horizon=horizon, num_samples=args.num_samples, step=args.step,
        checkpoint=args.ckpt, pad_mode="drop", batch_size=args.batch_size, max_rows=args.max_rows,
    )
    print(f"[build] {summary}")

    # sanity: the cache round-trips through the surval reader
    _, _, actions, pred, obs_feat, *_ = _load_cache_hdf5(args.out, load_obs_features=True)
    print(f"[check] actions={actions.shape} pred={pred.shape} obs_features={obs_feat.shape}")
    print("OK: wrote a canonical seqcache surval.sequential_validate can score.")


if __name__ == "__main__":
    main()
