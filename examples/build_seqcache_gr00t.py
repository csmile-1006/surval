"""Example: build a surval seqcache from NVIDIA Isaac-GR00T on its demo_data.

Runs the public GR00T N1.5 VLA over the public `demo_data/robot_sim.PickNPlace`
LeRobot dataset (GR1 humanoid, 26-dim arms+hands action) and writes a canonical
seqcache surval can score — a fully open-source, surval-only example (no
customized training repo needed).

Like the openpi/RLDS example, GR00T has its own video-aware data pipeline, so we
drive it directly and use the low-level `surval.cache_io.write_seqcache_hdf5`.
Each `policy.get_action` call at step s gives one cache row:
  - `actions`      : the dataset's ground-truth action chunk at s, `[T, A]`
  - `pred_actions` : the policy's predicted chunk,                  `[S, T, A]`
  - `obs_features` : the GR00T backbone (Eagle VLM) embedding at s, mean-pooled
                     over tokens -> `[F]` (the model's own feature output)

Run in the `gr00t` env (where `import gr00t` works), from the Isaac-GR00T root so
`demo_data/...` resolves, on a GPU:

    cd /path/to/Isaac-GR00T
    python /path/to/surval/examples/build_seqcache_gr00t.py \
        --out /tmp/gr00t_cache/seqcache_step_000000.hdf5 --step 0 \
        --max-rows-per-demo 12

Then score it (in any surval env) with the built-in GR1 layout below, e.g.
`scripts/score_seqcache.py --action-space gr1 ...` — or this script's `--score`.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))
from surval.action_spaces import GR1  # noqa: E402  (the 26-D arms+hands layout)
from surval.cache_io import write_seqcache_hdf5  # noqa: E402

from gr00t.data.dataset import LeRobotSingleDataset  # noqa: E402
from gr00t.experiment.data_config import DATA_CONFIG_MAP  # noqa: E402
from gr00t.model.policy import Gr00tPolicy  # noqa: E402

# Action vector = concat of these keys (GR1 arms+hands), fixed order matching the
# block layout in surval.action_spaces.GR1. -> A = 26.
ACTION_KEYS = GR1["block_names"]  # ["left_arm", "right_arm", "left_hand", "right_hand"]


def _concat_action(d):
    """Concat the per-limb action arrays into [T, A] in ACTION_KEYS order."""
    parts = [np.asarray(d[f"action.{k}"], dtype=np.float32) for k in ACTION_KEYS]
    parts = [p[0] if p.ndim == 3 else p for p in parts]  # drop a leading batch dim if present
    return np.concatenate(parts, axis=-1)  # [T, A]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="e.g. .../seqcache_step_000000.hdf5")
    ap.add_argument("--step", type=int, default=0, help="cache time index (checkpoint step)")
    ap.add_argument("--dataset-path", default="demo_data/robot_sim.PickNPlace/")
    ap.add_argument("--model-path", default="nvidia/GR00T-N1.5-3B")
    ap.add_argument("--data-config", default="fourier_gr1_arms_only")
    ap.add_argument("--embodiment-tag", default="gr1")
    ap.add_argument("--num-samples", type=int, default=1, help="predicted chunks per row (S)")
    ap.add_argument("--max-rows-per-demo", type=int, default=None, help="cap rows per trajectory")
    ap.add_argument("--max-demos", type=int, default=None)
    ap.add_argument("--score", action="store_true", help="score the cache after building (needs surval env)")
    args = ap.parse_args()

    out_parent = os.path.dirname(os.path.abspath(args.out))
    if out_parent:
        os.makedirs(out_parent, exist_ok=True)

    data_config = DATA_CONFIG_MAP[args.data_config]
    modality_config = data_config.modality_config()
    policy = Gr00tPolicy(
        model_path=args.model_path, embodiment_tag=args.embodiment_tag,
        modality_config=modality_config, modality_transform=data_config.transform(),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    dataset = LeRobotSingleDataset(
        dataset_path=args.dataset_path, modality_configs=modality_config,
        video_backend="decord", transforms=None, embodiment_tag=args.embodiment_tag,
    )
    horizon = len(data_config.action_indices)  # action chunk length T
    print(f"[load] model={args.model_path} config={args.data_config} "
          f"trajs={len(dataset.trajectory_ids)} horizon={horizon}")

    # Hook the backbone to capture the per-observation VLM embedding (mean-pooled
    # over tokens) -> obs_features.
    feat = {}

    def _hook(_m, _inp, out):
        bf = out["backbone_features"]                      # [1, T_tok, F]
        mask = out.get("backbone_attention_mask", None)
        if mask is not None:
            m = mask[..., None].to(bf.dtype)
            pooled = (bf * m).sum(1) / m.sum(1).clamp(min=1)
        else:
            pooled = bf.mean(1)
        feat["v"] = pooled[0].float().cpu().numpy()        # [F]

    handle = policy.model.backbone.register_forward_hook(_hook)

    demo_ids, index_in_demo, gt_chunks, feat_rows = [], [], [], []
    pred_chunks = [[] for _ in range(args.num_samples)]
    traj_ids = list(dataset.trajectory_ids)[: args.max_demos] if args.max_demos else list(dataset.trajectory_ids)

    for tid in traj_ids:
        t_len = int(dataset.trajectory_lengths[list(dataset.trajectory_ids).index(tid)])
        starts = list(range(0, t_len - horizon, horizon))
        if args.max_rows_per_demo:
            starts = starts[: args.max_rows_per_demo]
        for s in starts:
            dp = dataset.get_step_data(tid, s)
            gt = _concat_action(dp)[:horizon]              # [T, A]
            samples = []
            for _ in range(args.num_samples):
                pred = _concat_action(policy.get_action(dp))[:horizon]  # [T, A]; sets feat["v"]
                samples.append(pred.astype(np.float32))
            demo_ids.append(f"demo_{int(tid)}")
            index_in_demo.append(int(s))
            gt_chunks.append(gt.astype(np.float32))
            feat_rows.append(feat["v"].astype(np.float32))
            for si in range(args.num_samples):
                pred_chunks[si].append(samples[si])
        print(f"[build] demo_{int(tid)}: {len([x for x in demo_ids if x == f'demo_{int(tid)}'])} rows")

    handle.remove()

    actions = np.stack(gt_chunks, axis=0)                  # [N, T, A]
    pred_list = [np.stack(p, axis=0) for p in pred_chunks]  # S x [N, T, A]
    obs_features = np.stack(feat_rows, axis=0)              # [N, F]
    write_seqcache_hdf5(
        args.out, demo_ids=np.array(demo_ids),
        index_in_demo=np.array(index_in_demo, np.int64),
        actions=actions, pred_actions_list=pred_list, obs_features=obs_features,
        checkpoint=args.model_path, step=args.step,
    )
    print(f"[build] wrote {args.out}: rows={actions.shape[0]} T={actions.shape[1]} "
          f"A={actions.shape[2]} S={len(pred_list)} F={obs_features.shape[1]}")

    if args.score:
        try:
            import argparse as _ap
            import glob
            import json

            from surval.sequential_validate import add_common_args, run, validate_common_args
        except ImportError as e:
            print(f"[score] skipped (surval.sequential_validate unavailable: {e}). Score elsewhere:\n"
                  f"  python scripts/score_seqcache.py --action-space gr1 --cache-dir {out_parent} "
                  f"--output-dir {out_parent}/out --block-scale-method inter")
            return
        p = _ap.ArgumentParser(); add_common_args(p)
        a = p.parse_args(["--cache-dir", out_parent, "--output-dir", f"{out_parent}/out",
                          "--block-scale-method", "inter"])
        validate_common_args(a, num_blocks=len(GR1["block_names"]))
        run(a, GR1)
        s = json.load(open(glob.glob(f"{out_parent}/out/*group_summary.json")[0]))[0]
        print(f"[score] inter (state-free)  PrefixSurvival = {s.get('mean_prefix_survival_score')}")


if __name__ == "__main__":
    main()
