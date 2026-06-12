# Extending surval — a new dataset, model, or action space

surval scores a cache. To support a new setup you wire up at most three things:
where the **ground-truth actions** come from (dataset), where the **predictions
and feature** come from (model), and how the action vector is **split into
blocks** (action space). Each is independent.

## New dataset

Two paths, depending on whether you can iterate raw trajectories.

### A. Per-episode dataset → subclass `EpisodeReader`

If you can yield raw per-episode arrays, subclass
`surval.ingest.EpisodeReader`; `build_seqcache` handles the horizon chunking,
batching, and writing.

```python
from surval.ingest import Episode, EpisodeReader, build_seqcache

class MyReader(EpisodeReader):
    def __iter__(self):
        for ep in my_dataset:                       # one item per demo / trajectory
            yield Episode(
                demo_id=ep.id,                      # str
                actions=ep.actions,                 # [T_ep, A] ground-truth, per step
                obs={"image": ep.images, "state": ep.proprio},   # each [T_ep, ...]
                state=ep.proprio,                   # [T_ep, D] optional obs_features fallback
            )

build_seqcache("out/seqcache_step_000600.hdf5", MyReader(),
               predict_fn=my_predict, feature_fn=my_encoder,
               horizon=16, num_samples=8, step=600)
```

`build_seqcache` slices `actions[s : s+horizon]` per start frame `s` (with
`stride` / `pad_mode` options), gathers the start-frame observations into
`obs_batch`, and calls your callbacks. See `RobomimicHDF5Reader` /
`LeRobotReader` in `src/surval/ingest/` for full readers, and
[`examples/build_seqcache_robomimic.py`](../examples/build_seqcache_robomimic.py).

### B. Model with its own data pipeline → low-level writer

If the model brings its own (e.g. video-aware) loader that already pre-chunks
rows, skip the reader: collect the row arrays yourself and call
`surval.cache_io.write_seqcache_hdf5(...)` directly.

```python
from surval.cache_io import write_seqcache_hdf5

write_seqcache_hdf5(
    "out/seqcache_step_000600.hdf5",
    demo_ids=demo_ids,                 # [N] str (one per row)
    index_in_demo=index_in_demo,       # [N] int64 (row order within a demo)
    actions=gt_actions,                # [N, T, A] ground-truth chunks
    pred_actions_list=[pred],          # list of S arrays, each [N, T, A]
    obs_features=obs_features,         # [N, F] REQUIRED — the model's feature output
    checkpoint="...", step=600,
)
```

Examples: [`examples/build_seqcache_rlds_openpi.py`](../examples/build_seqcache_rlds_openpi.py)
(openpi pi0/pi05, RLDS DROID) and
[`examples/build_seqcache_gr00t.py`](../examples/build_seqcache_gr00t.py)
(NVIDIA Isaac-GR00T N1.5 on its public `demo_data`).

## New model

The model-specific part is two callbacks (path A) or producing their outputs
inline (path B):

- **`predict_fn(obs_batch) -> [S, B, T, A]`** — your policy's predicted action
  chunks. `obs_batch` is `{obs_key: [B, ...]}` for the `B` rows in a batch, at
  each row's start frame; return `S` samples per row (a 3-D `[B, T, A]` is
  auto-wrapped to `S = 1`). `T` must equal `horizon`, `A` the action dim.
- **`feature_fn(obs_batch) -> [B, F]`** — the model's **own** per-row feature
  output, written as the required `obs_features` (a VLM prefix / backbone
  embedding, an encoder output, ...). It is the retrieval key for the
  state-conditional thresholds (`state_inter` / `state_intra`). Omit it only if
  the reader supplies `Episode.state` (used as the fallback).

If the feature isn't exposed by a public method, capture it with a forward hook
and mean-pool over tokens — see the backbone hook in
[`examples/build_seqcache_gr00t.py`](../examples/build_seqcache_gr00t.py):

```python
feat = {}
def _hook(_m, _in, out):
    bf = out["backbone_features"]            # [1, T_tok, F]
    feat["v"] = bf.mean(1)[0].float().cpu().numpy()
handle = policy.model.backbone.register_forward_hook(_hook)
pred = policy.get_action(obs)                # populates feat["v"]
handle.remove()
```

Keep GT and predictions in the **same** space (e.g. normalize the GT the way the
policy was trained, or unnormalize the predictions) so the per-block errors are
meaningful — the robomimic example normalizes GT with the checkpoint's stats.

## New action space

Add a layout to `surval.action_spaces.ACTION_SPACES`; then it is usable as
`scripts/score_seqcache.py --action-space <name>` and as the `ACTION_SPACE`
argument to `surval.sequential_validate.run`.

```python
MY_ROBOT = {
    "action_dim": A,
    "block_names": ["pos", "rot", "grip"],                  # semantic blocks
    "block_slices": {"pos": slice(0, 3), "rot": slice(3, 9), "grip": slice(9, 10)},
    "block_dims": {"pos": 3, "rot": 6, "grip": 1},
    "arm_pairs": [],                                         # (left, right) block pairs
    "scale_groups": [{"blocks": ["pos"], "summary_key": "s_pos"}],  # blocks sharing one S_g
    "summary_scale_fields": [("ActionBlockScale_pos", "s_pos")],
    "block_types": {"rot": "rot6d"},                        # geodesic for 6-D rotations
}
ACTION_SPACES["my_robot"] = MY_ROBOT
```

- Each block is scored independently; the error is L2 over the slice, or SO(3)
  geodesic distance when `block_types[k] = "rot6d"` (a 6-D continuous rotation).
- `scale_groups` decide which blocks pool into one threshold `S_g` (e.g.
  left/right arms share a scale); `summary_key` names the pooled scale.
- The block layout must match the order in which your producer concatenated the
  action components — the slices index the flat `[..., A]` action vector.

Gripper / finger channels are usually left out of the blocks (excluded from the
metric). See the built-in `droid` / `gripper` / `dex` / `humanoid` / `gr1`
layouts in `src/surval/action_spaces.py`.
