"""Generic seqcache builder: dataset reader + policy callbacks -> canonical HDF5.

This is the format-agnostic half of cache production. The dataset-specific part
lives in the readers (:mod:`surval.ingest.robomimic`, :mod:`surval.ingest.lerobot`);
the model-specific part is the user's ``predict_fn`` / ``feature_fn`` (which can
never be generic — only your policy knows how to predict). :func:`build_seqcache`
glues them: it chunks each episode into horizon rows, batches the row-start
observations through your callbacks, and writes the result with
:func:`surval.cache_io.write_seqcache_hdf5`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np

from ..cache_io import write_seqcache_hdf5
from .base import EpisodeReader
from .chunking import chunk_actions

# A batch of row-start observations: obs_key -> [B, ...] (B rows of this batch).
ObsBatch = Mapping[str, np.ndarray]
# Returns predicted actions [S, B, T_h, A] (or [B, T_h, A], auto-wrapped to S=1).
PredictFn = Callable[[ObsBatch], np.ndarray]
# Returns per-row state features [B, F].
FeatureFn = Callable[[ObsBatch], np.ndarray]


def build_seqcache(
    out_path: str,
    reader: EpisodeReader,
    *,
    predict_fn: PredictFn,
    horizon: int,
    step: int,
    checkpoint: str = "",
    feature_fn: FeatureFn | None = None,
    num_samples: int = 1,
    stride: int = 1,
    pad_mode: str = "repeat_last",
    batch_size: int = 256,
    max_rows: int | None = None,
) -> dict:
    """Build one canonical seqcache HDF5 from a dataset reader + policy callbacks.

    For every episode the reader yields, this slices ``horizon``-length
    ground-truth action chunks (one row per ``stride`` frames), gathers the
    observation at each row's start frame, batches those observations through
    ``predict_fn`` (and ``feature_fn``), then writes a step-mode canonical cache.

    Parameters
    ----------
    out_path
        Destination ``.hdf5`` (conventionally ``seqcache_step_{step:06d}.hdf5``).
    reader
        Any :class:`~surval.ingest.base.EpisodeReader`.
    predict_fn
        ``predict_fn(obs_batch) -> [S, B, T_h, A]`` (or ``[B, T_h, A]``). Receives
        a dict of stacked row-start observations ``{obs_key: [B, ...]}`` and must
        return ``num_samples`` predicted action chunks per row, horizon-aligned
        with the ground truth.
    horizon
        ``T_h`` — action-chunk length per row.
    step
        Training step (or epoch); stored as the cache's time index.
    checkpoint
        Free-form checkpoint descriptor stored in the cache.
    feature_fn
        ``feature_fn(obs_batch) -> [B, F]`` producing the required
        ``obs_features``. If ``None``, falls back to the episode ``state`` at each
        row's start frame; an error is raised if neither is available.
    num_samples
        Expected ``S`` (sanity-checks the ``predict_fn`` output count).
    stride, pad_mode
        Forwarded to :func:`~surval.ingest.chunking.chunk_actions`.
    batch_size
        Rows per ``predict_fn`` / ``feature_fn`` call.
    max_rows
        Optional cap on total rows written (stops early once reached).

    Returns
    -------
    dict
        Summary ``{"num_rows", "num_demos", "horizon", "ac_dim", "num_samples",
        "obs_feat_dim", "out_path"}``.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1; got {horizon}")
    if num_samples < 1:
        raise ValueError(f"num_samples must be >= 1; got {num_samples}")

    demo_ids: list[str] = []
    index_in_demo: list[int] = []
    gt_chunks: list[np.ndarray] = []           # each [n, T_h, A]
    pred_chunks: list[np.ndarray] = []         # each [S, n, T_h, A]
    feat_chunks: list[np.ndarray] = []         # each [n, F]
    n_demos = 0
    n_rows = 0

    for episode in reader:
        starts, chunks = chunk_actions(
            episode.actions, horizon, stride=stride, pad_mode=pad_mode
        )
        if starts.shape[0] == 0:
            continue
        n_demos += 1

        ep_preds: list[np.ndarray] = []
        ep_feats: list[np.ndarray] = []
        for b0 in range(0, starts.shape[0], batch_size):
            batch_starts = starts[b0 : b0 + batch_size]
            obs_batch = {k: np.asarray(v)[batch_starts] for k, v in episode.obs.items()}

            pred = np.asarray(predict_fn(obs_batch))
            if pred.ndim == 3:
                pred = pred[None]  # [B, T_h, A] -> [1, B, T_h, A]
            if pred.ndim != 4:
                raise ValueError(
                    f"predict_fn must return [S, B, T_h, A] or [B, T_h, A]; got {pred.shape}"
                )
            if pred.shape[0] != num_samples:
                raise ValueError(
                    f"predict_fn returned S={pred.shape[0]} samples but num_samples={num_samples}"
                )
            if pred.shape[1] != batch_starts.shape[0] or pred.shape[2] != horizon:
                raise ValueError(
                    f"predict_fn output {pred.shape} inconsistent with batch "
                    f"(B={batch_starts.shape[0]}, T_h={horizon})"
                )
            ep_preds.append(pred.astype(np.float32))

            feats = _resolve_features(feature_fn, obs_batch, episode, batch_starts)
            ep_feats.append(feats)

        demo_ids.extend([episode.demo_id] * starts.shape[0])
        index_in_demo.extend(starts.tolist())
        gt_chunks.append(chunks)
        pred_chunks.append(np.concatenate(ep_preds, axis=1))  # cat over B
        feat_chunks.append(np.concatenate(ep_feats, axis=0))
        n_rows += starts.shape[0]

        if max_rows is not None and n_rows >= max_rows:
            break

    if n_rows == 0:
        raise RuntimeError("Reader produced no rows (no episodes, or all shorter than the horizon).")

    actions = np.concatenate(gt_chunks, axis=0)              # [N, T_h, A]
    pred_all = np.concatenate(pred_chunks, axis=1)           # [S, N, T_h, A]
    obs_features = np.concatenate(feat_chunks, axis=0)       # [N, F]
    pred_actions_list = [pred_all[s] for s in range(pred_all.shape[0])]

    if max_rows is not None and actions.shape[0] > max_rows:
        actions = actions[:max_rows]
        obs_features = obs_features[:max_rows]
        pred_actions_list = [p[:max_rows] for p in pred_actions_list]
        demo_ids = demo_ids[:max_rows]
        index_in_demo = index_in_demo[:max_rows]

    write_seqcache_hdf5(
        out_path,
        demo_ids=np.asarray(demo_ids),
        index_in_demo=np.asarray(index_in_demo, dtype=np.int64),
        actions=actions,
        pred_actions_list=pred_actions_list,
        obs_features=obs_features,
        checkpoint=checkpoint,
        step=step,
    )

    return {
        "num_rows": int(actions.shape[0]),
        "num_demos": int(n_demos),
        "horizon": int(horizon),
        "ac_dim": int(actions.shape[2]),
        "num_samples": int(len(pred_actions_list)),
        "obs_feat_dim": int(obs_features.shape[1]),
        "out_path": out_path,
    }


def _resolve_features(feature_fn, obs_batch, episode, batch_starts) -> np.ndarray:
    """obs_features for one batch: feature_fn output, else episode.state fallback."""
    if feature_fn is not None:
        feats = np.asarray(feature_fn(obs_batch), dtype=np.float32)
        if feats.ndim != 2 or feats.shape[0] != batch_starts.shape[0]:
            raise ValueError(
                f"feature_fn must return [B, F] aligned with the batch "
                f"(B={batch_starts.shape[0]}); got {feats.shape}"
            )
        return feats
    if episode.state is not None:
        return np.asarray(episode.state[batch_starts], dtype=np.float32)
    raise ValueError(
        f"obs_features is required but episode {episode.demo_id!r} has no state and "
        "no feature_fn was provided. Pass feature_fn=... or give the reader "
        "state_keys / state_key so it can populate Episode.state."
    )
