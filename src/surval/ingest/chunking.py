"""Turn an episode's per-step actions into fixed-horizon seqcache rows.

A seqcache row holds a length-``T_h`` *action chunk* (the horizon a policy
predicts from one state), so building ground-truth rows means sliding a
horizon-sized window over the episode's ``[T_ep, A]`` action sequence.
"""

from __future__ import annotations

import numpy as np

_PAD_MODES = ("repeat_last", "zero", "drop")


def chunk_actions(
    actions: np.ndarray,
    horizon: int,
    *,
    stride: int = 1,
    pad_mode: str = "repeat_last",
    min_valid: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Slice ``[T_ep, A]`` actions into horizon windows.

    Parameters
    ----------
    actions
        ``[T_ep, A]`` per-step actions for one episode.
    horizon
        ``T_h`` — number of action steps per row.
    stride
        Gap between consecutive window start frames (default 1 = every frame).
    pad_mode
        How to handle windows that run past the episode end:

        - ``"repeat_last"`` (default): repeat the last real action to fill.
        - ``"zero"``: pad with zeros.
        - ``"drop"``: discard windows that cannot be filled completely.
    min_valid
        Drop a tail window unless it has at least this many real (un-padded)
        steps. Ignored when ``pad_mode="drop"``.

    Returns
    -------
    starts
        ``[N]`` int64 start-frame index of each kept window (the row's
        ``index_in_demo``).
    chunks
        ``[N, T_h, A]`` float32 ground-truth action chunks.
    """
    actions = np.asarray(actions)
    if actions.ndim != 2:
        raise ValueError(f"actions must be [T_ep, A]; got {actions.shape}")
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1; got {horizon}")
    if stride < 1:
        raise ValueError(f"stride must be >= 1; got {stride}")
    if pad_mode not in _PAD_MODES:
        raise ValueError(f"pad_mode must be one of {_PAD_MODES}; got {pad_mode!r}")

    t_ep, a_dim = actions.shape
    starts: list[int] = []
    chunks: list[np.ndarray] = []
    for t in range(0, t_ep, stride):
        end = t + horizon
        if end <= t_ep:
            chunk = actions[t:end]
        else:
            if pad_mode == "drop":
                continue
            real = actions[t:t_ep]
            n_valid = real.shape[0]
            if n_valid < min_valid:
                continue
            n_pad = horizon - n_valid
            if pad_mode == "repeat_last":
                pad = np.repeat(real[-1:], n_pad, axis=0)
            else:  # "zero"
                pad = np.zeros((n_pad, a_dim), dtype=real.dtype)
            chunk = np.concatenate([real, pad], axis=0)
        starts.append(t)
        chunks.append(chunk)

    if not chunks:
        return (
            np.zeros((0,), dtype=np.int64),
            np.zeros((0, horizon, a_dim), dtype=np.float32),
        )
    return (
        np.asarray(starts, dtype=np.int64),
        np.stack(chunks, axis=0).astype(np.float32),
    )
