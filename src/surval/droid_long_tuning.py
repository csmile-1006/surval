"""Finite c=1 sweep: native chunk length, exact neighbors, quantile, top-k, LSE.

Reuses SURVAL geometry and the earlier sweep's thresholds; no new score formula.
"""

from itertools import product
import numpy as np
from scipy.stats import rankdata

from .droid import action_space_config
from .droid_tuning import grouped_rows
from .sequential_validate import _compute_per_block_scaled_step_errors


AXES = {"ta": list(range(1, 16)),
        "k": [10, 15, 20, 25, 35, 50, 75, 100, 150, 200, 300, 400],
        "quantile": [.01, .025, .05, .1, .15, .2, .25, .35, .5, .65, .75, .85, .9, .95, .99, 1.],
        "lse_tau": [.03, .1, .3, 1., 3., 10., 30.]}
FIXED = {"scale_multiplier": 1., "every_step": True, "num_samples": 8,
         "temporal_radius": 5, "min_neighbors": 10, "exact_neighbors": True,
         "seed": 0, "split": "val30", "horizon": 15}


def hp_grid(axes=AXES):
    if set(axes) != set(AXES) or any(not v or len(v) != len(set(v)) for v in axes.values()):
        raise ValueError("Expected unique nonempty ta/k/quantile/lse_tau axes; c is fixed at one")
    grid = []
    for ta, k, q, tau in product(*(axes[name] for name in AXES)):
        if (ta != int(ta) or not 1 <= ta <= 15 or k != int(k) or k < 10
                or not 0 < q <= 1 or not np.isfinite(tau) or tau <= 0):
            raise ValueError("Invalid HP or horizon beyond native15")
    for ta in axes["ta"]:
        for topk in range(1, ta + 1):
            # Interior of the ceil(T*frac)=topk bin avoids floating boundary errors.
            for k, q, tau in product(axes["k"], axes["quantile"], axes["lse_tau"]):
                grid.append(dict(ta=ta, chunk_top_k=topk, chunk_top_frac=(topk-.5)/ta,
                                 k=k, quantile=q, lse_tau=tau, scale_multiplier=1.))
    return grid


def score_grid(cache, tables, axes=AXES, progress=None):
    hp_grid(axes)  # validate before any expensive work
    cfg = action_space_config(cache["droid_action_space"])
    raw = _compute_per_block_scaled_step_errors(
        cache["pred_actions"], cache["actions"][None],
        np.ones((len(cache["actions"]), len(cfg["block_names"])), np.float32),
        cfg["block_names"], cfg["block_slices"], block_types=cfg.get("block_types"))
    if max(axes["ta"]) > raw.shape[2]:
        raise ValueError("Chunk exceeds real cached predictions")
    groups = grouped_rows(cache)
    scale = np.stack([tables[k][:, :, qi] for k in axes["k"]
                      for qi in range(len(axes["quantile"]))])
    scale = np.maximum(scale.astype(np.float32), 1e-12)
    tau = np.asarray(axes["lse_tau"], np.float64)[:, None, None, None]
    output = []
    for ta in axes["ta"]:
        # Sorting each prefix once covers every worst-action count exactly.
        worst = np.sort(raw[:, :, :ta], axis=2)[:, :, ::-1]
        for topk in range(1, ta + 1):
            error = worst[:, :, :topk].mean(axis=2).mean(axis=0)
            block = np.empty((len(scale), len(tau)), np.float64)
            for start in range(0, len(scale), 16):
                scaled = (error[None] / scale[start:start+16]).astype(np.float64)
                p = np.exp(-np.maximum(0., scaled - 1.))
                neg = -tau * p[None]
                maximum = neg.max(axis=-1, keepdims=True)
                survival = -(np.log(np.exp(neg - maximum).mean(axis=-1)) + maximum[..., 0]) / tau[..., 0]
                values = np.zeros(survival.shape[:2])
                for rows in groups:
                    values += np.cumprod(survival[:, :, rows], axis=-1).mean(axis=-1)
                block[start:start+16] = (values / len(groups)).T
            output.append(block.ravel())
        if progress:
            progress(ta)
    result = np.concatenate(output)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite scores")
    return result


def metric_arrays(actual, scores):
    """Vectorized original primary metrics, including strict ties and eligibility."""
    actual = np.asarray(actual, np.float64)
    scores = np.asarray(scores, np.float64)
    if scores.ndim != 2 or scores.shape[1] != len(actual) or not np.isfinite(scores).all():
        raise ValueError("Expected finite [HP,checkpoint] scores")
    ranks = rankdata(scores, axis=1, method="average")
    ranks -= ranks.mean(axis=1, keepdims=True)
    ar = rankdata(actual) - (len(actual)+1)/2
    denominator = np.sqrt((ranks*ranks).sum(axis=1) * (ar*ar).sum())
    rho = np.divide(ranks @ ar, denominator, out=np.full(len(scores), np.nan), where=denominator>0)
    selected = np.argmax(scores, axis=1)
    span = float(np.ptp(actual))
    regret = (actual.max() - actual[selected]) / (span + 1e-12)
    distance = abs(actual[:, None] - actual[None, :])
    violation = ((scores[:, :, None] < scores[:, None, :]) != (actual[:, None] < actual[None, :]))
    mmrv = (distance[None] * violation).max(axis=2).mean(axis=1)
    eligible = (np.ptp(scores, axis=1) > 1e-8) & np.isfinite(rho) & (span > 0)
    return dict(nregret=regret, mmrv=mmrv, spearman=rho, selected_index=selected,
                selected_success_rate=actual[selected], regret_pp=(actual.max()-actual[selected])*100,
                eligible=eligible, balanced_loss=(regret + mmrv/span + (1-rho)/2)/3)
