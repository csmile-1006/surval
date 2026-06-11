"""
Per-seed proxy metrics + seed-level aggregation helpers.

Metric set (computed on aligned series A = ground truth, B = proxy):
  - spearman, delta_spearman
  - kendall_tau, delta_kendall_tau (optional)
  - hit@k for each k in k_list
  - nregret = (max A - A[argmax B]) / (max A - min A + eps)
  - rank_pct of argmax A under proxy B
  - mmrv = Mean Maximum Rank Violation (Li et al., 2024, arxiv:2405.05941)

MMRV is treated as a proxy-quality metric where A is the ground-truth signal
(e.g., real success rate) and B is the proxy signal (e.g., seqval score).
Lower is better; range [0, max|A_i - A_j|].
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from tqdm import trange


# ---------------------------------------------------------------------------
# Reductions tolerant of NaNs
# ---------------------------------------------------------------------------


def nanmean(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0 or np.all(np.isnan(x)):
        return np.nan
    return float(np.nanmean(x))


def nanstd(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0 or np.all(np.isnan(x)):
        return np.nan
    return float(np.nanstd(x))


def nanmedian(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0 or np.all(np.isnan(x)):
        return np.nan
    return float(np.nanmedian(x))


# ---------------------------------------------------------------------------
# Correlation primitives
# ---------------------------------------------------------------------------


def _average_rank(x):
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    ranks = np.empty_like(x, dtype=np.float64)
    n = len(x)
    start = 0
    while start < n:
        end = start + 1
        while end < n and sorted_x[end] == sorted_x[start]:
            end += 1
        avg_rank = 0.5 * ((start + 1) + end)
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def _pearson_simple(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - np.mean(x)
    y = y - np.mean(y)
    sx = np.sqrt(np.sum(x**2))
    sy = np.sqrt(np.sum(y**2))
    if sx == 0.0 or sy == 0.0:
        return np.nan
    return float(np.sum(x * y) / (sx * sy))


def _spearman(a, b):
    return _pearson_simple(_average_rank(a), _average_rank(b))


def _kendall_tau_b(a, b):
    n = len(a)
    if n < 2:
        return np.nan
    c = 0
    d = 0
    n1 = 0
    n2 = 0
    for i in range(n - 1):
        da = a[i + 1 :] - a[i]
        db = b[i + 1 :] - b[i]
        tie_a = da == 0
        tie_b = db == 0
        n1 += int(np.sum(tie_a))
        n2 += int(np.sum(tie_b))
        valid = (~tie_a) & (~tie_b)
        if not np.any(valid):
            continue
        prod = da[valid] * db[valid]
        c += int(np.sum(prod > 0))
        d += int(np.sum(prod < 0))
    n0 = n * (n - 1) / 2.0
    denom = np.sqrt((n0 - n1) * (n0 - n2))
    if denom == 0:
        return np.nan
    return float((c - d) / denom)


def _safe_corr_metric(a, b, fn, nan_policy):
    if np.any(np.isnan(a)) or np.any(np.isnan(b)):
        if nan_policy == "raise":
            raise ValueError("NaN found while computing correlation metric.")
        return np.nan
    if len(a) < 2 or np.all(a == a[0]) or np.all(b == b[0]):
        if nan_policy == "raise":
            raise ValueError("Undefined correlation metric on constant (or too short) sequence.")
        return np.nan
    return fn(a, b)


# ---------------------------------------------------------------------------
# Top-k / argmax helpers
# ---------------------------------------------------------------------------


def _argmax_set(x, tie_tol=0.0):
    x = np.asarray(x, dtype=np.float64)
    max_x = float(np.max(x))
    if tie_tol <= 0.0:
        return {int(np.argmax(x))}
    idx = np.where(x >= (max_x - tie_tol))[0]
    if idx.size == 0:
        return {int(np.argmax(x))}
    return set(int(i) for i in idx.tolist())


def _topk_indices_desc_with_tie_break(x, k):
    x = np.asarray(x, dtype=np.float64)
    t = len(x)
    if t == 0:
        return []
    order = np.lexsort((np.arange(t), -x))
    k_eff = min(int(k), t)
    return [int(i) for i in order[:k_eff]]


# ---------------------------------------------------------------------------
# MMRV (Mean Maximum Rank Violation) — Li et al., 2024 (arXiv:2405.05941)
# ---------------------------------------------------------------------------


def mmrv(A, B):
    """
    Mean Maximum Rank Violation.

        RankViolation(i, j) = |A_i - A_j| * 1[(B_i < B_j) != (A_i < A_j)]
        MMRV(A, B) = (1/N) * sum_i max_j RankViolation(i, j)

    A is the ground-truth signal (real-world success rate); B is the proxy
    (simulator score / seqval score). Lower is better. Returns NaN if N < 2.

    Note: strict `<` follows the paper. Equal proxy values count as a
    disagreement when the ground-truth side has a strict ordering (this
    penalizes proxies that compress informative differences to a tie).
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    n = len(A)
    if n < 2:
        return np.nan
    A_lt = A[:, None] < A[None, :]
    B_lt = B[:, None] < B[None, :]
    disagree = (A_lt != B_lt).astype(np.float64)
    abs_diff = np.abs(A[:, None] - A[None, :])
    rv = abs_diff * disagree
    return float(np.mean(np.max(rv, axis=1)))


# ---------------------------------------------------------------------------
# Per-seed metric computation
# ---------------------------------------------------------------------------


def _compute_seed_metrics_from_arrays(
    A,
    B,
    k_list,
    eps,
    tie_tol,
    nan_policy,
    compute_kendall=True,
    compute_delta_kendall=False,
):
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if A.ndim != 1 or B.ndim != 1 or len(A) != len(B):
        raise ValueError("A and B must be 1D arrays with same length.")
    t = len(A)
    if t < 2:
        raise ValueError("Need at least 2 checkpoints per seed.")

    out = {}
    out["spearman"] = _safe_corr_metric(A, B, _spearman, nan_policy)
    if compute_kendall:
        out["kendall_tau"] = _safe_corr_metric(A, B, _kendall_tau_b, nan_policy)

    dA = np.diff(A)
    dB = np.diff(B)
    out["delta_spearman"] = _safe_corr_metric(dA, dB, _spearman, nan_policy)
    if compute_delta_kendall:
        out["delta_kendall_tau"] = _safe_corr_metric(dA, dB, _kendall_tau_b, nan_policy)

    t_star = int(np.argmax(A))
    hat_t = int(np.argmax(B))
    s_a = _argmax_set(A, tie_tol=tie_tol)

    out["hit@1"] = 1.0 if hat_t in s_a else 0.0
    for k in k_list:
        topk = _topk_indices_desc_with_tie_break(B, k)
        out[f"hit@{int(k)}"] = 1.0 if any((idx in s_a) for idx in topk) else 0.0

    a_best = float(np.max(A))
    a_worst = float(np.min(A))
    a_sel = float(A[hat_t])
    out["nregret"] = (a_best - a_sel) / (a_best - a_worst + eps)

    if t > 1:
        rank = 1 + int(np.sum(B > B[t_star]))
        out["rank_pct"] = (rank - 1) / float(t - 1)
    else:
        out["rank_pct"] = np.nan

    out["mmrv"] = mmrv(A, B)
    return out


def compute_seed_metrics(
    a_rate,
    b_vals,
    k_list,
    eps,
    tie_tol,
    nan_policy,
    use_posterior_success=False,
    n_rollouts=200,
    num_mc=2000,
    beta_prior_a=1.0,
    beta_prior_b=1.0,
    a_success_counts=None,
    compute_kendall=True,
    compute_delta_kendall=False,
    rng=None,
):
    """
    Per-seed metric computation.

    If use_posterior_success=True, sample A from a Beta(k+a, n-k+b) posterior
    per checkpoint and return MC expectations of all metrics.
    """
    a_rate = np.asarray(a_rate, dtype=np.float64)
    b_vals = np.asarray(b_vals, dtype=np.float64)
    if rng is None:
        rng = np.random.default_rng(0)

    if not use_posterior_success:
        return _compute_seed_metrics_from_arrays(
            A=a_rate,
            B=b_vals,
            k_list=k_list,
            eps=eps,
            tie_tol=tie_tol,
            nan_policy=nan_policy,
            compute_kendall=compute_kendall,
            compute_delta_kendall=compute_delta_kendall,
        )

    if a_success_counts is None:
        k_success = np.rint(a_rate * float(n_rollouts)).astype(np.int64)
    else:
        k_success = np.asarray(a_success_counts, dtype=np.int64)
    k_success = np.clip(k_success, 0, int(n_rollouts))

    metric_samples = defaultdict(list)
    for _ in trange(int(num_mc), desc="MC posterior", leave=False):
        a_sample = rng.beta(
            k_success.astype(np.float64) + float(beta_prior_a),
            (float(n_rollouts) - k_success.astype(np.float64)) + float(beta_prior_b),
        )
        m = _compute_seed_metrics_from_arrays(
            A=a_sample,
            B=b_vals,
            k_list=k_list,
            eps=eps,
            tie_tol=tie_tol,
            nan_policy=nan_policy,
            compute_kendall=compute_kendall,
            compute_delta_kendall=compute_delta_kendall,
        )
        for key, val in m.items():
            metric_samples[key].append(val)

    return {key: nanmean(vals) for key, vals in metric_samples.items()}


# ---------------------------------------------------------------------------
# Seed-level aggregation helpers
# ---------------------------------------------------------------------------


def bootstrap_ci_of_mean(values, num_bootstrap=2000, ci_level=0.95, rng=None):
    """Seed-level bootstrap CI for the mean."""
    values = np.asarray(values, dtype=np.float64)
    values = values[~np.isnan(values)]
    n = len(values)
    if n == 0:
        return np.nan, np.nan
    if n == 1:
        v = float(values[0])
        return v, v
    if rng is None:
        rng = np.random.default_rng(0)

    boot_means = np.empty(num_bootstrap, dtype=np.float64)
    for i in range(num_bootstrap):
        sample = rng.choice(values, size=n, replace=True)
        boot_means[i] = np.mean(sample)

    alpha = 1.0 - ci_level
    lo = float(np.percentile(boot_means, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))
    return lo, hi


def leave_one_seed_out_stats(values):
    values = np.asarray(values, dtype=np.float64)
    valid = values[~np.isnan(values)]
    n = len(valid)
    if n <= 1:
        return {"min_mean": np.nan, "max_mean": np.nan}
    loo_means = []
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        loo_means.append(float(np.mean(valid[mask])))
    return {"min_mean": float(np.min(loo_means)), "max_mean": float(np.max(loo_means))}
