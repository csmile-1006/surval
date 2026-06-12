"""TensorBoard scalar I/O + proxy-metric helpers.

- ``tb_io``: read TensorBoard event scalars (used by ``scripts/extract_tb_metrics.py``).
- ``metrics``: per-seed proxy metrics (spearman / hit@k / nregret / mmrv) and
  seed-level bootstrap CI (used by ``scripts/correlate_metrics.py``).
"""

from .metrics import (
    bootstrap_ci_of_mean,
    compute_seed_metrics,
    leave_one_seed_out_stats,
    mmrv,
    nanmean,
    nanmedian,
    nanstd,
)
from .tb_io import (
    CachedScalarLoader,
    cache_path_for_seqval,
    cache_path_for_train,
    extract_tb_scalars,
    load_cache_file,
    save_cache_file,
)

__all__ = [
    "CachedScalarLoader",
    "bootstrap_ci_of_mean",
    "cache_path_for_seqval",
    "cache_path_for_train",
    "compute_seed_metrics",
    "extract_tb_scalars",
    "leave_one_seed_out_stats",
    "load_cache_file",
    "mmrv",
    "nanmean",
    "nanmedian",
    "nanstd",
    "save_cache_file",
]
