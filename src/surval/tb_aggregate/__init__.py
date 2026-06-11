"""
TensorBoard-driven seqval hyperparameter aggregation.

Two-stage workflow:
  1. `extract_tb_scalars.py` walks the train + eval TB trees and dumps a JSON
     cache (one file per tb_dir) under --cache_dir.
  2. `aggregate_seqval_hparams.py` loads the cache (no tensorboard import needed)
     and computes per-seed + aggregated proxy metrics across hyperparameter dirs.

The HP dir format aggregated is the `local_threshold_seqval` format produced by
`slurm_local_threshold_seqval_*` scripts:

    tq{tq}_lsetau{lsetau}_ctf{ctf}_ta{ta}_nds{nds}_{share|noshare}_{skip|noskip}_{L2|maha}

Each HP dir contains one or more `seqcache_ds<...>_mode<...>_split<...>_vkey<...>_efd<...>`
group dirs whose `tb/` subdir has the seqval scalars.
"""

from .layouts import (
    CACHE_GROUP_DIR_PATTERN,
    LOCAL_THRESHOLD_HP_DIR_PATTERN,
    collect_seed_dirs_by_seed,
    find_train_tb_dir,
    find_tb_event_dir,
    iter_seqval_eval_dirs,
    parse_hparam,
)
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
    cache_path_for_train,
    cache_path_for_seqval,
    extract_tb_scalars,
    load_cache_file,
    save_cache_file,
)

__all__ = [
    "CACHE_GROUP_DIR_PATTERN",
    "CachedScalarLoader",
    "LOCAL_THRESHOLD_HP_DIR_PATTERN",
    "bootstrap_ci_of_mean",
    "cache_path_for_seqval",
    "cache_path_for_train",
    "collect_seed_dirs_by_seed",
    "compute_seed_metrics",
    "extract_tb_scalars",
    "find_tb_event_dir",
    "find_train_tb_dir",
    "iter_seqval_eval_dirs",
    "leave_one_seed_out_stats",
    "load_cache_file",
    "mmrv",
    "nanmean",
    "nanmedian",
    "nanstd",
    "parse_hparam",
    "save_cache_file",
]
