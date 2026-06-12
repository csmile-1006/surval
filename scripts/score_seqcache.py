"""Score a directory of seqcache HDF5s with surval — and nothing else.

Runs ``surval.sequential_validate`` over a cache dir using a built-in action
space (droid/gripper/dex/humanoid), then saves the per-checkpoint proxy values
(SurVAL_Score, valid_loss, ActionL2_mean) as a small ``.npz`` keyed by
step. That npz is the proxy "B" consumed by ``correlate_metrics.py``.

This is the whole metric-from-cache path in one surval-only command — no openpi
or robomimic source needed. Threshold method is selectable:

    # state-free (no state DB; default method is 'inter'):
    python scripts/score_seqcache.py --cache-dir <caches> --output-dir <out> \
        --action-space droid --block-scale-method inter

    # state-conditional (needs a state DB from build_state_db_from_cache.py):
    python scripts/score_seqcache.py --cache-dir <caches> --output-dir <out> \
        --action-space droid --block-scale-method state_inter \
        --state-db-dir <db> --threshold-quantile 0.9
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_HERE, os.pardir, "src"))
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from surval.action_spaces import ACTION_SPACES, get_action_space  # noqa: E402
from surval.sequential_validate import add_common_args, run, validate_common_args  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--action-space", required=True, choices=sorted(ACTION_SPACES),
                        help="Built-in block layout to score with.")
    parser.add_argument("--proxy-out", default=None,
                        help="Path for the per-step proxy .npz (default: <output-dir>/proxy_metrics.npz).")
    add_common_args(parser)  # --cache-dir, --output-dir, --block-scale-method, --threshold-quantile, ...
    args = parser.parse_args()

    action_space = get_action_space(args.action_space)
    validate_common_args(args, num_blocks=len(action_space["block_names"]))
    run(args, action_space)

    # run() writes a results JSON per cache file; collect (step -> proxies).
    results_path = os.path.join(os.path.abspath(os.path.expanduser(args.output_dir)),
                                "sequential_validation_from_cache_results.json")
    results = json.load(open(results_path))
    rows = []
    for r in results:
        s = r.get("summary", {})
        rows.append((int(r["step"]),
                     _f(s.get("SurVAL_Score")),
                     _f(s.get("valid_loss")),
                     _f(s.get("ActionL2_mean"))))
    rows.sort(key=lambda x: x[0])
    steps = np.array([r[0] for r in rows], dtype=np.int64)
    surval = np.array([r[1] for r in rows], dtype=np.float64)
    valid_loss = np.array([r[2] for r in rows], dtype=np.float64)
    action_l2 = np.array([r[3] for r in rows], dtype=np.float64)

    proxy_out = args.proxy_out or os.path.join(os.path.abspath(os.path.expanduser(args.output_dir)),
                                               "proxy_metrics.npz")
    np.savez(proxy_out, steps=steps, surval=surval,
             valid_loss=valid_loss, action_l2=action_l2,
             action_space=args.action_space, block_scale_method=str(args.block_scale_method))
    print(f"\n[score] {args.action_space} | method={args.block_scale_method} | {len(steps)} checkpoint(s)")
    print(f"{'step':>8} {'SurVAL':>15} {'valid_loss':>12} {'ActionL2':>12}")
    for st, ps, vl, al in zip(steps, surval, valid_loss, action_l2):
        print(f"{st:>8} {ps:>15.6f} {vl:>12.6f} {al:>12.6f}")
    print(f"[score] saved proxy npz -> {proxy_out}")


def _f(v):
    return float(v) if v is not None else float("nan")


if __name__ == "__main__":
    main()
