"""Correlate a proxy against the real success rate — the whole metric step.

Loads the ground-truth npz from ``extract_tb_metrics.py`` (success_rate = A) and
the proxy npz from ``score_seqcache.py`` (B = prefix_survival / valid_loss /
action_l2), aligns them by training step, and reports how well the proxy ranks
checkpoints the way success rate does (spearman / hit@k / nregret / mmrv).

Single seed:
    python scripts/correlate_metrics.py --gt gt.npz --proxy proxy.npz

Multiple seeds (paired by order) -> per-seed metrics + bootstrap CI of the mean:
    python scripts/correlate_metrics.py --gt s0_gt.npz s1_gt.npz --proxy s0_p.npz s1_p.npz
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_HERE, os.pardir, "src"))
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from surval.tb_aggregate.metrics import (  # noqa: E402
    _compute_seed_metrics_from_arrays,
    bootstrap_ci_of_mean,
    nanmean,
)

_LOWER_IS_BETTER = {"valid_loss", "action_l2"}


def _aligned_AB(gt_path, proxy_path, proxy_key, negate):
    gt = np.load(gt_path, allow_pickle=True)
    pr = np.load(proxy_path, allow_pickle=True)
    a_by_step = dict(zip(gt["success_steps"].tolist(), gt["success_rate"].tolist()))
    b_by_step = dict(zip(pr["steps"].tolist(), pr[proxy_key].tolist()))
    common = sorted(set(a_by_step) & set(b_by_step))
    if len(common) < 2:
        raise SystemExit(f"{os.path.basename(proxy_path)}: <2 common steps with GT "
                         f"(gt steps={sorted(a_by_step)[:5]}..., proxy steps={sorted(b_by_step)[:5]}...).")
    A = np.array([a_by_step[s] for s in common], dtype=np.float64)
    B = np.array([b_by_step[s] for s in common], dtype=np.float64)
    if negate:
        B = -B
    return A, B, common


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", nargs="+", required=True, help="Ground-truth npz(s) from extract_tb_metrics.py.")
    ap.add_argument("--proxy", nargs="+", required=True, help="Proxy npz(s) from score_seqcache.py (paired by order).")
    ap.add_argument("--proxy-key", default="prefix_survival",
                    choices=["prefix_survival", "valid_loss", "action_l2"])
    ap.add_argument("--negate", choices=["auto", "yes", "no"], default="auto",
                    help="Negate B so higher=better. 'auto' negates the lower-is-better proxies.")
    ap.add_argument("--k-list", nargs="+", type=int, default=[1, 2, 3, 5])
    ap.add_argument("--num-boot", type=int, default=5000)
    ap.add_argument("--ci", type=float, default=0.95)
    args = ap.parse_args()

    if len(args.gt) != len(args.proxy):
        raise SystemExit(f"--gt ({len(args.gt)}) and --proxy ({len(args.proxy)}) must be paired 1:1.")
    negate = args.negate == "yes" or (args.negate == "auto" and args.proxy_key in _LOWER_IS_BETTER)

    hit_ks = sorted({1, *args.k_list})
    metric_keys = ["spearman", "kendall_tau"] + [f"hit@{k}" for k in hit_ks] + ["nregret", "mmrv"]
    per_seed: dict[str, list[float]] = {k: [] for k in metric_keys}
    print(f"proxy={args.proxy_key} (negate={negate})  seeds={len(args.gt)}\n")
    for gt_path, proxy_path in zip(args.gt, args.proxy):
        A, B, common = _aligned_AB(gt_path, proxy_path, args.proxy_key, negate)
        m = _compute_seed_metrics_from_arrays(A, B, k_list=args.k_list, eps=1e-8, tie_tol=0.0,
                                              nan_policy="omit", compute_kendall=True)
        print(f"- {os.path.basename(proxy_path)}: {len(common)} ckpts | "
              f"spearman={m['spearman']:.3f} hit@1={m['hit@1']:.0f} "
              f"hit@{args.k_list[-1]}={m.get(f'hit@{args.k_list[-1]}', float('nan')):.0f} "
              f"nregret={m['nregret']:.3f} mmrv={m['mmrv']:.3f}")
        for k in metric_keys:
            if k in m:
                per_seed[k].append(m[k])

    print(f"\n=== aggregate over {len(args.gt)} seed(s) (mean [{int(args.ci*100)}% CI]) ===")
    for k in metric_keys:
        vals = per_seed[k]
        if not vals:
            continue
        lo, hi = bootstrap_ci_of_mean(vals, num_bootstrap=args.num_boot, ci_level=args.ci)
        print(f"  {k:>12}: {nanmean(vals):.4f}  [{lo:.4f}, {hi:.4f}]")


if __name__ == "__main__":
    main()
