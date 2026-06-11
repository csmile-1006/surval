"""
Pre-extract per-(seed, dataset) scalars from seqcache HDF5 files into the
same JSON cache directory used by extract_tb_scalars.py.

For each `<seqcache_root>/<seed_name>/<cache_group_dir>/seqcache_epoch_*.hdf5`,
this writes:

    <cache_dir>/seqcache_metrics/<seed_name>/<cache_group_dir>.json

with three tags as (steps, values) series sorted by training epoch:

  Cache/Valid/Loss
  Cache/Valid/Off_Manifold_Norm
  Cache/Valid/MSE                # mean((actions - pred_actions)**2) over all
                                  # cache samples / demos / T / A dims

These are HP-invariant per (seed, dataset), so one JSON per (seed, dataset)
suffices. The aggregator's CachedScalarLoader falls back to this cache when
a tag is missing from the seqval-TB payload.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_HERE, os.pardir, "src"))
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from surval.tb_aggregate.seqcache_metrics import (  # noqa: E402
    extract_seqcache_metrics_for_group,
    iter_seed_groups,
)
from surval.tb_aggregate.tb_io import (  # noqa: E402
    cache_path_for_seqcache_metrics,
    save_cache_file,
)


def _build_argparser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seqcache_root", type=str, required=True,
                   help="Root dir with <seed>/<cache_group_dir>/seqcache_epoch_*.hdf5.")
    p.add_argument("--cache_dir", type=str, required=True,
                   help="Output cache dir (will create cache_dir/seqcache_metrics/).")
    p.add_argument("--seed_glob_prefix", type=str, default="")
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--strict", action="store_true")
    return p


def _do_one(payload):
    label, group_path, out_path, meta, skip_existing = payload
    try:
        if skip_existing and os.path.exists(out_path):
            return label, "skipped_existing", None, meta
        result = extract_seqcache_metrics_for_group(group_path)
        result["meta"] = meta
        n_steps = len(next(iter(result["tags"].values()))["steps"])
        save_cache_file(result, out_path)
        return label, "ok", n_steps, meta
    except Exception as e:  # noqa: BLE001
        return label, "error", str(e), meta


def main(argv=None):
    args = _build_argparser().parse_args(argv)
    args.cache_dir = os.path.abspath(args.cache_dir)
    os.makedirs(args.cache_dir, exist_ok=True)

    jobs = []
    for seed_name, group_name, group_path in iter_seed_groups(
        args.seqcache_root, args.seed_glob_prefix
    ):
        out_path = cache_path_for_seqcache_metrics(args.cache_dir, seed_name, group_name)
        jobs.append((
            f"{seed_name}/{group_name}",
            group_path,
            out_path,
            {"seed_name": seed_name, "cache_group_dir": group_name},
            args.skip_existing,
        ))
    print(f"[seqcache] enumerated {len(jobs)} (seed, dataset) groups.")
    if not jobs:
        print(f"[seqcache] nothing to do under {args.seqcache_root!r} "
              f"(seed_glob_prefix={args.seed_glob_prefix!r}).")
        return

    results = []
    if args.num_workers <= 1:
        for p in jobs:
            results.append(_do_one(p))
            label, status, info, _meta = results[-1]
            print(f"  {label}: {status} ({info})")
            if status == "error" and args.strict:
                raise RuntimeError(f"seqcache extraction failed for {label}: {info}")
    else:
        with ProcessPoolExecutor(max_workers=int(args.num_workers)) as ex:
            futures = {ex.submit(_do_one, p): p[0] for p in jobs}
            for fut in as_completed(futures):
                results.append(fut.result())
                label, status, info, _meta = results[-1]
                print(f"  {label}: {status} ({info})")
                if status == "error" and args.strict:
                    raise RuntimeError(f"seqcache extraction failed for {label}: {info}")

    counts = {"ok": 0, "skipped_existing": 0, "error": 0}
    errors = []
    for label, status, info, meta in results:
        counts[status] = counts.get(status, 0) + 1
        if status == "error":
            errors.append({"label": label, "error": info, "meta": meta})

    manifest_path = os.path.join(args.cache_dir, "seqcache_metrics_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump({
            "args": vars(args),
            "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
            "counts": counts,
            "errors": errors,
        }, f, indent=2)
    print(f"[seqcache] wrote {manifest_path}")
    print(f"[seqcache] done. ok={counts.get('ok', 0)} "
          f"skipped_existing={counts.get('skipped_existing', 0)} "
          f"error={counts.get('error', 0)}")


if __name__ == "__main__":
    main()
