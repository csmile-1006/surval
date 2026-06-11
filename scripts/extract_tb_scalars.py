"""
Pre-extract TensorBoard scalars for local_threshold seqval runs into a JSON cache.

Output layout (mirrors the input filesystem):

    <cache_dir>/manifest.json
    <cache_dir>/train/<seed_name>.json
    <cache_dir>/eval/<seed_name>/<hp_dir>/<cache_group_dir>.json

Each per-tb-dir file stores every scalar tag's (steps, values). The downstream
aggregate script reads only the tags it needs; you can extract once and run many
metric sweeps without touching the original event files.

Example:

    python scripts/extract_tb_scalars.py \
      --train_root_dir /.../valid_flow_action_dict \
      --eval_root_dir  /.../20260513/local_threshold_seqval \
      --cache_dir      /.../tb_cache/local_threshold_seqval_20260513 \
      --seed_glob_prefix flow_policy_image_ds_two_arm_drawer_cleanup_D0_n_demo_500 \
      --num-workers 8

You can restrict extraction to specific tags with --tag (repeatable). By
default every scalar tag is dumped — that's typically a few KB per TB and lets
you experiment with proxy choices later without re-extracting.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Iterable

# Allow `python scripts/extract_tb_scalars.py` from the repo root without install.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_HERE, os.pardir, "src"))
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from surval.tb_aggregate.layouts import (  # noqa: E402
    collect_seed_dirs_by_seed,
    find_tb_event_dir,
    find_train_tb_dir,
    iter_seqval_eval_dirs,
)
from surval.tb_aggregate.tb_io import (  # noqa: E402
    cache_path_for_seqval,
    cache_path_for_train,
    extract_tb_scalars,
    save_cache_file,
)


def _build_argparser():
    p = argparse.ArgumentParser(
        description="Pre-extract TensorBoard scalars into a JSON cache for fast aggregation."
    )
    p.add_argument("--train_root_dir", type=str, required=True)
    p.add_argument("--eval_root_dir", type=str, required=True)
    p.add_argument(
        "--cache_dir", type=str, required=True,
        help="Output directory for the JSON cache. Created if missing.",
    )
    p.add_argument("--seed_glob_prefix", type=str, default="")
    p.add_argument(
        "--tag", action="append", default=None,
        help="Limit extraction to this scalar tag. Repeat to keep multiple tags. "
             "Default: keep all scalar tags.",
    )
    p.add_argument(
        "--use_only_latest", type=lambda s: s.lower() in {"1", "true", "yes"},
        default=True,
        help="Load only the most recent event file per tb dir (matches v7 behaviour).",
    )
    p.add_argument(
        "--num-workers", type=int, default=1,
        help="Process pool size. Each task handles one tb_dir. Default: 1 (sequential).",
    )
    p.add_argument(
        "--skip_existing", action="store_true",
        help="If the output JSON already exists, skip extracting that tb dir.",
    )
    p.add_argument(
        "--strict", action="store_true",
        help="Raise on the first extraction error instead of skipping it.",
    )
    return p


def _enumerate_jobs(args):
    """
    Yield extraction jobs as (label, tb_dir, out_path, kind, meta).
    kind: 'train' or 'seqval'.
    meta: extra fields used for the manifest.
    """
    train_seed_dirs = collect_seed_dirs_by_seed(args.train_root_dir, args.seed_glob_prefix)
    eval_seed_dirs = collect_seed_dirs_by_seed(args.eval_root_dir, args.seed_glob_prefix)
    common_seeds = sorted(set(train_seed_dirs.keys()) & set(eval_seed_dirs.keys()))

    summary = {
        "num_train_seed_dirs": len(train_seed_dirs),
        "num_eval_seed_dirs": len(eval_seed_dirs),
        "num_common_seed_dirs": len(common_seeds),
        "common_seeds": [int(s) for s in common_seeds],
    }

    jobs = []
    skipped_train: list = []
    skipped_eval: list = []

    for seed in common_seeds:
        train_seed_dir = train_seed_dirs[seed]
        eval_seed_dir = eval_seed_dirs[seed]
        seed_name = os.path.basename(eval_seed_dir)

        train_tb = find_train_tb_dir(train_seed_dir)
        if train_tb is None:
            skipped_train.append({"seed_name": seed_name, "reason": "train_tb_not_found"})
        else:
            out_path = cache_path_for_train(args.cache_dir, seed_name)
            jobs.append((
                f"train/{seed_name}",
                train_tb,
                out_path,
                "train",
                {"seed": int(seed), "seed_name": seed_name},
            ))

        for child_path, hp_dir, group_dir in iter_seqval_eval_dirs(eval_seed_dir):
            tb_dir = find_tb_event_dir(child_path)
            if tb_dir is None:
                skipped_eval.append({
                    "seed_name": seed_name,
                    "hp_dir": hp_dir,
                    "cache_group_dir": group_dir,
                    "reason": "seqval_tb_not_found",
                })
                continue
            out_path = cache_path_for_seqval(args.cache_dir, seed_name, hp_dir, group_dir)
            jobs.append((
                f"eval/{seed_name}/{hp_dir}/{group_dir}",
                tb_dir,
                out_path,
                "seqval",
                {
                    "seed": int(seed),
                    "seed_name": seed_name,
                    "hp_dir": hp_dir,
                    "cache_group_dir": group_dir,
                },
            ))

    summary["skipped_train"] = skipped_train
    summary["skipped_eval"] = skipped_eval
    return jobs, summary


def _do_one_job(payload):
    label, tb_dir, out_path, kind, meta, tag_filter, use_only_latest, skip_existing = payload
    try:
        if skip_existing and os.path.exists(out_path):
            return label, kind, "skipped_existing", None, meta
        result = extract_tb_scalars(
            tb_dir, tag_filter=tag_filter, use_only_latest=use_only_latest,
        )
        # Annotate with the job metadata so the cache is self-describing.
        result["kind"] = kind
        result["meta"] = meta
        save_cache_file(result, out_path)
        return label, kind, "ok", len(result["tags"]), meta
    except Exception as e:  # noqa: BLE001
        return label, kind, "error", str(e), meta


def main(argv: Iterable[str] | None = None):
    args = _build_argparser().parse_args(argv)
    args.cache_dir = os.path.abspath(args.cache_dir)
    os.makedirs(args.cache_dir, exist_ok=True)

    jobs, summary = _enumerate_jobs(args)
    print(
        f"[extract] enumerated {len(jobs)} extraction jobs "
        f"(train={sum(1 for j in jobs if j[3] == 'train')}, "
        f"seqval={sum(1 for j in jobs if j[3] == 'seqval')})."
    )
    if summary["skipped_train"]:
        print(f"[extract] {len(summary['skipped_train'])} train TBs skipped (not found).")
    if summary["skipped_eval"]:
        print(f"[extract] {len(summary['skipped_eval'])} seqval TBs skipped (not found).")

    tag_filter = args.tag  # list or None
    payloads = [
        (
            label, tb_dir, out_path, kind, meta,
            tag_filter, args.use_only_latest, args.skip_existing,
        )
        for (label, tb_dir, out_path, kind, meta) in jobs
    ]

    results = []
    if args.num_workers <= 1:
        for p in payloads:
            results.append(_do_one_job(p))
            label, kind, status, info, _meta = results[-1]
            print(f"  [{kind}] {label}: {status} ({info})")
            if status == "error" and args.strict:
                raise RuntimeError(f"extraction failed for {label}: {info}")
    else:
        with ProcessPoolExecutor(max_workers=int(args.num_workers)) as ex:
            futures = {ex.submit(_do_one_job, p): p[0] for p in payloads}
            for fut in as_completed(futures):
                label, kind, status, info, _meta = fut.result()
                results.append((label, kind, status, info, _meta))
                print(f"  [{kind}] {label}: {status} ({info})")
                if status == "error" and args.strict:
                    raise RuntimeError(f"extraction failed for {label}: {info}")

    counts = {"ok": 0, "skipped_existing": 0, "error": 0}
    errors = []
    for label, kind, status, info, meta in results:
        counts[status] = counts.get(status, 0) + 1
        if status == "error":
            errors.append({"label": label, "kind": kind, "error": info, "meta": meta})

    manifest = {
        "args": vars(args),
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "summary": summary,
        "counts": counts,
        "errors": errors,
    }
    manifest_path = os.path.join(args.cache_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[extract] wrote {manifest_path}")
    print(f"[extract] done. ok={counts.get('ok', 0)} "
          f"skipped_existing={counts.get('skipped_existing', 0)} "
          f"error={counts.get('error', 0)}")


if __name__ == "__main__":
    main()
