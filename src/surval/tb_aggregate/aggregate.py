"""
Seed-level aggregation orchestration for the cached two-stage workflow.

The aggregate step reads the JSON cache (no tensorboard import needed),
computes per-seed proxy metrics for each (HP, seqval_tag) combination, and
emits aggregated JSON / CSV outputs.

Supports multiple --seqval_tag values by including the tag in the hp_key
(`tag=...|dataset=...|...`), so downstream pivots can split on it.
"""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from .layouts import iter_seqval_eval_dirs
from .metrics import (
    bootstrap_ci_of_mean,
    compute_seed_metrics,
    leave_one_seed_out_stats,
    nanmean,
    nanmedian,
    nanstd,
)
from .tb_io import CachedScalarLoader


# ---------------------------------------------------------------------------
# Per-seed driver
# ---------------------------------------------------------------------------


def _build_skip(reason: str, **fields):
    rec = {"reason": reason}
    rec.update(fields)
    return rec


def process_one_seed(
    args,
    seed,
    train_seed_dir,
    eval_seed_dir,
    loader: CachedScalarLoader,
    parse_fn,
    iter_fn,
):
    """
    For one seed, compute per-hp metrics across every seqval_tag.

    Returns:
        per_hp: dict[hp_key_with_tag -> row]   (caller merges by seed)
        skipped: list[skip records]
    """
    per_hp: dict = {}
    skipped: list = []
    seed_name = os.path.basename(eval_seed_dir)

    # --- train cache ---
    try:
        train_payload = loader.load_train(seed_name)
    except FileNotFoundError as e:
        skipped.append(_build_skip(
            "train_cache_not_found",
            seed_name=seed_name, train_seed_dir=train_seed_dir, error=str(e),
        ))
        return per_hp, skipped

    try:
        _, train_vals = loader.get_series(train_payload, args.train_tag)
    except KeyError as e:
        skipped.append(_build_skip(
            "train_tag_load_failed",
            seed_name=seed_name, train_seed_dir=train_seed_dir, error=str(e),
        ))
        return per_hp, skipped

    train_success_counts = None
    if args.train_success_count_tag is not None:
        try:
            _, train_success_counts = loader.get_series(
                train_payload, args.train_success_count_tag
            )
        except KeyError as e:
            skipped.append(_build_skip(
                "train_success_count_tag_load_failed",
                seed_name=seed_name, train_seed_dir=train_seed_dir, error=str(e),
            ))
            return per_hp, skipped
        if len(train_success_counts) != len(train_vals):
            skipped.append(_build_skip(
                "train_success_count_length_mismatch",
                seed_name=seed_name,
                train_seed_dir=train_seed_dir,
                train_rate_len=int(len(train_vals)),
                train_count_len=int(len(train_success_counts)),
            ))
            return per_hp, skipped

    # --- per HP / cache-group / seqval_tag ---
    for child_path, hp_dir_name, cache_group_dir_name in iter_fn(eval_seed_dir):
        parsed = parse_fn(hp_dir_name, cache_group_dir_name)
        if parsed is None:
            skipped.append(_build_skip(
                "hparam_parse_failed",
                seed_name=seed_name,
                eval_seed_dir=eval_seed_dir,
                seqval_dir=child_path,
            ))
            continue
        hp_key_base, hp_dict = parsed

        try:
            seq_payload = loader.load_seqval(
                seed_name, hp_dir_name, cache_group_dir_name
            )
        except FileNotFoundError as e:
            skipped.append(_build_skip(
                "seqval_cache_not_found",
                seed_name=seed_name,
                eval_seed_dir=eval_seed_dir,
                seqval_dir=child_path,
                error=str(e),
            ))
            continue

        force_train = set(getattr(args, "force_train_for_tag", None) or [])
        force_seqcache = set(getattr(args, "force_seqcache_for_tag", None) or [])

        def _try_seqcache_metrics(tag):
            try:
                payload = loader.load_seqcache_metrics(seed_name, cache_group_dir_name)
            except FileNotFoundError as exc:
                raise KeyError(str(exc)) from exc
            return loader.get_series(payload, tag)

        for seqval_tag in args.seqval_tag:
            # Fallback order:
            #   --force_seqcache_for_tag → seqcache_metrics cache only
            #   --force_train_for_tag    → train cache only
            #   default                  → seqval TB cache, then seqcache
            #                              metrics (per-dataset HDF5-derived),
            #                              then train TB cache.
            seq_source = None
            seq_vals = None
            err_chain: list = []
            if seqval_tag in force_seqcache:
                try:
                    _, seq_vals = _try_seqcache_metrics(seqval_tag)
                    seq_source = "seqcache_forced"
                except KeyError as e:
                    err_chain.append(("seqcache_metrics", str(e)))
            elif seqval_tag in force_train:
                try:
                    _, seq_vals = loader.get_series(train_payload, seqval_tag)
                    seq_source = "train_forced"
                except KeyError as e:
                    err_chain.append(("train", str(e)))
            else:
                try:
                    _, seq_vals = loader.get_series(seq_payload, seqval_tag)
                    seq_source = "seqval"
                except KeyError as e:
                    err_chain.append(("seqval", str(e)))
                    try:
                        _, seq_vals = _try_seqcache_metrics(seqval_tag)
                        seq_source = "seqcache_fallback"
                    except KeyError as e2:
                        err_chain.append(("seqcache_metrics", str(e2)))
                        try:
                            _, seq_vals = loader.get_series(train_payload, seqval_tag)
                            seq_source = "train_fallback"
                        except KeyError as e3:
                            err_chain.append(("train", str(e3)))
            if seq_source is None:
                skipped.append(_build_skip(
                    "seqval_tag_load_failed",
                    seed_name=seed_name,
                    eval_seed_dir=eval_seed_dir,
                    seqval_dir=child_path,
                    seqval_tag=seqval_tag,
                    error_chain=err_chain,
                ))
                continue

            negate_all = bool(getattr(args, "negate_seqval_tag", False))
            negate_set = set(getattr(args, "negate_tag", None) or [])
            if negate_all or (seqval_tag in negate_set):
                seq_vals = -seq_vals

            if len(train_vals) != len(seq_vals):
                skipped.append(_build_skip(
                    "length_mismatch",
                    seed_name=seed_name,
                    train_seed_dir=train_seed_dir,
                    eval_seed_dir=eval_seed_dir,
                    seqval_dir=child_path,
                    seqval_tag=seqval_tag,
                    train_len=int(len(train_vals)),
                    seqval_len=int(len(seq_vals)),
                ))
                continue

            if len(train_vals) < args.min_steps:
                skipped.append(_build_skip(
                    "not_enough_steps",
                    seed_name=seed_name,
                    train_seed_dir=train_seed_dir,
                    eval_seed_dir=eval_seed_dir,
                    seqval_dir=child_path,
                    seqval_tag=seqval_tag,
                    num_steps=int(len(train_vals)),
                ))
                continue

            if args.train_tag_is_rate and (
                np.any(train_vals < 0.0) or np.any(train_vals > 1.0)
            ):
                skipped.append(_build_skip(
                    "train_rate_out_of_range",
                    seed_name=seed_name,
                    train_seed_dir=train_seed_dir,
                    eval_seed_dir=eval_seed_dir,
                    seqval_dir=child_path,
                    seqval_tag=seqval_tag,
                ))
                continue

            try:
                flat = compute_seed_metrics(
                    a_rate=train_vals,
                    b_vals=seq_vals,
                    k_list=args.k_list,
                    eps=args.eps,
                    tie_tol=args.tie_tol,
                    nan_policy=args.nan_policy,
                    use_posterior_success=args.use_posterior_success,
                    n_rollouts=args.n_rollouts,
                    num_mc=args.num_mc,
                    beta_prior_a=args.beta_prior_a,
                    beta_prior_b=args.beta_prior_b,
                    a_success_counts=train_success_counts,
                    compute_kendall=(not args.disable_kendall),
                    compute_delta_kendall=args.enable_delta_kendall,
                    rng=np.random.default_rng(args.bootstrap_seed + int(seed)),
                )
            except Exception as e:
                skipped.append(_build_skip(
                    "metric_compute_failed",
                    seed_name=seed_name,
                    train_seed_dir=train_seed_dir,
                    eval_seed_dir=eval_seed_dir,
                    seqval_dir=child_path,
                    seqval_tag=seqval_tag,
                    error=str(e),
                ))
                continue

            hp_dict_tag = dict(hp_dict)
            hp_dict_tag["seqval_tag"] = seqval_tag
            hp_dict_tag["proxy_source"] = seq_source
            hp_key = f"tag={seqval_tag}|{hp_key_base}"
            per_hp[hp_key] = {
                "seed_name": seed_name,
                "train_seed_dir": train_seed_dir,
                "eval_seed_dir": eval_seed_dir,
                "seqval_dir": child_path,
                "hparams": hp_dict_tag,
                "n_steps": int(len(train_vals)),
                "metrics": flat,
            }

    return per_hp, skipped


# ---------------------------------------------------------------------------
# Worker / loop
# ---------------------------------------------------------------------------


def _mp_worker(payload):
    args, seed, train_seed_dir, eval_seed_dir, cache_dir = payload
    # Recreate non-picklable objects in worker.
    loader = CachedScalarLoader(cache_dir)
    from .layouts import iter_seqval_eval_dirs as _iter, parse_hparam as _parse
    per_hp, sk = process_one_seed(
        args, seed, train_seed_dir, eval_seed_dir,
        loader=loader, parse_fn=_parse, iter_fn=_iter,
    )
    return seed, per_hp, sk


def run_seed_loop(
    args,
    common_seeds,
    train_seed_dirs,
    eval_seed_dirs,
    cache_dir: str,
    parse_fn,
    iter_fn,
):
    per_hp_per_seed: dict = defaultdict(dict)
    skipped: list = []

    if args.num_workers <= 1:
        loader = CachedScalarLoader(cache_dir)
        for seed in common_seeds:
            per_hp, sk = process_one_seed(
                args, seed, train_seed_dirs[seed], eval_seed_dirs[seed],
                loader=loader, parse_fn=parse_fn, iter_fn=iter_fn,
            )
            for hk, row in per_hp.items():
                per_hp_per_seed[hk][seed] = row
            skipped.extend(sk)
        return per_hp_per_seed, skipped

    nw = min(int(args.num_workers), len(common_seeds))
    payloads = [
        (args, seed, train_seed_dirs[seed], eval_seed_dirs[seed], cache_dir)
        for seed in common_seeds
    ]
    with ProcessPoolExecutor(max_workers=nw) as ex:
        futures = [ex.submit(_mp_worker, p) for p in payloads]
        for fut in as_completed(futures):
            seed, per_hp, sk = fut.result()
            for hk, row in per_hp.items():
                per_hp_per_seed[hk][seed] = row
            skipped.extend(sk)

    return per_hp_per_seed, skipped


# ---------------------------------------------------------------------------
# Printing / saving
# ---------------------------------------------------------------------------


def _fmt_float(x, digits=4):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "nan"
    return f"{float(x):.{digits}f}"


def _get_print_metric_names(aggregated, print_all_metrics):
    if print_all_metrics:
        return sorted({m for info in aggregated.values() for m in info["metrics"].keys()})
    default = [
        "spearman", "delta_spearman", "kendall_tau",
        "hit@1", "hit@3", "hit@5",
        "nregret", "rank_pct", "mmrv",
    ]
    return [m for m in default if any(m in info["metrics"] for info in aggregated.values())]


def _sorted_aggregate_items(aggregated, sort_by):
    # Lower-is-better metrics — sort ascending.
    lower_is_better = {"mmrv", "nregret", "rank_pct"}
    reverse = sort_by not in lower_is_better

    def _key(item):
        _, hp_info = item
        if sort_by in hp_info["metrics"]:
            v = hp_info["metrics"][sort_by]["mean"]
            if v is None or np.isnan(v):
                return -np.inf if reverse else np.inf
            return v
        return -np.inf if reverse else np.inf

    return sorted(aggregated.items(), key=_key, reverse=reverse)


def print_aggregated_table(aggregated, print_all_metrics=False, sort_by="spearman"):
    if not aggregated:
        print("\n[Aggregate] No aggregated results.")
        return

    metric_names = _get_print_metric_names(aggregated, print_all_metrics)
    items = _sorted_aggregate_items(aggregated, sort_by)
    print(f"\n[Aggregate] Hyperparameter groups: {len(items)}")
    print(f"[Aggregate] Sorted by: {sort_by}")
    print(f"[Aggregate] Showing metrics: {', '.join(metric_names)}\n")

    for rank, (hp_key, hp_info) in enumerate(items, 1):
        print(f"[{rank:02d}] {hp_key} (num_seeds={hp_info['num_seeds']})")
        parts = []
        for m in metric_names:
            if m not in hp_info["metrics"]:
                continue
            stat = hp_info["metrics"][m]
            parts.append(
                f"{m}=mean {_fmt_float(stat['mean'])} "
                f"[CI {_fmt_float(stat['ci95_low'])}..{_fmt_float(stat['ci95_high'])}], "
                f"median {_fmt_float(stat['median'])}, "
                f"std {_fmt_float(stat['std'])} (n={stat['n']})"
            )
        print("     " + " | ".join(parts))
    print("")


def save_print_aggregate_csv(aggregated, out_path, print_all_metrics=False, sort_by="spearman"):
    if not aggregated:
        return False
    metric_names = _get_print_metric_names(aggregated, print_all_metrics)
    items = _sorted_aggregate_items(aggregated, sort_by)
    rows = []

    def _display_label(hparams):
        keys = [
            "seqval_tag", "dataset", "cache_mode", "cache_split",
            "cache_vkey", "eval_first_n_demos",
            "tq", "lsetau", "ctf", "ta", "nds",
            "share", "skip", "dist",
        ]
        present = [k for k in keys if k in hparams]
        return ",".join(f"{k}={hparams[k]}" for k in present) or "hparams_unknown"

    for rank, (hp_key, hp_info) in enumerate(items, 1):
        row = {
            "rank": rank,
            "hp_key": hp_key,
            "display_label": _display_label(hp_info["hparams"]),
            "num_seeds": hp_info["num_seeds"],
            "sort_by": sort_by,
        }
        row.update(hp_info["hparams"])
        for m in metric_names:
            stat = hp_info["metrics"].get(m, {})
            row[f"{m}_mean"] = stat.get("mean", np.nan)
            row[f"{m}_ci95_low"] = stat.get("ci95_low", np.nan)
            row[f"{m}_ci95_high"] = stat.get("ci95_high", np.nan)
            row[f"{m}_median"] = stat.get("median", np.nan)
            row[f"{m}_std"] = stat.get("std", np.nan)
            row[f"{m}_n"] = stat.get("n", 0)
        rows.append(row)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return True


def finalize_aggregate(
    args,
    per_hp_per_seed,
    skipped,
    train_seed_dirs,
    eval_seed_dirs,
    common_seeds,
):
    rng = np.random.default_rng(args.bootstrap_seed)

    aggregated = {}
    for hp_key, seed_map in per_hp_per_seed.items():
        seed_items = sorted(seed_map.items(), key=lambda kv: kv[0])
        metric_names = sorted(seed_items[0][1]["metrics"].keys()) if seed_items else []
        per_metric_values = {m: [] for m in metric_names}
        for _, item in seed_items:
            for m in metric_names:
                per_metric_values[m].append(item["metrics"].get(m, np.nan))

        agg_metrics = {}
        for m in metric_names:
            vals = np.asarray(per_metric_values[m], dtype=np.float64)
            valid = vals[~np.isnan(vals)]
            stat = {
                "mean": nanmean(vals),
                "std": nanstd(vals),
                "median": nanmedian(vals),
                "n": int(len(valid)),
            }
            ci_low, ci_high = bootstrap_ci_of_mean(
                valid,
                num_bootstrap=args.num_boot,
                ci_level=args.ci_level,
                rng=rng,
            )
            stat["ci95_low"] = ci_low
            stat["ci95_high"] = ci_high
            if args.report_loo:
                stat["loo"] = leave_one_seed_out_stats(valid)
            agg_metrics[m] = stat

        aggregated[hp_key] = {
            "hparams": seed_items[0][1]["hparams"] if seed_items else {},
            "num_seeds": len(seed_items),
            "seeds": [int(s) for s, _ in seed_items],
            "metrics": agg_metrics,
        }

    output = {
        "args": vars(args),
        "num_train_seed_dirs": len(train_seed_dirs),
        "num_eval_seed_dirs": len(eval_seed_dirs),
        "num_common_seed_dirs": len(common_seeds),
        "common_seeds": [int(s) for s in common_seeds],
        "num_hparam_groups": len(aggregated),
        "aggregated": aggregated,
        "per_hparam_per_seed": {
            hp_key: {str(seed): data for seed, data in seed_map.items()}
            for hp_key, seed_map in per_hp_per_seed.items()
        },
        "skipped": skipped,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved JSON to {args.out_json}")

    if args.print_aggregate:
        print_aggregated_table(
            aggregated=aggregated,
            print_all_metrics=args.print_all_metrics,
            sort_by=args.print_sort_by,
        )

    print_csv_path = args.out_print_csv
    if (print_csv_path is None) and args.out_csv:
        root, ext = os.path.splitext(args.out_csv)
        ext = ext if ext else ".csv"
        print_csv_path = root + "_print" + ext
    if print_csv_path is not None:
        ok = save_print_aggregate_csv(
            aggregated=aggregated,
            out_path=print_csv_path,
            print_all_metrics=args.print_all_metrics,
            sort_by=args.print_sort_by,
        )
        if ok:
            print(f"Saved print-style aggregate CSV to {print_csv_path}")
        else:
            print("No aggregated rows to write for print-style aggregate CSV.")

    if args.out_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)) or ".", exist_ok=True)
        rows = []
        for hp_key, hp_info in aggregated.items():
            base = {"hp_key": hp_key, "num_seeds": hp_info["num_seeds"]}
            base.update(hp_info["hparams"])
            for metric_name, stat in hp_info["metrics"].items():
                row = dict(base)
                row["metric"] = metric_name
                row["mean"] = stat["mean"]
                row["std"] = stat["std"]
                row["median"] = stat.get("median", np.nan)
                row["ci95_low"] = stat.get("ci95_low", np.nan)
                row["ci95_high"] = stat.get("ci95_high", np.nan)
                row["n"] = stat["n"]
                rows.append(row)

        if rows:
            # Unify fieldnames across rows (hparams may differ across hp_keys).
            fieldnames: list = []
            seen = set()
            for r in rows:
                for k in r.keys():
                    if k not in seen:
                        seen.add(k)
                        fieldnames.append(k)
            with open(args.out_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            print(f"Saved CSV to {args.out_csv}")
        else:
            print("No rows to write for CSV.")
