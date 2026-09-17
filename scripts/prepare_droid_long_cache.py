"""Infer native 15-step val30 caches and extend frozen DINO expert chunks."""

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import time

from cache_droid_datasets import REPO, execute, save_json
from tune_droid_hp import sha256

sys.path.insert(0, str(REPO / "src"))


def input_rows(path):
    return json.loads(path.read_text())["results"]


def cache_matrix(args):
    grouped = {}
    for row in input_rows(args.baseline_json):
        if row["dataset"].endswith("val30"):
            grouped.setdefault(row["dataset"], []).append(row)
    if len(grouped) != 3:
        raise ValueError("Expected three val30 tasks")
    jobs = []
    for index, (dataset, rows) in enumerate(sorted(grouped.items())):
        rows = sorted(rows, key=lambda r: r["epoch"])
        if [r["epoch"] for r in rows] != list(range(5, 51, 5)):
            raise ValueError("Need all ten checkpoints")
        name = dataset.replace("/droid/", "_")
        directory = args.output_root / "conditions" / name / "seed_0"
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--root", str(args.root),
                   "--baseline-json", str(args.baseline_json), "--worker-index", str(index)]
        jobs.append({"name": name, "dataset": dataset, "val_demos": 30, "epochs": list(range(5, 51, 5)),
                     "job_dir": str(directory), "checkpoint_dir": str(Path(rows[0]["checkpoint"]).parent),
                     "command": command, "standard_validation": False, "cache_ta": 15})
    return {"jobs": jobs, "total_jobs": 3, "missing_checkpoint_tasks": [], "horizon": 15,
            "samples": 8, "seed": 0, "gpu": "0", "standard_validation": False}


def infer_worker(job):
    import torch
    # Shared GPU: do not evict unrelated jobs; cap allocation and wait for headroom.
    while torch.cuda.mem_get_info()[0] < 6 * 2**30:
        print("WAIT GPU free memory <6GiB", flush=True)
        time.sleep(10)
    torch.cuda.set_per_process_memory_fraction(0.06)
    from robomimic.scripts.sequential_cache_checkpoints import main
    main(["--checkpoint-dir", job["checkpoint_dir"], "--data-dir", "/workspace/data",
          "--dataset-name", job["dataset"], "--cache-dir", str(Path(job["job_dir"]) / "cache"),
          "--cache-ta", "15", "--num-cache-samples", "8", "--batch-size", "32", "--seed", "0",
          "--num-threads", "4", "--loader-parallelism", "4", "--no-run-standard-validation"])


def prepare(args, task=None):
    import numpy as np
    from surval.droid_dino import align_dino_cache, dataset_db_path, load_shared_dino_db, save_shared_dino_db
    from surval.droid import load_policy_cache
    rows = [r for r in input_rows(args.baseline_json) if r["dataset"].endswith("val30")]
    if task is not None:
        rows = [r for r in rows if task in r["dataset"].split("/")[0].split("_")]
        if len(rows) != 10:
            raise ValueError("Expected ten val30 rows for requested task")
    input_hashes = {row["cache_file"]: sha256(row["cache_file"]) for row in rows}
    sources = {}
    for job in cache_matrix(args)["jobs"]:
        for path in (Path(job["job_dir"]) / "cache").rglob("seqcache_epoch_*.hdf5"):
            epoch = int(path.stem.rsplit("_", 1)[1])
            sources[job["dataset"].split("/droid/")[0], epoch] = path
    expected = {(r["dataset"].split("/droid/")[0], r["epoch"]) for r in rows}
    if not expected.issubset(sources):
        raise ValueError("Need all requested completed long caches")
    suffix = "" if task is None else "_" + task
    result, databases, proofs = [], {}, []
    for row in sorted(rows, key=lambda r: (r["dataset"], r["epoch"])):
        task = row["dataset"].split("/droid/")[0]
        target = sources[task, row["epoch"]]
        cache, historical = load_policy_cache(target), load_policy_cache(row["cache_file"])
        if cache["checkpoint"] != historical["checkpoint"] or cache["epoch"] != historical["epoch"]:
            raise ValueError("Checkpoint identity changed")
        if cache["actions"].shape[1:] != (15, 10) or cache["pred_actions"].shape[0] != 8:
            raise ValueError("Expected genuine15-step/eight-sample predictions")
        for key in ("demo_ids", "index_in_demo", "action_scale", "action_offset"):
            np.testing.assert_array_equal(cache[key], historical[key])
        np.testing.assert_array_equal(cache["actions"][:, :8], historical["actions"])
        dataset = row["dataset"]
        if dataset not in databases:
            old_path = dataset_db_path(args.dino_root, dataset)
            db, old_chunks, meta = load_shared_dino_db(old_path)
            prefix = {**cache, "actions": cache["actions"][:, :8]}
            order = align_dino_cache(db, old_chunks, meta, prefix)
            inverse = np.argsort(order)
            np.testing.assert_array_equal(cache["actions"][inverse, 0], db.records.actions)
            cfg = replace(db.cfg, action_chunk_size=15)
            db.cfg = cfg
            destination = dataset_db_path(args.root / "shared_dino", dataset)
            metadata = {**meta, "parent_dino_content_sha256": meta["content_sha256"],
                        "long_gt_source_sha256": sha256(target), "action_chunk_size": 15}
            if not destination.exists():
                save_shared_dino_db(destination, db, cache["actions"][inverse], metadata)
            extended, chunks, extended_meta = load_shared_dino_db(destination)
            np.testing.assert_array_equal(extended.embeddings, db.embeddings)
            np.testing.assert_array_equal(chunks[:, :8], old_chunks)
            databases[dataset] = extended, chunks, extended_meta
            proofs.append({"dataset": dataset, "rows": db.n_states, "embeddings_bitwise_unchanged": True,
                           "gt_prefix_bitwise_unchanged": True, "parent_dino": str(old_path)})
        align_dino_cache(*databases[dataset], cache)
        # Historical baselines are explicitly separate; long caches have no inherited Loss/OMN.
        result.append({**row, "cache_file": str(target.resolve()), "historical_cache_file": row["cache_file"],
                       "historical_surval_score": row["surval_score"], "num_rows": len(cache["actions"]),
                       "num_samples": 8, "ta": 15})
    if {p: sha256(p) for p in input_hashes} != input_hashes:
        raise AssertionError("Historical inputs changed")
    save_json(args.root / f"cache_index{suffix}.json", {"results": result, "horizon": 15,
                                               "sampling": "val30-only-seed0", "baseline_values": "historical-only"})
    save_json(args.root / f"cache_verification{suffix}.json", {"complete": True, "checkpoints": len(rows),
                                                     "inferred_checkpoints": len(rows), "num_samples": 8,
                                                     "horizon": 15, "datasets": proofs,
                                                     "historical_cache_sha256": input_hashes})
    print(f"PREPARE complete {len(rows)} caches, {len(databases)} unchanged DINO DBs, 15-stepGT", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO / "outputs/droid_c1_long_20260913")
    parser.add_argument("--baseline-json", type=Path, default=REPO / "outputs/droid_baselines_20260913/checkpoint_metrics.json")
    parser.add_argument("--dino-root", type=Path, default=REPO / "outputs/shared_dino_val")
    parser.add_argument("--worker-index", type=int)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    args.root = args.root.resolve()
    args.output_root = args.root / "cache_generation"
    args.gpu, args.num_cache_samples, args.num_shards, args.shard_idx = "0", 8, 1, 0
    specification = cache_matrix(args)
    if args.list or os.environ.get("DRY_RUN") == "1":
        print(json.dumps(specification, indent=2))
        return 0
    if args.worker_index is not None:
        infer_worker(specification["jobs"][args.worker_index])
        return 0
    if not args.prepare_only:
        code = execute(args, specification)
        if code:
            return code
    prepare(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
