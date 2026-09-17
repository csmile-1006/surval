"""Resumable CPU grid scoring, one worker per task/validation split."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
import threading
import time

from cache_droid_datasets import REPO, now, run_worker, save_json

sys.path.insert(0, str(REPO / "src"))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def matrix(args):
    from surval.droid_tuning import AXES, FIXED, VERSION, hp_grid
    axes = json.loads(args.axes_json.read_text()) if args.axes_json else AXES
    grid = hp_grid(axes)
    baseline = json.loads(args.baseline_json.read_text())
    grouped = defaultdict(list)
    for row in baseline["results"]:
        grouped[row["dataset"]].append(row)
    jobs = []
    for index, (dataset, rows) in enumerate(sorted(grouped.items())):
        task = [t for t in ("apple", "pan", "pet") if t in dataset.split("/")[0].split("_")]
        if len(task) != 1:
            raise ValueError(f"Unexpected task {dataset}")
        rows = sorted(rows, key=lambda row: row["epoch"])
        if [r["epoch"] for r in rows] != list(range(5, 51, 5)):
            raise ValueError(f"Expected ten checkpoints at epochs5..50: {dataset}")
        name = dataset.replace("/droid/", "_")
        directory = args.output_root / "conditions" / name / "seed_0"
        command = [sys.executable, "-u", str(Path(__file__).resolve()),
                   "--baseline-json", str(args.baseline_json), "--dino-root", str(args.dino_root),
                   "--output-root", str(args.output_root), "--worker-index", str(index)]
        if args.axes_json:
            command += ["--axes-json", str(args.axes_json)]
        jobs.append({"index": index, "name": name, "dataset": dataset, "task": task[0],
                     "val": int(dataset.rsplit("val", 1)[1]), "rows": rows,
                     "job_dir": str(directory), "command": command})
    return {"version": VERSION, "axes": axes, "fixed": FIXED, "grid_size": len(grid),
            "jobs": jobs, "total_jobs": len(jobs), "device": "cpu", "wandb": None,
            "seed_meaning": "cached diffusion draws only; not a new training seed"}


def score_worker(args, job, specification):
    import numpy as np
    from surval.droid import load_policy_cache, score_policy_cache
    from surval.droid_dino import align_dino_cache, dataset_db_path, load_shared_dino_db
    from surval.droid_tuning import (canonical_score, canonical_thresholds, expert_distance_matrices,
                                    hp_grid, local_scale_table, score_grid)
    from surval.local_threshold.threshold import _block_pair_distances, query_threshold_neighbors
    root = Path(job["job_dir"])
    root.mkdir(parents=True, exist_ok=True)
    output = root / "eval"
    output.mkdir(exist_ok=True)
    started = time.monotonic()
    cache_hashes = {r["cache_file"]: sha256(r["cache_file"]) for r in job["rows"]}
    code_paths = ["src/surval/droid_tuning.py", "src/surval/sequential_validate.py",
                  "src/surval/local_threshold/threshold.py", "src/surval/droid.py",
                  "src/surval/droid_dino.py", "scripts/tune_droid_hp.py"]
    codes = {p: sha256(REPO / p) for p in code_paths}
    db, chunks, meta = load_shared_dino_db(dataset_db_path(args.dino_root, job["dataset"]))
    signature = {"axes": specification["axes"], "fixed": specification["fixed"],
                 "cache_sha256": cache_hashes, "code_sha256": codes,
                 "db_content_sha256": meta["content_sha256"]}
    previous = output / "verification.json"
    if previous.exists():
        saved = json.loads(previous.read_text())
        if saved["signature"] != signature:
            raise ValueError("Completed worker provenance changed; use a new output root")
        if sha256(output / "scores.npz") != saved["scores_sha256"]:
            raise ValueError("Saved scores checksum mismatch")
        print("RESUME verified complete worker", flush=True)
        return
    save_json(root / "manifest.json", {**job, "signature": signature})
    axes = specification["axes"]
    matrices = expert_distance_matrices(db)
    tables, cfgs, threshold_error = {}, {}, 0.0
    for k in axes["k"]:
        tables[k], cfgs[k] = local_scale_table(db, matrices, k, axes["quantile"])
        neighbors = query_threshold_neighbors(db, cfgs[k], exact_neighbors=True)
        # Independent original pairwise routine checks 16 states for EVERY k/q/block.
        for row in np.linspace(0, db.n_states - 1, min(16, db.n_states), dtype=int):
            for bi, (name, sl) in enumerate(db.cfg.block_slice_dict().items()):
                pairs = _block_pair_distances(db.records.actions[neighbors[row]], sl, pairwise=True,
                                             block_type=db.cfg.block_type_dict().get(name))
                expected = np.quantile(pairs, axes["quantile"]).astype(np.float32)
                threshold_error = max(threshold_error, float(np.max(abs(expected - tables[k][row, bi]))))
                np.testing.assert_allclose(tables[k][row, bi], expected, rtol=1e-6, atol=1e-7)
        print(f"THRESHOLD k={k} rows={db.n_states} elapsed={time.monotonic()-started:.1f}s", flush=True)
    del matrices
    scores = np.empty((len(hp_grid(axes)), len(job["rows"])), np.float64)
    old_every_step, old_stride, order_signatures = [], [], []
    smoke_error = 0.0
    for col, row in enumerate(job["rows"]):
        cache = load_policy_cache(row["cache_file"])
        if cache["pred_actions"].shape[0] != 8 or cache["actions"].shape[1] != 8:
            raise ValueError("Expected all eight samples and eight cached action offsets")
        order = align_dino_cache(db, chunks, meta, cache)
        order_signatures.append(hashlib.sha256(order.tobytes()).hexdigest())
        scores[:, col] = score_grid(cache, {k: v[order] for k, v in tables.items()}, axes)
        control, episodes = score_policy_cache(cache, row["state_db_dir"], every_step=True)
        if sum(e["T"] for e in episodes) != db.n_states:
            raise AssertionError("Buffered control skipped rows")
        old_every_step.append(control["PrefixSurvival_Score"])
        old_stride.append(row["surval_score"])
        if col == 0:
            hp = {"ta": axes["ta"][-1], "k": axes["k"][0], "quantile": axes["quantile"][0],
                  "scale_multiplier": axes["scale_multiplier"][0]}
            target = output / "canonical_smoke"
            threshold = canonical_thresholds(db, axes["quantile"], hp["k"], target)
            reference = canonical_score(cache, target, threshold, hp)
            fast = scores[hp_grid(axes).index(hp), col]
            smoke_error = abs(reference - fast)
            np.testing.assert_allclose(fast, reference, atol=2e-7, rtol=2e-6)
        print(f"SCORE epoch={row['epoch']} configs={len(scores)} elapsed={time.monotonic()-started:.1f}s", flush=True)
    if {p: sha256(p) for p in cache_hashes} != cache_hashes:
        raise AssertionError("Source cache changed during scoring")
    temporary = output / "scores.npz.tmp"
    with temporary.open("wb") as f:
        np.savez_compressed(f, scores=scores, old_every_step=old_every_step, old_stride=old_stride,
                            epochs=[r["epoch"] for r in job["rows"]])
    temporary.replace(output / "scores.npz")
    with (output / "scales.npz").open("wb") as f:
        np.savez_compressed(f, **{str(k): v for k, v in tables.items()})
    save_json(previous, {"complete": True, "signature": signature,
                        "scores_sha256": sha256(output / "scores.npz"),
                        "scales_sha256": sha256(output / "scales.npz"),
                        "grid_size": len(scores), "checkpoints": scores.shape[1], "rows": db.n_states,
                        "num_samples": 8, "horizon": 8, "every_step": True, "exact_k": True,
                        "fallback_rows": 0, "threshold_max_abs_error": threshold_error,
                        "canonical_smoke_abs_error": smoke_error, "cache_order_hashes": order_signatures,
                        "elapsed_seconds": time.monotonic() - started})


def execute(args, specification):
    root = args.output_root
    root.mkdir(parents=True, exist_ok=True)
    suffix = f".shard_{args.shard_idx}" if args.num_shards > 1 else ""
    jobs = [j for j in specification["jobs"] if j["index"] % args.num_shards == args.shard_idx]
    with (root / f"runner{suffix}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        save_json(root / f"run_manifest{suffix}.json", {**specification, "num_shards": args.num_shards,
                                                       "shard_idx": args.shard_idx, "workers": args.workers})
        states = {j["name"]: {"name": j["name"], "state": "queued"} for j in jobs}
        mutex = threading.Lock()

        def emit(event, job=None, **details):
            with mutex:
                if job is not None:
                    states[job["name"]].update(details, updated_at=now())
                    save_json(Path(job["job_dir"]) / "status.json", states[job["name"]])
                counts = dict(Counter(s["state"] for s in states.values()))
                record = {"time": now(), "event": event, "counts": counts,
                          "job": job["name"] if job else None, **details}
                line = json.dumps(record)
                print(line, flush=True)
                for name in (f"events{suffix}.jsonl", f"orchestrator{suffix}.log"):
                    with (root / name).open("a") as f:
                        f.write(line + "\n")
                save_json(root / f"run_status{suffix}.json",
                          {"state": "failed" if counts.get("failed") else
                           "complete" if counts.get("complete", 0) == len(jobs) else "running",
                           "updated_at": now(), "counts": counts, "jobs": list(states.values())})

        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2",
               "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "TF_CPP_MIN_LOG_LEVEL": "3",
               "PYTHONUNBUFFERED": "1"}

        def worker(job):
            directory = Path(job["job_dir"])
            directory.mkdir(parents=True, exist_ok=True)
            emit("START", job, state="running")
            try:
                code = run_worker(job["command"], directory / "job.log", env,
                                  lambda pid: emit("STATUS", job, pid=pid))
                if code:
                    tail = "\n".join((directory / "job.log").read_text().splitlines()[-12:])
                    emit("FAIL", job, state="failed", exit_code=code, tail=tail)
                    return False
                verification = json.loads((directory / "eval/verification.json").read_text())
                if not verification["complete"]:
                    raise ValueError("Worker did not verify completion")
                emit("DONE", job, state="complete", exit_code=0)
                return True
            except Exception as error:
                emit("FAIL", job, state="failed", error=str(error))
                return False

        emit("MATRIX", selected_jobs=len(jobs), grid_size=specification["grid_size"])
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(worker, jobs))
        emit("FINISH")
        return 0 if all(results) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-json", type=Path, default=REPO / "outputs/droid_baselines_20260913/checkpoint_metrics.json")
    parser.add_argument("--dino-root", type=Path, default=REPO / "outputs/shared_dino_val")
    parser.add_argument("--output-root", type=Path, default=REPO / "outputs/droid_hp_every_step_20260913")
    parser.add_argument("--axes-json", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-idx", type=int, default=0)
    parser.add_argument("--worker-index", type=int)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.num_shards < 1 or not 0 <= args.shard_idx < args.num_shards:
        parser.error("Invalid workers/shard configuration")
    args.output_root = args.output_root.resolve()
    specification = matrix(args)
    if args.list or os.environ.get("DRY_RUN") == "1":
        selected = [j for j in specification["jobs"] if j["index"] % args.num_shards == args.shard_idx]
        print(json.dumps({k: v for k, v in specification.items() if k != "jobs"}))
        print(f"selected_jobs={len(selected)} output_root={args.output_root}")
        for job in selected:
            print(shlex.join(job["command"]))
        return 0
    if args.worker_index is not None:
        score_worker(args, specification["jobs"][args.worker_index], specification)
        return 0
    return execute(args, specification)


if __name__ == "__main__":
    raise SystemExit(main())
