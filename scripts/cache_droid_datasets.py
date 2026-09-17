"""Run task-matched DROID validation caches, one task/split per GPU worker."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time


REPO = Path(__file__).resolve().parents[1]
VAL_SIZES = (5, 10, 20, 30)


def now():
    return datetime.now(timezone.utc).isoformat()


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def build_matrix(args):
    tasks = sorted(p.parent.name for p in args.data_dir.glob("*/droid_split_manifest.json"))
    if not tasks:
        raise ValueError(f"No task manifests under {args.data_dir}")
    runs = {task: [] for task in tasks}
    for config_path in sorted(args.checkpoint_root.glob("*/*/config.json")):
        config = json.loads(config_path.read_text())
        names = [n for n in config["train"].get("dataset_names", []) if n != "droid"]
        if len(names) != 1 or names[0].split("/")[0] not in runs:
            continue
        checkpoints = list((config_path.parent / "models").glob("model_epoch_*.pth"))
        epochs = sorted(int(re.fullmatch(r"model_epoch_(\d+)\.pth", p.name)[1]) for p in checkpoints)
        if epochs:
            runs[names[0].split("/")[0]].append((config_path.parent, epochs))
    cells, absent, jobs = [], [], []
    for size in VAL_SIZES:
        for task in tasks:
            dataset_name = f"{task}/droid/val{size}"
            dataset = args.data_dir / dataset_name
            if not dataset.is_dir():
                if size != 30:
                    raise FileNotFoundError(f"Required validation split missing: {dataset}")
                absent.append(dataset_name)
                continue
            infos = list(dataset.glob("*/dataset_info.json"))
            if len(infos) != 1:
                raise ValueError(f"Expected one TFDS version under {dataset}")
            splits = json.loads(infos[0].read_text())["splits"]
            val = next((s for s in splits if s["name"] == "val"), None)
            if val is None or sum(map(int, val["shardLengths"])) != size:
                raise ValueError(f"TFDS val count differs from val{size}: {dataset}")
            cells.append({"dataset": dataset_name, "status": "ready" if runs[task] else "missing_checkpoint"})
            for run, epochs in runs[task]:
                name = f"{task}_val{size}_{run.name}"
                job_dir = args.output_root / "conditions" / name / f"seed_{args.seed}"
                command = [sys.executable, "-u", str(REPO / "scripts/cache_droid_checkpoints.py"),
                           "--checkpoint-dir", str(run / "models"), "--data-dir", str(args.data_dir),
                           "--dataset-name", dataset_name, "--cache-dir", str(job_dir / "cache"),
                           "--num-cache-samples", str(args.num_cache_samples), "--batch-size", str(args.batch_size),
                           "--valid-num-steps", str(args.valid_num_steps), "--run-standard-validation",
                           "--manifold-k", "5", "--seed", str(args.seed), "--num-threads", "4",
                           "--loader-parallelism", "4"]
                jobs.append({"name": name, "dataset": dataset_name, "val_demos": size,
                             "epochs": epochs, "job_dir": str(job_dir), "command": command})
    selected = [j for i, j in enumerate(jobs) if i % args.num_shards == args.shard_idx]
    return {"data_dir": str(args.data_dir), "checkpoint_root": str(args.checkpoint_root),
            "cells": cells, "absent_val30": absent, "jobs": selected,
            "missing_checkpoint_tasks": [task for task in tasks if not runs[task]],
            "total_jobs": len(jobs), "num_shards": args.num_shards, "shard_idx": args.shard_idx,
            "gpu": args.gpu, "samples": args.num_cache_samples, "batch_size": args.batch_size,
            "seed": args.seed, "valid_num_steps": args.valid_num_steps}


def verify_job(job, samples):
    sys.path.insert(0, str(REPO / "src"))
    from surval.droid import load_policy_cache
    import numpy as np

    manifests = list((Path(job["job_dir"]) / "cache").rglob("seqcache_manifest.json"))
    if len(manifests) != 1:
        raise ValueError(f"Expected one completed producer manifest, found {len(manifests)}")
    manifest = json.loads(manifests[0].read_text())
    if sorted(item["epoch"] for item in manifest["files"]) != job["epochs"]:
        raise ValueError("Completed epochs differ from the requested checkpoint inventory")
    summary = []
    for item in manifest["files"]:
        cache = load_policy_cache(item["cache_file"])
        provenance = cache["provenance"]
        if provenance["dataset_name"] != job["dataset"] or cache["epoch"] != item["epoch"]:
            raise ValueError("Cache dataset/epoch identity mismatch")
        if provenance["max_rows"] is not None or provenance["max_demos"] is not None:
            raise ValueError("A capped smoke cache cannot satisfy a full-split job")
        if len(set(cache["demo_ids"])) != job["val_demos"] or cache["pred_actions"].shape[0] != samples:
            raise ValueError("Incomplete episode or prediction-sample coverage")
        if job.get("cache_ta") is not None and cache["actions"].shape[1] != job["cache_ta"]:
            raise ValueError("Cached horizon differs from requested horizon")
        if job.get("standard_validation", True) and not all(
                v is not None and np.isfinite(v) for v in (cache["valid_loss"], cache["valid_omn"])):
            raise ValueError("Validation Loss/OMN missing or non-finite")
        summary.append({"epoch": item["epoch"], "rows": len(cache["actions"]),
                        "cache_file": item["cache_file"], "loss": cache["valid_loss"] if cache["valid_loss"] is not None and np.isfinite(cache["valid_loss"]) else None,
                        "omn": cache["valid_omn"] if cache["valid_omn"] is not None and np.isfinite(cache["valid_omn"]) else None})
    return summary


def run_worker(command, log_path, env, heartbeat):
    with log_path.open("a") as log:
        log.write(f"\n[{now()}] COMMAND {shlex.join(command)}\n")
        log.flush()
        with subprocess.Popen(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT) as child:
            while True:
                try:
                    return child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    heartbeat(child.pid)


def execute(args, matrix):
    args.output_root.mkdir(parents=True, exist_ok=True)
    suffix = f".shard_{args.shard_idx}" if args.num_shards > 1 else ""
    # One writer per shard; producer provenance checks handle completed epochs on resume.
    with (args.output_root / f"runner{suffix}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        save_json(args.output_root / f"run_manifest{suffix}.json", matrix)
        status = {"started_at": now(), "state": "running", "jobs": [],
                  "missing_checkpoint_tasks": matrix["missing_checkpoint_tasks"]}
        status_path = args.output_root / f"run_status{suffix}.json"
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": args.gpu, "JAX_PLATFORMS": "cpu",
               "HF_HOME": str(REPO / ".venv-droid/model-cache/huggingface"),
               "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TF_CPP_MIN_LOG_LEVEL": "3",
               "TF_NUM_INTEROP_THREADS": "2", "TF_NUM_INTRAOP_THREADS": "4",
               "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "PYTHONUNBUFFERED": "1"}

        def emit(event, **details):
            status["updated_at"] = now()
            counts = dict(Counter(j["state"] for j in status["jobs"]))
            record = {"time": status["updated_at"], "event": event, "counts": counts, **details}
            message = json.dumps(record)
            print(message, flush=True)
            with (args.output_root / f"events{suffix}.jsonl").open("a") as stream:
                stream.write(message + "\n")
            save_json(status_path, status)

        for job in matrix["jobs"]:
            job_dir = Path(job["job_dir"])
            job_dir.mkdir(parents=True, exist_ok=True)
            save_json(job_dir / "manifest.json", job)
            current = {"name": job["name"], "state": "running", "started_at": now()}
            status["jobs"].append(current)
            save_json(job_dir / "status.json", current)
            emit("START", job=job["name"], epochs=len(job["epochs"]))
            started = time.monotonic()
            try:
                code = run_worker(job["command"], job_dir / "job.log", env,
                                  lambda pid: emit("STATUS", job=job["name"], pid=pid))
                current["exit_code"] = code
                if code:
                    raise RuntimeError(f"Cache producer exited with status {code}; see job.log")
                current["caches"] = verify_job(job, args.num_cache_samples)
                current["state"] = "completed"
            except Exception as error:
                current.update(state="failed", error=str(error))
            current.update(finished_at=now(), seconds=time.monotonic() - started)
            save_json(job_dir / "status.json", current)
            emit("DONE" if current["state"] == "completed" else "FAIL", job=job["name"])
            if current["state"] == "failed":
                print("\n".join((job_dir / "job.log").read_text(errors="replace").splitlines()[-15:]), flush=True)
                # Stop on infrastructure failure; do not spend the remaining queue repeating it.
                break
        status["state"] = ("failed" if any(j["state"] == "failed" for j in status["jobs"])
                           else "completed_available_missing_checkpoints" if matrix["missing_checkpoint_tasks"]
                           else "completed")
        emit("FINISH", state=status["state"])
        return 1 if status["state"] == "failed" else 2 if matrix["missing_checkpoint_tasks"] else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("/workspace/data"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("/workspace/droid_policy_learning/checkpoints"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--num-cache-samples", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--valid-num-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-idx", type=int, default=0)
    parser.add_argument("--list", "--dry-run", dest="list_only", action="store_true")
    args = parser.parse_args()
    if min(args.num_cache_samples, args.batch_size, args.valid_num_steps, args.num_shards) < 1:
        parser.error("Sample, batch, validation and shard counts must be positive")
    if not 0 <= args.shard_idx < args.num_shards:
        parser.error("shard-idx must be between zero and num-shards - 1")
    args.data_dir, args.checkpoint_root, args.output_root = (
        path.resolve() for path in (args.data_dir, args.checkpoint_root, args.output_root))
    matrix = build_matrix(args)
    if args.list_only or os.environ.get("DRY_RUN") == "1":
        print(json.dumps(matrix, indent=2))
        return 0
    return execute(args, matrix)


if __name__ == "__main__":
    raise SystemExit(main())
