"""Small stdlib checks for matrix coverage and supervised child exit handling."""

import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest


spec = importlib.util.spec_from_file_location(
    "cache_sweep", Path(__file__).resolve().parents[1] / "scripts/cache_droid_datasets.py")
sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep)


class CacheSweepTest(unittest.TestCase):
    def test_matrix_missing_checkpoints_and_shards(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for i in range(6):
                task = root / "data" / f"task_{i}"
                task.mkdir(parents=True)
                (task / "droid_split_manifest.json").write_text("{}")
                for size in (5, 10, 20, 30):
                    if i == 5 and size == 30:
                        continue
                    version = task / "droid" / f"val{size}" / "1.0.0"
                    version.mkdir(parents=True)
                    (version / "dataset_info.json").write_text(json.dumps(
                        {"splits": [{"name": "val", "shardLengths": [str(size)]}]}))
                if i < 3:
                    run = root / "checkpoints" / f"run_{i}" / "timestamp"
                    (run / "models").mkdir(parents=True)
                    (run / "config.json").write_text(json.dumps(
                        {"train": {"dataset_names": ["droid", f"task_{i}/droid/val30"]}}))
                    for epoch in range(5, 51, 5):
                        (run / "models" / f"model_epoch_{epoch}.pth").touch()
            args = SimpleNamespace(data_dir=root / "data", checkpoint_root=root / "checkpoints",
                                   output_root=root / "output", seed=0, num_cache_samples=8,
                                   batch_size=32, valid_num_steps=50, num_shards=1, shard_idx=0, gpu="0")
            matrix = sweep.build_matrix(args)
            self.assertEqual(len(matrix["cells"]), 23)
            self.assertEqual(len(matrix["jobs"]), 12)
            self.assertEqual(sum(len(j["epochs"]) for j in matrix["jobs"]), 120)
            self.assertEqual(matrix["missing_checkpoint_tasks"], ["task_3", "task_4", "task_5"])
            self.assertFalse(args.output_root.exists())
            expected = {j["name"] for j in matrix["jobs"]}
            names = []
            for shard in range(3):
                args.num_shards, args.shard_idx = 3, shard
                names.extend(j["name"] for j in sweep.build_matrix(args)["jobs"])
            self.assertEqual(len(names), len(set(names)))
            self.assertEqual(set(names), expected)

    def test_worker_logs_success_and_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            for code in (0, 3):
                log = Path(temporary) / f"worker_{code}.log"
                command = [sys.executable, "-c", f"print('worker-output'); raise SystemExit({code})"]
                self.assertEqual(sweep.run_worker(command, log, os.environ.copy(), lambda pid: None), code)
                self.assertIn("COMMAND", log.read_text())
                self.assertIn("worker-output", log.read_text())


if __name__ == "__main__":
    unittest.main()
