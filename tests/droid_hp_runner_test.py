"""CPU launch/report contracts, including real success/failure subprocesses."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from report_droid_hp import best_index, pareto_indices
from tune_droid_hp import execute


class RunnerContract(unittest.TestCase):
    def test_selection_eligibility_and_pareto(self):
        rows = [dict(eligible=True, nregret=1., mmrv=.2, spearman=.1, balanced_loss=.6),
                dict(eligible=True, nregret=.1, mmrv=.1, spearman=.7, balanced_loss=.2),
                dict(eligible=False, nregret=0., mmrv=0., spearman=1., balanced_loss=0.)]
        self.assertEqual(best_index(rows), 1)
        self.assertEqual(best_index(rows, "spearman"), 1)
        self.assertEqual(pareto_indices(rows), [1])
        with self.assertRaises(ValueError):
            best_index([rows[2]])

    def test_real_worker_exit_codes_and_logs(self):
        with tempfile.TemporaryDirectory() as td, redirect_stdout(io.StringIO()):
            for code in (0, 7):
                out = Path(td) / str(code)
                jobdir = out / "conditions/probe/seed_0"
                (jobdir / "eval").mkdir(parents=True)
                (jobdir / "eval/verification.json").write_text(json.dumps({"complete": True}))
                job = {"index": 0, "name": "probe", "job_dir": str(jobdir), "command": [sys.executable,
                       "-c", f'print("worker-output"); raise SystemExit({code})']}
                args = SimpleNamespace(output_root=out, num_shards=1, shard_idx=0, workers=1)
                self.assertEqual(execute(args, {"jobs": [job], "grid_size": 1}), 0 if code == 0 else 1)
                status = json.loads((out / "run_status.json").read_text())
                self.assertEqual(status["state"], "complete" if code == 0 else "failed")
                self.assertIn("worker-output", (jobdir / "job.log").read_text())
                self.assertTrue((out / "events.jsonl").is_file())
                self.assertTrue((jobdir / "status.json").is_file())

    def test_dry_run_and_documented_shards_do_not_write(self):
        root = Path(__file__).resolve().parents[1]
        # This test uses the existing baseline manifest but never loads caches.
        if not (root / "outputs/droid_baselines_20260913/checkpoint_metrics.json").exists():
            self.skipTest("Recorded DROID baseline manifest not available")
        with tempfile.TemporaryDirectory() as td:
            output = Path(td) / "not-created"
            base = [sys.executable, str(root / "scripts/tune_droid_hp.py"), "--list", "--output-root", str(output)]
            all_lines = subprocess.check_output(base, text=True).splitlines()
            self.assertEqual(json.loads(all_lines[0])["total_jobs"], 12)
            self.assertEqual(json.loads(all_lines[0])["grid_size"], 3584)
            all_jobs = set(all_lines[2:])
            shards = []
            for shard in (0, 1):
                lines = subprocess.check_output(base + ["--num-shards", "2", "--shard-idx", str(shard)],
                                                text=True).splitlines()
                self.assertIn("selected_jobs=6", lines[1])
                shards.append(set(lines[2:]))
            self.assertFalse(shards[0] & shards[1])
            self.assertEqual(shards[0] | shards[1], all_jobs)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
