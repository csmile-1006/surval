"""Run with PYTHONPATH=src:tests:scripts python -m unittest droid_long_tuning_test."""

from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from droid_tuning_test import synthetic
from surval.droid import score_policy_cache
from surval.droid_tuning import canonical_thresholds, expert_distance_matrices, local_scale_table, primary_metrics
from surval.droid_long_tuning import hp_grid, score_grid, metric_arrays
from cache_droid_datasets import verify_job
from prepare_droid_long_cache import cache_matrix


class LongTuningTest(unittest.TestCase):
    def test_long_formula_all_topk_and_temperature(self):
        db, cache = synthetic()
        cache['actions'] = np.concatenate([cache['actions'], cache['actions'][:, :7]], axis=1)
        cache['pred_actions'] = np.concatenate([cache['pred_actions'], cache['pred_actions'][:, :, :7]], axis=2)
        axes = dict(ta=[1, 9, 15], k=[10], quantile=[.1, .95], lse_tau=[.03, 1., 30.])
        table = local_scale_table(db, expert_distance_matrices(db), 10, axes['quantile'])[0]
        scores = score_grid(cache, {10: table}, axes)
        with tempfile.TemporaryDirectory() as td:
            canonical_thresholds(db, axes['quantile'], 10, td)
            for i, hp in enumerate(hp_grid(axes)):
                summary, episodes = score_policy_cache(cache, td, ta=hp['ta'], quantile=hp['quantile'],
                    chunk_top_frac=hp['chunk_top_frac'], lse_tau=hp['lse_tau'], every_step=True)
                np.testing.assert_allclose(scores[i], summary['PrefixSurvival_Score'], atol=2e-7, rtol=2e-6)
                self.assertEqual(sum(e['T'] for e in episodes), len(cache['actions']))
        with self.assertRaises(ValueError):
            hp_grid({**axes, 'scale_multiplier': [2]})
        self.assertEqual(len(hp_grid()), 161280)
        self.assertTrue(all(np.ceil(h['ta']*h['chunk_top_frac']) == h['chunk_top_k'] for h in hp_grid()))

    def test_metrics_with_ties_and_constant_scores(self):
        actual = np.array([2, 8, 11, 12, 14, 15, 9, 11, 15, 12])/30
        scores = np.random.default_rng(42).integers(0, 6, (100, 10)).astype(float)
        scores[0] = 1
        fast = metric_arrays(actual, scores)
        for i, s in enumerate(scores):
            ref = primary_metrics(actual, s)
            for key in fast:
                np.testing.assert_allclose(fast[key][i], ref[key], atol=1e-14, equal_nan=True)

    def test_selection_objectives(self):
        from tune_droid_long_hp import choices
        grid = hp_grid(dict(ta=[2, 9], k=[10], quantile=[.1, .9], lse_tau=[1.]))
        rng = np.random.default_rng(123)
        actual = np.array([1, 2, 2, 4, 7, 7, 6, 4, 8, 10])/30
        metrics = {t: metric_arrays(actual, rng.integers(0, 20, (len(grid), 10)))
                   for t in ('apple', 'pan', 'pet')}
        selected = choices(grid, metrics)
        self.assertEqual(len(selected), 51)
        for choice in selected:
            domain, scope = choice['domain'].split('/')
            index, target = choice['hp_index'], choice['target']
            mask = np.array([True if scope == 'all' else h['ta'] <= 8 if scope == 'short'
                             else h['ta'] > 8 for h in grid])
            self.assertTrue(mask[index])
            if domain in metrics:
                values = metrics[domain]
                mask &= values['eligible']
                objective = values[target]
            else:
                mask &= np.logical_and.reduce([v['eligible'] for v in metrics.values()])
                objective = (np.max([v['balanced_loss'] for v in metrics.values()], axis=0)
                             if domain == 'common_minimax' else
                             np.mean([v[target] for v in metrics.values()], axis=0))
            if target == 'spearman':
                objective = -objective
            self.assertEqual(objective[index], objective[mask].min())

    def test_val30_only_and_optional_baseline_contract(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            rows = [dict(dataset=f'{t}/droid/val{v}', epoch=e, checkpoint=f'/models/{e}.pth')
                    for t in ('apple', 'pan', 'pet') for v in (5, 10, 20, 30) for e in range(5, 51, 5)]
            baseline = root/'baseline.json'
            baseline.write_text(json.dumps(dict(results=rows)))
            jobs = cache_matrix(SimpleNamespace(root=root, output_root=root/'out', baseline_json=baseline))['jobs']
            self.assertEqual(len(jobs), 3)
            self.assertTrue(all(j['dataset'].endswith('val30') for j in jobs))
            self.assertFalse((root/'out').exists())
            (root/'cache').mkdir()
            (root/'cache/seqcache_manifest.json').write_text(json.dumps(dict(files=[dict(epoch=5, cache_file='fake')])))
            job = dict(job_dir=str(root), epochs=[5], dataset='apple/droid/val30', val_demos=30, cache_ta=15)
            cache = dict(provenance=dict(dataset_name=job['dataset'], max_rows=None, max_demos=None), epoch=5,
                         demo_ids=np.arange(30), actions=np.zeros((30,15,10)), pred_actions=np.zeros((8,30,15,10)),
                         valid_loss=float('nan'), valid_omn=None)
            with patch('surval.droid.load_policy_cache', return_value=cache):
                with self.assertRaises(ValueError):
                    verify_job(job, 8)
                result = verify_job({**job, 'standard_validation': False}, 8)
                self.assertIsNone(result[0]['loss'])
                json.dumps(result, allow_nan=False)
                with self.assertRaises(ValueError):
                    verify_job({**job, 'standard_validation': False, 'cache_ta': 8}, 8)


if __name__ == '__main__':
    unittest.main()
