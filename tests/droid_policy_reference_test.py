"""CPU contracts for frozen epoch5/25/50 references and fixed-half/LSE1 scoring."""

import copy
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from droid_tuning_test import synthetic
from surval.droid import score_policy_cache
from surval.droid_dino import compute_dino_omn, dino_neighbors
from surval.droid_policy_reference import (align_policy_reference, build_policy_reference,
                                            fixed_grid, score_fixed_grid)
from surval.droid_tuning import canonical_thresholds, expert_distance_matrices, local_scale_table


def reference_fixture():
    db, cache = synthetic()
    cache.update(epoch=25, checkpoint='/test/model_epoch_25.pth', obs_features=db.embeddings.copy(),
        action_start_offset=1, action_scale=np.ones(10, np.float32), action_offset=np.zeros(10, np.float32),
        provenance=dict(run='task', run_timestamp='run1', dataset_name='task/droid/val30', dataset_files=[],
            observation_horizon=2, feature_source='ema.policy.obs_encoder', feature_pool='flatten_observation_horizon'))
    cache['actions'] = np.concatenate([cache['actions'], cache['actions'][:, :7]], axis=1)
    cache['pred_actions'] = np.concatenate([cache['pred_actions'], cache['pred_actions'][:, :, :7]], axis=2)
    return cache


class FrozenPolicyContract(unittest.TestCase):
    def test_report_rejects_unsupported_reference(self):
        from types import SimpleNamespace
        from report_droid_policy_reference import REPO, report
        with self.assertRaisesRegex(ValueError, 'epoch25'):
            report(SimpleNamespace(reference_epoch=15), {})
        for reference, protected in ((5, 25), (5, 50), (50, 25), (25, 50)):
            with self.assertRaisesRegex(ValueError, 'preserve'):
                report(SimpleNamespace(reference_epoch=reference,
                    output_root=REPO/f'outputs/droid_policy{protected}_val30_20260915'), {})

    def test_cli_no_write_and_three_shards(self):
        root = Path(__file__).resolve().parents[1]
        if not (root/'outputs/droid_c1_long_20260913/cache_index.json').exists():
            self.skipTest('Recorded DROID manifest not available')
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)/'not-created'
            command = ['bash', str(root/'scripts/tune_droid_policy_reference.sh'),
                       '--output-root', str(output), '--reference-epoch', '5']
            env = {**os.environ, 'PYTHON': sys.executable, 'DRY_RUN': '1', 'SKIP_ARTIFACT_CHECK': '1'}
            lines = subprocess.check_output(command, env=env, text=True).splitlines()
            spec = json.loads(lines[0])
            self.assertEqual((spec['total_jobs'], spec['grid_size']), (3, 2880))
            self.assertEqual(spec['fixed']['reference_epoch'], 5)
            shards = []
            for index in range(3):
                selected = subprocess.check_output(command + ['--list', '--num-shards', '3',
                    '--shard-idx', str(index)], env=env, text=True).splitlines()
                self.assertIn('selected_jobs=1', selected[1])
                shards.extend(selected[2:])
            self.assertEqual(len(shards), len(set(shards)))
            self.assertEqual(set(shards), set(lines[2:]))
            self.assertFalse(output.exists())

    def test_epoch5_and50_references_are_frozen(self):
        for epoch in (5, 50):
            cache = reference_fixture()
            cache.update(epoch=epoch, checkpoint=f'/test/model_epoch_{epoch}.pth')
            db, chunks, meta = build_policy_reference(cache, reference_epoch=epoch)
            self.assertEqual(meta['reference_epoch'], epoch)
            self.assertIn(f'epoch{epoch}', db.cfg.encoder_name)
            target = {**cache, 'epoch': 25, 'obs_features': -100*cache['obs_features']}
            np.testing.assert_array_equal(align_policy_reference(db, chunks, meta, target), np.arange(48))
            with self.assertRaises(ValueError):
                build_policy_reference(cache, reference_epoch=25)

    def test_frozen_features_alignment_and_guards(self):
        cache = reference_fixture()
        db, chunks, meta = build_policy_reference(cache)
        target = copy.deepcopy(cache)
        target['epoch'] = 50
        target['obs_features'] *= -100
        np.testing.assert_array_equal(align_policy_reference(db, chunks, meta, target), np.arange(48))
        for mutate in (lambda c: c.__setitem__('epoch', 5),
                       lambda c: c['obs_features'].__setitem__(0, 0),
                       lambda c: c['obs_features'].__setitem__((0, 0), np.nan),
                       lambda c: c['provenance'].__setitem__('max_rows', 12),
                       lambda c: c['index_in_demo'].__setitem__(1, c['index_in_demo'][0])):
            bad = copy.deepcopy(cache); mutate(bad)
            with self.assertRaises(ValueError):
                build_policy_reference(bad)
        for mutate in (lambda c: c['provenance'].__setitem__('run_timestamp', 'other'),
                       lambda c: c['actions'].__setitem__((0, 0, 0), 99),
                       lambda c: c['index_in_demo'].__setitem__(0, 999)):
            bad = copy.deepcopy(target); mutate(bad)
            with self.assertRaises(ValueError):
                align_policy_reference(db, chunks, meta, bad)

    def test_fixed_grid_original_scorer_and_row_order(self):
        cache = reference_fixture()
        db, chunks, meta = build_policy_reference(cache)
        axes = dict(ta=[1, 2, 9, 15], k=[10, 20], quantile=[.1, .95])
        matrices = expert_distance_matrices(db)
        tables = {k: local_scale_table(db, matrices, k, axes['quantile'])[0] for k in axes['k']}
        fast = score_fixed_grid(cache, tables, axes)
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            for k in axes['k']:
                threshold = canonical_thresholds(db, axes['quantile'], k, Path(temp)/str(k))
                np.testing.assert_allclose(threshold.thresholds, tables[k], atol=1e-7, rtol=1e-6)
            for i, hp in enumerate(fixed_grid(axes)):
                summary, episodes = score_policy_cache(cache, str(Path(temp)/str(hp['k'])),
                    ta=hp['ta'], quantile=hp['quantile'], chunk_top_frac=.5, lse_tau=1., every_step=True)
                np.testing.assert_allclose(fast[i], summary['PrefixSurvival_Score'], atol=2e-7, rtol=2e-6)
                self.assertEqual(sum(e['T'] for e in episodes), len(chunks))
        order = np.random.default_rng(2).permutation(len(chunks))
        target = {**cache, **{k: cache[k][order] for k in ('actions', 'demo_ids', 'index_in_demo', 'obs_features')},
                  'pred_actions': cache['pred_actions'][:, order], 'epoch': 50}
        aligned = align_policy_reference(db, chunks, meta, target)
        np.testing.assert_array_equal(aligned, order)
        np.testing.assert_allclose(score_fixed_grid(target, {k: v[order] for k, v in tables.items()}, axes),
                                   fast, rtol=0, atol=1e-14)
        self.assertEqual(len(fixed_grid()), 2880)
        self.assertTrue(all(h['chunk_top_frac'] == .5 and h['lse_tau'] == 1 and
                           h['chunk_top_k'] == int(np.ceil(h['ta']/2)) for h in fixed_grid()))
        with self.assertRaises(ValueError):
            fixed_grid({**axes, 'lse_tau': [3.]})

    def test_omn_ignores_target_features(self):
        cache = reference_fixture()
        db, chunks, meta = build_policy_reference(cache)
        neighbors = dino_neighbors(db, k=5, temporal_radius=5)
        for i, nb in enumerate(neighbors):
            self.assertEqual(len(nb), 5)
            self.assertFalse(np.any((db.records.demo_id_int[nb] == db.records.demo_id_int[i]) &
                                   (abs(db.records.t[nb]-db.records.t[i]) <= 5)))
        order = align_policy_reference(db, chunks, meta, cache)
        first, info = compute_dino_omn(cache, chunks, order, neighbors)
        other = {**cache, 'obs_features': cache['obs_features']*-10, 'epoch': 50}
        second, _ = compute_dino_omn(other, chunks, align_policy_reference(db, chunks, meta, other), neighbors)
        self.assertEqual(first, second)
        self.assertEqual(info['rows_without_neighbors'], 0)


if __name__ == '__main__':
    unittest.main()
