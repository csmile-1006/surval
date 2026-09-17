"""CPU-only frozen-policy reference sweep; default epoch25, full val30, half/tau1."""

import argparse
import csv
import json
import os
from pathlib import Path
import shlex
import sys
import time
from types import SimpleNamespace

from cache_droid_datasets import REPO, save_json
from tune_droid_hp import execute, sha256
from tune_droid_long_hp import matrix as long_matrix, save_npz

sys.path.insert(0, str(REPO/'src'))
import numpy as np
from surval.droid_policy_reference import AXES, POLICY_OMN, build_policy_reference, align_policy_reference, fixed_grid, score_fixed_grid
from surval.droid_long_tuning import FIXED as LONG_FIXED

FIXED = {**LONG_FIXED, 'chunk_top_frac': .5, 'lse_tau': 1., 'baseline_horizon': 8,
         'omn_k': 5, 'representation': 'frozen_policy_conditioning'}


def matrix(args):
    old = long_matrix(SimpleNamespace(root=args.source_root, output_root=args.source_root/'tuning'))
    jobs = []
    for job in old['jobs']:
        rows = job['rows']
        refs = [r for r in rows if r['epoch'] == args.reference_epoch]
        if len(refs) != 1:
            raise ValueError('Exactly one reference checkpoint is required per task')
        command = [sys.executable, '-u', str(Path(__file__).resolve()),
            '--source-root', str(args.source_root), '--output-root', str(args.output_root),
            '--baseline-csv', str(args.baseline_csv), '--reference-epoch', str(args.reference_epoch),
            '--worker-index', str(job['index'])]
        jobs.append({**job, 'dino_eval_dir': str(Path(job['job_dir'])/'eval'),
                     'reference_cache': refs[0]['cache_file'], 'command': command,
                     'job_dir': str(args.output_root/'conditions'/job['name']/'seed_0')})
    return dict(version=1, jobs=jobs, total_jobs=3, axes=AXES,
        fixed={**FIXED, 'reference_epoch': args.reference_epoch}, grid_size=len(fixed_grid()),
        device='cpu', wandb=None, seeds=[0], new_inference=False,
        outcomes_used_for_tuning=True, checkpoints_per_task=10)


def score_worker(args, job, spec):
    from surval.droid import load_policy_cache, score_policy_cache
    from surval.droid_dino import (DINO_OMN, _content_hash, align_dino_cache, compute_dino_omn,
                                   dataset_db_path, dino_neighbors, load_shared_dino_db)
    from surval.droid_tuning import (canonical_thresholds, expert_distance_matrices, local_scale_table)
    from surval.local_threshold.threshold import _block_pair_distances, query_threshold_neighbors
    output = Path(job['job_dir'])/'eval'
    output.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    reference = load_policy_cache(job['reference_cache'])
    db, chunks, meta = build_policy_reference(reference, args.reference_epoch)
    dino, dino_chunks, dino_meta = load_shared_dino_db(dataset_db_path(args.source_root/'shared_dino', job['dataset']))
    dino_order = align_dino_cache(dino, dino_chunks, dino_meta, reference)
    assert len(set(reference['demo_ids'])) == 30 and chunks.shape[1] == 15
    with args.baseline_csv.open(newline='') as stream:
        baseline_rows = list(csv.DictReader(stream))
    key = lambda r: (r['run'], r['run_timestamp'], r['dataset'], int(r['epoch']))
    old = {key(r): r for r in baseline_rows}
    for row in job['rows']:
        assert old[key(row)]['checkpoint'] == row['checkpoint']
    code_paths = ['src/surval/droid_policy_reference.py', 'src/surval/droid_tuning.py',
        'src/surval/droid_long_tuning.py', 'src/surval/droid.py', 'src/surval/droid_dino.py',
        'src/surval/sequential_validate.py', 'src/surval/cache_io.py',
        'src/surval/local_threshold/database.py', 'src/surval/local_threshold/threshold.py',
        'src/surval/local_threshold/rotation.py', 'scripts/tune_droid_policy_reference.py']
    signature = dict(axes=AXES, fixed=spec['fixed'], reference_metadata=meta,
        policy_db_content_sha256=_content_hash(db, chunks), dino_db_content_sha256=dino_meta['content_sha256'],
        baseline_csv_sha256=sha256(args.baseline_csv),
        cache_sha256={r['cache_file']: sha256(r['cache_file']) for r in job['rows']},
        code_sha256={p: sha256(REPO/p) for p in code_paths})
    complete = output/'verification.json'
    if complete.exists():
        proof = json.loads(complete.read_text())
        if proof['signature'] != signature:
            raise ValueError('Completed source/signature changed; use another output root')
        for name, digest in proof['outputs_sha256'].items():
            assert sha256(output/name) == digest, name
        print('RESUME verified complete', flush=True)
        return
    progress_path = output/'progress.json'
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else dict(signature=signature, epochs={})
    if progress['signature'] != signature:
        raise ValueError('Resume source/signature changed; use another output root')
    save_json(Path(job['job_dir'])/'manifest.json', {**job, 'signature': signature})
    save_json(progress_path, progress)
    reference_dir = output/'reference_db'
    if not reference_dir.exists():
        db.save(str(reference_dir))
        np.save(reference_dir/'expert_action_chunks.npy', chunks)
        save_json(reference_dir/'reference_meta.json', {**meta, 'complete': True,
            'content_sha256': signature['policy_db_content_sha256'], 'source_cache': job['reference_cache']})
    # Frozen full-split neighbors are computed once, then reused across targets.
    policy_neighbors = dino_neighbors(db, k=5, temporal_radius=5)
    dino_nbs = dino_neighbors(dino, k=5, temporal_radius=5)
    assert all(len(nb) == 5 for nb in (*policy_neighbors, *dino_nbs))
    if 'scales_sha256' in progress:
        assert sha256(output/'scales.npz') == progress['scales_sha256']
        with np.load(output/'scales.npz') as saved:
            tables = {k: saved[str(k)] for k in AXES['k']}
    else:
        matrices = expert_distance_matrices(db)
        tables, maximum = {}, 0.
        for k in AXES['k']:
            tables[k], cfg = local_scale_table(db, matrices, k, AXES['quantile'])
            neighbors = query_threshold_neighbors(db, cfg, exact_neighbors=True)
            for row in np.linspace(0, db.n_states-1, 16, dtype=int):
                for bi, (name, sl) in enumerate(db.cfg.block_slice_dict().items()):
                    pairs = _block_pair_distances(db.records.actions[neighbors[row]], sl, pairwise=True,
                                                 block_type=db.cfg.block_type_dict().get(name))
                    expected = np.quantile(pairs, AXES['quantile']).astype(np.float32)
                    maximum = max(maximum, float(abs(expected-tables[k][row, bi]).max()))
                    np.testing.assert_allclose(expected, tables[k][row, bi], atol=1e-7, rtol=1e-6)
            print(f'THRESHOLD k={k} elapsed={time.monotonic()-start:.1f}s', flush=True)
        del matrices
        save_npz(output/'scales.npz', **{str(k): v for k, v in tables.items()})
        progress.update(scales_sha256=sha256(output/'scales.npz'), threshold_max_abs_error=maximum)
        save_json(progress_path, progress)
    score_rows, baselines = [], []
    grid = fixed_grid()
    for col, row in enumerate(job['rows']):
        file = output/f'epoch_{row["epoch"]:03d}.npz'
        epoch_key = str(row['epoch'])
        if epoch_key in progress['epochs']:
            assert sha256(file) == progress['epochs'][epoch_key]
            with np.load(file) as saved:
                values = {k: saved[k].copy() for k in saved.files}
        else:
            cache = load_policy_cache(row['cache_file'])
            assert cache['pred_actions'].shape == (8, db.n_states, 15, 10)
            order = align_policy_reference(db, chunks, meta, cache)
            target_dino_order = align_dino_cache(dino, dino_chunks, dino_meta, cache)
            scores = score_fixed_grid(cache, {k: t[order] for k, t in tables.items()})
            # H8 for both OMNs and MSE: exactly the same samples/rows/offsets.
            short = {**cache, 'actions': cache['actions'][:, :8], 'pred_actions': cache['pred_actions'][:, :, :8]}
            policy_omn, pi_info = compute_dino_omn(short, chunks[:, :8], order, policy_neighbors)
            dino_omn, di_info = compute_dino_omn(short, dino_chunks[:, :8], target_dino_order, dino_nbs)
            assert pi_info['rows_without_neighbors'] == di_info['rows_without_neighbors'] == 0
            scale, offset = cache['action_scale'], cache['action_offset']
            gt = (short['actions']-offset)/scale
            pred = (short['pred_actions']-offset)/scale
            mse = float(np.square(pred.mean(axis=0)-gt).mean())
            values = dict(scores=scores, **{POLICY_OMN: policy_omn, DINO_OMN: dino_omn, 'MSE_mean_pred': mse,
                'Loss': float(old[key(row)]['Loss']), 'Off_Manifold_Norm': float(old[key(row)]['Off_Manifold_Norm'])})
            assert all(np.isfinite(v).all() for v in values.values())
            if col == 0:
                smoke_dir = output/'canonical_smoke'
                original = canonical_thresholds(db, AXES['quantile'], 10, smoke_dir)
                np.testing.assert_allclose(original.thresholds, tables[10], atol=1e-7, rtol=1e-6)
                errors = []
                for ta, q in ((1, .1), (8, .5), (15, .95)):
                    i = next(i for i, h in enumerate(grid) if (h['ta'], h['k'], h['quantile']) == (ta, 10, q))
                    summary, episodes = score_policy_cache(cache, str(smoke_dir), ta=ta, quantile=q,
                        chunk_top_frac=.5, lse_tau=1., num_samples=8, every_step=True)
                    assert sum(e['T'] for e in episodes) == db.n_states
                    reference_score = summary['PrefixSurvival_Score']
                    np.testing.assert_allclose(scores[i], reference_score, atol=2e-7, rtol=2e-6)
                    errors.append(abs(scores[i]-reference_score))
                progress['canonical_smoke_max_abs_error'] = max(errors)
            save_npz(file, **values)
            progress['epochs'][epoch_key] = sha256(file)
            save_json(progress_path, progress)
        score_rows.append(values['scores'])
        baselines.append(dict(task=job['task'], **{k: row[k] for k in ('run', 'run_timestamp', 'dataset', 'epoch', 'checkpoint')},
            **{k: float(v) for k, v in values.items() if k != 'scores'},
            reference_epoch=args.reference_epoch, baseline_horizon=8, num_rows=db.n_states,
            legacy_loss_policy_omn_rows=min(1600, db.n_states)))
        print(f'SCORE epoch={row["epoch"]} HPs={len(grid)} elapsed={time.monotonic()-start:.1f}s', flush=True)
    save_npz(output/'scores.npz', scores=np.stack(score_rows, axis=1), epochs=[r['epoch'] for r in job['rows']])
    save_json(output/'baselines.json', baselines)
    for path, digest in signature['cache_sha256'].items():
        assert sha256(path) == digest
    save_json(complete, dict(complete=True, signature=signature, grid_size=len(grid), checkpoints=10,
        rows=db.n_states, feature_dim=db.embeddings.shape[1], num_demos=30, fallback_rows=0,
        elapsed_seconds=time.monotonic()-start, threshold_max_abs_error=progress['threshold_max_abs_error'],
        canonical_smoke_max_abs_error=progress['canonical_smoke_max_abs_error'],
        outputs_sha256={n: sha256(output/n) for n in ('scores.npz', 'scales.npz', 'baselines.json')}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=REPO/'outputs/droid_c1_long_20260913')
    parser.add_argument('--output-root', type=Path, default=REPO/'outputs/droid_policy25_val30_20260915')
    parser.add_argument('--baseline-csv', type=Path, default=REPO/'outputs/droid_baselines_20260913/checkpoint_metrics.csv')
    parser.add_argument('--reference-epoch', type=int, default=25)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--shard-idx', type=int, default=0)
    parser.add_argument('--worker-index', type=int)
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--report', action='store_true')
    args = parser.parse_args()
    if args.workers < 1 or args.num_shards < 1 or not 0 <= args.shard_idx < args.num_shards:
        parser.error('Invalid worker or shard parameters')
    for name in ('source_root', 'output_root', 'baseline_csv'):
        setattr(args, name, getattr(args, name).resolve())
    if args.output_root == args.source_root or args.output_root.is_relative_to(args.source_root):
        parser.error('Use a new output root outside the archived DINO run')
    spec = matrix(args)
    if args.list or os.environ.get('DRY_RUN') == '1':
        print(json.dumps({k: v for k, v in spec.items() if k != 'jobs'}))
        jobs = [j for j in spec['jobs'] if j['index'] % args.num_shards == args.shard_idx]
        print(f'selected_jobs={len(jobs)} output_root={args.output_root}')
        for job in jobs:
            print(shlex.join(job['command']))
        return 0
    if args.report:
        from report_droid_policy_reference import report
        report(args, spec)
        return 0
    if args.worker_index is not None:
        if not 0 <= args.worker_index < len(spec['jobs']):
            parser.error('Invalid worker index')
        score_worker(args, spec['jobs'][args.worker_index], spec)
        return 0
    return execute(args, spec)


if __name__ == '__main__':
    raise SystemExit(main())
