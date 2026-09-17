"""val30-only c=1 long-chunk sweep and audited exploratory reports."""

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time

from cache_droid_datasets import REPO, save_json
from tune_droid_hp import execute, sha256

sys.path.insert(0, str(REPO / 'src'))
from surval.droid_long_tuning import AXES, FIXED, hp_grid


def matrix(args):
    baseline = args.root / 'cache_index.json'
    # Listing is possible before inference, with no output creation.
    if not baseline.exists():
        baseline = REPO / 'outputs/droid_baselines_20260913/checkpoint_metrics.json'
    grouped = {}
    for row in json.loads(baseline.read_text())['results']:
        if row['dataset'].endswith('val30'):
            grouped.setdefault(row['dataset'], []).append(row)
    if len(grouped) != 3:
        raise ValueError('Expected exactly three val30 datasets')
    jobs = []
    for index, (dataset, rows) in enumerate(sorted(grouped.items())):
        task = next(t for t in ('apple', 'pan', 'pet') if t in dataset.split('/')[0].split('_'))
        rows.sort(key=lambda r: r['epoch'])
        if [r['epoch'] for r in rows] != list(range(5, 51, 5)):
            raise ValueError('Expected epochs5..50')
        name = dataset.replace('/droid/', '_')
        directory = args.output_root / 'conditions' / name / 'seed_0'
        command = [sys.executable, '-u', str(Path(__file__).resolve()), '--root', str(args.root),
                   '--worker-index', str(index)]
        jobs.append(dict(index=index, name=name, dataset=dataset, task=task, val=30, rows=rows,
                         job_dir=str(directory), command=command))
    return dict(version=1, axes=AXES, fixed=FIXED, grid_size=len(hp_grid()), jobs=jobs,
                total_jobs=3, device='cpu', wandb=None, outcomes_used_for_tuning=True)


def save_npz(path, **arrays):
    import numpy as np
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def score_worker(args, job):
    import numpy as np
    from surval.droid import load_policy_cache, score_policy_cache
    from surval.droid_dino import align_dino_cache, dataset_db_path, load_shared_dino_db
    from surval.droid_tuning import canonical_thresholds, expert_distance_matrices, local_scale_table
    from surval.droid_long_tuning import score_grid
    from surval.local_threshold.threshold import _block_pair_distances, query_threshold_neighbors
    from types import SimpleNamespace
    from prepare_droid_long_cache import prepare
    generation = args.root/'cache_generation/conditions'/job['name']/'seed_0/cache'
    while len(list(generation.rglob('seqcache_epoch_*.hdf5'))) < 10:
        status_path = args.root/"cache_generation/run_status.json"
        if status_path.exists() and json.loads(status_path.read_text())["state"] == "failed":
            raise RuntimeError("Cache producer failed; inspect cache_generation logs")
        print(f"WAIT complete val30 caches: {job['task']}", flush=True)
        time.sleep(30)
    prepare(SimpleNamespace(root=args.root, output_root=args.root/'cache_generation',
        baseline_json=REPO/'outputs/droid_baselines_20260913/checkpoint_metrics.json',
        dino_root=REPO/'outputs/shared_dino_val'), task=job['task'])
    job['rows'] = json.loads((args.root/f"cache_index_{job['task']}.json").read_text())['results']
    output = Path(job['job_dir'])/'eval'
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    db, chunks, meta = load_shared_dino_db(dataset_db_path(args.root/'shared_dino', job['dataset']))
    paths = ['src/surval/droid_long_tuning.py', 'src/surval/droid_tuning.py',
             'src/surval/sequential_validate.py', 'src/surval/droid.py',
             'src/surval/local_threshold/threshold.py', 'scripts/tune_droid_long_hp.py']
    signature = dict(axes=AXES, fixed=FIXED, db_sha256=meta['content_sha256'],
                     cache_sha256={r['cache_file']: sha256(r['cache_file']) for r in job['rows']},
                     code_sha256={p: sha256(REPO/p) for p in paths})
    progress_path = output/'progress.json'
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else dict(signature=signature, epochs={})
    if progress['signature'] != signature:
        raise ValueError('Resume provenance changed; use a new root')
    verified = output/'verification.json'
    if verified.exists():
        saved = json.loads(verified.read_text())
        for file in ('scores.npz', 'scales.npz'):
            if sha256(output/file) != saved[file+'_sha256']:
                raise ValueError('Completed output checksum mismatch')
        print('RESUME verified complete', flush=True)
        return
    if 'scales_sha256' in progress:
        if sha256(output/'scales.npz') != progress['scales_sha256']:
            raise ValueError('Scale checksum mismatch')
        with np.load(output/'scales.npz') as data:
            tables = {k: data[str(k)] for k in AXES['k']}
    else:
        matrices = expert_distance_matrices(db)
        tables, error = {}, 0.
        for k in AXES['k']:
            tables[k], cfg = local_scale_table(db, matrices, k, AXES['quantile'])
            neighbors = query_threshold_neighbors(db, cfg, exact_neighbors=True)
            for row in np.linspace(0, db.n_states-1, 16, dtype=int):
                for bi, (name, sl) in enumerate(db.cfg.block_slice_dict().items()):
                    pairs = _block_pair_distances(db.records.actions[neighbors[row]], sl, pairwise=True,
                                                 block_type=db.cfg.block_type_dict().get(name))
                    expected = np.quantile(pairs, AXES['quantile']).astype(np.float32)
                    error = max(error, float(abs(tables[k][row, bi]-expected).max()))
                    np.testing.assert_allclose(tables[k][row, bi], expected, atol=1e-7, rtol=1e-6)
            print(f'THRESHOLD k={k} elapsed={time.monotonic()-started:.1f}s', flush=True)
        del matrices
        save_npz(output/'scales.npz', **{str(k): v for k,v in tables.items()})
        progress.update(scales_sha256=sha256(output/'scales.npz'), threshold_max_abs_error=error)
        save_json(progress_path, progress)
    scores = np.empty((len(hp_grid()), 10), np.float64)
    for col, row in enumerate(job['rows']):
        file = output/f"epoch_{row['epoch']:03d}.npz"
        key = str(row['epoch'])
        if key in progress['epochs']:
            if sha256(file) != progress['epochs'][key]:
                raise ValueError('Checkpoint score checksum mismatch')
            with np.load(file) as data:
                scores[:, col] = data['scores']
            continue
        cache = load_policy_cache(row['cache_file'])
        if cache['pred_actions'].shape != (8, db.n_states, 15, 10):
            raise ValueError('Need all eight samples, every val30 row, and genuine15-step chunks')
        order = align_dino_cache(db, chunks, meta, cache)
        scores[:, col] = score_grid(cache, {k: v[order] for k,v in tables.items()},
            progress=lambda ta: print(f"SCORE epoch={row['epoch']} Ta={ta} elapsed={time.monotonic()-started:.1f}s", flush=True))
        if col == 0:
            target = output/'canonical_smoke'
            original = canonical_thresholds(db, AXES['quantile'], 10, target)
            np.testing.assert_allclose(original.thresholds, tables[10], atol=1e-7, rtol=1e-6)
            maximum = 0.
            grid = hp_grid()
            for ta, q, tau, topk in ((9,.1,.03,1), (15,.95,30.,15), (15,.5,1.,8)):
                index = next(i for i,h in enumerate(grid) if (h['ta'],h['k'],h['quantile'],h['lse_tau'],h['chunk_top_k']) == (ta,10,q,tau,topk))
                hp = grid[index]
                summary, episodes = score_policy_cache(cache, str(target), ta=ta, quantile=q,
                    chunk_top_frac=hp['chunk_top_frac'], lse_tau=tau, every_step=True)
                ref = summary['PrefixSurvival_Score']
                maximum = max(maximum, abs(ref-scores[index,col]))
                np.testing.assert_allclose(ref, scores[index,col], atol=2e-7, rtol=2e-6)
                assert sum(e['T'] for e in episodes) == db.n_states
            progress['canonical_smoke_abs_error'] = maximum
        save_npz(file, scores=scores[:, col])
        progress['epochs'][key] = sha256(file)
        save_json(progress_path, progress)
    if {p: sha256(p) for p in signature['cache_sha256']} != signature['cache_sha256']:
        raise AssertionError('Cache changed during scoring')
    save_npz(output/'scores.npz', scores=scores, epochs=[r['epoch'] for r in job['rows']])
    save_json(verified, dict(complete=True, signature=signature, grid_size=len(scores),
        checkpoints=10, rows=db.n_states, horizon=15, num_samples=8, fallback_rows=0,
        threshold_max_abs_error=progress['threshold_max_abs_error'],
        canonical_smoke_abs_error=progress['canonical_smoke_abs_error'],
        elapsed_seconds=time.monotonic()-started,
        **{f+'_sha256':sha256(output/f) for f in ('scores.npz','scales.npz')}))


def choices(grid, metrics):
    import numpy as np
    selected = []
    def choose(domain, subset, values, eligible):
        for target in ('balanced_loss','nregret','mmrv','spearman'):
            objective = -values[target] if target == 'spearman' else values[target]
            ids = np.flatnonzero(subset & eligible)
            if len(ids):
                index = int(ids[np.lexsort((ids, values['balanced_loss'][ids], objective[ids]))[0]])
                selected.append(dict(domain=domain, target=target, hp_index=index))
    ta = np.array([h['ta'] for h in grid])
    for scope, subset in [('all',ta>0),('short',ta<=8),('long',ta>8)]:
        for task, values in metrics.items():
            choose(task+'/'+scope, subset, values, values['eligible'])
        eligible = np.logical_and.reduce([m['eligible'] for m in metrics.values()])
        macro = {key: np.mean([m[key] for m in metrics.values()], axis=0)
                 for key in ('balanced_loss','nregret','mmrv','spearman')}
        choose('common_macro/'+scope, subset, macro, eligible)
        worst = np.max([m['balanced_loss'] for m in metrics.values()], axis=0)
        ids = np.flatnonzero(subset & eligible)
        index = int(ids[np.lexsort((ids,macro['balanced_loss'][ids],worst[ids]))[0]])
        selected.append(dict(domain='common_minimax/'+scope, target='balanced_loss', hp_index=index))
    return selected


def report(args, specification):
    import numpy as np
    from surval.droid_long_tuning import metric_arrays
    from surval.droid_tuning import primary_metrics, canonical_thresholds
    from surval.droid import load_policy_cache, score_policy_cache
    from surval.droid_dino import dataset_db_path, load_shared_dino_db
    root = args.output_root
    labels_path = REPO/'outputs/droid_real_world_20260913/success_counts.json'
    labels = json.loads(labels_path.read_text())
    grid, scores, metrics, actual = hp_grid(), {}, {}, {}
    for job in specification['jobs']:
        task, output = job['task'], Path(job['job_dir'])/'eval'
        verification = json.loads((output/'verification.json').read_text())
        if not verification['complete'] or sha256(output/'scores.npz') != verification['scores.npz_sha256']:
            raise ValueError('Missing or corrupted worker output')
        with np.load(output/'scores.npz') as data:
            scores[task] = data['scores']
            np.testing.assert_array_equal(data['epochs'], labels['epoch_order'])
        if scores[task].shape != (len(grid),10):
            raise ValueError('Incomplete grid')
        actual[task] = np.array(labels['successes'][task])/labels['trials_per_checkpoint']
        metrics[task] = metric_arrays(actual[task], scores[task])
        save_npz(output/'metrics.npz', **metrics[task])
    selection = choices(grid, metrics)
    # Canonical scorer audit of EVERY published HP, all10 checkpoints per task.
    audited, maximum = {}, 0.
    for job in specification['jobs']:
        task = job['task']
        ids = sorted({c['hp_index'] for c in selection if c['domain'].startswith(task+'/') or c['domain'].startswith('common_')})
        output = Path(job['job_dir'])/'eval'
        db, _, _ = load_shared_dino_db(dataset_db_path(args.root/'shared_dino', job['dataset']))
        for k in sorted({grid[i]['k'] for i in ids}):
            path = output/'audit_thresholds'/str(k)
            # Full original thresholds, not only sampled-row checks, for published k.
            threshold = canonical_thresholds(db, AXES['quantile'], k, path)
            with np.load(output/'scales.npz') as data:
                np.testing.assert_allclose(threshold.thresholds, data[str(k)], atol=1e-7, rtol=1e-6)
            audit_ids = [i for i in ids if grid[i]['k']==k]
            references = np.empty((len(audit_ids),10))
            for col, row in enumerate(job['rows']):
                cache = load_policy_cache(row['cache_file'])
                for j,index in enumerate(audit_ids):
                    h = grid[index]
                    summary, episodes = score_policy_cache(cache, str(path), ta=h['ta'], quantile=h['quantile'],
                        chunk_top_frac=h['chunk_top_frac'], lse_tau=h['lse_tau'], every_step=True)
                    assert sum(e['T'] for e in episodes)==db.n_states
                    references[j,col]=summary['PrefixSurvival_Score']
            np.testing.assert_allclose(references,scores[task][audit_ids],atol=2e-7,rtol=2e-6)
            maximum=max(maximum,float(abs(references-scores[task][audit_ids]).max()))
            for j,index in enumerate(audit_ids):
                reference=primary_metrics(actual[task],references[j])
                for key in metrics[task]:
                    np.testing.assert_allclose(reference[key],metrics[task][key][index],atol=1e-12,equal_nan=True)
                audited[f'{task}:{index}']=references[j].tolist()
            print(f'AUDIT {task} k={k} HPs={len(audit_ids)} maxerror={maximum:.3g}',flush=True)
    # Independent original metric routine on every HP row; catches strict-tie changes.
    for task in metrics:
        for index, values in enumerate(scores[task]):
            ref=primary_metrics(actual[task],values)
            for key in metrics[task]:
                a,b=ref[key],metrics[task][key][index]
                if not (a==b or (np.isnan(a) and np.isnan(b)) or abs(a-b)<=1e-12):
                    raise AssertionError((task,index,key,a,b))
        print(f'METRICS independently verified {task}: {len(grid)} rows',flush=True)
    rows=[]
    for choice in selection:
        index=choice['hp_index']
        tasks=list(metrics) if choice['domain'].startswith('common_') else [choice['domain'].split('/')[0]]
        for task in tasks:
            values={k:v[index].item() for k,v in metrics[task].items()}
            rows.append({**choice,'task':task,**grid[index],**values,
                         'selected_epoch':labels['epoch_order'][values['selected_index']]})
    def write_rows(path, iterable):
        iterator=iter(iterable)
        first=next(iterator)
        with path.open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(first));writer.writeheader();writer.writerow(first);writer.writerows(iterator)
    write_rows(root/'selected_metrics.csv',rows)
    write_rows(root/'all_metrics.csv',({ 'task':task,'hp_index':i,**hp,
        **{k:v[i].item() for k,v in metrics[task].items()}}
        for task in metrics for i,hp in enumerate(grid)))
    save_json(root/'selection.json',dict(posthoc=True,labels_sha256=sha256(labels_path),choices=rows))
    save_json(root/'canonical_scores.json',audited)
    save_json(root/'verification.json',dict(complete=True,grid_size=len(grid),tasks=3,split='val30',
        checkpoint_scores=len(grid)*30,metric_rows=len(grid)*3,canonical_checkpoint_scores=len(audited)*10,
        canonical_max_abs_error=maximum,all_metrics_original_definition_verified=True,scale_multiplier=1.,
        labels_sha256=sha256(labels_path),selected_csv_sha256=sha256(root/'selected_metrics.csv')))
    lines=['# c=1 · val30-only SurVAL 튜닝','',
        '실제 성공률을 보고 선택한 **사후 탐색 결과**이며 독립 평가 성능이 아닙니다.',
        '3태스크 × 10체크포인트, HP 161,280개. c=1, every_step=True, diffusion samples=8.',
        'DINO 표현/val30 reference는 고정. 원 모델 Tp16/To2의 실제 예측 15-step을 사용했습니다.',
        'NRegret=(최고SR−선택SR)/(최고SR−최저SR); MMRV는 SR 0–1 단위. 낮을수록 좋고 Spearman은 높을수록 좋습니다.',
        '균형 선택 손실=(NRegret + MMRV/SR범위 + (1−Spearman)/2)/3.', '',
        '| 범위 | 태스크 | Ta | k | q | worst 개수 | LSE τ | NRegret↓ | MMRV↓ | Spearman↑ | 선택 epoch |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        if row['target']=='balanced_loss':
            lines.append('| '+ ' | '.join([row['domain'],row['task'],str(row['ta']),str(row['k']),
                str(row['quantile']),str(row['chunk_top_k']),str(row['lse_tau']),f"{row['nregret']:.4f}",
                f"{row['mmrv']:.4f}",f"{row['spearman']:.4f}",str(row['selected_epoch'])])+' |')
    lines += ['', '각 지표별 최적 HP는 selected_metrics.csv, 전체 탐색은 all_metrics.csv와 conditions/*/seed_0/eval/scores.npz에 보존했습니다.',
              f'공개 선택값 모두 원래 scorer로 재계산: {len(audited)*10}점, 최대 절대오차 {maximum:.3g}; 지표/선택 epoch 일치.',
              '기존 H8 캐시와 재생성 H15 캐시의 확률적 예측은 미세하게 다릅니다. short/long 비교는 모두 같은 새 H15 캐시의 prefix를 사용합니다.',
              '범위 내 최적값일 뿐 전역 최적 보장은 없습니다. 독립 task/robot evaluation 또는 사전 고정 HP로 재검증해야 합니다.']
    (root/'REPORT.ko.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=REPO/'outputs/droid_c1_long_20260913')
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--num-shards',type=int,default=1)
    parser.add_argument('--shard-idx',type=int,default=0)
    parser.add_argument('--worker-index',type=int)
    parser.add_argument('--list',action='store_true')
    parser.add_argument('--report',action='store_true')
    args=parser.parse_args()
    if args.workers<1 or args.num_shards<1 or not 0<=args.shard_idx<args.num_shards:
        parser.error('Invalid workers/shard configuration')
    args.root=args.root.resolve();args.output_root=args.root/'tuning'
    specification=matrix(args)
    if args.list or os.environ.get('DRY_RUN')=='1':
        print(json.dumps({**specification,'jobs':[j for j in specification['jobs'] if j['index']%args.num_shards==args.shard_idx]},indent=2))
        return 0
    if args.report:
        report(args,specification);return 0
    if args.worker_index is not None:
        score_worker(args,specification['jobs'][args.worker_index]);return 0
    return execute(args,specification)


if __name__=='__main__':
    raise SystemExit(main())
