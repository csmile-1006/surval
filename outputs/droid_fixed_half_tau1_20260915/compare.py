"""Freeze worst-half/tau1; select Ta/k/q from verified val30 scores, no inference."""

from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
sys.path[:0] = [str(REPO/'src'), str(REPO/'scripts')]
from cache_droid_datasets import save_json
from compare_droid_chunk_lengths import best_index, markdown_table
from report_droid_hp import write_csv
from tune_droid_hp import sha256
from tune_droid_long_hp import matrix
from surval.droid import load_policy_cache, score_policy_cache
from surval.droid_dino import align_dino_cache, dataset_db_path, load_shared_dino_db
from surval.droid_long_tuning import AXES, FIXED, hp_grid, metric_arrays
from surval.droid_tuning import canonical_thresholds, primary_metrics
from surval.local_threshold.threshold import LocalThresholdMap
from surval.sequential_validate import _aggregate_chunk_time


def read_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    # Small explicit snapshots preserve the previous publication tables as requested.
    prior = REPO/'outputs/droid_ta_comparison_20260914'
    baseline = REPO/'outputs/droid_optimal_vs_baselines_20260915'
    source = REPO/'outputs/droid_c1_long_20260913'
    labels_path = REPO/'outputs/droid_real_world_20260913/success_counts.json'
    labels = json.loads(labels_path.read_text())
    prior_proof = json.loads((prior/'verification.json').read_text())
    assert prior_proof['complete'] and prior_proof['labels_sha256'] == sha256(labels_path)
    saved_hashes = {str(labels_path): sha256(labels_path)}
    snapshots = {}
    for folder in (prior, baseline):
        proof = json.loads((folder/'verification.json').read_text())
        assert proof['complete']
        for name, expected in proof['outputs_sha256'].items():
            assert sha256(folder/name) == expected, name
        target = ROOT/'prior_results'/folder.name
        target.mkdir(parents=True, exist_ok=True)
        for path in [folder/'verification.json', *(folder/name for name in proof['outputs_sha256'])]:
            digest = sha256(path)
            saved_hashes[str(path)] = digest
            copy = target/path.name
            if copy.exists():
                assert sha256(copy) == digest, 'Do not overwrite an incompatible snapshot'
            else:
                shutil.copy2(path, copy)
            assert sha256(copy) == digest
            snapshots[str(copy.relative_to(ROOT))] = dict(source=str(path), sha256=digest)
    save_json(ROOT/'prior_results/manifest.json', snapshots)
    print(f'PRESERVED {len(snapshots)} prior artifacts', flush=True)

    grid = hp_grid()
    ta = np.array([h['ta'] for h in grid])
    half = np.array([h['chunk_top_k'] == (h['ta']+1)//2 and h['lse_tau'] == 1. for h in grid])
    requested = np.isin(ta, [2, 4, 8, 10, 12, 14, 15])
    assert half.sum() == 2880 and (half & requested).sum() == 1344
    assert len({(h['ta'], h['k'], h['quantile']) for h, keep in zip(grid, half) if keep}) == 2880
    # Runnable check: the library uses ceil, including odd native horizons.
    for length in AXES['ta']:
        values = np.arange(length, dtype=np.float32).reshape(1, 1, length)
        count = (length+1)//2
        np.testing.assert_array_equal(_aggregate_chunk_time(values, agg='top_k_mean', top_frac=.5),
                                      values[:, :, -count:].mean(axis=2))
    jobs = matrix(SimpleNamespace(root=source, output_root=source/'tuning'))['jobs']
    scores, metrics, actual = {}, {}, {}
    for job in jobs:
        task, directory = job['task'], Path(job['job_dir'])/'eval'
        proof = json.loads((directory/'verification.json').read_text())
        assert proof['complete'] and proof['signature']['fixed'] == FIXED and proof['signature']['axes'] == AXES
        inputs = {**proof['signature']['cache_sha256'],
                  **{str(REPO/p): v for p, v in proof['signature']['code_sha256'].items()},
                  **{str(directory/n): proof[n+'_sha256'] for n in ('scores.npz', 'scales.npz')}}
        for path, expected in inputs.items():
            assert sha256(path) == expected, path
        saved_hashes.update(inputs)
        with np.load(directory/'scores.npz') as data:
            scores[task] = data['scores']
            np.testing.assert_array_equal(data['epochs'], labels['epoch_order'])
        actual[task] = np.array(labels['successes'][task], np.float64)/labels['trials_per_checkpoint']
        metrics[task] = metric_arrays(actual[task], scores[task])
        with np.load(directory/'metrics.npz') as data:
            for key, value in metrics[task].items():
                np.testing.assert_allclose(value, data[key], atol=1e-12, rtol=0, equal_nan=True)
        # Independently verify every candidate of this restricted search.
        for index in np.flatnonzero(half):
            for key, value in primary_metrics(actual[task], scores[task][index]).items():
                np.testing.assert_allclose(value, metrics[task][key][index], atol=1e-12, rtol=0, equal_nan=True)
        print(f'VERIFIED {task}: 2880 fixed candidates', flush=True)

    metric_names = ('nregret', 'mmrv', 'spearman', 'balanced_loss', 'regret_pp', 'selected_success_rate')
    common = {key: np.mean([v[key] for v in metrics.values()], axis=0) for key in metric_names}
    eligible = np.logical_and.reduce([v['eligible'] for v in metrics.values()])
    previous = read_csv(prior/'selected_metrics.csv')
    selections, rows, aggregates = [], [], []
    for scope, mask in [('all', ta > 0), ('requested', requested)]:
        for regime, allowed in [('free', mask), ('fixed_half_tau1', mask & half)]:
            for domain in ('common', *metrics):
                values = common if domain == 'common' else metrics[domain]
                index = best_index(allowed, eligible if domain == 'common' else values['eligible'],
                                   values['balanced_loss'], values['balanced_loss'])
                hp = dict(grid[index])
                if regime != 'free':
                    hp['chunk_top_frac'] = .5  # Actual scorer input; same integer top-k as saved grid.
                    assert int(np.ceil(hp['ta']*.5)) == hp['chunk_top_k'] and hp['lse_tau'] == 1.
                selected = dict(scope=scope, regime=regime, domain=domain, hp_index=index, **hp)
                selections.append(selected)
                tasks = list(metrics) if domain == 'common' else [domain]
                for task in tasks:
                    if regime == 'free':
                        old = [r for r in previous if (r['scope'], r['domain'], r['task'], r['target']) ==
                            ('previous_all' if scope == 'all' else scope,
                             'common_macro' if domain == 'common' else domain, task, 'balanced_loss')]
                        assert len(old) == 1 and int(old[0]['hp_index']) == index
                    values = primary_metrics(actual[task], scores[task][index])
                    rows.append(dict(**selected, task=task, **values,
                                     selected_epoch=labels['epoch_order'][values['selected_index']]))
            for selection in ('common', 'task_specific'):
                subset = [r for r in rows if r['scope'] == scope and r['regime'] == regime
                          and (r['domain'] == 'common' if selection == 'common' else r['domain'] == r['task'])]
                assert len(subset) == 3
                aggregates.append(dict(scope=scope, regime=regime, selection=selection,
                    **{key: float(np.mean([r[key] for r in subset])) for key in metric_names}))
    write_csv(ROOT/'selected_hp.csv', selections)
    write_csv(ROOT/'selected_metrics.csv', rows)
    write_csv(ROOT/'aggregate_metrics.csv', aggregates)
    write_csv(ROOT/'fixed_grid_metrics.csv', [dict(task=task, hp_index=int(i), **{**grid[i], 'chunk_top_frac': .5},
        **{key: value[i].item() for key, value in metrics[task].items()})
        for task in metrics for i in np.flatnonzero(half)])

    canonical, max_error = [], 0.
    for job in jobs:
        task, directory = job['task'], Path(job['job_dir'])/'eval'
        ids = sorted({r['hp_index'] for r in rows if r['task'] == task and r['regime'] == 'fixed_half_tau1'})
        db, chunks, metadata = load_shared_dino_db(dataset_db_path(source/'shared_dino', job['dataset']))
        proof = json.loads((directory/'verification.json').read_text())
        assert metadata['content_sha256'] == proof['signature']['db_sha256']
        paths = {}
        with np.load(directory/'scales.npz') as data:
            for k in sorted({grid[i]['k'] for i in ids}):
                candidates = [directory/'audit_thresholds'/str(k), prior/'audit_thresholds'/task/str(k),
                              ROOT/'audit_thresholds'/task/str(k)]
                path = next((p for p in candidates if (p/'thresholds/local_threshold_map.npz').exists()), None)
                if path is None:
                    path = candidates[-1]
                    print(f'COMPUTE original thresholds: {task} k={k}', flush=True)
                    threshold = canonical_thresholds(db, AXES['quantile'], k, path)
                else:
                    threshold = LocalThresholdMap.load(str(path/'thresholds'))
                np.testing.assert_array_equal(np.load(path/'demo_id_int.npy'), db.records.demo_id_int)
                np.testing.assert_array_equal(np.load(path/'t.npy'), db.records.t)
                assert {int(k): v for k, v in json.loads((path/'demo_id_str.json').read_text()).items()} == db.records.demo_id_str_by_int
                assert threshold.quantiles == tuple(AXES['quantile'])
                assert not threshold.fallback_used.any() and (threshold.n_neighbors_used == k).all()
                np.testing.assert_allclose(threshold.thresholds, data[str(k)], rtol=1e-6, atol=1e-7)
                paths[k] = path
                print(f'THRESHOLDS verified {task} k={k}', flush=True)
        refs = np.empty((len(ids), 10))
        for col, row in enumerate(job['rows']):
            cache = load_policy_cache(row['cache_file'])
            assert cache['pred_actions'].shape == (8, db.n_states, 15, 10)
            align_dino_cache(db, chunks, metadata, cache)
            for j, index in enumerate(ids):
                h = grid[index]
                with redirect_stdout(io.StringIO()):
                    summary, episodes = score_policy_cache(cache, str(paths[h['k']]), ta=h['ta'],
                        quantile=h['quantile'], chunk_top_frac=.5, lse_tau=1., num_samples=8, every_step=True)
                assert sum(e['T'] for e in episodes) == db.n_states
                refs[j, col] = summary['PrefixSurvival_Score']
                error = abs(refs[j, col]-scores[task][index, col])
                max_error = max(max_error, float(error))
                canonical.append(dict(task=task, hp_index=index, epoch=row['epoch'],
                    original_score=float(refs[j, col]), grid_score=float(scores[task][index, col]), abs_error=float(error)))
            print(f'AUDIT {task} epoch={row["epoch"]}: {len(ids)} HPs', flush=True)
        np.testing.assert_allclose(refs, scores[task][ids], atol=2e-7, rtol=2e-6)
        for j, index in enumerate(ids):
            for key, value in primary_metrics(actual[task], refs[j]).items():
                np.testing.assert_allclose(value, metrics[task][key][index], atol=1e-12, rtol=0)
    write_csv(ROOT/'canonical_audit.csv', canonical)
    checkpoint_rows = []
    for row in rows:
        for col, epoch in enumerate(labels['epoch_order']):
            checkpoint_rows.append({k: row[k] for k in ('scope', 'regime', 'domain', 'task', 'hp_index')}
                | dict(epoch=epoch, surval_score=float(scores[row['task']][row['hp_index'], col]),
                       successes=labels['successes'][row['task']][col], trials=labels['trials_per_checkpoint']))
    write_csv(ROOT/'checkpoint_scores.csv', checkpoint_rows)
    lines = ['# SurVAL: worst-half / LSE=1 고정 재튜닝', '',
        '기존 결과를 prior_results/에 별도 복사하고 SHA256을 보존했다. 원본 결과·캐시·코드는 수정하지 않았다.',
        'c=1, val30, frozen DINO/full-val30 reference, every_step=True, diffusion samples8 고정.',
        'worst-action=ceil(Ta/2), chunk_top_frac=0.5, LSE τ=1. Ta/k/q만 선택한다.',
        'Ta 전체1..15: 2880조합. 최근 후보2/4/8/10/12/14/15: 1344조합. k/q 후보는 기존과 동일하다.',
        f'k={AXES["k"]}', f'q={AXES["quantile"]}',
        '모든 before/after는 동일 H15 예측 캐시를 사용한다. 추가 정책 추론/학습 없음.',
        '균형 손실=(NRegret + MMRV/SR범위 + (1−Spearman)/2)/3. 공통은 태스크 균형 손실 평균을 최소화한다.',
        '동률은 균형 손실→HP인덱스. 표는 세 태스크의 지표를 동일 가중 평균한 값이다.', '',
        '## Aggregate 비교', '']
    lines += markdown_table(aggregates, ['scope', 'regime', 'selection', 'nregret', 'mmrv', 'spearman', 'balanced_loss'])
    lines += ['', '## 고정 조건에서 선택된 HP', '']
    lines += markdown_table([s for s in selections if s['regime'] != 'free'],
        ['scope', 'domain', 'ta', 'k', 'quantile', 'chunk_top_k', 'chunk_top_frac', 'lse_tau'])
    lines += ['', '## 검증 및 보고 범위', '',
        f'전체고정 grid 8640 task/HP행을 원 scalar metric으로 검증. 선택 HP의 원 scorer {len(canonical)}점 대조, 최대 오차 {max_error:.3g}.',
        '모든 선택 metric/epoch 일치. 고정값0.5를 원 scorer에 직접 전달하여 홀수/짝수 ceil 동작까지 확인했다.',
        '이 결과도 같은 real-world 성공률을 이용한 사후 튜닝값이다. 독립 평가/통계적 유의성 주장이 아니다.',
        'aggregate_metrics.csv: 요청한 aggregate 결과. selected_hp.csv: 공통/태스크별 설정.',
        'fixed_grid_metrics.csv: Ta/k/q 전체 후보. selected_metrics.csv/checkpoint_scores.csv는 재현용 상세자료.',
        'prior_results/: 이전 자유튜닝/베이스라인 비교 결과의 별도 스냅샷.', '',
        '재현: `.venv-droid/bin/python outputs/droid_fixed_half_tau1_20260915/compare.py`']
    (ROOT/'REPORT.ko.md').write_text('\n'.join(lines)+'\n')
    for path, expected in saved_hashes.items():
        assert sha256(path) == expected, path
    save_json(ROOT/'verification.json', dict(complete=True, fixed={**FIXED, 'chunk_top_frac': .5, 'lse_tau': 1.},
        tuned_axes={k: AXES[k] for k in ('ta', 'k', 'quantile')}, full_candidates=2880, requested_candidates=1344,
        independently_verified_task_hp_rows=8640, aggregate_rows=len(aggregates), canonical_scores=len(canonical),
        canonical_max_abs_error=max_error, canonical_metrics_and_selection_unchanged=True,
        source_sha256=saved_hashes, previous_results_preserved=snapshots,
        output_sha256={p.name: sha256(p) for p in ROOT.iterdir() if p.is_file() and p.name != 'verification.json'},
        posthoc=True, new_inference=False))
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
