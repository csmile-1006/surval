"""Compare requested Ta prefixes using the completed c=1/val30 grid; no inference."""

import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from cache_droid_datasets import REPO, save_json
from report_droid_hp import write_csv
from tune_droid_hp import sha256
from tune_droid_long_hp import matrix
from surval.droid import load_policy_cache, score_policy_cache
from surval.droid_dino import align_dino_cache, dataset_db_path, load_shared_dino_db
from surval.droid_long_tuning import AXES, FIXED, hp_grid, metric_arrays
from surval.droid_tuning import canonical_thresholds, primary_metrics
from surval.local_threshold.threshold import LocalThresholdMap


def best_index(mask, eligible, objective, balanced):
    ids = np.flatnonzero(mask & eligible)
    if not len(ids):
        raise ValueError("No eligible HP in requested subset")
    index = int(ids[np.lexsort((ids, balanced[ids], objective[ids]))[0]])
    assert objective[index] == objective[ids].min()
    tied = ids[objective[ids] == objective[index]]
    assert balanced[index] == balanced[tied].min()
    assert index == tied[balanced[tied] == balanced[index]].min()
    return index


def markdown_table(rows, columns):
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in rows:
        values = [f"{row[k]:.6f}" if isinstance(row[k], float) else str(row[k]) for k in columns]
        lines.append("| " + " | ".join(values) + " |")
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=REPO/'outputs/droid_c1_long_20260913')
    parser.add_argument('--output', type=Path, default=REPO/'outputs/droid_ta_comparison_20260914')
    parser.add_argument('--ta', type=int, nargs='+', default=[2, 4, 8, 10, 12, 14, 15])
    args = parser.parse_args()
    if len(set(args.ta)) != len(args.ta) or not set(args.ta) <= set(AXES['ta']):
        parser.error('Use distinct native chunk lengths 1..15; Ta16 is unavailable for Tp16/To2')
    args.root = args.root.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)  # Never overwrite a previous report.
    root = args.root/'tuning'
    labels_path = REPO/'outputs/droid_real_world_20260913/success_counts.json'
    labels = json.loads(labels_path.read_text())
    prior = json.loads((root/'verification.json').read_text())
    assert prior['complete'] and prior['all_metrics_original_definition_verified']
    assert prior['labels_sha256'] == sha256(labels_path)
    assert prior['selected_csv_sha256'] == sha256(root/'selected_metrics.csv')
    jobs = matrix(SimpleNamespace(root=args.root, output_root=root))['jobs']
    grid = hp_grid()
    ta = np.array([h['ta'] for h in grid])
    requested = np.isin(ta, args.ta)
    assert requested.sum() == sum(args.ta)*len(AXES['k'])*len(AXES['quantile'])*len(AXES['lse_tau'])
    scores, metrics, actual, source_hashes = {}, {}, {}, {}
    for job in jobs:
        task, directory = job['task'], Path(job['job_dir'])/'eval'
        proof = json.loads((directory/'verification.json').read_text())
        assert proof['complete'] and proof['signature']['fixed'] == FIXED
        assert proof['signature']['axes'] == AXES
        for path, expected in proof['signature']['code_sha256'].items():
            assert sha256(REPO/path) == expected, path
        for path, expected in proof['signature']['cache_sha256'].items():
            assert sha256(path) == expected, path
            source_hashes[path] = expected
        for name in ('scores.npz', 'scales.npz'):
            expected = proof[name+'_sha256']
            assert sha256(directory/name) == expected
            source_hashes[str(directory/name)] = expected
        with np.load(directory/'scores.npz') as data:
            scores[task] = data['scores']
            np.testing.assert_array_equal(data['epochs'], labels['epoch_order'])
        assert scores[task].shape == (len(grid), 10)
        actual[task] = np.asarray(labels['successes'][task])/labels['trials_per_checkpoint']
        metrics[task] = metric_arrays(actual[task], scores[task])
        with np.load(directory/'metrics.npz') as data:
            for key, value in metrics[task].items():
                np.testing.assert_allclose(value, data[key], rtol=0, atol=1e-12, equal_nan=True)
        print(f'VERIFIED source and metrics: {task}', flush=True)

    targets = ('balanced_loss', 'nregret', 'mmrv', 'spearman')
    macro = {key: np.mean([m[key] for m in metrics.values()], axis=0) for key in targets}
    common_eligible = np.logical_and.reduce([m['eligible'] for m in metrics.values()])
    worst = np.max([m['balanced_loss'] for m in metrics.values()], axis=0)
    selections, selected_rows, common_rows = [], [], []
    scopes = [(f'Ta{length}', ta == length) for length in sorted(args.ta)]
    scopes += [('requested', requested), ('previous_all', ta > 0), ('previous_long', ta > 8)]
    for scope, mask in scopes:
        for domain in (*metrics, 'common_macro', 'common_minimax'):
            values = metrics.get(domain, macro)
            eligible = metrics[domain]['eligible'] if domain in metrics else common_eligible
            wanted = ('balanced_loss',) if scope.startswith('Ta') or domain == 'common_minimax' else targets
            for target in wanted:
                objective = worst if domain == 'common_minimax' else values[target]
                if target == 'spearman':
                    objective = -objective
                index = best_index(mask, eligible, objective, values['balanced_loss'])
                selection = dict(scope=scope, domain=domain, target=target, hp_index=index)
                selections.append(selection)
                tasks = [domain] if domain in metrics else list(metrics)
                for task in tasks:
                    ref = primary_metrics(actual[task], scores[task][index])
                    for key, value in ref.items():
                        np.testing.assert_allclose(value, metrics[task][key][index], rtol=0, atol=1e-12)
                    selected_rows.append(dict(**selection, task=task, **grid[index], **ref,
                        selected_epoch=labels['epoch_order'][ref['selected_index']]))
                if domain.startswith('common_'):
                    common_rows.append(dict(**selection, **grid[index],
                        **{key: float(value[index]) for key, value in macro.items()},
                        worst_task_loss=float(worst[index])))
    write_csv(args.output/'selected_metrics.csv', selected_rows)
    write_csv(args.output/'common_metrics.csv', common_rows)
    checkpoint_rows = []
    for row in selected_rows:
        for col, epoch in enumerate(labels['epoch_order']):
            checkpoint_rows.append({key: row[key] for key in ('scope', 'domain', 'target', 'task', 'hp_index')}
                | grid[row['hp_index']] | dict(epoch=epoch, surval_score=float(scores[row['task']][row['hp_index'], col]),
                    successes=labels['successes'][row['task']][col], trials=labels['trials_per_checkpoint'],
                    success_rate=float(actual[row['task']][col])))
    write_csv(args.output/'checkpoint_scores.csv', checkpoint_rows)

    # Re-score all newly published choices, all checkpoints, with the library.
    # Previously published optima are comparison inputs, already canonically audited.
    canonical, max_error, threshold_error = [], 0., 0.
    for job in jobs:
        task, directory = job['task'], Path(job['job_dir'])/'eval'
        ids = sorted({r['hp_index'] for r in selected_rows if r['task'] == task
                      and not r['scope'].startswith('previous_')})
        db, chunks, meta = load_shared_dino_db(dataset_db_path(args.root/'shared_dino', job['dataset']))
        proof = json.loads((directory/'verification.json').read_text())
        assert meta['content_sha256'] == proof['signature']['db_sha256']
        paths = {}
        with np.load(directory/'scales.npz') as data:
            for k in sorted({grid[i]['k'] for i in ids}):
                path = directory/'audit_thresholds'/str(k)
                if path.exists():
                    threshold = LocalThresholdMap.load(str(path/'thresholds'))
                else:
                    path = args.output/'audit_thresholds'/task/str(k)
                    threshold = canonical_thresholds(db, AXES['quantile'], k, path)
                np.testing.assert_array_equal(np.load(path/'demo_id_int.npy'), db.records.demo_id_int)
                np.testing.assert_array_equal(np.load(path/'t.npy'), db.records.t)
                saved_ids = json.loads((path/'demo_id_str.json').read_text())
                assert {int(k): v for k, v in saved_ids.items()} == db.records.demo_id_str_by_int
                assert threshold.quantiles == tuple(AXES['quantile'])
                assert not threshold.fallback_used.any() and (threshold.n_neighbors_used == k).all()
                np.testing.assert_allclose(threshold.thresholds, data[str(k)], rtol=1e-6, atol=1e-7)
                threshold_error = max(threshold_error, float(abs(threshold.thresholds-data[str(k)]).max()))
                paths[k] = path
                print(f'THRESHOLDS verified {task} k={k}', flush=True)
        references = np.empty((len(ids), 10))
        for col, row in enumerate(job['rows']):
            cache = load_policy_cache(row['cache_file'])
            assert cache['pred_actions'].shape == (8, db.n_states, 15, 10)
            align_dino_cache(db, chunks, meta, cache)
            for j, index in enumerate(ids):
                hp = grid[index]
                with redirect_stdout(io.StringIO()):
                    summary, episodes = score_policy_cache(cache, str(paths[hp['k']]), ta=hp['ta'],
                        quantile=hp['quantile'], chunk_top_frac=hp['chunk_top_frac'], lse_tau=hp['lse_tau'],
                        num_samples=8, every_step=True)
                assert sum(e['T'] for e in episodes) == db.n_states
                ref = references[j, col] = summary['PrefixSurvival_Score']
                error = abs(ref-scores[task][index, col])
                max_error = max(max_error, float(error))
                canonical.append(dict(task=task, hp_index=index, epoch=row['epoch'],
                    canonical_score=ref, grid_score=float(scores[task][index, col]), abs_error=float(error)))
            print(f'AUDIT {task} epoch={row["epoch"]} HPs={len(ids)} max_error={max_error:.3g}', flush=True)
        np.testing.assert_allclose(references, scores[task][ids], rtol=2e-6, atol=2e-7)
        for j, index in enumerate(ids):
            for key, value in primary_metrics(actual[task], references[j]).items():
                np.testing.assert_allclose(value, metrics[task][key][index], rtol=0, atol=1e-12)
    write_csv(args.output/'canonical_audit.csv', canonical)
    for path, expected in source_hashes.items():
        assert sha256(path) == expected, path

    lines = ['# SurVAL chunk 길이 비교: c=1, val30 only', '',
        f'요청 길이: {sorted(args.ta)}. Tp16/To2의 실제 미래 action은 최대15개이므로16은15로 대체.',
        f'기존 전체161,280 HP 중 {int(requested.sum()):,} HP ×3태스크 ×10체크포인트를 재사용.',
        '각 Ta에서 다른 HP도 재최적화한 결과이며, 나머지 HP를 고정한 단일변수 ablation이 아니다.',
        '모든 비교는 동일 H15 캐시의 prefix: c=1, every_step=True, samples=8, frozen DINO/full-val30 reference.',
        '정규화는 pos L2 / rot SO(3)의 local expert-pair quantile을 각각 사용; gripper는 제외.',
        'k=이웃 수; q=그룹별 expert-pair quantile; worst=chunk 내 큰 오차를 평균할 action 수.',
        'chunk_top_frac=(worst−0.5)/Ta는 ceil(Ta×frac)=worst를 정확히 구현하는 입력값이다.',
        'LSE τ는 그룹 survival의 soft-min 강도이며, 클수록 최악 그룹에 집중한다.',
        'NRegret=(최고SR−선택SR)/(최고SR−최저SR); regret_pp는 원래 성공률 손실(%p).',
        'MMRV는 SR 0–1 단위, Spearman은 동률 평균 rank; SurVAL score가 클수록 좋다.',
        '균형 손실=(NRegret + MMRV/SR범위 + (1−Spearman)/2)/3. 낮을수록 좋다.',
        'common_macro는 세 태스크 균형 손실 평균을, common_minimax는 가장 나쁜 태스크 손실을 최소화한다.',
        '동률은 균형 손실→HP 인덱스 순으로 선택(minimax는 평균 손실→인덱스).', '',
        '## 탐색 범위', '', json.dumps(AXES, ensure_ascii=False),
        'worst 개수는 각 Ta에서1..Ta 전체. 이번 비교에서 새로운 HP 범위를 추가하지 않았다.', '',
        '## 길이별 태스크 최적 HP', '']
    columns = ['scope', 'task', 'ta', 'k', 'quantile', 'chunk_top_k', 'lse_tau',
               'nregret', 'mmrv', 'spearman', 'selected_epoch']
    lines += markdown_table([r for r in selected_rows if r['scope'].startswith('Ta') and r['domain'] in metrics], columns)
    lines += ['', '## 길이별 공통 HP: macro / minimax', '']
    common_columns = ['scope', 'domain', 'ta', 'k', 'quantile', 'chunk_top_k', 'lse_tau',
                      'nregret', 'mmrv', 'spearman', 'balanced_loss', 'worst_task_loss']
    lines += markdown_table([r for r in common_rows if r['scope'].startswith('Ta')], common_columns)
    lines += ['', '## 요청 범위 최적값과 이전 최적값', '']
    lines += markdown_table([r for r in selected_rows if not r['scope'].startswith('Ta')
                             and r['target'] == 'balanced_loss'], ['scope', 'domain'] + columns[1:])
    lines += ['', '### 공통 HP의 macro 평균', '']
    lines += markdown_table([r for r in common_rows if not r['scope'].startswith('Ta')
                             and r['target'] == 'balanced_loss'], common_columns)
    lines += ['', '## 지표별 별도 최적값 (요청 범위)', '',
        '각 행은 해당 지표만 최우선으로 최적화한 하나의 HP이다. 서로 다른 행의 최저값을 한 조합처럼 합치면 안 된다.', '']
    lines += markdown_table([r for r in selected_rows if r['scope'] == 'requested'
                             and r['target'] != 'balanced_loss'], ['target', 'domain'] + columns[1:])
    lines += ['', '## 검증 및 해석', '',
        f'원 scorer로 신규 선택 HP의 모든 체크포인트 {len(canonical)}점을 재계산: 최대 절대오차 {max_error:.3g}.',
        '원 scalar metric과 벡터 집계, 선택 epoch 일치. 캐시·score·scale 원본 SHA256 전후 일치.',
        'pan은 요청한 모든 Ta에서 task별 최적 균형 지표가 동일하다. Ta2는 tie-break 대표값이다.',
        'success rate를 보고 HP를 고른 사후 튜닝 결과이며 독립 평가 성능이 아니다. 유한 grid 내 최적값이다.',
        '논문에는 탐색 범위·선택 기준·사용한 real-world 결과를 명시하고, 별도 held-out 평가로 확인해야 한다.', '',
        'selected_metrics.csv: 모든 선택의 태스크별 지표. common_metrics.csv: 공통 HP의 macro 평균.',
        'checkpoint_scores.csv: 선택 HP별10개 checkpoint score/실제 성공 횟수. canonical_audit.csv: 원 scorer 대조.', '',
        '재현 (새 --output 경로 필요):', '', '```bash', 'cd /workspace/surval',
        'PYTHONPATH=src:scripts CUDA_VISIBLE_DEVICES=\'\' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \\',
        '  .venv-droid/bin/python scripts/compare_droid_chunk_lengths.py --output /workspace/surval/outputs/ta_comparison_rerun',
        '```']
    (args.output/'REPORT.ko.md').write_text('\n'.join(lines)+'\n')
    save_json(args.output/'verification.json', dict(complete=True, requested_ta=sorted(args.ta),
        fixed=FIXED, axes=AXES, requested_hp_count=int(requested.sum()),
        reused_checkpoint_scores=int(requested.sum())*30, selection_objectives=len(selections),
        selected_metric_rows=len(selected_rows), checkpoint_rows=len(checkpoint_rows),
        canonical_checkpoint_scores=len(canonical), canonical_max_abs_error=max_error,
        threshold_max_abs_error=threshold_error, primary_metrics_and_selection_unchanged=True,
        labels_sha256=sha256(labels_path), script_sha256=sha256(__file__), source_sha256=source_hashes,
        outputs_sha256={p.name: sha256(p) for p in args.output.iterdir() if p.is_file()},
        posthoc=True, original_outputs_unchanged=True))
    print(f'PASS: {len(selections)} objectives; {len(canonical)} canonical scores; {args.output}', flush=True)


if __name__ == '__main__':
    main()
