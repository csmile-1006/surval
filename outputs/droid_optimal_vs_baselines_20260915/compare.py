"""Reproduce val30-only optimal SurVAL vs recorded baseline comparison; no inference."""

import csv
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
sys.path[:0] = [str(REPO/'src'), str(REPO/'scripts')]
from cache_droid_datasets import save_json
from report_droid_hp import write_csv
from tune_droid_hp import sha256
from surval.droid_tuning import primary_metrics


def read_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    baseline_root = REPO/'outputs/droid_baselines_20260913'
    real_root = REPO/'outputs/droid_real_world_20260913'
    tuned_root = REPO/'outputs/droid_ta_comparison_20260914'
    baseline = baseline_root/'checkpoint_metrics.csv'
    labels_path = real_root/'success_counts.json'
    labels = json.loads(labels_path.read_text())
    epochs, trials = labels['epoch_order'], labels['trials_per_checkpoint']
    assert epochs == list(range(5, 51, 5)) and trials == 30
    old_proof = json.loads((real_root/'verification.json').read_text())
    assert sha256(baseline) == old_proof['baseline_csv_sha256_unchanged']
    proof = json.loads((tuned_root/'verification.json').read_text())
    assert proof['complete'] and proof['labels_sha256'] == sha256(labels_path)
    for name, expected in proof['outputs_sha256'].items():
        assert sha256(tuned_root/name) == expected, name
    for path, expected in proof['source_sha256'].items():
        assert sha256(path) == expected, path
    source_paths = [baseline, labels_path, real_root/'primary_metrics.csv',
                    tuned_root/'selected_metrics.csv', tuned_root/'checkpoint_scores.csv',
                    REPO/'outputs/droid_c1_long_20260913/cache_index.json']
    source_hashes = {str(p): sha256(p) for p in source_paths}
    baseline_rows = [r for r in read_csv(baseline) if r['dataset'].endswith('/val30')]
    assert len(baseline_rows) == 30
    old_metrics = {(r['task'], r['method']): r for r in read_csv(real_root/'primary_metrics.csv')
                   if r['val_split'] == 'val30'}
    tuned_rows = read_csv(tuned_root/'selected_metrics.csv')
    tuned_scores = read_csv(tuned_root/'checkpoint_scores.csv')
    cache_index = json.loads(source_paths[-1].read_text())['results']
    key = lambda r: (r['run'], r['run_timestamp'], r['dataset'], int(r['epoch']), r['checkpoint'])
    assert {key(r) for r in cache_index} == {key(r) for r in baseline_rows}
    methods = {'Loss': ('Loss', -1), 'MSE': ('MSE_mean_pred', -1),
               'Policy OMN': ('Off_Manifold_Norm', -1), 'DINO OMN': ('Off_Manifold_Norm_DINO', -1),
               'SurVAL original': ('surval_score', 1)}
    selections = {'SurVAL common/all': ('previous_all', 'common_macro'),
                  'SurVAL task/all': ('previous_all', None),
                  'SurVAL common/requested': ('requested', 'common_macro'),
                  'SurVAL task/requested': ('requested', None)}
    results, checkpoints = [], []

    def add(task, method, raw, direction, expected, hp, coverage):
        actual = np.array(labels['successes'][task], np.float64)/trials
        raw = np.asarray(raw, np.float64)
        assert raw.shape == (10,) and np.isfinite(raw).all()
        oriented = direction*raw
        values = primary_metrics(actual, oriented)
        for name in ('nregret', 'mmrv', 'spearman', 'regret_pp', 'selected_success_rate'):
            np.testing.assert_allclose(values[name], float(expected[name]), rtol=0, atol=1e-12)
        np.testing.assert_allclose(values['spearman'], spearmanr(actual, oriented).statistic, atol=1e-12)
        mmrv = np.mean([max(abs(actual[i]-actual[j]) if
            ((oriented[i] < oriented[j]) != (actual[i] < actual[j])) else 0.
            for j in range(10)) for i in range(10)])
        np.testing.assert_allclose(values['mmrv'], mmrv, atol=1e-12)
        chosen = values['selected_index']
        assert epochs[chosen] == int(expected['selected_epoch'])
        counts = labels['successes'][task]
        np.testing.assert_allclose(values['nregret'],
            (max(counts)-counts[chosen])/(max(counts)-min(counts)), atol=1e-10)
        results.append(dict(task=task, method=method, val_split='val30', **values,
            selected_epoch=epochs[chosen], selected_successes=counts[chosen], trials=trials,
            hp=hp, coverage=coverage))
        for i, epoch in enumerate(epochs):
            checkpoints.append(dict(task=task, method=method, epoch=epoch, proxy_value=float(raw[i]),
                higher_is_better=direction > 0, successes=counts[i], trials=trials, success_rate=float(actual[i])))

    for task in labels['successes']:
        rows = sorted([r for r in baseline_rows if task in r['dataset'].split('/')[0].split('_')],
                      key=lambda r: int(r['epoch']))
        assert [int(r['epoch']) for r in rows] == epochs
        assert all(int(r['num_demos']) == 30 and int(r['ta']) == 8 for r in rows)
        for name, (column, direction) in methods.items():
            coverage = 'first1600' if column in ('Loss', 'Off_Manifold_Norm') else 'full_val30'
            add(task, name, [float(r[column]) for r in rows], direction, old_metrics[task, column],
                'recorded H8; DINO OMN k=5' if name == 'DINO OMN' else 'recorded H8', coverage)
        for name, (scope, domain) in selections.items():
            selected = [r for r in tuned_rows if (r['scope'], r['domain'], r['task'], r['target']) ==
                        (scope, domain or task, task, 'balanced_loss')]
            assert len(selected) == 1
            row = selected[0]
            subset = sorted([r for r in tuned_scores if (r['scope'], r['domain'], r['task'], r['target']) ==
                (scope, domain or task, task, 'balanced_loss')], key=lambda r: int(r['epoch']))
            assert [int(r['epoch']) for r in subset] == epochs
            assert all(r['hp_index'] == row['hp_index'] for r in subset)
            assert [int(r['successes']) for r in subset] == labels['successes'][task]
            hp = ', '.join(f'{k}={row[k]}' for k in ('ta', 'k', 'quantile', 'chunk_top_k', 'lse_tau', 'scale_multiplier'))
            add(task, name, [float(r['surval_score']) for r in subset], 1, row, hp, 'full_val30')
    names = [*methods, *selections]
    macro = []
    for name in names:
        rows = [r for r in results if r['method'] == name]
        assert len(rows) == 3
        macro.append(dict(method=name, num_tasks=3,
            **{k: float(np.mean([r[k] for r in rows])) for k in
               ('nregret', 'mmrv', 'spearman', 'regret_pp', 'selected_success_rate', 'balanced_loss')}))
    write_csv(ROOT/'task_metrics.csv', results)
    write_csv(ROOT/'macro_metrics.csv', macro)
    write_csv(ROOT/'checkpoint_scores.csv', checkpoints)
    lines = ['# 최적 SurVAL vs baseline: val30', '',
        '동일3태스크 ×10체크포인트(epoch5..50), checkpoint당30회 실측. 추가 추론/튜닝 없음.',
        'all=Ta1..15 전체 탐색; requested=Ta2/4/8/10/12/14/15 제한 탐색.',
        '최적은 기존 균형 손실 (NRegret + MMRV/SR범위 + (1−Spearman)/2)/3 기준이다.',
        'NRegret/MMRV는 낮을수록 좋고 Spearman은 높을수록 좋다. Loss/MSE/OMN은 부호 반전.',
        'NRegret은 성공률 gap/range, raw regret은 CSV의 regret_pp(%p). MMRV는 SR0–1 단위.', '',
        '## 태스크별 비교', '', '각 셀: NRegret / MMRV / Spearman.', '',
        '| Method | apple | pan | pet |', '|---|---|---|---|']
    lookup = {(r['task'], r['method']): r for r in results}
    for name in names:
        cells = [name]
        for task in ('apple', 'pan', 'pet'):
            r = lookup[task, name]
            cells.append(f"{r['nregret']:.4f} / {r['mmrv']:.4f} / {r['spearman']:.4f}")
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines += ['', '## 태스크 동일 가중 평균', '',
        '| Method | NRegret↓ | MMRV↓ | Spearman↑ | 선택 성공률 |', '|---|---:|---:|---:|---:|']
    for r in macro:
        lines.append(f"| {r['method']} | {r['nregret']:.4f} | {r['mmrv']:.4f} | {r['spearman']:.4f} | {r['selected_success_rate']:.2%} |")
    lines += ['', '## 선택 HP 및 checkpoint', '',
        '| Method | Task | HP | epoch | 실측 성공 횟수 |', '|---|---|---|---:|---:|']
    for r in results:
        if r['method'] in selections:
            lines.append(f"| {r['method']} | {r['task']} | {r['hp']} | {r['selected_epoch']} | {r['selected_successes']}/30 |")
    lines += ['', '## 해석 제한', '',
        '- 최적 SurVAL은 동일 real-world 결과로 HP를 고른 사후 튜닝값이다. Baseline은 기존 고정 설정이며 튜닝 예산이 다르다.',
        '- Loss와 policy OMN은 첫1600행: apple52.5%, pan52.8%, pet38.8%. 전체-val 지표로 표기하면 안 된다.',
        '- MSE와 DINO OMN은 full-val30/H8, tuned SurVAL은 새 H15 캐시 prefix를 쓴다. 확률적 예측 realization도 달라 엄밀한 동일-cache ablation이 아니다.',
        '- MSE/OMN은 normalized10D(gripper 포함), SurVAL은 physical pos/rot 그룹(gripper 제외).',
        '- DINO OMN은 full-val reference k5, policy OMN은 policy-encoder/batch-local이다.',
        '- 동일 checkpoint/run/timestamp/dataset 조인과 epoch coverage를 검증했다. Baseline의 저장값을 새로 평가하지 않았다.',
        '- 태스크별 지표를 평균했다. 세 태스크 checkpoint를 한데 섞은 Spearman이 아니며 독립 seed CI/유의성을 주장하지 않는다.',
        '- 논문 최종 비교에는 HP 선택과 독립된 평가, 동일 coverage 및 명시적인 horizon/예측-cache 조건이 필요하다.', '',
        '재현: `.venv-droid/bin/python outputs/droid_optimal_vs_baselines_20260915/compare.py`',
        '원본 유지; task_metrics.csv, macro_metrics.csv, checkpoint_scores.csv, verification.json에 검증 결과 보존.']
    (ROOT/'REPORT.ko.md').write_text('\n'.join(lines)+'\n')
    for path, expected in source_hashes.items():
        assert sha256(path) == expected
    assert len(results) == 27 and len(checkpoints) == 270
    save_json(ROOT/'verification.json', dict(complete=True, task_metric_rows=27, checkpoint_rows=270,
        tasks=3, checkpoints_per_task=10, split='val30', trials_per_checkpoint=30,
        original_metric_and_scipy_spearman_scalar_mmrv_checks=True, source_sha256=source_hashes,
        outputs_sha256={p.name: sha256(p) for p in ROOT.iterdir() if p.is_file() and p.name != 'verification.json'},
        optimal_surval_posthoc=True, baseline_partial_coverage=['Loss', 'Policy OMN'],
        matching_policy_keys=True, new_inference=False))
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
