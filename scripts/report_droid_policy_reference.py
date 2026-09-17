"""Compare DINO controls, frozen-policy transfer, and Ta/k/q retuning."""

from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path

import numpy as np

from cache_droid_datasets import REPO, save_json
from compare_droid_chunk_lengths import best_index, markdown_table
from report_droid_hp import METRICS, write_csv
from tune_droid_hp import sha256
from surval.droid import load_policy_cache, score_policy_cache
from surval.droid_dino import (_content_hash, align_dino_cache, dataset_db_path, load_shared_dino_db)
from surval.droid_long_tuning import hp_grid, metric_arrays
from surval.droid_policy_reference import (AXES, POLICY_OMN, align_policy_reference,
                                            build_policy_reference, fixed_grid)
from surval.droid_tuning import canonical_thresholds, primary_metrics
from surval.local_threshold.database import StateDatabase
from surval.local_threshold.threshold import LocalThresholdMap


def select_indices(metrics, mask, common):
    tasks = list(metrics)
    if common:
        loss = np.mean([metrics[t]['balanced_loss'] for t in tasks], axis=0)
        eligible = np.logical_and.reduce([metrics[t]['eligible'] for t in tasks])
        index = best_index(mask, eligible, loss, loss)
        return {t: index for t in tasks}
    return {t: best_index(mask, metrics[t]['eligible'], metrics[t]['balanced_loss'],
                          metrics[t]['balanced_loss']) for t in tasks}


def report(args, spec):
    if args.reference_epoch not in (5, 25, 50):
        raise ValueError('This comparison report supports reference epoch5, epoch25 or epoch50')
    policy_label = f'Policy{args.reference_epoch}'
    prior25_root = REPO/'outputs/droid_policy25_val30_20260915'
    output = args.output_root
    for epoch in (25, 50):
        archived = REPO/f'outputs/droid_policy{epoch}_val30_20260915'
        if args.reference_epoch != epoch and output.resolve().is_relative_to(archived.resolve()):
            raise ValueError(f'Use a separate output root; preserve the epoch{epoch} comparison')
    grid = fixed_grid()
    old_grid = hp_grid()
    old_ids = [i for i, h in enumerate(old_grid)
               if h['chunk_top_k'] == (h['ta']+1)//2 and h['lse_tau'] == 1.]
    assert [{**old_grid[i], 'chunk_top_frac': .5} for i in old_ids] == grid
    labels_path = REPO/'outputs/droid_real_world_20260913/success_counts.json'
    labels = json.loads(labels_path.read_text())
    previous = REPO/'outputs/droid_fixed_half_tau1_20260915'
    previous_proof = json.loads((previous/'verification.json').read_text())
    assert previous_proof['complete']
    sources = {str(labels_path): sha256(labels_path)}

    def remember(path, expected=None):
        actual = sha256(path)
        if expected is not None:
            assert actual == expected, str(path)
        sources[str(path)] = actual

    for path, digest in previous_proof['source_sha256'].items():
        remember(path, digest)
    for path in previous.iterdir():
        if path.is_file():
            remember(path)
    prior25_baselines = []
    if args.reference_epoch != 25:
        prior25_proof = json.loads((prior25_root/'verification.json').read_text())
        assert prior25_proof['complete'] and prior25_proof['canonical_metrics_and_selection_unchanged']
        for name, digest in prior25_proof['outputs_sha256'].items():
            remember(prior25_root/name, digest)
        remember(prior25_root/'verification.json')
    scores, metrics, actual, proofs, baseline_checkpoints = {}, {}, {}, {}, []
    for rep in dict.fromkeys(('DINO', policy_label, 'Policy25')):
        scores[rep], metrics[rep] = {}, {}
    for job in spec['jobs']:
        task, directory = job['task'], Path(job['job_dir'])/'eval'
        proof = proofs[task] = json.loads((directory/'verification.json').read_text())
        assert proof['complete'] and proof['signature']['fixed'] == spec['fixed']
        assert proof['signature']['axes'] == AXES and proof['num_demos'] == 30
        for path, digest in proof['signature']['code_sha256'].items():
            remember(REPO/path, digest)
        for path, digest in proof['signature']['cache_sha256'].items():
            remember(path, digest)
        remember(args.baseline_csv, proof['signature']['baseline_csv_sha256'])
        for name, digest in proof['outputs_sha256'].items():
            remember(directory/name, digest)
        old_directory = Path(job['dino_eval_dir'])
        old_proof = json.loads((old_directory/'verification.json').read_text())
        assert old_proof['complete']
        for name in ('scores.npz', 'scales.npz'):
            remember(old_directory/name, old_proof[name+'_sha256'])
        actual[task] = np.array(labels['successes'][task])/labels['trials_per_checkpoint']
        loaders = [(policy_label, directory/'scores.npz', slice(None)),
                   ('DINO', old_directory/'scores.npz', old_ids)]
        if args.reference_epoch != 25:
            prior_dir = prior25_root/'conditions'/job['name']/'seed_0/eval'
            prior_proof = json.loads((prior_dir/'verification.json').read_text())
            assert prior_proof['complete']
            assert prior_proof['signature']['fixed'] == {**spec['fixed'], 'reference_epoch': 25}
            for name in ('axes', 'cache_sha256', 'code_sha256', 'baseline_csv_sha256', 'dino_db_content_sha256'):
                assert prior_proof['signature'][name] == proof['signature'][name], name
            for name, digest in prior_proof['outputs_sha256'].items():
                remember(prior_dir/name, digest)
            remember(prior_dir/'verification.json')
            loaders.append(('Policy25', prior_dir/'scores.npz', slice(None)))
            prior25_baselines.extend(json.loads((prior_dir/'baselines.json').read_text()))
        for rep, path, ids in loaders:
            with np.load(path) as saved:
                np.testing.assert_array_equal(saved['epochs'], labels['epoch_order'])
                scores[rep][task] = saved['scores'][ids]
            assert scores[rep][task].shape == (2880, 10)
            metrics[rep][task] = metric_arrays(actual[task], scores[rep][task])
        baseline_checkpoints.extend(json.loads((directory/'baselines.json').read_text()))
        print(f'INPUT VERIFIED {task}', flush=True)

    # Independent scalar definitions check every published grid metric, including ties.
    grid_rows = []
    for task, values in scores[policy_label].items():
        for i, row in enumerate(values):
            ref = primary_metrics(actual[task], row)
            for key, value in ref.items():
                np.testing.assert_allclose(value, metrics[policy_label][task][key][i],
                                           rtol=0, atol=1e-12, equal_nan=True)
            grid_rows.append(dict(task=task, hp_index=i, **grid[i], **ref))
        print(f'SCALAR VERIFIED {task}: {len(values)} HPs', flush=True)
    write_csv(output/'fixed_grid_metrics.csv', grid_rows)
    selected, aggregate, checkpoints = [], [], []
    ta = np.array([h['ta'] for h in grid])
    for scope, mask in [('all', ta > 0), ('requested', np.isin(ta, [2, 4, 8, 10, 12, 14, 15]))]:
        for selection in ('common', 'task_specific'):
            dino_ids = select_indices(metrics['DINO'], mask, selection == 'common')
            policy_ids = select_indices(metrics[policy_label], mask, selection == 'common')
            choices = [('DINO_optimal', 'DINO', dino_ids),
                       (f'{policy_label}_same_HP', policy_label, dino_ids),
                       (f'{policy_label}_retuned', policy_label, policy_ids)]
            if args.reference_epoch != 25:
                prior_ids = select_indices(metrics['Policy25'], mask, selection == 'common')
                choices += [('Policy25_optimal', 'Policy25', prior_ids),
                            (f'{policy_label}_same_Policy25_HP', policy_label, prior_ids)]
            for regime, rep, ids in choices:
                group = []
                for task, index in ids.items():
                    ref = primary_metrics(actual[task], scores[rep][task][index])
                    row = dict(scope=scope, regime=regime, selection=selection,
                               representation=rep, task=task, hp_index=index, **grid[index], **ref,
                               selected_epoch=labels['epoch_order'][ref['selected_index']])
                    selected.append(row)
                    group.append(row)
                    for col, epoch in enumerate(labels['epoch_order']):
                        checkpoints.append({k: row[k] for k in ('scope', 'regime', 'selection', 'task', 'hp_index')}
                            | dict(epoch=epoch, surval_score=float(scores[rep][task][index, col]),
                                   successes=labels['successes'][task][col],
                                   trials=labels['trials_per_checkpoint'], success_rate=float(actual[task][col])))
                aggregate.append(dict(scope=scope, regime=regime, selection=selection,
                    **{key: float(np.mean([r[key] for r in group])) for key in METRICS}))
    # Reproduce the already published DINO comparison exactly before claiming a delta.
    with (previous/'aggregate_metrics.csv').open() as stream:
        previous_rows = list(csv.DictReader(stream))
    for row in aggregate:
        if row['regime'] == 'DINO_optimal':
            old = next(r for r in previous_rows if r['regime'] == 'fixed_half_tau1'
                       and r['scope'] == row['scope'] and r['selection'] == row['selection'])
            for key in METRICS:
                np.testing.assert_allclose(row[key], float(old[key]), rtol=0, atol=1e-12)
    if args.reference_epoch != 25:
        with (prior25_root/'aggregate_metrics.csv').open() as stream:
            prior_rows = list(csv.DictReader(stream))
        for row in aggregate:
            if row['regime'] == 'Policy25_optimal':
                old = next(r for r in prior_rows if r['regime'] == 'Policy25_retuned'
                           and r['scope'] == row['scope'] and r['selection'] == row['selection'])
                for key in METRICS:
                    np.testing.assert_allclose(row[key], float(old[key]), rtol=0, atol=1e-12)
        old_baselines = {(r['task'], r['epoch']): r for r in prior25_baselines}
        for row in baseline_checkpoints:
            old = old_baselines[row['task'], row['epoch']]
            for key in ('run', 'run_timestamp', 'dataset', 'checkpoint', 'baseline_horizon', 'num_rows',
                        'MSE_mean_pred', 'Off_Manifold_Norm_DINO', 'Loss', 'Off_Manifold_Norm'):
                assert row[key] == old[key], (row['task'], row['epoch'], key)
    write_csv(output/'aggregate_metrics.csv', aggregate)
    write_csv(output/'selected_metrics.csv', selected)
    hp_fields = ('scope', 'regime', 'selection', 'task', 'hp_index', *grid[0])
    write_csv(output/'selected_hp.csv', [{k: r[k] for k in hp_fields} for r in selected])
    write_csv(output/'checkpoint_scores.csv', checkpoints)
    write_csv(output/'baseline_checkpoint_metrics.csv', baseline_checkpoints)
    baseline_rows, baseline_macro = [], []
    names = [('MSE_mean_pred', 'matched_H8_full_val30'),
             ('Off_Manifold_Norm_DINO', 'matched_H8_full_val30'),
             (POLICY_OMN, 'matched_H8_full_val30'),
             ('Loss', 'legacy_first1600'), ('Off_Manifold_Norm', 'legacy_first1600_batch_neighbors')]
    if args.reference_epoch != 25:
        names.append((POLICY_OMN+'_Epoch25', 'matched_H8_full_val30'))
    for name, coverage in names:
        group = []
        data = prior25_baselines if name.endswith('_Epoch25') else baseline_checkpoints
        source_name = POLICY_OMN if name.endswith('_Epoch25') else name
        display_name = f'{name}_Epoch{args.reference_epoch}' if name == POLICY_OMN and args.reference_epoch != 25 else name
        for task in actual:
            rows = sorted([r for r in data if r['task'] == task], key=lambda r: r['epoch'])
            assert [r['epoch'] for r in rows] == labels['epoch_order']
            ref = primary_metrics(actual[task], -np.array([r[source_name] for r in rows]))
            row = dict(metric=display_name, coverage=coverage, task=task, **ref,
                       selected_epoch=labels['epoch_order'][ref['selected_index']])
            group.append(row)
            baseline_rows.append(row)
        baseline_macro.append(dict(metric=display_name, coverage=coverage,
                                   **{key: float(np.mean([r[key] for r in group])) for key in METRICS}))
    write_csv(output/'baseline_task_metrics.csv', baseline_rows)
    write_csv(output/'baseline_aggregate_metrics.csv', baseline_macro)

    canonical, maximum, threshold_error = [], 0., 0.
    for job in spec['jobs']:
        task, directory = job['task'], Path(job['job_dir'])/'eval'
        reference = load_policy_cache(job['reference_cache'])
        policy, chunks, meta = build_policy_reference(reference, args.reference_epoch)
        persisted = StateDatabase.load(str(directory/'reference_db'), policy.cfg)
        persisted_chunks = np.load(directory/'reference_db/expert_action_chunks.npy')
        assert _content_hash(persisted, persisted_chunks) == _content_hash(policy, chunks)
        assert _content_hash(policy, chunks) == proofs[task]['signature']['policy_db_content_sha256']
        for path in (directory/'reference_db').iterdir():
            if path.is_file():
                remember(path)
        dino, dino_chunks, dino_meta = load_shared_dino_db(dataset_db_path(args.source_root/'shared_dino', job['dataset']))
        assert dino_meta['content_sha256'] == proofs[task]['signature']['dino_db_content_sha256']
        for rep, db, expert, metadata, scale_path in [
            (policy_label, policy, chunks, meta, directory/'scales.npz'),
            ('DINO', dino, dino_chunks, dino_meta, Path(job['dino_eval_dir'])/'scales.npz')]:
            ids = sorted({r['hp_index'] for r in selected if r['task'] == task and r['representation'] == rep})
            paths = {}
            with np.load(scale_path) as tables:
                for k in sorted({grid[i]['k'] for i in ids}):
                    target = output/'audit_thresholds'/rep/task/str(k)
                    candidates = [target]
                    if rep == 'DINO':
                        candidates += [Path(job['dino_eval_dir'])/'audit_thresholds'/str(k),
                            previous/'audit_thresholds'/task/str(k),
                            REPO/'outputs/droid_ta_comparison_20260914/audit_thresholds'/task/str(k)]
                    available = next((p for p in candidates if (p/'thresholds').is_dir()), None)
                    if available is not None:
                        target = available
                        threshold = LocalThresholdMap.load(str(target/'thresholds'))
                    else:
                        threshold = canonical_thresholds(db, AXES['quantile'], k, target)
                    np.testing.assert_array_equal(np.load(target/'demo_id_int.npy'), db.records.demo_id_int)
                    np.testing.assert_array_equal(np.load(target/'t.npy'), db.records.t)
                    mapping = json.loads((target/'demo_id_str.json').read_text())
                    assert {int(k): v for k, v in mapping.items()} == db.records.demo_id_str_by_int
                    assert threshold.quantiles == tuple(AXES['quantile'])
                    assert not threshold.fallback_used.any() and (threshold.n_neighbors_used == k).all()
                    np.testing.assert_allclose(threshold.thresholds, tables[str(k)], atol=1e-7, rtol=1e-6)
                    threshold_error = max(threshold_error, float(abs(threshold.thresholds-tables[str(k)]).max()))
                    paths[k] = target
                    print(f'THRESHOLD AUDIT {rep} {task} k={k}', flush=True)
            original = np.empty((len(ids), 10))
            for col, row in enumerate(job['rows']):
                cache = load_policy_cache(row['cache_file'])
                align = align_policy_reference if rep == policy_label else align_dino_cache
                align(db, expert, metadata, cache)
                for j, index in enumerate(ids):
                    hp = grid[index]
                    with redirect_stdout(io.StringIO()):
                        summary, episodes = score_policy_cache(cache, str(paths[hp['k']]), ta=hp['ta'],
                            quantile=hp['quantile'], chunk_top_frac=.5, lse_tau=1., num_samples=8, every_step=True)
                    assert sum(e['T'] for e in episodes) == db.n_states
                    ref = original[j, col] = summary['PrefixSurvival_Score']
                    error = float(abs(ref-scores[rep][task][index, col]))
                    maximum = max(maximum, error)
                    canonical.append(dict(representation=rep, task=task, hp_index=index, epoch=row['epoch'],
                                          canonical_score=ref, grid_score=scores[rep][task][index, col], abs_error=error))
                print(f'SCORE AUDIT {rep} {task} epoch={row["epoch"]} HPs={len(ids)} max_error={maximum:.3g}', flush=True)
            np.testing.assert_allclose(original, scores[rep][task][ids], atol=2e-7, rtol=2e-6)
            for j, index in enumerate(ids):
                for key, value in primary_metrics(actual[task], original[j]).items():
                    np.testing.assert_allclose(value, metrics[rep][task][key][index], atol=1e-12, rtol=0)
    write_csv(output/'canonical_audit.csv', canonical)
    for path, digest in sources.items():
        assert sha256(path) == digest, path
    lines = [f'# Frozen epoch{args.reference_epoch} policy feature: val30 재튜닝', '',
        f'apple/pan/pet 각각 epoch{args.reference_epoch} EMA policy obs_encoder의 1024D conditioning을 고정한다.',
        '두 카메라와 proprio를 fusion한 512D feature × observation history 2개이며, image-only DINO와 입력 구성이 다르다.',
        f'reference와 query 모두 epoch{args.reference_epoch} feature를 사용하고, target checkpoint는 예측 action만 제공한다.',
        'full-val30 reference, L2-normalized feature의 exact kNN, self 및 동일 demo ±5 frame 제외. k명 모두 확보, fallback 0.',
        'worst=ceil(Ta/2), LSE=1, c=1, samples=8, every_step=True 고정. Ta/k/q만 탐색.',
        f'탐색 범위: {json.dumps(AXES)}. 총 2,880 HP × 3태스크 × 10체크포인트.',
        'all=Ta1..15; requested=Ta2/4/8/10/12/14/15 (현재 캐시는 최대15).',
        '그룹별 scale과 local threshold는 같은 expert-pair quantile이며 pos L2 / rot SO(3)에 각각 적용, gripper 제외.',
        '최적화 기준: 평균 [(NRegret + MMRV/SR범위 + (1−Spearman)/2)/3]. 동률은 기존 HP 순서.',
        'NRegret=(best SR−selected SR)/(best SR−worst SR); regret_pp는 성공률 손실 %p. MMRV는 SR 0–1 단위.',
        'checkpoint score 동률은 epoch 오름차순 첫 최대값(np.argmax)을 선택하며, Spearman은 평균 rank를 사용한다.',
        '아래는 태스크별 지표를 동일 가중치로 평균한 macro 결과이며 30 checkpoint를 pooled correlation한 것이 아니다.', '',
        '## SurVAL aggregate', '']
    if args.reference_epoch != 25:
        lines += [f'{policy_label}_same_HP는 DINO 최적 HP를 유지한 결과; {policy_label}_same_Policy25_HP는 epoch25 최적 HP를 유지한 결과다.',
                  f'Policy25_optimal은 보존된 epoch25 결과를 그대로 대조했으며, {policy_label}_retuned만 epoch{args.reference_epoch} 공간에서 재튜닝한다.', '']
    lines += markdown_table(aggregate, ['scope', 'regime', 'selection', 'nregret', 'mmrv', 'spearman', 'regret_pp'])
    peak_ties = {}
    for task in actual:
        choice = next(r for r in selected if r['scope'] == 'all' and r['selection'] == 'task_specific'
                      and r['regime'] == f'{policy_label}_retuned' and r['task'] == task)
        values = scores[policy_label][task][choice['hp_index']]
        winners = [labels['epoch_order'][int(i)] for i in np.flatnonzero(values == values.max())]
        if len(winners) > 1:
            peak_ties[task] = winners
    if peak_ties:
        lines += ['', f'전체 Ta·태스크별 재튜닝의 최고점 동률 epoch: {json.dumps(peak_ties, sort_keys=True)}.',
                  '이 경우 NRegret은 동률 checkpoint 중 가장 이른 epoch를 선택한 값이다.']
    lines += ['', '## Baseline aggregate', '',
        '모든 baseline은 낮을수록 좋으므로 음수화한 뒤 지표 계산. SurVAL은 원 score 그대로 사용.',
        'matched 3종은 동일 H15 예측 캐시의 H8 prefix/S8/전체 val30을 사용한다. 기존 H8 캐시와 stochastic sample이 달라 이전 수치와 동일하지 않을 수 있다.',
        f'DINO OMN과 {policy_label} OMN은 동일 projection 코드와 k5/full-val 기준이며 이웃 표현만 다르다.',
        'Loss 및 기존 policy-batch OMN은 기존 first1600 row 결과를 그대로 보존한 legacy 비교다.', '']
    lines += markdown_table(baseline_macro, ['metric', 'coverage', 'nregret', 'mmrv', 'spearman', 'regret_pp'])
    lines += ['', '## 검증 및 파일', '',
        f'8640개 {policy_label} HP 지표를 원 scalar 정의와 대조. 신규 선택 및 DINO control {len(canonical)}점을 원 scorer로 재계산: 최대오차 {maximum:.3g}.',
        '원 threshold와 최적 HP의 전체 행 scale 일치, RLDS 행/GT/캐시 SHA256 및 이전 결과 보존 확인.',
        'selected_hp.csv에 공통·태스크별 HP, checkpoint_scores.csv에 실제 성공 횟수와 score, reference_db에 고정 표현 저장.',
        '실제 성공률을 사용한 사후 튜닝이며 독립적인 일반화 성능이 아니다. reference epoch 비교 역시 사후 분석으로 명시해야 한다.',
        'epoch25 기존 최적값은 이전 canonical audit와 저장 파일 SHA256 및 이번 scalar 지표 재현으로 확인한다.',
        '새 checkpoint action 캐시는 같은 run/split/행/GT 계약에 맞으면 고정 reference DB와 threshold를 재사용할 수 있다.', '']
    (output/'REPORT.ko.md').write_text('\n'.join(lines))
    save_json(output/'verification.json', dict(complete=True, fixed=spec['fixed'], axes=AXES,
        policy_grid_metric_rows=len(grid_rows), aggregate_rows=len(aggregate), baseline_aggregate_rows=len(baseline_macro),
        canonical_scores=len(canonical), canonical_max_abs_error=maximum, threshold_max_abs_error=threshold_error,
        canonical_metrics_and_selection_unchanged=True, source_sha256=sources, retuned_max_score_ties=peak_ties,
        script_sha256=sha256(__file__), original_outputs_unchanged=True, posthoc=True,
        outputs_sha256={p.name: sha256(p) for p in output.iterdir() if p.suffix == '.csv' or p.name == 'REPORT.ko.md'}))
    print(f'PASS report: {len(canonical)} canonical scores; {output}', flush=True)
