"""Reaggregate the verified epoch5 grid under every task-sharing constraint.

Run from /workspace/surval:
PYTHONPATH=src:scripts .venv-droid/bin/python outputs/droid_policy5_hp_sharing_20260915/compare.py
No inference, threshold construction, or source/default changes.
"""

import csv
import itertools
import json
from pathlib import Path

import numpy as np

from cache_droid_datasets import REPO, save_json
from compare_droid_chunk_lengths import markdown_table
from report_droid_hp import METRICS, write_csv
from report_droid_policy_reference import select_indices
from surval.droid_long_tuning import metric_arrays
from surval.droid_policy_reference import AXES, fixed_grid
from surval.droid_tuning import primary_metrics
from tune_droid_hp import sha256

NAMES = ('ta', 'k', 'quantile')
SHARING = [s for n in range(4) for s in itertools.combinations(NAMES, n)]


def profile(grid, metrics, shared, mask):
    """For each shared tuple, independently minimize the remaining task losses."""
    groups = {}
    for index in np.flatnonzero(mask):
        groups.setdefault(tuple(grid[index][name] for name in shared), []).append(index)
    choices = []
    for values, indices in groups.items():
        allowed = np.zeros(len(grid), dtype=bool)
        allowed[indices] = True
        if not all((allowed & m['eligible']).any() for m in metrics.values()):
            continue
        ids = select_indices(metrics, allowed, common=False)
        loss = float(np.mean([metrics[t]['balanced_loss'][i] for t, i in ids.items()]))
        choices.append((loss, values, tuple(ids.values()), ids))
    if not choices:
        raise ValueError('No eligible shared tuple')
    return sorted(choices, key=lambda row: row[:3])


def self_check():
    """Compare grouped optimization against exhaustive task assignments."""
    grid = [dict(zip(NAMES, values)) for values in itertools.product((1, 2), repeat=3)]
    rng = np.random.default_rng(17)
    metrics = {t: dict(balanced_loss=rng.random(8), eligible=np.ones(8, bool)) for t in ('a', 'b', 'c')}
    metrics['a']['eligible'][0] = False
    for shared in SHARING:
        got = profile(grid, metrics, shared, np.ones(8, bool))[0][0]
        exhaustive = []
        for ids in itertools.product(range(8), repeat=3):
            if not all(metrics[t]['eligible'][i] for t, i in zip(metrics, ids)):
                continue
            if all(len({grid[i][n] for i in ids}) == 1 for n in shared):
                exhaustive.append(np.mean([metrics[t]['balanced_loss'][i] for t, i in zip(metrics, ids)]))
        np.testing.assert_allclose(got, min(exhaustive), rtol=0, atol=1e-15)


def main():
    self_check()
    root = REPO/'outputs/droid_policy5_val30_20260915'
    output = Path(__file__).resolve().parent
    if (output/'verification.json').exists():
        raise FileExistsError('Preserve existing results; copy this script into a new output directory')
    sources = {}

    def remember(path, expected=None):
        path = Path(path).resolve()
        digest = sha256(path)
        if expected is not None:
            assert digest == expected, str(path)
        sources[str(path)] = digest
        return path

    prior = json.loads(remember(root/'verification.json').read_text())
    assert prior['complete'] and prior['canonical_metrics_and_selection_unchanged']
    assert prior['fixed']['reference_epoch'] == 5 and prior['axes'] == AXES
    for name, digest in prior['outputs_sha256'].items():
        remember(root/name, digest)
    spec = json.loads(remember(root/'run_manifest.json').read_text())
    assert spec['fixed'] == prior['fixed']
    labels_path = REPO/'outputs/droid_real_world_20260913/success_counts.json'
    labels = json.loads(remember(labels_path, prior['source_sha256'][str(labels_path)]).read_text())
    grid = fixed_grid()
    scores, metrics, actual = {}, {}, {}
    for job in spec['jobs']:
        task, directory = job['task'], Path(job['job_dir'])/'eval'
        proof = json.loads(remember(directory/'verification.json').read_text())
        assert proof['complete'] and proof['num_demos'] == 30
        assert proof['signature']['fixed'] == prior['fixed'] and proof['signature']['axes'] == AXES
        for name, digest in proof['signature']['code_sha256'].items():
            remember(REPO/name, digest)
        path = remember(directory/'scores.npz', proof['outputs_sha256']['scores.npz'])
        with np.load(path) as data:
            np.testing.assert_array_equal(data['epochs'], labels['epoch_order'])
            scores[task] = data['scores']
        assert scores[task].shape == (2880, 10)
        actual[task] = np.asarray(labels['successes'][task])/labels['trials_per_checkpoint']
        metrics[task] = metric_arrays(actual[task], scores[task])
    assert set(metrics) == {'apple', 'pan', 'pet'}
    checked = set()
    with (root/'fixed_grid_metrics.csv').open() as stream:
        for row in csv.DictReader(stream):
            task, index = row['task'], int(row['hp_index'])
            assert (task, index) not in checked
            checked.add((task, index))
            for key, value in grid[index].items():
                assert float(row[key]) == value
            scalar = primary_metrics(actual[task], scores[task][index])
            for key, value in scalar.items():
                np.testing.assert_allclose(value, metrics[task][key][index], rtol=0, atol=1e-12, equal_nan=True)
                saved = row[key] == 'True' if key == 'eligible' else float(row[key])
                np.testing.assert_allclose(value, saved, rtol=0, atol=1e-12, equal_nan=True)
    assert len(checked) == 8640
    print('Verified all 8640 grid metrics against scalar definitions and previous report.', flush=True)

    aggregate, selected, checkpoints, profiles = [], [], [], []

    def collect(scope, shared, choice, per_value=False):
        loss, values, _, ids = choice
        free = tuple(n for n in NAMES if n not in shared)
        group = dict(scope=scope, shared='+'.join(shared) or 'none',
                     task_specific='+'.join(free) or 'none', task_specific_types=len(free),
                     fitted_values=len(shared)+3*len(free),
                     common_values=json.dumps(dict(zip(shared, values))))
        rows = []
        for task, index in ids.items():
            hp, s, a = grid[index], scores[task][index], actual[task]
            ref = primary_metrics(a, s)
            ties = np.flatnonzero(s == s.max())
            tie_regrets = (a.max()-a[ties])/(np.ptp(a)+1e-12)
            rows.append(dict(**group, task=task, hp_index=index, **hp, **ref,
                selected_epoch=labels['epoch_order'][ref['selected_index']],
                max_score_epochs=json.dumps([labels['epoch_order'][i] for i in ties]),
                max_score_count=len(ties), nregret_tie_uniform=float(tie_regrets.mean()),
                nregret_tie_worst=float(tie_regrets.max())))
        result = dict(**group, **{key: float(np.mean([r[key] for r in rows])) for key in METRICS},
            nregret_tie_uniform=float(np.mean([r['nregret_tie_uniform'] for r in rows])),
            nregret_tie_worst=float(np.mean([r['nregret_tie_worst'] for r in rows])),
            tasks_with_max_ties='+'.join(r['task'] for r in rows if r['max_score_count'] > 1) or 'none')
        assert abs(result['balanced_loss']-loss) < 1e-12
        selected.extend(rows)
        for row in rows:
            for col, epoch in enumerate(labels['epoch_order']):
                checkpoints.append(dict(scope=scope, shared=group['shared'], task=row['task'],
                    hp_index=row['hp_index'], epoch=epoch,
                    surval_score=float(scores[row['task']][row['hp_index'], col]),
                    success_rate=float(actual[row['task']][col])))
        (profiles if per_value else aggregate).append(result)
        return result

    ta = np.asarray([hp['ta'] for hp in grid])
    for scope, mask in [('all', ta > 0), ('requested', np.isin(ta, [2, 4, 8, 10, 12, 14, 15]))]:
        optima = {}
        for shared in SHARING:
            choices = profile(grid, metrics, shared, mask)
            optima[shared] = choices[0][0]
            collect(scope, shared, choices[0])
            # One shared axis at a time: every fixed Ta, k, or q and the best remaining HPs.
            if scope == 'all' and len(shared) == 1:
                for choice in sorted(choices, key=lambda r: r[1]):
                    collect(f'fixed_{shared[0]}={choice[1][0]}', shared, choice, per_value=True)
        for smaller in SHARING:
            for larger in SHARING:
                if set(smaller) <= set(larger):
                    assert optima[smaller] <= optima[larger]+1e-12
    with (root/'aggregate_metrics.csv').open() as stream:
        old = list(csv.DictReader(stream))
    for row in aggregate:
        if row['shared'] not in ('none', '+'.join(NAMES)):
            continue
        selection = 'task_specific' if row['shared'] == 'none' else 'common'
        expected = next(r for r in old if r['scope'] == row['scope'] and r['selection'] == selection
                        and r['regime'] == 'Policy5_retuned')
        for key in METRICS:
            np.testing.assert_allclose(row[key], float(expected[key]), rtol=0, atol=1e-12)

    write_csv(output/'aggregate_metrics.csv', aggregate)
    write_csv(output/'fixed_value_profiles.csv', profiles)
    write_csv(output/'selected_hp_metrics.csv', selected)
    write_csv(output/'checkpoint_scores.csv', checkpoints)
    lines = ['# Epoch5: 태스크 간 HP 공유 비교', '',
        '- apple/pan/pet, val30, 각 task epoch5 EMA policy encoder 고정. 기존 점수 grid 재집계; 신규 추론 없음.',
        '- c=1, LSE tau=1, worst=ceil(Ta/2), prediction samples=8, 기타 조건은 원본과 동일.',
        '- shared는 태스크 공통값, task_specific은 태스크별 선택. 공통값도 실측 성공률로 선택한 값이며 사전 고정값이 아니다.',
        '- 선택 기준: task별 (NRegret + MMRV / success-rate range + (1-Spearman)/2)/3의 macro 평균 최소화.',
        '- fitted_values는 공통값 1개/태스크별값 3개를 센 수이며, HP 종류는 여전히 Ta/k/q 세 가지다.',
        '- NRegret/MMRV는 낮을수록, Spearman은 높을수록 좋음. 세 task 동일 가중 평균.', '',
        '## 전체 Ta=1..15', '']
    columns = ('shared', 'task_specific', 'common_values', 'nregret', 'mmrv', 'spearman', 'balanced_loss')
    lines += markdown_table([r for r in aggregate if r['scope'] == 'all'], columns)
    lines += ['', '## 요청했던 Ta={2,4,8,10,12,14,15} 범위', '']
    lines += markdown_table([r for r in aggregate if r['scope'] == 'requested'], columns)
    lines += ['', '## 태스크별 설정 (메트릭은 위 macro만 표시)', '']
    lines += markdown_table([r for r in selected if r['scope'] == 'all'],
                            ('shared', 'task', 'ta', 'k', 'quantile', 'selected_epoch', 'max_score_epochs'))
    lines += ['', '## 해석 및 검증', '',
        '- 최소 개별 HP 한 종류 중 기존 balanced 기준 최적: Ta=3, q=.99 공통 + k만 개별. NRegret는 0이 아님.',
        '- NRegret=0 및 최고점 동률 없음이 우선이면 k=300, q=.35 공통 + Ta만 개별(apple/pan=6, pet=13).',
        '- 기본 checkpoint 선택은 최고 score의 첫 epoch. pan 동률이 있는 행의 낮은 regret에는 이 규칙의 영향이 있다.',
        '- aggregate CSV에 동률 균등 선택 기대 NRegret와 최악 NRegret를 별도 기록했다. 본 메트릭 정의는 변경하지 않았다.',
        '- 같은 30-checkpoint 성공률로 선택하고 평가한 post-hoc 결과. 독립 일반화 성능이나 확정 논문 test 결과가 아니다.',
        '- 원본 점수/보고서 해시 확인, 8640 grid 메트릭의 scalar 재계산 일치, 공유 최적화 exhaustive self-check,',
        '  제약 추가 시 objective 비개선 확인, 기존 공통/완전개별 optimum 재현. 신규 canonical score 계산은 하지 않았다.',
        '- 원본 보고서에는 400개 canonical score 검증이 기록되어 있다. 이번 분석은 저장된 전체 grid를 재집계했다.',
        '- fixed_value_profiles.csv에 각 Ta/k/q 공통값별로 나머지 두 HP를 태스크별 최적화한 전체 profile을 저장.', '',
        '재현: 이 compare.py를 새 결과 디렉터리에 복사하고, 저장소에서 PYTHONPATH=src:scripts .venv-droid/bin/python <compare.py> 실행.', '']
    (output/'REPORT.ko.md').write_text('\n'.join(lines))
    for path, digest in sources.items():
        assert sha256(path) == digest, f'Source changed during reporting: {path}'
    outputs = {p.name: sha256(p) for p in output.iterdir() if p.is_file()}
    save_json(output/'verification.json', dict(complete=True, fixed=prior['fixed'], axes=AXES,
        all_grid_scalar_metrics_verified=len(checked), source_sha256=sources, outputs_sha256=outputs,
        self_check='exhaustive tiny-grid sharing optimization', restriction_monotonicity=True,
        existing_endpoint_optima_reproduced=True, canonical_scores_recomputed=0,
        source_canonical_scores=prior['canonical_scores'], aggregate_rows=len(aggregate),
        fixed_value_profile_rows=len(profiles), posthoc=True))
    print('\n'.join(markdown_table([r for r in aggregate if r['scope'] == 'all'], columns)), flush=True)
    print(f'Saved {output}', flush=True)


if __name__ == '__main__':
    main()
