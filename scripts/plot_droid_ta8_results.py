"""Plot saved frozen-policy / val30 / fixed-Ta results with NRegret; no tuning."""

import argparse
import csv
import json
from pathlib import Path
import re
import sys
import zipfile

import numpy as np

REPO = Path(__file__).resolve().parents[1]
STYLE = Path('/workspace/surval_openpi/scripts/plot_pi05_ta12_results.py')
sys.path.insert(0, str(STYLE.parent))
import plot_pi05_ta12_results as style
from cache_droid_datasets import save_json
from report_droid_hp import METRICS, write_csv
from surval.droid_policy_reference import fixed_grid
from surval.droid_tuning import primary_metrics
from tune_droid_hp import sha256

TASKS = ('apple', 'pan', 'pet')
NAMES = dict(zip(TASKS, ('PnP Apple', 'Put Pan on Stove', 'Put Pet on Shelf')))
COLORS = dict(zip(TASKS, (style.BLUE, style.RED, style.GREEN)))
MARKERS = dict(zip(TASKS, ('s', '^', 'D')))
METHODS = ('Loss', 'Off_Manifold_Norm_PolicyReference', 'MSE_mean_pred', 'SurVAL')
FILENAMES = dict(zip(METHODS, ('val_loss', 'omn', 'val_mse', 'ours')))
LABELS = dict(zip(METHODS, ('−Validation Loss (z-score)', '−Off-Manifold Norm (z-score)',
                          '−Validation MSE (z-score)', 'RACE (z-score)')))
EPOCHS = np.arange(5, 51, 5)
COVERAGE = dict(zip(METHODS, ('legacy_first1600', 'matched_H8_full_val30',
                            'matched_H8_full_val30', 'full_val30_selected_Ta')))


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def load_data(ta=8):
    if ta not in range(1, 16):
        raise ValueError('Saved chunk lengths are 1..15')
    scope = f'fixed_ta={ta}'
    root = REPO/'outputs/droid_policy5_val30_20260915'
    selected_root = REPO/'outputs/droid_policy5_hp_sharing_20260915'
    sources = {}

    def remember(path, expected=None):
        path = Path(path).resolve()
        digest = sha256(path)
        assert expected is None or digest == expected, str(path)
        sources[str(path)] = digest
        return path

    proof = json.loads(remember(root/'verification.json').read_text())
    selected_proof = json.loads(remember(selected_root/'verification.json').read_text())
    assert proof['complete'] and selected_proof['complete']
    assert proof['fixed'] == selected_proof['fixed'] and proof['fixed']['reference_epoch'] == 5
    for directory, metadata, names in [
        (root, proof, ('baseline_checkpoint_metrics.csv', 'baseline_aggregate_metrics.csv', 'baseline_task_metrics.csv')),
        (selected_root, selected_proof, ('selected_hp_metrics.csv', 'checkpoint_scores.csv', 'fixed_value_profiles.csv'))]:
        for name in names:
            remember(directory/name, metadata['outputs_sha256'][name])
    labels_path = REPO/'outputs/droid_real_world_20260913/success_counts.json'
    labels = json.loads(remember(labels_path, proof['source_sha256'][str(labels_path)]).read_text())
    np.testing.assert_array_equal(labels['epoch_order'], EPOCHS)
    assert labels['trials_per_checkpoint'] == 30
    run_path = root/'run_manifest.json'
    run = json.loads(remember(run_path, selected_proof['source_sha256'][str(run_path)]).read_text())
    choices = [r for r in read_csv(selected_root/'selected_hp_metrics.csv') if r['scope'] == scope]
    assert len(choices) == 3 and {r['task'] for r in choices} == set(TASKS)
    score_rows = [r for r in read_csv(selected_root/'checkpoint_scores.csv') if r['scope'] == scope]
    baseline = read_csv(root/'baseline_checkpoint_metrics.csv')
    old_metrics = read_csv(root/'baseline_task_metrics.csv')
    grid = fixed_grid()
    data, metrics, hps, datasets = {}, [], [], []
    for task in TASKS:
        hp = next(r for r in choices if r['task'] == task)
        index = int(hp['hp_index'])
        assert hp['shared'] == 'ta' and hp['task_specific'] == 'k+quantile'
        assert all(float(hp[k]) == v for k, v in grid[index].items())
        assert grid[index]['ta'] == ta and grid[index]['chunk_top_k'] == (ta+1)//2
        hps.append(dict(task=task, hp_index=index, **grid[index]))
        job = next(j for j in run['jobs'] if j['task'] == task)
        directory = Path(job['job_dir'])/'eval'
        worker = json.loads(remember(directory/'verification.json').read_text())
        assert worker['complete'] and worker['signature']['fixed'] == proof['fixed']
        path = remember(directory/'scores.npz', worker['outputs_sha256']['scores.npz'])
        with np.load(path) as saved:
            np.testing.assert_array_equal(saved['epochs'], EPOCHS)
            ours = saved['scores'][index].copy()
        points = sorted([r for r in score_rows if r['task'] == task], key=lambda r: int(r['epoch']))
        rows = sorted([r for r in baseline if r['task'] == task], key=lambda r: int(r['epoch']))
        assert [int(r['epoch']) for r in points] == [int(r['epoch']) for r in rows] == EPOCHS.tolist()
        np.testing.assert_array_equal(ours, [float(r['surval_score']) for r in points])
        gt = np.array(labels['successes'][task])/30
        np.testing.assert_array_equal(gt, [float(r['success_rate']) for r in points])
        assert all(int(r['hp_index']) == index for r in points)
        for row, cached in zip(rows, job['rows']):
            assert all(row[k] == str(cached[k]) for k in ('run', 'run_timestamp', 'dataset', 'epoch', 'checkpoint'))
            assert int(row['reference_epoch']) == 5 and int(row['baseline_horizon']) == 8
        config_path = Path(rows[0]['checkpoint']).parents[1]/'config.json'
        config = json.loads(remember(config_path).read_text())
        task_long = rows[0]['dataset'].split('/')[0]
        split_path = Path('/workspace/data')/task_long/'droid_split_manifest.json'
        split = json.loads(remember(split_path).read_text())['configs']['val30']
        assert not set(split['train']) & set(split['val']) and len(split['val']) == 30
        datasets.append(dict(task=task, dataset=rows[0]['dataset'], run=rows[0]['run'],
            run_timestamp=rows[0]['run_timestamp'], configured_datasets=config['train']['dataset_names'],
            task_train_demos=len(split['train']), val_demos=len(split['val']),
            num_rows=int(rows[0]['num_rows']), reference_cache=job['reference_cache']))
        data[task] = {'success_rate': gt, **{m: -np.array([float(r[m]) for r in rows]) for m in METHODS[:-1]},
                      'SurVAL': ours}
        for method in METHODS:
            x = data[task][method]
            assert x.shape == gt.shape == (10,) and np.isfinite(x).all() and np.ptp(x) > 0
            result = primary_metrics(gt, x)
            old_name = method+'_Epoch5' if method == 'Off_Manifold_Norm_PolicyReference' else method
            expected = hp if method == 'SurVAL' else next(r for r in old_metrics if r['task'] == task and r['metric'] == old_name)
            for key in METRICS:
                np.testing.assert_allclose(result[key], float(expected[key]), atol=1e-12, rtol=0)
            assert int(EPOCHS[result['selected_index']]) == int(expected['selected_epoch'])
            metrics.append(dict(task=task, method=method, coverage=COVERAGE[method], **result,
                selected_epoch=int(EPOCHS[result['selected_index']]),
                max_score_epochs=json.dumps(EPOCHS[x == x.max()].tolist())))
    aggregate = {m: {k: float(np.mean([r[k] for r in metrics if r['method'] == m])) for k in METRICS} for m in METHODS}
    target = next(r for r in read_csv(selected_root/'fixed_value_profiles.csv') if r['scope'] == scope)
    for key in METRICS:
        np.testing.assert_allclose(aggregate['SurVAL'][key], float(target[key]), atol=1e-12, rtol=0)
    for method in METHODS[:-1]:
        name = method+'_Epoch5' if method == 'Off_Manifold_Norm_PolicyReference' else method
        target = next(r for r in read_csv(root/'baseline_aggregate_metrics.csv') if r['metric'] == name)
        for key in METRICS:
            np.testing.assert_allclose(aggregate[method][key], float(target[key]), atol=1e-12, rtol=0)
    for path in (Path(__file__), STYLE, REPO/'src/surval/droid_tuning.py', REPO/'src/surval/tb_aggregate/metrics.py'):
        remember(path)
    return data, metrics, aggregate, hps, datasets, sources, labels


def save_comparison(fig, path):
    # Data-driven limits; only add headroom to panels whose annotations hide points.
    for _ in range(16):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        blocked_axes = []
        for ax in fig.axes:
            assert ax.xaxis.label.get_fontsize() == ax.yaxis.label.get_fontsize() == 15
            assert all(t.get_fontsize() == 15 for t in ax.artists[-1].get_texts())
            assert all(t.get_fontsize() == 12 for t in ax.get_xticklabels()+ax.get_yticklabels())
            assert ax.xaxis.label.get_fontweight() == 'bold'
            assert ax.xaxis.label.get_window_extent(renderer).width <= ax.get_window_extent(renderer).width, 'Axis label too wide'
            ours = ax.get_xlabel() == LABELS['SurVAL']
            assert (ax.get_facecolor()[:3] != (1., 1., 1.)) == ours, 'Only Ours should be highlighted'
            assert all(s.get_linewidth() == (1.6 if ours else .5) for s in ax.spines.values())
            show_ylabel = ax.get_xlabel() in (LABELS['Loss'], LABELS['Off_Manifold_Norm_PolicyReference'])
            assert ax.get_ylabel() == ('Success Rate (z-normalized)' if show_ylabel else '')
            assert (ax.get_legend() is not None) == (ax.get_xlabel() == LABELS['Loss'])
            boxes = [ax.artists[-1].get_window_extent(renderer)]
            if ax.get_legend() is not None:
                assert all(t.get_fontsize() == 15 for t in ax.get_legend().get_texts())
                assert [t.get_text() for t in ax.get_legend().get_texts()] == [NAMES[t] for t in TASKS]
                boxes.append(ax.get_legend().get_window_extent(renderer))
                assert not boxes[0].overlaps(boxes[1]), 'Statistics and legend overlap'
            padding = 7*fig.dpi/72
            points = np.concatenate([ax.transData.transform(c.get_offsets()) for c in ax.collections])
            assert all(ax.get_window_extent(renderer).padded(-padding).contains(x, y) for x, y in points), 'Marker clipped'
            if any(box.padded(padding).contains(x, y) for box in boxes for x, y in points):
                blocked_axes.append(ax)
        if not blocked_axes:
            break
        for ax in blocked_axes:
            bottom, top = ax.get_ylim()
            ax.set_ylim(bottom, top+.10*(top-bottom))
    else:
        raise ValueError('Could not clear annotations with bounded automatic axis adjustment')
    limits = [dict(label=ax.get_xlabel(), ylabel=ax.get_ylabel(), task_legend=ax.get_legend() is not None,
                   xlim=list(ax.get_xlim()), ylim=list(ax.get_ylim())) for ax in fig.axes]
    style.save(fig, path)
    return limits


def render(output, ta=8):
    data, metrics, aggregate, hps, datasets, sources, labels = load_data(ta)
    axis_limits = {}
    fonts = style.setup_style()
    output.mkdir(parents=True, exist_ok=False)
    normalized = {t: {} for t in TASKS}
    for task in TASKS:
        for name, values in data[task].items():
            assert values.std(ddof=1) > 0
            z = normalized[task][name] = (values-values.mean())/values.std(ddof=1)
            np.testing.assert_allclose([z.mean(), z.std(ddof=1)], [0, 1], atol=1e-12, rtol=0)
            assert abs(z).max() < 2.9
    epoch_norm = style.Normalize(EPOCHS.min(), EPOCHS.max())
    legend = [style.Line2D([0], [0], marker=MARKERS[t], color='w', markerfacecolor=COLORS[t],
        markeredgecolor=style.BLACK, markeredgewidth=.5, markersize=11, label=NAMES[t]) for t in TASKS]

    def axis(ax, method):
        xs, ys = [], []
        for task in TASKS:
            x, y = normalized[task][method], normalized[task]['success_rate']
            ax.scatter(x, y, s=125, c=style.epoch_cmap(COLORS[task])(epoch_norm(EPOCHS)),
                       marker=MARKERS[task], edgecolor=style.BLACK, linewidth=.5, alpha=.9)
            xs.extend(x); ys.extend(y)
        style.fit(ax, np.asarray(xs), np.asarray(ys))
        m = aggregate[method]
        style.stats_box(ax, [('Spearman ρ', f"{m['spearman']:+.3f}"),
                             ('MMRV', f"{m['mmrv']:.3f}"), ('NRegret', f"{m['nregret']:.3f}")])
        ax.legend_ = None  # Statistics remain in ax.artists; reserve the task-legend slot.
        ax.set_xlabel(LABELS[method], fontsize=15, fontweight='bold')
        ax.set_ylabel('Success Rate (z-normalized)' if method in METHODS[:2] else '',
                      fontsize=15, fontweight='bold')
        ax.margins(x=.10, y=.10)
        style.style_axis(ax)
        ax.tick_params(axis='both', labelsize=12)
        if method == 'Loss':
            ax.legend(handles=legend, loc='lower left',
                      fontsize=15, framealpha=.9, labelspacing=.25, handlelength=.9, handletextpad=.4)
        if method == 'SurVAL':
            style.emphasize_ours(ax, 'Ours')
            ax.xaxis.label.set_color(style.GREEN)

    fig, axes = style.plt.subplots(1, 4, figsize=(20, 4.7), layout='constrained')
    for ax, method in zip(axes.flat, METHODS):
        axis(ax, method)
    axis_limits['comparison'] = save_comparison(fig, output/'aggregate'/'comparison')
    for method in METHODS:
        fig, ax = style.plt.subplots(figsize=(5, 4), layout='constrained')
        axis(ax, method)
        axis_limits[method] = save_comparison(fig, output/'aggregate'/FILENAMES[method])
    for task in TASKS:
        fig, ax = style.plt.subplots(figsize=(5, .6), layout='constrained')
        bar = fig.colorbar(style.ScalarMappable(norm=epoch_norm, cmap=style.epoch_cmap(COLORS[task])),
                          cax=ax, orientation='horizontal')
        bar.set_label('Training epoch — '+NAMES[task], fontsize=10, fontweight='bold', labelpad=2)
        bar.set_ticks(EPOCHS); bar.ax.tick_params(labelsize=7, width=.5, length=2.5, pad=2)
        bar.outline.set_linewidth(.5)
        style.save(fig, output/'colorbars'/task)
    write_csv(output/'plot_data.csv', [dict(task=t, epoch=int(e), successes=labels['successes'][t][i], trials=30,
        **{k: float(v[i]) for k, v in data[t].items()}) for t in TASKS for i, e in enumerate(EPOCHS)])
    write_csv(output/'plot_data_zscore.csv', [dict(task=t, epoch=int(e),
        **{k: float(v[i]) for k, v in normalized[t].items()}) for t in TASKS for i, e in enumerate(EPOCHS)])
    write_csv(output/'selected_hp.csv', hps)
    write_csv(output/'task_metrics.csv', metrics)
    write_csv(output/'aggregate_metrics.csv', [dict(method=m, coverage=COVERAGE[m], **aggregate[m]) for m in METHODS])
    (output/'README.ko.md').write_text('\n'.join([
        f'# DROID Ta={ta} 고정 결과 그래프', '',
        'droid_policy_learning diffusion policies / task train20 / val30 / frozen epoch5 EMA policy encoder.',
        f'각 task checkpoint epoch5,10,...,50 모두 포함. Ta={ta}, worst{(ta+1)//2}, LSE=1, c=1, samples=8. k/q만 기존 task별 선택값 사용.',
        f'설정: {json.dumps(hps)}', '',
        'aggregate/comparison.pdf/png: 네 방법 비교. aggregate/val_loss, omn, val_mse, ours.pdf/png: 방법별 5×4인치. colorbars/: task별 epoch 범례. 데이터·메트릭·manifest는 실험 폴더 루트, ZIP은 실험 폴더 옆에 같은 이름으로 저장.',
        'surval_openpi와 같은 outputs/surval_figures/<model>_train20_val30_encoder<epoch>_ta<Ta>_qk_per_task 폴더 규칙. 요청한 aggregate 결과만 유지하며 per_task 그림을 새로 생성하지 않음.',
        '제목 없음. surval_openpi와 같은 Iosevka NF 실제 폰트 등록 및 PDF 임베딩. 축 라벨 15pt bold, task legend 15pt, 축 눈금 12pt, 내부 통계 15pt. PNG 500dpi / 벡터 PDF.',
        'SurVAL 표시명은 RACE (z-score): 해당 패널만 연한 초록 배경, 초록 1.6pt 테두리, 초록 x축 라벨로 강조. 모든 방법의 축 글꼴 크기와 태스크별 점 색상은 동일. CSV 내부 키는 SurVAL로 보존.',
        '태스크 legend는 Validation Loss 패널에만 표시. y축 라벨 Success Rate (z-normalized)는 Validation Loss와 Off-Manifold Norm에만 표시하며 MSE/RACE는 라벨만 생략(눈금 유지). 통합/단독 그림 모두 동일 규칙.',
        f'태스크 표시명 및 legend 순서: {" → ".join(NAMES[t] for t in TASKS)}. 색상표에도 동일한 표시명을 사용하며, CSV의 task ID apple/pan/pet은 그대로 유지.',
        '점 색상/마커: apple 파랑/사각형, pan 빨강/삼각형, pet 초록/다이아몬드. 밝은 색=초기 epoch.',
        '태스크별 성공률 및 각 방법 점수에 sample std(ddof=1) z-score 적용. 원점수 상관/선택 메트릭은 변경하지 않음.',
        '그림의 Spearman ρ/MMRV/NRegret은 원래 task별 곡선에서 계산한 뒤 3 task 동일 가중 평균. pooled 통계 미표시.',
        'MMRV는 성공률 0–1 단위. NRegret=(best-success_selected)/(best-worst+1e-12)인 무차원 값이며 퍼센트가 아님. raw regret_pp는 CSV만 보존.',
        'x/y 한계를 고정하지 않음: 패널 데이터 범위+10% margin에서 시작하고, 통계/범례가 점을 가리는 패널만 상단을 확장. 실제 범위는 manifest의 axis_limits에 저장.',
        '같은 성공률을 쓰므로 데이터 기반 y 범위가 일부 패널에서 같을 수 있음. 축 범위 차이는 데이터 또는 주석 여백만 반영.',
        'plot_data.csv의 baseline 점수는 음수화된 plotting 값. 원본 baseline 값은 이를 한 번 부호 반전하면 복원됨.',
        'Loss는 legacy first1600 rows. OMN은 Off_Manifold_Norm_PolicyReference: SurVAL과 같은 epoch5 encoder이며 H8/full-val30 baseline 값을 유지.',
        'MSE도 H8/full-val30. original OMN 및 DINO OMN은 제외. x 라벨은 −Off-Manifold Norm (z-score), 다른 축과 같은 15pt bold. 실제 좌표는 기존처럼 -OMN의 within-task z-score.',
        '학습 config의 dataset_names에 droid와 task val30 데이터 경로가 포함되며, 해당 task split manifest의 train20/val30을 확인함.',
        '동률은 가장 이른 epoch 선택. 최고점 동률 epoch는 task_metrics.csv에 보존; Ta3 pan은 epoch30/35/40/45/50 동률이므로 NRegret=0 해석에 주의.',
        '회귀선은 pooled z-score 점들의 시각적 추세선일 뿐 그림 내부 통계의 계산 근거가 아님.',
        'GT를 보고 선택한 HP의 사후 분석. Bootstrap/유의성 검정/재튜닝/새 cache 생성 없음. 이전 집계/기본값 변경 없음.',
        '기존 pi05 렌더러의 순수 스타일 함수만 재사용; pi05 데이터/HP/메트릭 정의는 사용하지 않음.',
        '재현: PYTHONPATH=src:scripts MPLCONFIGDIR=/tmp/droid_ta8_plot_mpl CUDA_VISIBLE_DEVICES=\'\' '
        f'/venv/droid_policy/bin/python scripts/plot_droid_ta8_results.py --ta {ta} --output <새 출력 디렉터리>', '']))
    pdfs = sorted(output.rglob('*.pdf'))
    assert len(pdfs) == len(list(output.rglob('*.png'))) == 8
    assert {p.stem for p in (output/'aggregate').glob('*.pdf')} == {'comparison', 'val_loss', 'omn', 'val_mse', 'ours'}
    for pdf in pdfs:
        blob = pdf.read_bytes()
        names = re.findall(rb'/BaseFont /([^\s/]+)', blob)
        assert names and all(b'Iosevka' in n for n in names) and b'/FontFile2' in blob, pdf
    for path, digest in sources.items():
        assert sha256(path) == digest, f'Source changed during plotting: {path}'
    save_json(output/'manifest.json', dict(status='passed', variant=f'Ta{ta}_task_specific_k_quantile',
        reference_epoch=5, epochs=EPOCHS.tolist(), exclusions=[], hp=hps, datasets=datasets,
        coverage=dict(tasks=3, checkpoints_per_task=10, trials_per_checkpoint=30, points_per_method=30, methods=4),
        source_sha256=sources, font_files={str(p): sha256(p) for p in fonts},
        output_sha256={str(p.relative_to(output)): sha256(p) for p in output.rglob('*') if p.is_file()},
        pdf_count=8, png_count=8, all_pdf_fonts_embedded_iosevka=True, aggregate=aggregate,
        normalization='within-task, within-method z-score; ddof=1; metrics on original task curves',
        statistics='equal-weight task means only', displayed_statistics=['Spearman ρ', 'MMRV', 'NRegret'],
        mmrv_units='0-1', nregret_units='dimensionless, not percent', axis_limits=axis_limits,
        axes='per-panel data-driven limits with bounded annotation clearance',
        omn_axis_label=LABELS['Off_Manifold_Norm_PolicyReference'], omn_plotted_values='negative OMN, within-task z-score (ddof=1)',
        ours_axis_label=LABELS['SurVAL'], ours_highlight='pale green panel, green 1.6pt frame and x-axis label',
        axis_label_fontsize=15, task_legend_fontsize=15, tick_label_fontsize=12, statistics_fontsize=15,
        font_family='Iosevka NF', png_dpi=500, pdf_fonttype=42,
        standalone_size_inches=[5, 4], comparison_size_inches=[20, 4.7],
        method_pdf_files={m: 'aggregate/'+stem+'.pdf' for m, stem in FILENAMES.items()},
        task_display_names=NAMES, legend_order=[NAMES[t] for t in TASKS],
        task_legend_methods=['Loss'], success_rate_axis_label='Success Rate (z-normalized)',
        success_rate_axis_label_methods=list(METHODS[:2]),
        baseline_coverage=COVERAGE, bootstrap=False, posthoc=True))
    archive = output.with_name(output.name+'.zip')
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(output.rglob('*')):
            if path.is_file():
                bundle.write(path, path.relative_to(output))
        bundle.write(Path(__file__), 'plot_droid_ta8_results.py')
        bundle.write(STYLE, 'reference/plot_pi05_ta12_results.py')
    print(json.dumps(dict(output=str(output), archive=str(archive), aggregate=aggregate)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ta', type=int, choices=range(1, 16), default=8)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or REPO/'outputs/surval_figures'/f'droid_train20_val30_encoder5_ta{args.ta}_qk_per_task'
    render(output.resolve(), args.ta)
