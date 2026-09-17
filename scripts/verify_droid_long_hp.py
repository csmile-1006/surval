"""Verify published val30 optima and export selected per-checkpoint scores."""

import argparse
import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from cache_droid_datasets import REPO, save_json
from tune_droid_hp import sha256
from report_droid_hp import write_csv
from tune_droid_long_hp import matrix

sys.path.insert(0, str(REPO/'src'))
import numpy as np
from surval.droid_long_tuning import hp_grid, metric_arrays


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=REPO/'outputs/droid_c1_long_20260913')
    args=parser.parse_args()
    root=args.root/'tuning'
    labels_path=REPO/'outputs/droid_real_world_20260913/success_counts.json'
    labels=json.loads(labels_path.read_text())
    verified=json.loads((root/'verification.json').read_text())
    assert verified['complete'] and verified['labels_sha256']==sha256(labels_path)
    assert verified['all_metrics_original_definition_verified']
    assert sha256(root/'selected_metrics.csv')==verified['selected_csv_sha256']
    grid=hp_grid();metrics={};scores={};epochs={};cache_count=0
    manifest=matrix(SimpleNamespace(root=args.root.resolve(),output_root=root.resolve()))
    for job in manifest['jobs']:
        task=job['task'];output=Path(job['job_dir'])/'eval'
        proof=json.loads((output/'verification.json').read_text())
        assert proof['complete'] and proof['signature']['fixed']['scale_multiplier']==1
        for path,expected in proof['signature']['code_sha256'].items():
            assert sha256(REPO/path)==expected, path
        for path,expected in proof['signature']['cache_sha256'].items():
            assert sha256(path)==expected, path
            cache_count+=1
        assert sha256(output/'scores.npz')==proof['scores.npz_sha256']
        assert sha256(output/'scales.npz')==proof['scales.npz_sha256']
        with np.load(output/'scores.npz') as data:
            scores[task]=data['scores'];epochs[task]=data['epochs']
        np.testing.assert_array_equal(epochs[task],labels['epoch_order'])
        actual=np.array(labels['successes'][task])/labels['trials_per_checkpoint']
        metrics[task]=metric_arrays(actual,scores[task])
        with np.load(output/'metrics.npz') as data:
            for key,value in metrics[task].items():
                np.testing.assert_allclose(value,data[key],atol=1e-12,rtol=0,equal_nan=True)
    with (root/'selected_metrics.csv').open() as stream:
        selected=list(csv.DictReader(stream))
    ta=np.array([h['ta'] for h in grid]);export=[];objectives=set()
    for row in selected:
        domain,scope=row['domain'].split('/');target=row['target'];i=int(row['hp_index']);task=row['task']
        mask=np.ones(len(grid),bool) if scope=='all' else ta<=8 if scope=='short' else ta>8
        assert mask[i] and grid[i]['scale_multiplier']==float(row['scale_multiplier'])==1
        for key,value in grid[i].items():
            assert abs(float(row[key])-value)<=1e-12,(key,row[key],value)
        if domain in metrics:
            mask &= metrics[domain]['eligible'];objective=metrics[domain][target]
        else:
            mask &= np.logical_and.reduce([m['eligible'] for m in metrics.values()])
            objective=(np.max([m['balanced_loss'] for m in metrics.values()],axis=0)
                       if domain=='common_minimax' else np.mean([m[target] for m in metrics.values()],axis=0))
        if target=='spearman':objective=-objective
        assert mask[i] and objective[i]==objective[mask].min(), row
        objectives.add((row['domain'],target))
        for key in ('nregret','mmrv','spearman','balanced_loss','selected_success_rate'):
            assert abs(float(row[key])-metrics[task][key][i])<=1e-12
        assert int(row['selected_epoch'])==epochs[task][np.argmax(scores[task][i])]
        for col,epoch in enumerate(epochs[task]):
            export.append(dict(domain=row['domain'],target=target,task=task,hp_index=i,**grid[i],
                epoch=int(epoch),surval_score=float(scores[task][i,col]),
                successes=labels['successes'][task][col],trials=labels['trials_per_checkpoint'],
                success_rate=labels['successes'][task][col]/labels['trials_per_checkpoint']))
    assert len(objectives)==51 and cache_count==30 and len(selected)==81
    # Verify all-metric CSV coverage/identity against its saved arrays, not only winners.
    seen={task:set() for task in metrics}
    with (root/'all_metrics.csv').open() as stream:
        for row in csv.DictReader(stream):
            task=row['task'];i=int(row['hp_index'])
            assert i not in seen[task] and float(row['scale_multiplier'])==1
            seen[task].add(i)
            for key,value in grid[i].items():assert abs(float(row[key])-value)<=1e-12
            for key in ('nregret','mmrv','spearman','balanced_loss'):
                a,b=float(row[key]),metrics[task][key][i]
                assert (np.isnan(a) and np.isnan(b)) or abs(a-b)<=1e-12
    assert all(ids==set(range(len(grid))) for ids in seen.values())
    preparation=json.loads((args.root/'cache_verification.json').read_text())
    assert preparation['complete'] and preparation['checkpoints']==30
    for path,expected in preparation['historical_cache_sha256'].items():assert sha256(path)==expected
    write_csv(root/'selected_checkpoint_scores.csv',export)
    save_json(root/'integrity_verification.json',dict(complete=True,independent_objectives=51,
        full_metric_csv_rows=len(grid)*3,selected_metric_rows=81,selected_checkpoint_rows=len(export),
        source_long_caches_unchanged=30,source_historical_caches_unchanged=30,
        labels_sha256=sha256(labels_path),all_metrics_sha256=sha256(root/'all_metrics.csv'),
        checkpoint_csv_sha256=sha256(root/'selected_checkpoint_scores.csv')))
    print('PASS:51 optima,483840 metric rows,810 selected checkpoint rows; all sources unchanged')


if __name__=='__main__':
    main()
