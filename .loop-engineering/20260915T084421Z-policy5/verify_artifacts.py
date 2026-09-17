"""Read-only epoch5 report and epoch25/50 preservation verification."""
import csv
import json
from pathlib import Path

import numpy as np

from tune_droid_hp import sha256

root = Path('outputs/droid_policy5_val30_20260915')
previous = [Path(f'outputs/droid_policy{epoch}_val30_20260915') for epoch in (25, 50)]
snapshot = json.loads(Path(__file__).with_name('preserved_previous_sha256.json').read_text())
assert {str(p) for directory in previous for p in directory.rglob('*') if p.is_file()} == set(snapshot)
for path, digest in snapshot.items():
    assert sha256(path) == digest, path
print('All', len(snapshot), 'epoch25/50 files preserved byte-for-byte PASS')
proof = json.loads((root/'verification.json').read_text())
assert proof['complete'] and proof['fixed']['reference_epoch'] == 5
assert proof['retuned_max_score_ties'] == {'pan': [30, 35, 40, 50]}
before_note = json.loads(Path(__file__).with_name('pre_note_csv_sha256.json').read_text())
for name, digest in before_note.items():
    assert sha256(root/name) == digest, name
assert 'np.argmax' in (root/'REPORT.ko.md').read_text()
print('Tie annotation and all numeric CSVs unchanged PASS')
assert proof['script_sha256'] == sha256('scripts/report_droid_policy_reference.py')
for name, digest in proof['outputs_sha256'].items():
    assert sha256(root/name) == digest, name
for path, digest in proof['source_sha256'].items():
    assert sha256(path) == digest, path

def read_csv(name):
    with (root/name).open() as stream:
        return list(csv.DictReader(stream))

for name, count in [('aggregate_metrics.csv', 20), ('baseline_aggregate_metrics.csv', 6),
                    ('fixed_grid_metrics.csv', 8640), ('baseline_checkpoint_metrics.csv', 30),
                    ('selected_metrics.csv', 60), ('checkpoint_scores.csv', 600)]:
    assert len(read_csv(name)) == count, name
    print(name, count, 'PASS')
selected = read_csv('selected_metrics.csv')
for row in read_csv('aggregate_metrics.csv'):
    group = [r for r in selected if all(r[k] == row[k] for k in ('scope', 'regime', 'selection'))]
    assert len(group) == 3 and {r['task'] for r in group} == {'apple', 'pan', 'pet'}
    for key in ('nregret', 'mmrv', 'spearman', 'balanced_loss', 'regret_pp'):
        np.testing.assert_allclose(float(row[key]), np.mean([float(r[key]) for r in group]), rtol=0, atol=1e-12)
wanted = {(r['representation'], r['task'], int(r['hp_index']), epoch)
          for r in selected if r['representation'] in ('Policy5', 'DINO')
          for epoch in range(5, 51, 5)}
canonical = read_csv('canonical_audit.csv')
actual = {(r['representation'], r['task'], int(r['hp_index']), int(r['epoch'])) for r in canonical}
assert actual == wanted and len(actual) == len(canonical) == proof['canonical_scores']
assert proof['canonical_metrics_and_selection_unchanged'] and proof['threshold_max_abs_error'] == 0.
assert json.loads((root/'run_status.json').read_text())['state'] == 'complete'
for job in json.loads((root/'run_manifest.json').read_text())['jobs']:
    result = json.loads((Path(job['job_dir'])/'eval/verification.json').read_text())
    assert result['complete'] and result['num_demos'] == 30 and result['feature_dim'] == 1024
    assert result['fallback_rows'] == 0
    assert result['signature']['reference_metadata']['reference_epoch'] == 5
print('Full canonical coverage', len(canonical), 'and macro means PASS')
print('Source/output checksums and epoch5 identity PASS; max score error', proof['canonical_max_abs_error'])
