"""Read-only final artifact verification; run from the SurVAL repository."""
import csv
import json
from pathlib import Path

from tune_droid_hp import sha256

root = Path('outputs/droid_policy25_val30_20260915')
proof = json.loads((root/'verification.json').read_text())
assert proof['complete']
assert proof['script_sha256'] == sha256('scripts/report_droid_policy_reference.py')
for name, digest in proof['outputs_sha256'].items():
    assert sha256(root/name) == digest, name
for path, digest in proof['source_sha256'].items():
    assert sha256(path) == digest, path
for name, count in [('aggregate_metrics.csv', 12), ('baseline_aggregate_metrics.csv', 5),
                    ('fixed_grid_metrics.csv', 8640), ('baseline_checkpoint_metrics.csv', 30),
                    ('canonical_audit.csv', 310)]:
    with (root/name).open() as stream:
        assert len(list(csv.DictReader(stream))) == count, name
    print(name, count, 'PASS')
assert json.loads((root/'run_status.json').read_text())['state'] == 'complete'
assert proof['threshold_max_abs_error'] == 0.
assert proof['canonical_metrics_and_selection_unchanged']
print('Source/output SHA256, counts, original metric/selection checks PASS')
