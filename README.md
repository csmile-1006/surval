# surval

Shared library for sequential-validation-from-cache metrics and state-conditional threshold tooling.

This library is consumed by:

- `surval_openpi` — DROID/openpi sequential-validation pipeline
- `custom-robomimic` — gripper / dex / humanoid sequential-validation wrappers

## Install

```bash
# From a sibling directory of this repo, in your project's venv:
pip install -e /home/changyeon/workspace/surval[encoder]
```

Extras:

- `encoder` — pulls in `torch`, needed by `surval.local_threshold.encoder.FrozenImageEncoder`
- `db` — pulls in `faiss-cpu`, used lazily inside `surval.local_threshold.database`
- `sanity` — pulls in `matplotlib`, used by the plotting helpers in `surval.local_threshold.sanity`
- `full` — all of the above
- `dev` — `pytest` for the test suite

## Public API

```python
from surval.sequential_validate import add_common_args, run, validate_common_args
from surval.local_threshold.threshold import LocalThresholdMap
from surval.local_threshold.config import LocalThresholdConfig
from surval.local_threshold.database import StateDatabase, StateRecords
```

The `sequential_validate` module exposes the entry-point trio (`add_common_args`,
`validate_common_args`, `run`) consumed by per-action-space wrapper scripts.

## Tests

```bash
cd /home/changyeon/workspace/surval
pip install -e .[dev,encoder]
pytest tests/ -v
```
